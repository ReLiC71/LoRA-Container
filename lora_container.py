"""lora_container: put cover images and metadata inside a .safetensors header.

  lora_container.py embed MODEL [COVER ...] -o OUT
  lora_container.py meta MODEL -m key=value --author NAME -o OUT
  lora_container.py show MODEL
  lora_container.py verify MODEL [MODEL ...]
  lora_container.py extract MODEL OUTDIR
  lora_container.py strip MODEL -o OUT
  lora_container.py batch FOLDER --out FOLDER2
  lora_container.py gui

Only the JSON header changes. Tensor data is copied byte for byte and checked.
Needs Python 3.8+. Pillow is needed to add images. The format is in SPEC.md.

Exit codes: 0 done, 1 a check failed, 2 bad input, 3 refused (would overwrite).
"""
import argparse, base64, binascii, datetime, hashlib, json, os, re, struct, sys, zlib

__version__ = "1.0.2"
FORMAT = "1"
MAX_COVERS = 6  # the convention: one cover (preview and icon) and five gallery images
MAX_HEADER = 100_000_000
IDENTITY_KEYS = ("model_sha256_original", "tensor_data_sha256")
OWN_KEYS = ("cover_count", "cover_format", "cm_info") + IDENTITY_KEYS
IMAGE_EXT = (".jpeg", ".jpg", ".png", ".webp", ".bmp", ".gif")
EXIT_OK, EXIT_FAILED, EXIT_INPUT, EXIT_REFUSED = 0, 1, 2, 3
CHUNK = 1 << 22
DATA_PREFIX = "data:image/jpeg;base64,"
FIELDS = {  # tool name -> cm-info name, is a list
    "title": ("ModelName", False),
    "author": ("AuthorUsername", False),
    "base_model": ("BaseModel", False),
    "trigger_words": ("TrainedWords", True),
    "tags": ("Tags", True),
    "description": ("ModelDescription", False),
    "model_type": ("ModelType", False),
}

# cm-info fields that are free text, where a leading slash is not a path
FREE_TEXT = ("ModelName", "ModelDescription", "VersionName", "VersionDescription", "UserTitle", "Tags", "TrainedWords")
_PATH_START = re.compile(r"(?:/(?!/)|~[/\\]|[A-Za-z]:[/\\]|\\\\|file:)")
_PATH_INSIDE = re.compile(r"(?:^|[\s\"'=:,\[{(])(?:/(?:home|Users|Volumes|private|root|mnt|media|run|var|tmp|opt|srv|data|content|workspace|kaggle)/"
                          r"|~[/\\]|[A-Za-z]:[/\\]|\\\\[^\\]|file://)")

try:
    from blake3 import blake3 as _blake3
except Exception:  # optional
    _blake3 = None


class Error(Exception):
    def __init__(self, msg, code=EXIT_INPUT):
        super().__init__(msg)
        self.code = code


# ---------- reading ----------

def read_header(path):
    """Return (header length in bytes, header as a dict)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if not 0 < n <= MAX_HEADER or 8 + n > size:
                raise ValueError("bad header length")
            hdr = json.loads(f.read(n).decode("utf-8"))
        if not isinstance(hdr, dict) or not isinstance(hdr.get("__metadata__", {}), dict):
            raise ValueError("header is not a JSON object")
    except OSError as e:
        raise Error("cannot read %s: %s" % (path, e.strerror or e))
    except (ValueError, struct.error) as e:
        raise Error("%s is not a safetensors file (%s)" % (path, e))
    return n, hdr


def _hash(path, skip=0, prefix=b"", full=False):
    sha = hashlib.sha256(prefix)
    crc = zlib.crc32(prefix) if full else 0
    b3 = _blake3(prefix) if full and _blake3 else None
    with open(path, "rb") as f:
        f.seek(skip)
        for b in iter(lambda: f.read(CHUNK), b""):
            sha.update(b)
            if full:
                crc = zlib.crc32(b, crc)
                if b3:
                    b3.update(b)
    out = {"sha256": sha.hexdigest()}
    if full:
        out["crc32"] = crc & 0xFFFFFFFF
        out["blake3"] = b3.hexdigest() if b3 else None
    return out


def file_hashes(path):
    """SHA-256, CRC32 and (if the blake3 package is present) BLAKE3 of the whole file."""
    return _hash(path, full=True)


def tensor_hash(path):
    """SHA-256 of everything after the header: the stable identity."""
    n, _ = read_header(path)
    return _hash(path, skip=8 + n)["sha256"]


def sm_hashes(h):
    """Hashes as Stability Matrix writes them in a .cm-info.json."""
    out = {"SHA256": h["sha256"].upper()}
    if h.get("crc32") is not None:
        out["CRC32"] = "%08X" % int.from_bytes(h["crc32"].to_bytes(4, "big"), "little")
    if h.get("blake3"):
        out["BLAKE3"] = h["blake3"].upper()
    return out


def decode_cover(value):
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise Error("a cover is not valid base64", EXIT_FAILED)


def get_covers(meta):
    """Embedded covers as a list of JPEG bytes. cover_1 (the preview and icon) is first."""
    try:
        count = int(meta.get("cover_count", 0))
    except ValueError:
        raise Error("cover_count is not a number", EXIT_FAILED)
    return [decode_cover(meta["cover_%d" % i]) for i in range(1, count + 1) if "cover_%d" % i in meta]


def get_cm_info(meta):
    if "cm_info" not in meta:
        return None
    try:
        j = json.loads(meta["cm_info"])
    except ValueError:
        raise Error("cm_info is not valid JSON", EXIT_FAILED)
    return j if isinstance(j, dict) else None


# ---------- writing ----------

def encode_cover(src, side=512, quality=46):
    """A cover as base64 JPEG. src is an image path, or JPEG bytes to keep as they are."""
    if isinstance(src, (bytes, bytearray)):
        if bytes(src[:2]) != b"\xff\xd8":
            raise Error("cover bytes are not a JPEG")
        return base64.b64encode(bytes(src)).decode()
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise Error("Pillow is needed to add images: pip install Pillow")
    import io
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            k = side / min(w, h)
            if k < 1:
                lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                im = im.resize((max(1, round(w * k)), max(1, round(h * k))), lanczos)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
    except Error:
        raise
    except Exception as e:
        raise Error("cannot read image %s: %s" % (src, e))
    return base64.b64encode(buf.getvalue()).decode()


def header_bytes(hdr, pad=True, ascii_only=False):
    try:
        hb = json.dumps(hdr, separators=(",", ":"), ensure_ascii=ascii_only).encode("utf-8")
    except UnicodeEncodeError:
        hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    if pad:
        hb += b" " * ((-len(hb)) % 8)
    return hb


def _write(src, dst, hb, n_old, tensor):
    tmp = dst + ".tmp"
    try:
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            fo.write(struct.pack("<Q", len(hb)))
            fo.write(hb)
            fi.seek(8 + n_old)
            for b in iter(lambda: fi.read(CHUNK), b""):
                fo.write(b)
    except OSError as e:
        raise Error("cannot write %s: %s" % (tmp, e.strerror or e))
    if tensor_hash(tmp) != tensor:
        raise Error("tensor data changed while writing: nothing replaced, bad copy left at " + tmp, EXIT_FAILED)
    os.replace(tmp, dst)


def _target(model, out, force):
    dst = out or model
    same = os.path.exists(dst) and os.path.samefile(dst, model)
    if same and not force:
        raise Error("this would overwrite %s: give -o OUT, or --force to change it in place" % model, EXIT_REFUSED)
    if not same and os.path.exists(dst) and not force:
        raise Error("%s exists: use --force to overwrite it" % dst, EXIT_REFUSED)
    return dst


def sidecar_path(model):
    return os.path.splitext(model)[0] + ".cm-info.json"


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        return j if isinstance(j, dict) else None
    except (OSError, ValueError):
        return None


def default_cm_info(model):
    name = os.path.splitext(os.path.basename(model))[0]
    return {
        "ModelId": None, "ModelName": name, "ModelDescription": "", "Nsfw": False, "Tags": [],
        "ModelType": "LORA", "VersionId": None, "VersionName": "", "VersionDescription": "",
        "AuthorUsername": "", "BaseModel": "", "RemoteFileName": os.path.basename(model),
        "RemoteFileId": None, "FileMetadata": {"fp": None, "size": None, "format": "SafeTensor"},
        "ImportedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "Hashes": {}, "TrainedWords": [],
        "Stats": {"favoriteCount": 0, "commentCount": 0, "thumbsUpCount": 0, "downloadCount": 0,
                  "ratingCount": 0, "rating": 0},
        "UserTitle": None, "ThumbnailImageUrl": None, "InferenceDefaults": None, "Source": None,
        "SourceUrl": None,
    }


def scrub_paths(obj, trail=""):
    """Blank every value in a cm-info that is a path on this machine. Returns the names of the fields blanked."""
    found = []
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            name = trail + "." + k if trail else k
            if k in FREE_TEXT:
                continue
            if isinstance(v, str) and _PATH_START.match(v):
                obj[k] = None
                found.append(name)
            else:
                found += scrub_paths(v, name)
    elif isinstance(obj, list):
        keep = [v for v in obj if not (isinstance(v, str) and _PATH_START.match(v))]
        if len(keep) != len(obj):
            obj[:] = keep
            found.append(trail)
        for v in obj:
            found += scrub_paths(v, trail)
    return found


def local_paths(md):
    """Names of the metadata keys whose value seems to hold a path on somebody's machine."""
    return sorted(k for k, v in md.items() if not re.fullmatch(r"cover_\d+", k) and _PATH_INSIDE.search(v))


def _as_list(v):
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _changelog(j, text, date=None):
    d = (date or datetime.date.today()).strftime("%d/%m/%y")
    lines = [x for x in str(j.get("Changelog") or "").splitlines() if x.strip()]
    lines.append("%d. %s. %s" % (len(lines) + 1, d, text))
    j["Changelog"] = "\n".join(lines)


def _update_sidecar(model, dst, original, embedded, text, date=None):
    """Bring the .cm-info.json beside dst up to date with the file as written."""
    path = sidecar_path(dst)
    j = _load_json(path) or _load_json(sidecar_path(model)) or dict(embedded or default_cm_info(dst))
    j.setdefault("OriginalHashes", original)
    j["Hashes"] = sm_hashes(file_hashes(dst))
    j["CurrentHash"] = dict(j["Hashes"])
    _changelog(j, text, date)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return path


def embed(model, covers=None, out=None, force=False, replace=False, meta=None, unset=None,
          fields=None, cm_info=True, sidecar=False, side=512, quality=46, dry_run=False):
    """Write covers and metadata into the header of model. Returns a report dict.

    covers: list of image paths or JPEG bytes; None leaves any existing covers alone.
    meta / unset: plain __metadata__ keys to set / remove.
    fields: common fields (see FIELDS), stored in the embedded cm-info.
    """
    n, hdr = read_header(model)
    dst = _target(model, out, force)
    md = hdr.setdefault("__metadata__", {})
    had = "cover_count" in md
    for k in list(meta or {}) + list(unset or []):
        if k in OWN_KEYS or re.fullmatch(r"cover_\d+", k):
            raise Error("%s is a reserved key" % k)
    if fields and not cm_info:
        raise Error("the common fields are stored in the cm-info: do not combine them with --no-cm-info")
    if covers is not None:
        if not covers:
            raise Error("no cover images given")
        if had and not replace:
            raise Error("%s already has covers: use --replace to change them" % model, EXIT_REFUSED)

    tensor = tensor_hash(model)
    if md.get("tensor_data_sha256", tensor) != tensor:
        raise Error("%s: tensor data does not match its recorded tensor_data_sha256" % model, EXIT_FAILED)
    old_cm = get_cm_info(md)
    disk_cm = _load_json(sidecar_path(model))
    if "model_sha256_original" in md:
        original = (old_cm or {}).get("Hashes") or (disk_cm or {}).get("OriginalHashes") or {}
        if original.get("SHA256", "").lower() != md["model_sha256_original"]:
            original = {"SHA256": md["model_sha256_original"].upper()}
    else:
        h = file_hashes(model)
        md["model_sha256_original"] = h["sha256"]
        original = sm_hashes(h)
    md["tensor_data_sha256"] = tensor

    if covers is not None:
        for k in [k for k in md if re.fullmatch(r"cover_\d+", k)]:
            del md[k]
        md["cover_count"] = str(len(covers))
        for i, c in enumerate(covers, 1):
            md["cover_%d" % i] = encode_cover(c, side, quality)
    if "cover_count" in md:
        md["cover_format"] = FORMAT
    for k in unset or []:
        md.pop(k, None)
    for k, v in (meta or {}).items():
        md[k] = str(v)

    embedded, scrubbed = old_cm, []
    if cm_info:
        embedded = dict(old_cm or disk_cm or default_cm_info(model))
        for k, v in (fields or {}).items():
            if k not in FIELDS:
                raise Error("unknown field %s" % k)
            name, is_list = FIELDS[k]
            embedded[name] = _as_list(v) if is_list else str(v)
        embedded["Hashes"] = original
        embedded["OriginalHashes"] = original
        embedded.pop("CurrentHash", None)
        embedded["ThumbnailImageUrl"] = None  # a local path: no use, and private, inside the file
        scrubbed = scrub_paths(embedded)
        md["cm_info"] = json.dumps(embedded, separators=(",", ":"), ensure_ascii=False)

    hb = header_bytes(hdr)
    if len(hb) > MAX_HEADER:
        raise Error("the header would be %.1f MiB; safetensors readers stop at %.1f MiB: use fewer or smaller covers"
                    % (len(hb) / 2**20, MAX_HEADER / 2**20))
    rep = {"model": model, "out": dst, "covers": int(md.get("cover_count", 0)), "header_old": n,
           "header_new": len(hb), "size_new": os.path.getsize(model) - n + len(hb),
           "tensor_data_sha256": tensor, "model_sha256_original": md["model_sha256_original"],
           "dry_run": dry_run, "sidecar": None, "scrubbed": scrubbed, "local_paths": local_paths(md)}
    if not dry_run:
        _write(model, dst, hb, n, tensor)
        if sidecar:
            text = "Embedded cover image and gallery" if covers is not None else "Edited embedded metadata"
            rep["sidecar"] = _update_sidecar(model, dst, original, embedded, text)
    return rep


def strip(model, out=None, force=False, sidecar=False, dry_run=False):
    """Undo: remove everything this tool adds. Reports whether the result is the original file."""
    n, hdr = read_header(model)
    dst = _target(model, out, force)
    md = hdr.get("__metadata__", {})
    original = md.get("model_sha256_original")
    tensor = tensor_hash(model)
    found = [k for k in md if k in OWN_KEYS or re.fullmatch(r"cover_\d+", k)]
    for k in found:
        del md[k]
    shapes = [hdr]
    if "__metadata__" in hdr and not md:
        bare = dict(hdr)
        del bare["__metadata__"]
        shapes = [bare, hdr]
    hb, exact = header_bytes(shapes[0]), None
    if original:
        exact = False
        seen = set()
        for shape in shapes:
            for pad in (True, False):
                for asc in (False, True):
                    cand = header_bytes(shape, pad, asc)
                    if cand in seen:
                        continue
                    seen.add(cand)
                    if _hash(model, skip=8 + n, prefix=struct.pack("<Q", len(cand)) + cand)["sha256"] == original:
                        hb, exact = cand, True
                        break
                if exact:
                    break
            if exact:
                break
    rep = {"model": model, "out": dst, "removed": len(found), "exact": exact, "dry_run": dry_run,
           "tensor_data_sha256": tensor, "sidecar": None}
    if not dry_run:
        _write(model, dst, hb, n, tensor)
        if sidecar:
            rep["sidecar"] = _update_sidecar(model, dst, {"SHA256": original.upper()} if original else {}, None,
                                             "Removed embedded covers and metadata")
    return rep


def verify(model):
    """Check a file against what its header records. status: ok, plain (nothing embedded) or failed."""
    n, hdr = read_header(model)
    md = hdr.get("__metadata__", {})
    problems, notes = [], []
    if not any(k in md for k in OWN_KEYS):
        return {"model": model, "status": "plain", "problems": [], "notes": ["nothing embedded"]}
    tensor = _hash(model, skip=8 + n)["sha256"]
    if "tensor_data_sha256" not in md:
        problems.append("no tensor_data_sha256 recorded")
    elif md["tensor_data_sha256"] != tensor:
        problems.append("tensor data does not match tensor_data_sha256")
    else:
        notes.append("tensor data matches")
    try:
        covers = get_covers(md)
        count = int(md.get("cover_count", 0))
        if len(covers) != count:
            problems.append("cover_count is %d but %d covers are present" % (count, len(covers)))
        bad = [i for i, c in enumerate(covers, 1) if c[:2] != b"\xff\xd8"]
        if bad:
            problems.append("not JPEG: cover " + ", ".join(map(str, bad)))
        notes.append("%d covers" % len(covers))
    except Error as e:
        problems.append(str(e))
    if md.get("cover_format", FORMAT) != FORMAT:
        notes.append("cover_format %s is newer than this tool (%s)" % (md["cover_format"], FORMAT))
    try:
        cm = get_cm_info(md)
        if cm is not None:
            sha = (cm.get("Hashes") or {}).get("SHA256", "").lower()
            if sha and md.get("model_sha256_original") and sha != md["model_sha256_original"]:
                problems.append("cm_info Hashes do not match model_sha256_original")
            notes.append("cm-info present")
    except Error as e:
        problems.append(str(e))
    if n % 8:
        notes.append("header is not padded to 8 bytes")
    if local_paths(md):
        notes.append("possible local path in: " + ", ".join(local_paths(md)))
    return {"model": model, "status": "failed" if problems else "ok", "problems": problems, "notes": notes}


def extract(model, outdir, gnames=False):
    """Write the covers, and the cm-info if there is one, to outdir. Returns the paths.

    gnames: name them NAME.preview.jpeg (Stability Matrix's name), then NAME.gallery1.jpeg ... (this project's own).
    """
    _, hdr = read_header(model)
    md = hdr.get("__metadata__", {})
    stem = os.path.splitext(os.path.basename(model))[0]
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, data in enumerate(get_covers(md), 1):
        if gnames:
            name = stem + (".preview.jpeg" if i == 1 else ".gallery%d.jpeg" % (i - 1))
        else:
            name = "cover_%d.jpg" % i
        p = os.path.join(outdir, name)
        with open(p, "wb") as f:
            f.write(data)
        paths.append(p)
    cm = get_cm_info(md)
    if cm is not None:
        cm["Hashes"] = sm_hashes(file_hashes(model))
        cm["CurrentHash"] = dict(cm["Hashes"])
        if paths:
            cm["ThumbnailImageUrl"] = os.path.abspath(paths[0])
        p = os.path.join(outdir, stem + ".cm-info.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cm, f, ensure_ascii=False)
        paths.append(p)
    return paths


def show(model):
    n, hdr = read_header(model)
    md = hdr.get("__metadata__", {})
    return {"model": model, "header_bytes": n, "tensors": len(hdr) - ("__metadata__" in hdr),
            "covers": [len(md[k]) for k in sorted((k for k in md if re.fullmatch(r"cover_\d+", k)),
                                                  key=lambda k: int(k[6:]))],
            "cm_info": get_cm_info(md),
            "metadata": {k: v for k, v in md.items() if k != "cm_info" and not re.fullmatch(r"cover_\d+", k)}}


def _natural(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def find_covers(model):
    """Loose images that belong to a model, preview first.

    NAME.preview.EXT then NAME.gallery1.EXT ...; else the images in a folder called NAME;
    else NAME.EXT alone.
    """
    base = os.path.splitext(model)[0]
    folder, stem = os.path.split(base)
    names = os.listdir(folder or ".")
    def pick(pattern):
        hits = [x for x in names if re.fullmatch(pattern, x, re.I) and x.lower().endswith(IMAGE_EXT)]
        return [os.path.join(folder, x) for x in sorted(hits, key=_natural)]
    preview = pick(re.escape(stem) + r"\.preview\.\w+")[:1]
    if preview:
        return preview + pick(re.escape(stem) + r"\.gallery\d+\.\w+")
    if os.path.isdir(base):
        return [os.path.join(base, x) for x in sorted(os.listdir(base), key=_natural) if x.lower().endswith(IMAGE_EXT)]
    return pick(re.escape(stem) + r"\.\w+")[:1]


def batch(folder, out=None, force=False, replace=False, sidecar=False, side=512, quality=46, dry_run=False,
          max_covers=MAX_COVERS):
    """Embed for every .safetensors in folder, using find_covers and any NAME.meta.json. Returns a list of results."""
    if not os.path.isdir(folder):
        raise Error("%s is not a folder" % folder)
    if out and not dry_run:
        os.makedirs(out, exist_ok=True)
    results = []
    for name in sorted(os.listdir(folder), key=_natural):
        if not name.lower().endswith(".safetensors"):
            continue
        model = os.path.join(folder, name)
        res = {"model": model, "status": "ok", "message": ""}
        try:
            covers = find_covers(model)
            if max_covers:
                covers = covers[:max_covers]
            _, hdr = read_header(model)
            if not covers:
                res.update(status="skipped", message="no images found")
            elif "cover_count" in hdr.get("__metadata__", {}) and not replace:
                res.update(status="skipped", message="already has covers")
            else:
                extra = _load_json(os.path.splitext(model)[0] + ".meta.json") or {}
                rep = embed(model, covers, os.path.join(out, name) if out else None, force, replace,
                            {k: v for k, v in extra.items() if k not in FIELDS},
                            fields={k: v for k, v in extra.items() if k in FIELDS},
                            sidecar=sidecar, side=side, quality=quality, dry_run=dry_run)
                res["message"] = "%d covers, %.1f KiB added" % (rep["covers"], (rep["header_new"] - rep["header_old"]) / 1024)
        except Error as e:
            res.update(status="refused" if e.code == EXIT_REFUSED else "failed", message=str(e))
        results.append(res)
    return results


# ---------- command line ----------

def _pairs(items):
    out = {}
    for kv in items or []:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise Error("metadata must be key=value, got %s" % kv)
        out[k] = v
    return out


def _fields(a):
    return {k: getattr(a, k) for k in FIELDS if getattr(a, k, None) is not None}


def _say_write(rep):
    verb = "would write" if rep["dry_run"] else "wrote"
    print("%s %s: %d covers, header %.1f KiB -> %.1f KiB, file %.2f MiB" % (
        verb, rep["out"], rep["covers"], rep["header_old"] / 1024, rep["header_new"] / 1024, rep["size_new"] / 2**20))
    if rep["sidecar"]:
        print("sidecar", rep["sidecar"])
    if rep["scrubbed"]:
        print("left out of the embedded cm-info, as paths on this machine: " + ", ".join(rep["scrubbed"]))
    if rep["local_paths"]:
        print("note: these metadata keys seem to hold a path on somebody's machine, and travel with the file: "
              + ", ".join(rep["local_paths"]))


def cmd_embed(a):
    if not a.covers:
        a.covers = find_covers(a.model)[:a.max_covers or None]
        if not a.covers:
            stem = os.path.splitext(os.path.basename(a.model))[0]
            raise Error("no images given, and none found beside the model (%s.preview.jpeg, %s.gallery1.jpeg ...)" % (stem, stem))
    if a.max_covers and len(a.covers) > a.max_covers:
        raise Error("%d images given; the limit is %d (one cover and %d gallery): raise it with --max-covers, 0 for none"
                    % (len(a.covers), a.max_covers, a.max_covers - 1))
    _say_write(embed(a.model, a.covers, a.out, a.force, a.replace, _pairs(a.meta), None, _fields(a),
                     not a.no_cm_info, a.sidecar, a.side, a.quality, a.dry_run))


def cmd_meta(a):
    if not (a.meta or a.unset or _fields(a)):
        raise Error("nothing to change: give -m, --unset or a field such as --author")
    _say_write(embed(a.model, None, a.out, a.force, False, _pairs(a.meta), a.unset, _fields(a),
                     not a.no_cm_info, a.sidecar, dry_run=a.dry_run))


def cmd_show(a):
    s = show(a.model)
    if a.json:
        print(json.dumps(s, indent=1, ensure_ascii=False))
        return
    print("header %.1f KiB, %d tensors, %d covers" % (s["header_bytes"] / 1024, s["tensors"], len(s["covers"])))
    for i, n in enumerate(s["covers"], 1):
        print("  cover_%d: %.1f KiB%s" % (i, n * 0.75 / 1024, " (preview and icon)" if i == 1 else ""))
    for k, v in s["metadata"].items():
        print("  %s: %s" % (k, v if len(v) < 90 else v[:87] + "..."))
    for k, v in (s["cm_info"] or {}).items():
        v = json.dumps(v, ensure_ascii=False)
        print("  cm_info.%s: %s" % (k, v if len(v) < 90 else v[:87] + "..."))


def cmd_verify(a):
    bad = 0
    for m in a.models:
        try:
            r = verify(m)
        except Error as e:
            r = {"status": "failed", "problems": [str(e)], "notes": []}
        bad += r["status"] == "failed"
        print("%-6s %s: %s" % (r["status"], m, "; ".join(r["problems"] + r["notes"])))
    return EXIT_FAILED if bad else EXIT_OK


def cmd_extract(a):
    paths = extract(a.model, a.outdir, a.gnames)
    print("\n".join(paths) if paths else "nothing embedded in " + a.model)


def cmd_strip(a):
    r = strip(a.model, a.out, a.force, a.sidecar, a.dry_run)
    print("%s %s: %d keys removed" % ("would write" if r["dry_run"] else "wrote", r["out"], r["removed"]))
    print({True: "byte-identical to the original file",
           False: "tensor data intact; the header differs from the original, so the whole-file hash is not the original one",
           None: "tensor data intact; no original hash was recorded to compare with"}[r["exact"]])


def cmd_batch(a):
    res = batch(a.folder, a.out, a.force, a.replace, a.sidecar, a.side, a.quality, a.dry_run, a.max_covers)
    for r in res:
        print("%-7s %s%s" % (r["status"], os.path.basename(r["model"]), ": " + r["message"] if r["message"] else ""))
    n = lambda s: sum(r["status"] == s for r in res)
    print("%d done, %d skipped, %d refused, %d failed%s" % (n("ok"), n("skipped"), n("refused"), n("failed"),
                                                           " (dry run)" if a.dry_run else ""))
    return EXIT_FAILED if n("failed") else EXIT_REFUSED if n("refused") else EXIT_OK


def cmd_gui(a):
    import lora_container_gui
    lora_container_gui.main()


def main(argv=None):
    p = argparse.ArgumentParser(prog="lora_container", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    s = p.add_subparsers(dest="cmd", required=True)

    def writes(q, images=False):
        q.add_argument("-o", "--out", help="output file; without it --force is needed to change MODEL in place")
        q.add_argument("--force", action="store_true", help="allow overwriting")
        q.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
        q.add_argument("--sidecar", action="store_true", help="also update the .cm-info.json beside the output, with a Changelog line")
        if images:
            q.add_argument("--side", type=int, default=512, help="shortest side of a cover in px (default 512)")
            q.add_argument("--quality", type=int, default=46, help="JPEG quality (default 46)")
            q.add_argument("--max-covers", type=int, default=MAX_COVERS, help="most images per model (default 6; 0 = no limit)")

    def fields(q):
        q.add_argument("-m", "--meta", action="append", metavar="KEY=VALUE", help="plain metadata key (repeatable)")
        q.add_argument("--title")
        q.add_argument("--author")
        q.add_argument("--base-model")
        q.add_argument("--trigger-words", help="comma separated")
        q.add_argument("--tags", help="comma separated")
        q.add_argument("--description")
        q.add_argument("--model-type", help="default LORA")
        q.add_argument("--no-cm-info", action="store_true", help="do not embed a Stability Matrix cm-info")

    e = s.add_parser("embed", help="put covers (first = preview and icon) and metadata into MODEL")
    e.add_argument("model")
    e.add_argument("covers", nargs="*", metavar="cover",
                   help="images; with none, NAME.preview.jpeg and NAME.gallery1.jpeg ... beside MODEL are used")
    e.add_argument("--replace", action="store_true", help="replace covers the file already has")
    fields(e)
    writes(e, True)
    e.set_defaults(f=cmd_embed)

    m = s.add_parser("meta", help="change metadata only; covers are left alone")
    m.add_argument("model")
    m.add_argument("--unset", action="append", metavar="KEY", help="remove a plain metadata key (repeatable)")
    fields(m)
    writes(m)
    m.set_defaults(f=cmd_meta)

    w = s.add_parser("show", help="list what is in the header")
    w.add_argument("model")
    w.add_argument("--json", action="store_true")
    w.set_defaults(f=cmd_show)

    v = s.add_parser("verify", help="check tensor data, covers and cm-info against the header")
    v.add_argument("models", nargs="+", metavar="model")
    v.set_defaults(f=cmd_verify)

    x = s.add_parser("extract", help="write the covers and cm-info out as files")
    x.add_argument("model")
    x.add_argument("outdir")
    x.add_argument("--gnames", action="store_true", help="name them NAME.preview.jpeg, NAME.gallery1.jpeg ...")
    x.set_defaults(f=cmd_extract)

    t = s.add_parser("strip", help="undo: remove everything this tool added")
    t.add_argument("model")
    writes(t)
    t.set_defaults(f=cmd_strip)

    b = s.add_parser("batch", help="embed for every model in FOLDER, from loose images beside each one")
    b.add_argument("folder")
    b.add_argument("--replace", action="store_true")
    writes(b, True)
    b.set_defaults(f=cmd_batch)

    g = s.add_parser("gui", help="open the window")
    g.set_defaults(f=cmd_gui)

    a = p.parse_args(argv)
    try:
        return a.f(a) or EXIT_OK
    except Error as e:
        print("error: %s" % e, file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
