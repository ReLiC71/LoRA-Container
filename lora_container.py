"""lora_container: put cover images and metadata inside a .safetensors header.

  lora_container.py embed MODEL COVER [COVER ...] [-m key=value ...] [-o OUT]
  lora_container.py show MODEL
  lora_container.py extract MODEL OUTDIR
  lora_container.py strip MODEL -o OUT

Only the JSON header changes. Tensor data is copied byte for byte.
Needs Python 3.8+ and Pillow.
"""
import argparse, base64, hashlib, io, json, os, struct

KEYS = ("cover_count", "model_sha256_original", "tensor_data_sha256")
MAX_HEADER = 200_000_000


def read_header(fh):
    n = struct.unpack("<Q", fh.read(8))[0]
    if not 0 < n < MAX_HEADER:
        raise SystemExit("not a safetensors file (bad header length)")
    return n, json.loads(fh.read(n))


def whole_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tensor_hash(path):
    with open(path, "rb") as f:
        n, _ = read_header(f)
        h = hashlib.sha256()
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cover_keys(meta):
    return ["cover_%d" % i for i in range(1, int(meta.get("cover_count", 0)) + 1)]


def jpeg_b64(path, side, quality):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    k = side / min(w, h)
    if k < 1:
        im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def write(src, dst, header, n_old):
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    tmp = dst + ".tmp"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        fi.seek(8 + n_old)
        for b in iter(lambda: fi.read(1 << 22), b""):
            fo.write(b)
    if tensor_hash(tmp) != tensor_hash(src):
        raise SystemExit("tensor data changed: aborted, remove " + tmp)
    os.replace(tmp, dst)


def cmd_embed(a):
    with open(a.model, "rb") as f:
        n, hdr = read_header(f)
    meta = hdr.setdefault("__metadata__", {})
    if "cover_count" in meta:
        raise SystemExit("already has covers; run strip first")
    meta.setdefault("model_sha256_original", whole_hash(a.model))
    meta["tensor_data_sha256"] = tensor_hash(a.model)
    meta["cover_count"] = str(len(a.covers))
    for i, c in enumerate(a.covers, 1):
        meta["cover_%d" % i] = jpeg_b64(c, a.side, a.quality)
    for kv in a.meta or []:
        k, _, v = kv.partition("=")
        meta[k] = v
    out = a.o or a.model
    write(a.model, out, hdr, n)
    print("wrote", out, os.path.getsize(out), "bytes,", len(a.covers), "covers")


def cmd_show(a):
    with open(a.model, "rb") as f:
        n, hdr = read_header(f)
    meta = hdr.get("__metadata__", {})
    print("header bytes:", n, " tensors:", len(hdr) - ("__metadata__" in hdr))
    for k, v in meta.items():
        if k.startswith("cover_") and k != "cover_count":
            print("  %s: %d bytes (base64)" % (k, len(v)))
        else:
            print("  %s: %s" % (k, v if len(v) < 90 else v[:87] + "..."))


def cmd_extract(a):
    with open(a.model, "rb") as f:
        _, hdr = read_header(f)
    meta = hdr.get("__metadata__", {})
    os.makedirs(a.outdir, exist_ok=True)
    for k in cover_keys(meta):
        v = meta[k].split(",", 1)[-1] if meta[k].startswith("data:") else meta[k]
        p = os.path.join(a.outdir, k + ".jpg")
        with open(p, "wb") as o:
            o.write(base64.b64decode(v))
        print(p)


def cmd_strip(a):
    with open(a.model, "rb") as f:
        n, hdr = read_header(f)
    meta = hdr.get("__metadata__", {})
    for k in cover_keys(meta) + list(KEYS):
        meta.pop(k, None)
    if not meta:
        hdr.pop("__metadata__", None)
    write(a.model, a.o, hdr, n)
    print("wrote", a.o)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    e = s.add_parser("embed")
    e.add_argument("model")
    e.add_argument("covers", nargs="+")
    e.add_argument("-m", "--meta", action="append", help="extra key=value metadata")
    e.add_argument("-o", help="output file (default: overwrite MODEL)")
    e.add_argument("--side", type=int, default=512, help="shortest side in px")
    e.add_argument("--quality", type=int, default=46, help="JPEG quality")
    e.set_defaults(f=cmd_embed)
    w = s.add_parser("show")
    w.add_argument("model")
    w.set_defaults(f=cmd_show)
    x = s.add_parser("extract")
    x.add_argument("model")
    x.add_argument("outdir")
    x.set_defaults(f=cmd_extract)
    t = s.add_parser("strip")
    t.add_argument("model")
    t.add_argument("-o", required=True)
    t.set_defaults(f=cmd_strip)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
