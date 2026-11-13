import base64, hashlib, io, json, os, struct, sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lora_container as lc

Image = pytest.importorskip("PIL.Image")
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(os.path.dirname(HERE), "sample", "chaoslandscape.safetensors")
FIXTURE = os.environ.get("LORA_CONTAINER_FIXTURE", "")


def make_model(path, meta=None, pad=True, dtype="F32", raw_header=None):
    data = bytes(range(256)) * 4
    hdr = {}
    if meta is not None:
        hdr["__metadata__"] = meta
    width = 2 if dtype == "BF16" else 4
    hdr["lora.down"] = {"dtype": dtype, "shape": [len(data) // 2 // width], "data_offsets": [0, len(data) // 2]}
    hdr["lora.up"] = {"dtype": dtype, "shape": [len(data) // 2 // width], "data_offsets": [len(data) // 2, len(data)]}
    hb = raw_header or json.dumps(hdr, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if pad:
        hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + data)
    return str(path)


def make_image(path, size=(800, 600), colour=(200, 60, 120)):
    Image.new("RGB", size, colour).save(path)
    return str(path)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tensor_bytes(path):
    b = open(path, "rb").read()
    return b[8 + struct.unpack("<Q", b[:8])[0]:]


@pytest.fixture
def kit(tmp_path):
    model = make_model(tmp_path / "m.safetensors", {"ss_network_dim": "8"})
    imgs = [make_image(tmp_path / ("i%d.png" % i), colour=(40 * i, 90, 200)) for i in range(1, 4)]
    return tmp_path, model, imgs


def test_round_trip(kit):
    tmp, model, imgs = kit
    out = str(tmp / "out.safetensors")
    before = sha(model)
    rep = lc.embed(model, imgs, out)
    assert rep["covers"] == 3 and sha(model) == before
    assert tensor_bytes(out) == tensor_bytes(model)
    _, hdr = lc.read_header(out)
    md = hdr["__metadata__"]
    assert md["cover_count"] == "3" and md["cover_format"] == "1" and md["ss_network_dim"] == "8"
    assert md["model_sha256_original"] == before
    assert md["tensor_data_sha256"] == hashlib.sha256(tensor_bytes(model)).hexdigest()
    assert all(isinstance(v, str) for v in md.values())
    assert struct.unpack("<Q", open(out, "rb").read(8))[0] % 8 == 0
    paths = lc.extract(out, str(tmp / "x"))
    assert [os.path.basename(p) for p in paths] == ["cover_1.jpg", "cover_2.jpg", "cover_3.jpg", "out.cm-info.json"]
    im = Image.open(paths[0])
    assert im.format == "JPEG" and min(im.size) == 512
    assert lc.verify(out)["status"] == "ok"


@pytest.mark.parametrize("meta,pad", [({"a": "1"}, True), ({"a": "1"}, False), (None, True), (None, False),
                                      ({}, True), ({"name": "café 三"}, True)])
def test_strip_restores_the_original_bytes(tmp_path, meta, pad):
    model = make_model(tmp_path / "m.safetensors", meta, pad)
    img = make_image(tmp_path / "i.png")
    out, back = str(tmp_path / "o.safetensors"), str(tmp_path / "b.safetensors")
    lc.embed(model, [img], out, meta={"note": "x"} if meta else None)
    rep = lc.strip(out, back)
    if meta:  # the added plain key stays, so the bytes cannot match
        assert rep["exact"] is False and tensor_bytes(back) == tensor_bytes(model)
    else:
        assert rep["exact"] is True and sha(back) == sha(model)


def test_strip_exact_with_existing_metadata(tmp_path):
    model = make_model(tmp_path / "m.safetensors", {"a": "1", "name": "café"})
    out, back = str(tmp_path / "o.safetensors"), str(tmp_path / "b.safetensors")
    lc.embed(model, [make_image(tmp_path / "i.png")], out, fields={"author": "Rob"})
    assert lc.strip(out, back)["exact"] is True and sha(back) == sha(model)


def test_strip_exact_when_original_escaped_non_ascii(tmp_path):
    raw = json.dumps({"__metadata__": {"n": "café"}, "t": {"dtype": "F32", "shape": [256], "data_offsets": [0, 1024]}},
                     separators=(",", ":")).encode()
    model = make_model(tmp_path / "m.safetensors", raw_header=raw)
    out, back = str(tmp_path / "o.safetensors"), str(tmp_path / "b.safetensors")
    lc.embed(model, [make_image(tmp_path / "i.png")], out)
    assert lc.strip(out, back)["exact"] is True and sha(back) == sha(model)


def test_bf16(tmp_path):
    model = make_model(tmp_path / "m.safetensors", {"ss_mixed_precision": "bf16"}, dtype="BF16")
    out = str(tmp_path / "o.safetensors")
    lc.embed(model, [make_image(tmp_path / "i.png")], out)
    _, hdr = lc.read_header(out)
    assert hdr["lora.up"]["dtype"] == "BF16" and tensor_bytes(out) == tensor_bytes(model)
    assert lc.verify(out)["status"] == "ok"


def test_refuses_to_overwrite(kit):
    tmp, model, imgs = kit
    before = sha(model)
    with pytest.raises(lc.Error) as e:
        lc.embed(model, imgs)
    assert e.value.code == lc.EXIT_REFUSED and sha(model) == before
    out = str(tmp / "o.safetensors")
    lc.embed(model, imgs, out)
    with pytest.raises(lc.Error) as e:
        lc.embed(model, imgs, out)
    assert e.value.code == lc.EXIT_REFUSED
    lc.embed(model, imgs, out, force=True)
    lc.embed(model, imgs, force=True)
    assert lc.verify(model)["status"] == "ok" and not os.path.exists(model + ".tmp")


def test_replace_and_repeat_runs(kit):
    tmp, model, imgs = kit
    before = sha(model)
    out = str(tmp / "o.safetensors")
    lc.embed(model, imgs, out)
    with pytest.raises(lc.Error) as e:
        lc.embed(out, imgs[:1], force=True)
    assert e.value.code == lc.EXIT_REFUSED
    lc.embed(out, imgs[:1], force=True, replace=True)
    lc.embed(out, None, force=True, meta={"k": "v"})
    lc.embed(out, None, force=True, meta={"k": "v"})
    _, hdr = lc.read_header(out)
    md = hdr["__metadata__"]
    assert md["cover_count"] == "1" and "cover_2" not in md and md["k"] == "v"
    assert md["model_sha256_original"] == before
    assert json.loads(md["cm_info"])["Hashes"]["SHA256"] == before.upper()
    assert lc.verify(out)["status"] == "ok" and tensor_bytes(out) == tensor_bytes(model)


def test_existing_covers_pass_through_unchanged(kit):
    tmp, model, imgs = kit
    out = str(tmp / "o.safetensors")
    lc.embed(model, imgs, out)
    covers = lc.get_covers(lc.read_header(out)[1]["__metadata__"])
    lc.embed(out, [covers[2], covers[0]], force=True, replace=True)
    assert lc.get_covers(lc.read_header(out)[1]["__metadata__"]) == [covers[2], covers[0]]


def test_large_header(tmp_path, monkeypatch):
    model = make_model(tmp_path / "m.safetensors")
    imgs = []
    for i in range(12):
        p = tmp_path / ("n%d.png" % i)
        Image.frombytes("RGB", (1024, 1024), os.urandom(1024 * 1024 * 3)).save(p)
        imgs.append(str(p))
    out = str(tmp_path / "o.safetensors")
    rep = lc.embed(model, imgs, out, side=1024, quality=95)
    assert rep["header_new"] > 4 * 2**20 and lc.verify(out)["status"] == "ok"
    assert tensor_bytes(out) == tensor_bytes(model)
    monkeypatch.setattr(lc, "MAX_HEADER", 1000)
    with pytest.raises(lc.Error) as e:
        lc.embed(model, imgs[:1], str(tmp_path / "p.safetensors"))
    assert e.value.code == lc.EXIT_INPUT and not os.path.exists(tmp_path / "p.safetensors")


@pytest.mark.parametrize("content", [b"", b"abc", b"this is not a model file at all", struct.pack("<Q", 10) + b"notjson!!!",
                                     struct.pack("<Q", 10**12) + b"{}", struct.pack("<Q", 2) + b"[]"])
def test_not_safetensors(tmp_path, content):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(content)
    for call in (lambda: lc.read_header(str(p)), lambda: lc.verify(str(p)), lambda: lc.show(str(p)),
                 lambda: lc.strip(str(p), str(tmp_path / "o"))):
        with pytest.raises(lc.Error) as e:
            call()
        assert e.value.code == lc.EXIT_INPUT
    assert lc.main(["show", str(p)]) == lc.EXIT_INPUT
    assert lc.main(["verify", str(p)]) == lc.EXIT_FAILED


def test_bad_image_and_missing_file(kit):
    tmp, model, imgs = kit
    bad = tmp / "bad.png"
    bad.write_text("no")
    for covers in ([str(bad)], [str(tmp / "absent.png")]):
        with pytest.raises(lc.Error):
            lc.embed(model, covers, str(tmp / "o.safetensors"))
    assert not os.path.exists(tmp / "o.safetensors")
    with pytest.raises(lc.Error):
        lc.read_header(str(tmp / "absent.safetensors"))


def test_dry_run_writes_nothing(kit):
    tmp, model, imgs = kit
    before = sorted(os.listdir(tmp))
    rep = lc.embed(model, imgs, str(tmp / "o.safetensors"), dry_run=True, sidecar=True)
    assert rep["dry_run"] and rep["header_new"] > rep["header_old"] and sorted(os.listdir(tmp)) == before
    lc.embed(model, imgs, str(tmp / "o.safetensors"))
    mid = sorted(os.listdir(tmp))
    lc.strip(str(tmp / "o.safetensors"), str(tmp / "s.safetensors"), dry_run=True)
    assert lc.main(["batch", str(tmp), "--dry-run", "--force"]) == 0 and sorted(os.listdir(tmp)) == mid


def test_reserved_keys_rejected(kit):
    tmp, model, imgs = kit
    for k in ("cover_1", "cover_count", "tensor_data_sha256", "cm_info"):
        with pytest.raises(lc.Error):
            lc.embed(model, imgs, str(tmp / "o.safetensors"), meta={k: "x"})


def test_data_prefix_is_accepted(kit):
    tmp, model, imgs = kit
    out = str(tmp / "o.safetensors")
    lc.embed(model, imgs[:1], out)
    n, hdr = lc.read_header(out)
    plain = lc.get_covers(hdr["__metadata__"])
    hdr["__metadata__"]["cover_1"] = lc.DATA_PREFIX + hdr["__metadata__"]["cover_1"]
    hb = lc.header_bytes(hdr)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + tensor_bytes(model))
    assert lc.get_covers(lc.read_header(out)[1]["__metadata__"]) == plain and lc.verify(out)["status"] == "ok"


def test_cm_info(kit):
    tmp, model, imgs = kit
    before = sha(model)
    side = {"ModelId": 7, "ModelName": "Mine", "Hashes": {"SHA256": "OLD"}, "CurrentHash": {"SHA256": "OLD"},
            "ThumbnailImageUrl": "/home/someone/private/m.preview.jpeg", "Custom": "kept"}
    json.dump(side, open(lc.sidecar_path(model), "w"))
    out = str(tmp / "o.safetensors")
    lc.embed(model, imgs, out, fields={"author": "Rob", "trigger_words": "one, two", "tags": ["a", "b"]})
    cm = json.loads(lc.read_header(out)[1]["__metadata__"]["cm_info"])
    assert cm["ModelId"] == 7 and cm["ModelName"] == "Mine" and cm["Custom"] == "kept"
    assert cm["AuthorUsername"] == "Rob" and cm["TrainedWords"] == ["one", "two"] and cm["Tags"] == ["a", "b"]
    assert cm["Hashes"]["SHA256"] == before.upper() and cm["OriginalHashes"] == cm["Hashes"]
    assert len(cm["Hashes"]["CRC32"]) == 8 and "CurrentHash" not in cm and cm["ThumbnailImageUrl"] is None
    assert b"/home/someone" not in open(out, "rb").read(8 + lc.read_header(out)[0])
    paths = lc.extract(out, str(tmp / "x"), sm_names=True)
    assert [os.path.basename(p) for p in paths] == ["o.preview.jpeg", "o.gallery1.jpeg", "o.gallery2.jpeg", "o.cm-info.json"]
    j = json.load(open(paths[-1]))
    assert j["Hashes"]["SHA256"] == sha(out).upper() and j["OriginalHashes"]["SHA256"] == before.upper()
    assert j["ThumbnailImageUrl"] == os.path.abspath(paths[0])
    plain = str(tmp / "p.safetensors")
    lc.embed(model, imgs, plain, cm_info=False)
    assert "cm_info" not in lc.read_header(plain)[1]["__metadata__"]
    with pytest.raises(lc.Error):
        lc.embed(model, imgs, str(tmp / "q.safetensors"), cm_info=False, fields={"author": "x"})


def test_sidecar_changelog(kit):
    tmp, model, imgs = kit
    before = sha(model)
    lc.embed(model, imgs, force=True, sidecar=True)
    lc.embed(model, None, force=True, sidecar=True, meta={"k": "v"})
    j = json.load(open(lc.sidecar_path(model)))
    lines = j["Changelog"].splitlines()
    assert len(lines) == 2 and lines[0].startswith("1. ") and lines[0].endswith("Embedded cover image and gallery")
    assert lines[1].startswith("2. ") and j["OriginalHashes"]["SHA256"] == before.upper()
    assert j["Hashes"]["SHA256"] == sha(model).upper() == j["CurrentHash"]["SHA256"]


def test_verify_catches_damage(kit):
    tmp, model, imgs = kit
    out = str(tmp / "o.safetensors")
    lc.embed(model, imgs, out)
    b = bytearray(open(out, "rb").read())
    b[-1] ^= 1
    open(out, "wb").write(b)
    r = lc.verify(out)
    assert r["status"] == "failed" and "tensor data" in r["problems"][0]
    assert lc.main(["verify", out]) == lc.EXIT_FAILED
    with pytest.raises(lc.Error) as e:
        lc.embed(out, None, force=True, meta={"k": "v"})
    assert e.value.code == lc.EXIT_FAILED
    assert lc.verify(model)["status"] == "plain" and lc.main(["verify", model]) == 0


def test_batch(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_model(src / "a.safetensors")
    make_image(src / "a.preview.jpeg")
    for i in (1, 2, 10):
        make_image(src / ("a.gallery%d.jpeg" % i), colour=(i * 20, 0, 0))
    make_model(src / "b.safetensors")
    (src / "b").mkdir()
    make_image(src / "b" / "2.png")
    make_image(src / "b" / "1.png")
    json.dump({"author": "Rob", "note": "n"}, open(src / "b.meta.json", "w"))
    make_model(src / "c.safetensors")
    (src / "d.safetensors").write_text("junk")
    assert [os.path.basename(p) for p in lc.find_covers(str(src / "a.safetensors"))] == \
        ["a.preview.jpeg", "a.gallery1.jpeg", "a.gallery2.jpeg", "a.gallery10.jpeg"]
    res = {os.path.basename(r["model"]): r["status"] for r in lc.batch(str(src), str(dst))}
    assert res == {"a.safetensors": "ok", "b.safetensors": "ok", "c.safetensors": "skipped", "d.safetensors": "failed"}
    assert sorted(os.listdir(dst)) == ["a.safetensors", "b.safetensors"]
    md = lc.read_header(str(dst / "b.safetensors"))[1]["__metadata__"]
    assert md["cover_count"] == "2" and md["note"] == "n" and json.loads(md["cm_info"])["AuthorUsername"] == "Rob"
    assert lc.read_header(str(dst / "a.safetensors"))[1]["__metadata__"]["cover_count"] == "4"
    for i in (3, 4, 5, 6):
        make_image(src / ("a.gallery%d.jpeg" % i))
    lc.batch(str(src), str(dst), force=True)
    assert lc.read_header(str(dst / "a.safetensors"))[1]["__metadata__"]["cover_count"] == "6"
    assert lc.main(["batch", str(src)]) == lc.EXIT_FAILED
    again = {os.path.basename(r["model"]): r["status"] for r in lc.batch(str(dst), force=True)}
    assert again == {"a.safetensors": "skipped", "b.safetensors": "skipped"}


def test_cli(kit, capsys):
    tmp, model, imgs = kit
    out = str(tmp / "o.safetensors")
    assert lc.main(["embed", model] + imgs) == lc.EXIT_REFUSED
    assert lc.main(["embed", model, "-o", str(tmp / "none.safetensors")]) == lc.EXIT_INPUT
    named = make_model(tmp / "Joel Rea.safetensors")
    make_image(tmp / "Joel Rea.preview.jpeg")
    make_image(tmp / "Joel Rea.gallery1.jpeg")
    make_image(tmp / "Joel Rea other.preview.jpeg")
    assert lc.main(["embed", named, "-o", str(tmp / "named.safetensors")]) == 0
    assert lc.read_header(str(tmp / "named.safetensors"))[1]["__metadata__"]["cover_count"] == "2"
    assert lc.main(["embed", model] + imgs + ["-o", out, "-m", "k=v", "--author", "Rob", "--trigger-words", "x,y"]) == 0
    assert lc.main(["embed", model] + imgs * 3 + ["-o", str(tmp / "z.safetensors")]) == lc.EXIT_INPUT
    assert lc.main(["embed", model] + imgs * 3 + ["-o", str(tmp / "y.safetensors"), "--max-covers", "0"]) == 0
    assert lc.main(["embed", model, imgs[0], "-o", str(tmp / "z.safetensors"), "-m", "novalue"]) == lc.EXIT_INPUT
    assert lc.main(["meta", out, "--force"]) == lc.EXIT_INPUT
    assert lc.main(["meta", out, "--force", "--unset", "k", "--tags", "a,b"]) == 0
    capsys.readouterr()
    assert lc.main(["show", out, "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert len(s["covers"]) == 3 and "k" not in s["metadata"] and s["cm_info"]["Tags"] == ["a", "b"]
    assert s["cm_info"]["AuthorUsername"] == "Rob" and s["cm_info"]["TrainedWords"] == ["x", "y"]
    assert lc.main(["show", out]) == 0 and "preview and icon" in capsys.readouterr().out
    assert lc.main(["verify", out, model]) == 0
    assert lc.main(["extract", out, str(tmp / "x"), "--sm-names"]) == 0
    assert lc.main(["strip", out]) == lc.EXIT_REFUSED
    assert lc.main(["strip", out, "-o", str(tmp / "s.safetensors")]) == 0
    assert "byte-identical" in capsys.readouterr().out and sha(str(tmp / "s.safetensors")) == sha(model)
    with pytest.raises(SystemExit) as e:
        lc.main(["nonsense"])
    assert e.value.code == 2


@pytest.mark.parametrize("path", [p for p in (SAMPLE, FIXTURE) if p and os.path.exists(p)])
def test_real_files(tmp_path, path):
    assert lc.verify(path)["status"] == "ok"
    _, hdr = lc.read_header(path)
    md = hdr["__metadata__"]
    covers = lc.get_covers(md)
    assert len(covers) == int(md["cover_count"]) >= 1
    assert all(Image.open(io.BytesIO(c)).format == "JPEG" for c in covers)
    plain, again = str(tmp_path / "plain.safetensors"), str(tmp_path / "again.safetensors")
    lc.strip(path, plain)
    assert lc.verify(plain)["status"] == "plain" and tensor_bytes(plain) == tensor_bytes(path)
    lc.embed(plain, covers, again, fields={"author": "test"})
    md2 = lc.read_header(again)[1]["__metadata__"]
    assert lc.get_covers(md2) == covers and md2["tensor_data_sha256"] == md["tensor_data_sha256"]
    assert lc.verify(again)["status"] == "ok"


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample file not present")
def test_sample_is_version_1_and_undoes_exactly(tmp_path):
    md = lc.read_header(SAMPLE)[1]["__metadata__"]
    cm = json.loads(md["cm_info"])
    assert md["cover_format"] == "1" and md["cover_count"] == "6"
    assert cm["Hashes"]["SHA256"].lower() == md["model_sha256_original"] and cm["ThumbnailImageUrl"] is None
    plain = str(tmp_path / "plain.safetensors")
    assert lc.strip(SAMPLE, plain)["exact"] is True and sha(plain) == md["model_sha256_original"]


def test_local_paths_are_blanked_or_flagged(tmp_path, capsys):
    model = make_model(tmp_path / "m.safetensors", {"ss_dataset_dirs": '{"/home/someone/train/10_x": {"n_repeats": 10}}',
                                                    "ss_output_name": "x", "note": "ratio 3/4, see https://example.com/a"})
    side = {"ModelName": "/slash name", "ModelDescription": "/home/kept as text", "Tags": ["/r/kept"],
            "ThumbnailImageUrl": "/home/someone/m.preview.jpeg", "SourceUrl": "https://civitai.com/models/1",
            "Extra": {"LocalPath": "C:\\Users\\someone\\m.png", "Url": "//cdn.example.com/x", "List": ["~/a.png", "ok"]},
            "Other": "file:///home/someone/x"}
    json.dump(side, open(lc.sidecar_path(model), "w"))
    out = str(tmp_path / "o.safetensors")
    rep = lc.embed(model, [make_image(tmp_path / "i.png")], out)
    assert sorted(rep["scrubbed"]) == ["Extra.List", "Extra.LocalPath", "Other"] and rep["local_paths"] == ["cm_info", "ss_dataset_dirs"]
    cm = json.loads(lc.read_header(out)[1]["__metadata__"]["cm_info"])
    assert cm["ThumbnailImageUrl"] is None and cm["Other"] is None and cm["Extra"] == {"LocalPath": None, "Url": "//cdn.example.com/x", "List": ["ok"]}
    assert cm["ModelName"] == "/slash name" and cm["Tags"] == ["/r/kept"] and cm["SourceUrl"] == "https://civitai.com/models/1"
    assert "possible local path in: cm_info, ss_dataset_dirs" in lc.verify(out)["notes"] and lc.verify(out)["status"] == "ok"
    assert lc.main(["embed", model, str(tmp_path / "i.png"), "-o", str(tmp_path / "p.safetensors")]) == 0
    said = capsys.readouterr().out
    assert "left out of the embedded cm-info" in said and "ss_dataset_dirs" in said
    assert lc.local_paths({"a": "plain", "b": "1/2 and 3:4", "cover_1": "/home/x", "c": "https://example.com/Users/x",
                           "d": "v1-5-pruned.safetensors", "e": "//cdn.example.com/a"}) == []
    every = {"linux": "/home/a/b", "mac": "/Users/a/b", "mac_volume": "x /Volumes/Disk/a", "win": "C:\\Users\\a",
             "win_forward": "D:/train/a", "win_share": "\\\\nas\\share\\a", "tilde": "~/a", "url": "file:///C:/a",
             "in_json": '{"dir": "/mnt/c/train"}', "win_in_json": json.dumps({"dir": "E:\\data\\x"}), "colab": "/content/drive/x"}
    assert lc.local_paths(every) == sorted(every)
    for value in ("/home/a/b.png", "/Users/a/b.png", "/Volumes/D/b.png", "C:\\Users\\a\\b.png", "D:/a/b.png",
                  "\\\\nas\\share\\b.png", "~/b.png", "~\\b.png", "file:///home/a/b.png"):
        cm = {"Path": value, "Nested": [{"Path": value}]}
        assert lc.scrub_paths(cm) == ["Path", "Nested.Path"] and cm == {"Path": None, "Nested": [{"Path": None}]}
