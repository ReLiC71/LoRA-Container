import json, os, struct, sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tk = pytest.importorskip("tkinter")
pytest.importorskip("PIL.Image")
import lora_container as lc
from test_lora_container import make_image, make_model, sha, tensor_bytes


@pytest.fixture
def app():
    import lora_container_gui as gui
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    a = gui.App(root)
    a.quiet = True
    root.update()
    yield a
    root.destroy()


def test_gui_embed_reorder_save_reopen_strip(app, tmp_path):
    model = make_model(tmp_path / "m.safetensors", {"ss_network_dim": "8"})
    imgs = [make_image(tmp_path / ("i%d.png" % i), colour=(30 * i, 80, 160)) for i in range(1, 9)]
    before = sha(model)
    assert app.open_model(model) and not app.covers and not app.b_strip.on
    app.add_images(imgs)
    assert len(app.covers) == 6 and "left out" in app.status.cget("text")
    app.sel = 5
    app.move(-1)
    app.remove_selected()
    assert len(app.covers) == 5
    app.entries["author"].insert(0, "Rob")
    app.entries["trigger_words"].insert(0, "one, two")
    app.other.insert("1.0", "note=hello")
    out = str(tmp_path / "o.safetensors")
    app.save(out)
    app.wait()
    assert app.error is None and app.model == out and len(app.covers) == 5 and not app.dirty
    md = lc.read_header(out)[1]["__metadata__"]
    cm = json.loads(md["cm_info"])
    assert md["cover_count"] == "5" and md["note"] == "hello" and md["model_sha256_original"] == before
    assert cm["AuthorUsername"] == "Rob" and cm["TrainedWords"] == ["one", "two"]
    assert app.entries["author"].get() == "Rob" and lc.verify(out)["status"] == "ok" and sha(model) == before
    first = lc.get_covers(md)
    app.sel = 0
    app.move(1)
    app.save(out)
    app.wait()
    again = lc.get_covers(lc.read_header(out)[1]["__metadata__"])
    assert again[0] == first[1] and again[1] == first[0] and again[2:] == first[2:]
    app.verify()
    app.wait()
    assert "all good" in app.status.cget("text")
    app.extract(str(tmp_path / "x"))
    app.wait()
    assert "o.preview.jpeg" in os.listdir(tmp_path / "x") and "o.gallery4.jpeg" in os.listdir(tmp_path / "x")
    w = app.metadata_window()
    w.destroy()
    plain = str(tmp_path / "p.safetensors")
    app.strip(plain)
    app.wait()
    assert tensor_bytes(plain) == tensor_bytes(model) and lc.verify(plain)["status"] == "plain"


def test_gui_refuses_bad_input(app, tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_text("junk")
    assert app.open_model(str(bad)) is False and app.model is None and "not a safetensors" in app.error
    model = make_model(tmp_path / "m.safetensors")
    app.open_model(model)
    app.other.insert("1.0", "no equals sign")
    app.save(str(tmp_path / "o.safetensors"))
    assert "key=value" in app.error and not os.path.exists(tmp_path / "o.safetensors")
    app.other.delete("1.0", "end")
    app.add_images([str(bad)])
    assert not app.covers and "cannot read image" in app.error
    app.add_images([make_image(tmp_path / "i.png")])
    app.save(model)
    app.wait()
    app.remove_selected()
    app.save(model)
    assert "no images left" in app.error


def test_gui_one_click_from_loose_files(app, tmp_path):
    model = make_model(tmp_path / "Joel Rea.safetensors")
    make_image(tmp_path / "Joel Rea.preview.jpeg", colour=(1, 2, 3))
    for i in range(1, 8):
        make_image(tmp_path / ("Joel Rea.gallery%d.jpeg" % i))
    json.dump({"ModelName": "Joel Rea Style", "AuthorUsername": "someone", "BaseModel": "Flux.1 D", "TrainedWords": ["jr"],
               "ThumbnailImageUrl": "/home/x/y.jpeg"}, open(tmp_path / "Joel Rea.cm-info.json", "w"))
    assert app.open_model(model) and len(app.covers) == 6 and app.dirty
    assert app.covers[0].name == "Joel Rea.preview.jpeg" and app.covers[5].name == "Joel Rea.gallery5.jpeg"
    assert app.entries["title"].get() == "Joel Rea Style" and app.entries["trigger_words"].get() == "jr"
    out = str(tmp_path / "out.safetensors")
    app.save(out)
    app.wait()
    md = lc.read_header(out)[1]["__metadata__"]
    cm = json.loads(md["cm_info"])
    assert app.error is None and md["cover_count"] == "6" and cm["ModelName"] == "Joel Rea Style"
    assert cm["BaseModel"] == "Flux.1 D" and cm["ThumbnailImageUrl"] is None and not app.dirty
    assert lc.verify(out)["status"] == "ok" and all(c.embedded for c in app.covers)
