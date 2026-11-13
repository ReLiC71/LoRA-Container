LoRA-Container
==============

A method, with example tools, for containing an image gallery and metadata in a LoRA so that it is self-contained, without breaking the LoRA.

Concept by Robert Cunnington (ReLiC71), first discussed November 2025. Gallery format and tools built 13 November 2026 with Claude (Anthropic). Licensed under Apache 2.0.

The idea
--------
A .safetensors file starts with a JSON header that has a free-form `__metadata__` block of string values. Put the cover image and gallery in there, as base64 JPEGs. The images then travel with the model: they cannot be lost, separated or renamed away. Tensor data is not touched, and loaders that do not know about the covers simply ignore them.

Format
------
All keys live in `__metadata__`, and every value is a string.

| key | value |
|---|---|
| `cover_count` | number of covers, e.g. `"6"` |
| `cover_1` ... `cover_N` | base64 JPEG. `cover_1` is the preview/icon; the rest are the gallery |
| `model_sha256_original` | lower-case SHA-256 of the whole file before the covers were added |
| `tensor_data_sha256` | lower-case SHA-256 of everything after the header |

Readers should also accept a `data:image/jpeg;base64,` prefix on a cover. Any other metadata keys are left alone and may sit beside these.

The header is written as compact JSON, padded with spaces to a multiple of 8 bytes. Tensor offsets are relative to the start of the data region, so they do not change.

Practical notes
---------------
- Base64 adds about 33% to the JPEG size. At 512 px on the shortest side and quality 46, a cover is about 20-30 KiB, so six covers add roughly 0.2 MiB.
- Editing the header changes the whole-file hash, which breaks hash lookups on sites such as Civitai. `tensor_data_sha256` does not change and is the stable identity; `model_sha256_original` records the old whole-file hash.
- To undo, remove the keys above from the header (the `strip` command does this).

Tools
-----
`lora_container.py` (Python 3.8+, Pillow):

    python3 lora_container.py embed model.safetensors cover1.png cover2.png ... -m author=Name -o out.safetensors
    python3 lora_container.py show out.safetensors
    python3 lora_container.py extract out.safetensors outdir
    python3 lora_container.py strip out.safetensors -o plain.safetensors

`embed` resizes covers to 512 px on the shortest side at quality 46 (change with `--side` and `--quality`). It refuses to write if the tensor data would change, and writes through a temporary file.

`viewer.html` is a single page with no dependencies. Open it in a browser and drop in one or more .safetensors files to see their covers. Files are read locally and nothing is uploaded.

Related work
------------
Putting an image in a safetensors header is not new, and this convention builds on it:
- Stability AI's ModelSpec (https://github.com/Stability-AI/ModelSpec) defines `modelspec.thumbnail`: one very small (256x256 recommended) data-URI image.
- A `ssmd_cover_images` key, a JSON list of base64 images, is read by some model-manager code. Its origin and date have not been established here.
- A Civitai article proposing `sd.artifact.v1` (https://civitai.com/articles/24796, January 2026) suggests an embedded preview image in a self-describing block.

What this repository adds is a plain key-per-cover layout, a two-hash identity scheme that survives header edits, an undo route, and working writer and viewer tools.

Status
------
A proof of concept. The only model file here is `sample/chaoslandscape.safetensors`, a small SD 1.5 LoRA (trigger word `chaoslandscape`) trained by the author on his own generated images, with six covers embedded. Drop it into `viewer.html` to see them.
