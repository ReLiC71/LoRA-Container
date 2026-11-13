LoRA-Container format, version 1
================================

A way to carry a cover image, a gallery and descriptive metadata inside a .safetensors file, without touching the tensor data.

Where it lives
--------------
Everything is in the `__metadata__` object of the safetensors JSON header. Every value is a string, as safetensors requires.

| key | value |
|---|---|
| `cover_format` | `"1"`. If the key is absent, read the file as version 1 (files made before the key existed have none) |
| `cover_count` | number of covers, as a decimal string, e.g. `"6"` |
| `cover_1` ... `cover_N` | one image each: a base64 JPEG |
| `model_sha256_original` | lower-case hex SHA-256 of the whole file before anything was embedded |
| `tensor_data_sha256` | lower-case hex SHA-256 of every byte after the header |
| `cm_info` | optional. A Stability Matrix `.cm-info.json`, as JSON text |

Any other key in `__metadata__` is left alone.

Covers
------
- `cover_1` is the cover: the preview and the icon. It corresponds to Stability Matrix's `NAME.preview.jpeg`.
- `cover_2` onwards are the gallery, in order. They correspond to `NAME.gallery1.jpeg`, `NAME.gallery2.jpeg` and so on.
- The convention is one cover and five gallery images, six in all. Readers must not assume six: read `cover_count`.
- Each value is standard base64 (with padding) of a JPEG file. A reader must also accept the same data with a `data:image/jpeg;base64,` prefix.
- The reference writer scales an image down so that its shortest side is 512 px and saves it at JPEG quality 46. That is a default, not part of the format.

Identity: two hashes
--------------------
Editing the header changes the SHA-256 of the whole file, so a lookup by file hash (on Civitai, for example) stops working. Two keys deal with that.

- `tensor_data_sha256` covers only the tensor data, so no header edit changes it. It is the stable identity of the model, and it lets anyone check that the weights are untouched.
- `model_sha256_original` records the hash the file had before embedding: the hash that sites and tools already know the model by. A writer sets it once and never changes it on later edits.

A file cannot contain its own whole-file SHA-256, because writing the hash in changes the file being hashed. So the current whole-file hash is never stored in the header.

The embedded cm-info
--------------------
`cm_info` holds the same JSON object that Stability Matrix keeps in `NAME.cm-info.json` (`ModelName`, `AuthorUsername`, `BaseModel`, `TrainedWords`, `Tags`, `ModelDescription`, `Hashes` and the rest), with three rules:

- `Hashes` and `OriginalHashes` both describe the file before embedding: `SHA256` (upper-case hex), `CRC32`, and `BLAKE3` when the writer could calculate it.
- There is no `CurrentHash`, for the reason given above.
- `ThumbnailImageUrl` is `null`, and so is any other value that is a path on the writer's machine. In a sidecar such a path is of no use to anyone else, and private, inside a shared file. Free-text fields (names, descriptions, tags, trigger words) are left as they are.

A reader that writes the cm-info back out as a sidecar should set `Hashes` to the hashes of the file as it is now, and keep `OriginalHashes`.

Writing the header
------------------
- The header is compact JSON in UTF-8 (no spaces between tokens), padded at the end with spaces to a multiple of 8 bytes.
- Key order is kept. New keys go at the end of `__metadata__`.
- The header must not exceed 100,000,000 bytes, the limit in the safetensors library.
- Tensor offsets in the header are relative to the start of the data region, so they do not change. The tensor data is copied byte for byte.
- A writer should check `tensor_data_sha256` on the new file before it replaces anything.

Undo
----
Remove `cover_format`, `cover_count`, every `cover_N`, `model_sha256_original`, `tensor_data_sha256` and `cm_info`. If nothing else in the header was changed and the header was first written in the usual form (compact JSON, with or without the padding), the result is byte for byte the original file, and its SHA-256 equals `model_sha256_original`. A tool that undoes should check that and say which it achieved.

Reading
-------
- No `cover_count`: the file has no covers.
- A `cover_format` higher than the reader knows: still try to read the covers by the rules above.
- Ignore keys you do not know.

Size
----
In the sample file a cover takes about 45 KiB of header at the default size and quality, so six add about 0.3 MiB. Every program that opens the file reads and parses the whole header, so keep it modest: as a guide, under 10 MiB.

Optional sidecar convention
---------------------------
Where a `NAME.cm-info.json` is kept beside the model, a writer may record the change there: `OriginalHashes` (set once), `Hashes` and `CurrentHash` (the file as it is now), and `Changelog`, a string of numbered lines such as `1. 13/11/26. Embedded cover image and gallery`, kept so that a change can be traced and undone.
