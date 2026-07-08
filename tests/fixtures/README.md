# Test fixtures

Two real-world input pages used by the acceptance tests in `tests/test_acceptance.py`; the expected detection results live in `tests/expected/`.

| File | What it is | What it exercises |
|---|---|---|
| `louie_louie_preview.png` | 900×1164 low-resolution preview of a bass-clef trombone page (staff space ≈7px) | The automatic upscaling path, noteheads sitting on staff lines, key-signature reading at small scale, rejection of tempo/copyright text as notes |
| `misty_photo.jpg` | 4898×6530 phone photo of a printed trombone page (tilted, curved page, chord symbols, uneven light) | Deskew, page-curvature dewarp, adaptive thresholding, chord-symbol immunity, the native high-resolution path |

These images are excerpts of copyrighted arrangements and are included solely as detection test data for this project's regression suite — do not reuse them for any other purpose.
