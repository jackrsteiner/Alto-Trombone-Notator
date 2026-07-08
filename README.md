# Alto Trombone Position Annotator

Takes images of standard bass-clef trombone sheet music, detects the notes, and produces a PDF of the same pages with an Eb alto trombone slide position printed under (or above) every note. It runs two ways: as a website in your browser, or as a Python command-line script. Both produce identical output.

## The three annotation methods

| Method | What you play | What it sounds like | Number colour |
|---|---|---|---|
| `octave` (default) | The written note, one octave up, on alto | Same key, one octave above written | Green |
| `pitch` | Exactly the written note, on alto | As written | Blue |
| `fourth` | The written note *as if on tenor* (tenor positions on the alto) | A perfect fourth higher than written | Dark red |

The `fourth` method keeps your tenor reading reflexes but transposes the music; any accompaniment or chords must move up a fourth with you.

## The website

The site is a single static page that runs the Python script inside your browser using Pyodide (Python compiled to WebAssembly). There is no server: your images never leave your device. Processing runs in a background Web Worker, so the page stays responsive and the detection log streams live into a progress panel while pages are read.

### Hosting it on GitHub Pages

1. Put these three files in the root of a repository: `index.html`, `alto_annotate.py`, `README.md`.
2. In the repository settings, enable **Pages** and choose **Deploy from a branch**, branch `main`, folder `/ (root)`.
3. Open the published URL. Nothing else to configure.

Any other static host (Netlify, Cloudflare Pages, a plain web server) works the same way. The page must be served over HTTP(S) — opening `index.html` directly from disk will not load the script file.

### Using the site

1. Wait for the runtime to load. The **first visit downloads roughly 60–90 MB** (Python, numpy, OpenCV, Pillow); later visits use the browser cache and start much faster.
2. Choose one image per page of music, in page order. PNG or JPG.
3. Pick a method and where the numbers should go. The **key signature is read from the image automatically**; leave the key on *Auto-detect* unless the log's `detected key` line disagrees with the printed signature, in which case pick the major key yourself and run again.
4. Press **Annotate**. A few seconds per page is normal (longer on phones). The **Progress log** panel below the button streams every processing step as it happens — `·`-prefixed activity lines (loading, deskewing, flattening curvature, isolating each staff, reading each staff's notes, with their results) interleaved with the engine's readings, so you can see what is being done and what has already finished. The status line under the button always shows the current step. The panel is collapsible (click its header) and scrolls when the output gets long.
5. Review the log (including the `detected key` line), then download the PDF. Two MIDI files are offered alongside it — **even notes** (every note a quarter) and **guessed rhythm** (durations estimated from the engraving's note spacing) — so you can hear the piece at the sounding pitch of the chosen method.

The site works on mobile browsers (Safari, Chrome). The first-visit download is the main cost — do it on Wi-Fi. On a phone you can photograph the music directly from the file picker.

### Advanced settings

| Setting | Default | What it does |
|---|---|---|
| Ledger range | 4.5 | How many staff spaces above/below each staff to search for notes |
| Sensitivity | 1.0 | Notehead detector strictness. Lower (e.g. 0.85) finds fainter noteheads but risks false positives — try lowering it for poor photos |
| Skip left | 8.0 | Staff spaces skipped at the left of each staff (clef, key and time signature zone). Lower it if a pickup note at the very start of a line is missed |
| Ignore printed accidentals | off | Reads by key signature only. Use if a messy scan produces many false accidental detections |
| Skip page-curvature correction | off | Disables the automatic straightening of bowed staff lines in photos. Use if the correction itself distorts a page |

## The command-line script

```
pip install opencv-python numpy Pillow
python alto_annotate.py score.png -m octave          # key read from the image
python alto_annotate.py page1.jpg page2.jpg -k F -o out.pdf
python alto_annotate.py scan.png -k Bb --placement above --debug
```

| Flag | Meaning |
|---|---|
| `-m`, `--method` | `octave` (default), `pitch`, or `fourth` |
| `-k`, `--key` | Major key signature: `C F Bb Eb Ab Db Gb G D A E B F#`, or `auto` (the default) to read it from the printed signature. Pass a key explicitly if auto-detection reads it wrong |
| `-o`, `--output` | Output PDF path (default: next to the first input image) |
| `--placement` | `below` (default) or `above` the notes |
| `--ledger-range`, `--sensitivity`, `--skip-left` | Same as the website's advanced settings |
| `--no-accidentals` | Key-signature-only reading |
| `--no-dewarp` | Skip the automatic page-curvature correction |
| `--debug` | Also writes a `*_debug.png` overlay showing every detected staff line, notehead and barline — the fastest way to diagnose a bad result |

The terminal prints each staff's reading, e.g. `E3*:4  Eb3:1  F#3*:2`, which is the fastest way to proof the detection against the printed page.

Alongside the PDF, two MIDI files are always written next to it (names follow `-o` if given): `*_equal.mid` and `*_rhythm.mid` — see *What it does* below.

## Reading the output

- **Green / blue / dark red number** — a confident reading; the colour identifies the method (see table above).
- **`*` after a note name (terminal log only)** — the pitch was determined by a printed accidental (or one earlier in the same measure), not the key signature.
- **Orange number, `?` after the note name in the log** — something unreadable (often a rest, or a smudged glyph) sits where an accidental would be. The printed position assumes no accidental; check that note against the page by eye.
- **No number under a detected note, `X:?` in the log** — the note has no slide position in the chosen method's table (out of range, or a misread pitch). Nothing is printed so you can pencil the position in yourself. The per-page summary splits the counts, e.g. `3/57 notes flagged: 2 orange (verify by eye), 1 blank (no position for this method — pencil in)`.

## What it does

- Bass clef music: solo lines, one note at a time.
- **Key signature detection**: the printed flats or sharps after each clef are read and majority-voted across the page's staves (each page is detected independently). The result is printed as a `detected key` line; a manual key always overrides it.
- Solid (quarter/eighth), half and whole noteheads, on the staff and on ledger lines.
- Printed **sharps, flats and naturals**, applied for the rest of their measure like a human reader would; barlines are detected to know where measures end.
- Multi-page input to a single multi-page PDF.
- **Two Standard MIDI Files per run**, playing the detected notes at the **sounding pitch** of the chosen method (as written for `pitch`, an octave up for `octave`, a fourth up for `fourth`): one with every note an even quarter, one with durations **guessed from the engraving's note spacing** (quantized eighth through whole notes). Fixed tempo, quarter = 90 BPM — change it in any MIDI player or DAW.
- Deskewing of tilted photos, straightening of bowed staff lines (page curvature in phone photos), and adaptive thresholding for uneven lighting.

## What it doesn't do

- **No treble/tenor/alto clefs** — bass clef is assumed everywhere.
- **No chords** — stacked noteheads on one stem will not be read reliably.
- **No minor keys as such** — the signature is what's detected, which serves minor keys fine; when overriding manually, pass the relative major (for C minor, use `Eb`).
- **No mid-piece key changes** — one key per page. If the signature changes mid-page, split the pages or annotate the sections separately with explicit `-k` values. Cancellation naturals in a key change are not understood.
- **No double sharps/flats, no courtesy accidentals in parentheses.**
- **Ties across a barline don't carry their accidental** — the alteration resets at the bar, so the second tied note may be annotated a half step off. Ties within a measure are fine.
- **Rhythm is not read** — annotation is pitch-only; it doesn't know a quarter note from an eighth. The *guessed rhythm* MIDI estimates durations from horizontal note spacing, not from flags, beams or dots, so treat it as a sketch; rests are not detected and are not reproduced as silence.
- Lyrics, chord symbols, dynamics and other text are usually ignored correctly, but dense markings can occasionally produce a stray detection.
- Whole notes sitting directly on a staff line are occasionally missed.
- Page curvature in photos is straightened automatically before reading, but sharp creases, strong shadows or extreme perspective can still defeat detection — flatter and more evenly lit is always better. If the correction itself misbehaves on a page, turn it off (`--no-dewarp` / the advanced-settings checkbox).

Always proof the first page of anything against the printed music, especially the orange flags. The `--debug` overlay (CLI) shows exactly what was detected.

## Files

| File | Purpose |
|---|---|
| `alto_annotate.py` | The annotator. Used by the website and runnable directly from a terminal |
| `index.html` | The entire website: UI, Pyodide loading, and the glue that runs the script in-browser |
| `README.md` | This file |

Only annotate music you have the rights to.
