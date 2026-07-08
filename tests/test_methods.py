"""Multi-method output: one CV pass feeding several notation methods, the
repeatable -m flag, per-method output naming, and the shared run_job driver."""
import copy
import io
import contextlib
import os
from types import SimpleNamespace

import pytest

import alto_annotate as aa
from conftest import FIXTURES
from test_acceptance import parse_midi_note_ons

ALL_METHODS = ["octave", "pitch", "fourth"]


# ------------------------------------------------- analysis/render separation

class TestRenderSeparation:
    def test_three_renders_from_one_analysis(self, louie):
        pages = {m: aa.render_page(louie.analysis, m) for m in ALL_METHODS}
        sizes = {p.size for p in pages.values()}
        assert len(sizes) == 1, "all methods must render at the same page size"
        raw = {m: p.tobytes() for m, p in pages.items()}
        assert raw["octave"] != raw["pitch"]
        assert raw["octave"] != raw["fourth"]
        assert raw["pitch"] != raw["fourth"]

    def test_render_does_not_mutate_analysis(self, louie):
        before = copy.deepcopy(louie.analysis.notes)
        for m in ALL_METHODS:
            aa.render_page(louie.analysis, m)
        assert louie.analysis.notes == before

    def test_wrapper_matches_split_pipeline(self, louie):
        args = SimpleNamespace(method="octave", key="auto", placement="below",
                               ledger_range=4.5, sensitivity=1.0, skip_left=8.0,
                               no_accidentals=False, no_dewarp=False, debug=False)
        path = os.path.join(FIXTURES, "louie_louie_preview.png")
        with contextlib.redirect_stdout(io.StringIO()):
            img, events = aa.annotate_page(path, args, None)
        assert [e["midi"] for e in events] == [e["midi"] for e in louie.events]
        assert img.size == louie.img.size


# --------------------------------------------------------- per-method sounding

def test_midi_shift_per_method(louie):
    for m in ALL_METHODS:
        equal, rhythm = aa.build_midi_files(louie.events, m)
        want = [e["midi"] + aa.SOUNDING_SHIFT[m] for e in louie.events]
        assert parse_midi_note_ons(equal) == want, m
        assert parse_midi_note_ons(rhythm) == want, m


# ----------------------------------------------------------- method expansion

class TestExpandMethods:
    def test_default(self):
        assert aa.expand_methods(None) == ["octave"]
        assert aa.expand_methods([]) == ["octave"]

    def test_single_and_repeat(self):
        assert aa.expand_methods(["pitch"]) == ["pitch"]
        assert aa.expand_methods(["octave", "fourth"]) == ["octave", "fourth"]

    def test_all_and_dedupe(self):
        assert aa.expand_methods(["all"]) == ALL_METHODS
        assert aa.expand_methods(["pitch", "all"]) == ALL_METHODS
        assert aa.expand_methods(["pitch", "pitch", "octave"]) == ["pitch", "octave"]


# -------------------------------------------------------------- output naming

class TestResolveOutputs:
    def test_single_method_unchanged(self):
        assert (aa.resolve_outputs(["dir/score.png"], ["octave"], None)
                == {"octave": "dir/score_alto_octave.pdf"})
        assert (aa.resolve_outputs(["score.png"], ["pitch"], "out.pdf")
                == {"pitch": "out.pdf"})

    def test_multi_method_default_names(self):
        assert (aa.resolve_outputs(["score.png"], ALL_METHODS, None)
                == {m: f"score_alto_{m}.pdf" for m in ALL_METHODS})

    def test_multi_method_output_suffix(self):
        assert (aa.resolve_outputs(["score.png"], ["octave", "fourth"], "dir/out.pdf")
                == {"octave": "dir/out_octave.pdf", "fourth": "dir/out_fourth.pdf"})


# ------------------------------------------------------------------ end-to-end

def test_run_job_all_methods(tmp_path):
    args = SimpleNamespace(key="auto", placement="below", ledger_range=4.5,
                           sensitivity=1.0, skip_left=8.0, no_accidentals=False,
                           no_dewarp=False, debug=False)
    path = os.path.join(FIXTURES, "louie_louie_preview.png")
    analyzed = []
    real = aa.analyze_page
    aa.analyze_page = lambda *a, **k: (analyzed.append(1) or real(*a, **k))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            written = aa.run_job([path], ALL_METHODS, args, None,
                                 lambda m: str(tmp_path / f"out_{m}.pdf"))
    finally:
        aa.analyze_page = real
    assert len(analyzed) == 1, "the CV pass must run once for all methods"
    assert set(written) == set(ALL_METHODS)
    for m in ALL_METHODS:
        pdf, mid_eq, mid_rhy = written[m]
        assert pdf.endswith(f"out_{m}.pdf")
        assert open(pdf, "rb").read(5) == b"%PDF-"
        for mid in (mid_eq, mid_rhy):
            assert open(mid, "rb").read(4) == b"MThd"
    assert len(list(tmp_path.iterdir())) == 9   # 3 PDFs + 6 MIDI files
