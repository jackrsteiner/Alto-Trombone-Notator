"""Shared test harness: run the annotator once per fixture image and share
the parsed result across all test functions (a run costs several seconds)."""
import io
import json
import os
import re
import sys
import contextlib
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alto_annotate as aa

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected")


class PageResult:
    """One CV pass over a fixture page plus an octave-method render:
    structured events, the shared PageAnalysis (so multi-method tests can
    re-render without repeating detection), and the few log lines the script
    only reports as text (detected key, flagged summary)."""

    def __init__(self, path):
        progress = []
        aa.PROGRESS = progress.append
        try:
            args = SimpleNamespace(method="octave", key="auto", placement="below",
                                   ledger_range=4.5, sensitivity=1.0, skip_left=8.0,
                                   no_accidentals=False, no_dewarp=False, debug=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.analysis = aa.analyze_page(path, args, None)
                self.img = aa.render_page(self.analysis, "octave")
                self.events = self.analysis.notes
        finally:
            aa.PROGRESS = None
        self.log = buf.getvalue()
        self.progress = progress

        self.key_line = next((l.strip() for l in self.log.splitlines()
                              if "detected key" in l), "")
        m = re.search(r"flagged: (?:(\d+) orange)?[^\d]*(?:(\d+) blank)?",
                      self.log)
        self.flagged_orange = int(m.group(1) or 0) if m else 0
        self.flagged_blank = int(m.group(2) or 0) if m else 0

        self.staff_midis = {}
        for e in self.events:
            self.staff_midis.setdefault(e["staff"], []).append(e["midi"])

    @property
    def staff_note_counts(self):
        return [len(self.staff_midis[k]) for k in sorted(self.staff_midis)]

    def upscale_messages(self):
        return [m for m in self.progress if "upscaling" in m]


def load_expected(name):
    with open(os.path.join(EXPECTED, name + ".json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def louie():
    return PageResult(os.path.join(FIXTURES, "louie_louie_preview.png"))


@pytest.fixture(scope="session")
def misty():
    return PageResult(os.path.join(FIXTURES, "misty_photo.jpg"))
