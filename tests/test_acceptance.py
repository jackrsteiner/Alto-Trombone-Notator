"""Acceptance tests over the two committed fixture pages.

louie_louie_preview.png — clean 900px preview: strict criteria pinning the
low-resolution auto-upscale path (exact pitches, exact counts, exact key).
misty_photo.jpg — phone photo: tolerance-banded criteria for the deskew /
dewarp / native-resolution path, loose enough to survive minor library-version
drift, tight enough to catch real regressions.
The numbers live in tests/expected/*.json; update them deliberately when
detection behaviour is meant to change.
"""
import os

import pytest

import alto_annotate as aa
from conftest import load_expected


def parse_midi_note_ons(data):
    """Note-on pitches from a format-0 SMF as written by write_midi_bytes."""
    assert data[:4] == b"MThd"
    assert data[14:18] == b"MTrk"
    i, end = 22, 22 + int.from_bytes(data[18:22], "big")
    pitches = []
    while i < end:
        while data[i] & 0x80:               # delta time VLQ
            i += 1
        i += 1
        status = data[i]
        if status == 0xFF:                  # meta: type, VLQ length, payload
            length = data[i + 2]
            i += 3 + length
        elif status & 0xF0 == 0xC0:         # program change
            i += 2
        else:                               # note on/off: pitch, velocity
            if status & 0xF0 == 0x90 and data[i + 2] > 0:
                pitches.append(data[i + 1])
            i += 3
    return pitches


# ----------------------------------------------------------------- Louie Louie

class TestLouiePreviewStrict:
    exp = load_expected("louie_louie_preview")

    def test_upscale_path_taken(self, louie):
        msgs = louie.upscale_messages()
        assert msgs and self.exp["upscale_message"] in msgs[0]

    def test_key_detected(self, louie):
        assert louie.key_line == self.exp["key_line"]

    def test_staff_and_note_counts(self, louie):
        assert len(louie.staff_midis) == self.exp["staves"]
        assert louie.staff_note_counts == self.exp["staff_note_counts"]
        assert len(louie.events) == self.exp["total_notes"]

    def test_exact_pitches_per_staff(self, louie):
        for si, want in enumerate(self.exp["staff_midis"]):
            assert louie.staff_midis[si] == want, f"staff {si + 1} pitches differ"

    def test_repeated_lines_read_identically(self, louie):
        # the piece prints the same music on staves 1/6 and 2/7
        assert louie.staff_midis[0] == louie.staff_midis[5]
        assert louie.staff_midis[1] == louie.staff_midis[6]

    def test_nothing_flagged(self, louie):
        assert louie.flagged_orange <= self.exp["flagged_orange_max"]
        assert louie.flagged_blank <= self.exp["flagged_blank_max"]

    def test_output_restored_to_original_size(self, louie):
        assert list(louie.img.size) == self.exp["annotated_size"]

    def test_midi_files(self, louie):
        equal, rhythm = aa.build_midi_files(louie.events, "octave")
        for data in (equal, rhythm):
            pitches = parse_midi_note_ons(data)
            assert len(pitches) == self.exp["total_notes"]
        # octave method sounds one octave above written
        assert parse_midi_note_ons(equal) == [e["midi"] + 12 for e in louie.events]


# ----------------------------------------------------------------------- Misty

class TestMistyPhotoTolerant:
    exp = load_expected("misty_photo")

    def test_native_resolution_path(self, misty):
        assert not misty.upscale_messages()
        assert misty.img.width == self.exp["annotated_width"]

    def test_key_detected(self, misty):
        assert self.exp["key_contains"] in misty.key_line

    def test_staff_and_note_counts_within_band(self, misty):
        assert len(misty.staff_midis) == self.exp["staves"]
        tol = self.exp["staff_note_count_tolerance"]
        for si, want in enumerate(self.exp["staff_note_counts"]):
            got = len(misty.staff_midis.get(si, []))
            assert abs(got - want) <= tol, f"staff {si + 1}: {got} notes vs ~{want}"
        assert len(misty.events) >= self.exp["total_notes_min"]

    def test_pitches_plausible_for_trombone(self, misty):
        # phantom detections from chord symbols / text read as absurd pitches
        midis = [e["midi"] for e in misty.events]
        assert min(midis) >= self.exp["midi_min"]
        assert max(midis) <= self.exp["midi_max"]

    def test_flag_budget(self, misty):
        assert misty.flagged_orange <= self.exp["flagged_orange_max"]
        assert misty.flagged_blank <= self.exp["flagged_blank_max"]


# ------------------------------------------------------------- Caption helpers

class TestKeyCaption:
    def test_describe_key(self):
        assert (aa.describe_key("Eb", -3, "auto-detected")
                == "key of Eb major (3 flats), auto-detected")
        assert (aa.describe_key("F", -1, "set manually")
                == "key of F major (1 flat), set manually")
        assert (aa.describe_key("C", 0, "set manually")
                == "key of C major (no sharps or flats), set manually")

    def test_manual_key_normalisation(self):
        # user-typed key -> canonical caption name, via the same cleaning
        # key_accidentals uses
        for typed, name in (("eb", "Eb"), ("E♭", "Eb"), ("F#", "F#"),
                            ("FS", "F#"), ("c", "C"), ("bb", "Bb")):
            k = aa._key_number(typed)
            assert k is not None, typed
            canonical = aa.FLAT_KEYS[-k] if k < 0 else aa.SHARP_KEYS[k]
            assert canonical == name

    def test_unknown_key_is_none(self):
        assert aa._key_number("H") is None


# ---------------------------------------------------------------------- Shared

def test_pdf_assembly(louie, misty, tmp_path):
    # same call as main() and the website RUNNER
    out = tmp_path / "out.pdf"
    louie.img.save(out, save_all=True, append_images=[misty.img], resolution=150)
    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:5] == b"%PDF-"
