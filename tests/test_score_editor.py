import asyncio
import json

import pytest

from fl_yue2.abc_score import AbcError
from fl_yue2.score_editor import DEFAULT_SCORE, FL_YuE2_ScoreEditor, inspect_score, validate_score, build_score


def test_default_instrumental_and_node_output():
    info = inspect_score(DEFAULT_SCORE)
    assert info["bars"] == 8
    assert info["seconds"] == pytest.approx(32 * 60 / 88)
    assert info["instrumental"] and info["grid_available"]
    assert [s["name"] for s in info["sections"]] == ["verse", "outro"]
    assert FL_YuE2_ScoreEditor().score(DEFAULT_SCORE)["result"] == (DEFAULT_SCORE,)


def test_connected_score_is_authoritative_and_sent_to_editor():
    incoming = DEFAULT_SCORE.replace("=88", "=110")
    output = FL_YuE2_ScoreEditor().score("invalid local draft", incoming)
    assert output["result"] == (incoming,)
    assert output["ui"]["score_abc"] == [incoming]


def test_connected_empty_score_does_not_fall_back_to_local_score():
    with pytest.raises(AbcError):
        FL_YuE2_ScoreEditor().score(DEFAULT_SCORE, "")


def test_connected_edits_persist_until_upstream_score_changes():
    node = FL_YuE2_ScoreEditor()
    loaded = node.score("", DEFAULT_SCORE)
    source_hash = loaded["ui"]["source_score_hash"][0]
    edited = DEFAULT_SCORE.replace("=88", "=95")
    assert node.score(edited, DEFAULT_SCORE, source_hash)["result"] == (edited,)
    changed = DEFAULT_SCORE.replace("=88", "=110")
    refreshed = node.score(edited, changed, source_hash)
    assert refreshed["result"] == (changed,)
    assert refreshed["ui"]["source_score_hash"][0] != source_hash
    assert node.score(edited, None, source_hash)["result"] == (edited,)


def test_invalid_connected_edit_fails_instead_of_discarding_edit():
    node = FL_YuE2_ScoreEditor()
    source_hash = node.score("", DEFAULT_SCORE)["ui"]["source_score_hash"][0]
    with pytest.raises(AbcError):
        node.score("invalid edited score", DEFAULT_SCORE, source_hash)


@pytest.mark.parametrize("before,after,message", [
    ("E2A2c4B2A2G4", "E2A2c4B2A2G2", "duration"),
    ('"Am7"z16', '"Cmaj9"z16', "unsupported chord"),
    ("A8z8", "A8-A8-", "unresolved tie"),
    ("A8z8", "A8-B8", "tie changes pitch"),
    ("A8z8", "[ACE]16", "unsupported token"),
    ("A8z8", '"Am7"A8z8', "Native chord symbols belong in Vocal"),
])
def test_invalid_scores_fail_at_execution(before, after, message):
    with pytest.raises(AbcError, match=message):
        FL_YuE2_ScoreEditor().score(DEFAULT_SCORE.replace(before, after))


def test_raw_key_changes_preserved_without_grid_rewrite():
    score = DEFAULT_SCORE.replace('"Am7"z16', '[K:C]"Am7"z16', 1).replace("E2A2c4B2A2G4", "[K:C]E2A2c4B2A2G4")
    info = inspect_score(score)
    assert not info["grid_available"]
    assert info["abc"] == score


def test_compressed_rests_expand_for_grid():
    score = DEFAULT_SCORE.replace('"Am7"z16|"Dm7"z16|"G7"z16|"Cmaj7"z16|', "Z4|")
    assert inspect_score(score)["sections"][0]["vocal"] == ["Z"] * 4


def test_windows_newlines_and_trailing_space():
    assert inspect_score(DEFAULT_SCORE.replace("\n", "\r\n") + "\r\n")["abc"] == DEFAULT_SCORE


@pytest.mark.parametrize("value", [None, [], "x" * 200001], ids=["null", "list", "oversized"])
def test_input_boundary(value):
    with pytest.raises(AbcError):
        inspect_score(value)


@pytest.mark.parametrize("body", ['{"score_abc":', '[]', json.dumps({"score_abc": DEFAULT_SCORE.replace("Q:1/4=88", "Q:1/4=" + "9" * 5000)})],
                         ids=["malformed-json", "wrong-body-type", "oversized-number"])
def test_validation_route_reports_bad_requests(body):
    class Request:
        async def json(self):
            return json.loads(body)
    response = asyncio.run(validate_score(Request()))
    assert response.status == 400
    assert json.loads(response.body)["error"]


def test_piano_roll_roundtrip_preserves_pitches_timing_and_chords():
    original = inspect_score(DEFAULT_SCORE)
    result = build_score(original)
    assert result["roll"] == original["roll"]
    assert result["instrumental"]


def test_piano_roll_ties_across_bars_and_chord_changes():
    data = inspect_score(DEFAULT_SCORE)
    data["roll"]["tracks"]["Vocal"] = [{"start": 64, "duration": 2304, "pitch": 66}]
    data["roll"]["chords"].append({"start": 320, "symbol": "D7"})
    result = build_score(data)
    assert result["roll"]["tracks"] == data["roll"]["tracks"]
    assert sorted(result["roll"]["chords"], key=lambda c: c["start"]) == sorted(data["roll"]["chords"], key=lambda c: c["start"])


def test_piano_roll_preserves_short_notes_and_adjacent_attacks():
    data = inspect_score(DEFAULT_SCORE)
    data["roll"]["tracks"]["Ins"] = [{"start": 1, "duration": 1, "pitch": 0}, {"start": 2, "duration": 2, "pitch": 127}]
    result = build_score(data)
    assert result["roll"]["tracks"] == data["roll"]["tracks"]


@pytest.mark.parametrize("note,message", [
    ({"start": 1, "duration": 256, "pitch": 60}, "overlap"),
    ({"start": 8190, "duration": 256, "pitch": 60}, "past the end"),
    ({"start": 8000, "duration": 64, "pitch": 128}, "MIDI pitch"),
])
def test_piano_roll_rejects_invalid_notes(note, message):
    data = inspect_score(DEFAULT_SCORE)
    data["roll"]["tracks"]["Ins"].append(note)
    with pytest.raises(AbcError, match=message):
        build_score(data)


def test_piano_roll_key_change_does_not_transpose_notes():
    data = inspect_score(DEFAULT_SCORE)
    data["key"] = "F#"
    assert build_score(data)["roll"]["tracks"] == data["roll"]["tracks"]
