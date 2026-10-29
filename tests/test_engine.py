"""Model-free tests for the abstention and verification logic (python -m pytest tests)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from engine import Engine  # noqa: E402

SCORES = {
    "category": {"pred": "skirt", "confidence": 0.99, "threshold": 0.50},
    "color": {"pred": "black", "confidence": 0.40, "threshold": 0.60},
    "material": {"pred": "silk", "confidence": 0.70, "threshold": None},
}


def test_abstention_uses_thresholds():
    e = Engine()
    assert e.evidence(SCORES) == {"category": "skirt", "color": "unknown", "material": "unknown"}
    assert e.evidence(SCORES, abstain=False) == {"category": "skirt", "color": "black", "material": "silk"}


def test_check_flags_unknown_and_wrong_attributes():
    ev = {"category": "skirt", "color": "unknown", "material": "unknown"}
    problems = Engine.check("A red silk skirt for parties.", ev)
    assert {p["attribute"] for p in problems} == {"color", "material"}
    assert Engine.check("A flowing skirt for parties.", ev) == []
    wrong = Engine.check("A stylish dress.", ev)
    assert wrong and wrong[0]["attribute"] == "category"


def test_check_ignores_styling_and_detail_colors():
    ev = {"category": "skirt", "color": "black", "material": "unknown"}
    assert Engine.check("A black skirt with a white logo. Pair it with blue jeans.", ev) == []


def test_fabric_talk_removed_only_when_material_unknown():
    from engine import drop_fabric_talk
    text = "A blue kurta for festive days. Crafted from an unidentifiable material, it feels great."
    kept, dropped = drop_fabric_talk(text, {"category": "kurta", "color": "blue", "material": "unknown"})
    assert kept == "A blue kurta for festive days." and len(dropped) == 1
    assert drop_fabric_talk(text, {"category": "kurta", "color": "blue", "material": "cotton"})[0] == text


def test_well_formed_rejects_slot_lists():
    from engine import well_formed
    assert not well_formed("T-shirt: T-shirt\nblue: Blue")
    assert not well_formed("A blue t-shirt.")
    assert well_formed("Meet our blue t-shirt, an easy everyday essential.")
