"""
HUD detection categorization — maps a raw YOLO11n HUD-detector class name
(as produced by hud_inference.py's Model_Hud/best.onnx-class taxonomy) to a
semantic category (weapon / fire_mode / optic / barrel_laser / stock / mag /
hopup / turbo) plus its rarity tier, if any.

Deliberately pure Python with zero dependencies (no numpy/cv2/onnxruntime) —
unlike hud_inference.py itself, this module only interprets already-decoded
text lines (the "{name}: {score}%" strings hud_inference.get_hud_results()
already produces), so it stays importable and unit-testable without the
heavy runtime hud_inference.py needs, and the GUI is the only caller.

Built against the 91-class taxonomy baked into Model_Hud/best.onnx (weapon
name, scope-tier, attachment-tier-with-rarity, fire-mode). The smaller
shipped model variants (Apex_Weapon_584_*_Y11n.onnx, 25 classes) cover only
a subset of this — weapon names, optic tiers, and barrel rarity — so a name
they emit that isn't in this table simply falls through as "unknown" and the
caller shows a blank/"—" for that category, exactly as if nothing had been
detected. No caller needs to know which model produced a given name.
"""

from __future__ import annotations

import re

# ── Weapon names (no rarity suffix) ────────────────────────────────────────
_WEAPON_NAMES: frozenset[str] = frozenset({
    "30-30 Repeater", "Alternator SMG", "Bocek Compound Bow", "C.A.R. SMG",
    "Charge Rifle", "Devotion LMG", "EVA-8 Auto", "G7 Scout", "Havoc Rifle",
    "Hemlok Burst AR", "Kraber", "L-STAR EMG", "Longbow DMR", "M600 Spitfire",
    "Mastiff Shotgun", "Mozambique", "Mozambique Akimbo", "Nemesis Burst AR",
    "P2020", "P2020 Akimbo", "Peacekeeper", "Prowler Burst PDW",
    "R-301 Carbine", "R-99 SMG", "RE-45", "Rampage LMG", "Sentinel",
    "Triple Take", "VK-47 Flatline", "Volt SMG", "Wingman",
    # Leaner 25-class variants use shorter names for the same weapons.
    "Alternator", "Car", "Devotion", "Flatline", "G7", "Havoc", "Hemlock",
    "L-star", "Nemesis", "Prowler", "R301", "R99", "Rampage", "Re-45",
    "Spitfire", "Volt", "p2020",
})

# ── Fire mode (no rarity suffix; the displayed value IS the mode name) ─────
_FIRE_MODE_VALUES: dict[str, str] = {
    "Fire-mode 2": "2",
    "Fire-mode Auto": "Auto",
}

# ── Generic (rarity-less) hop-up presence flag ─────────────────────────────
_HOPUP_GENERIC = "Hop-Ups (Yes)"

# ── Base name (rarity suffix already stripped) -> category ────────────────
_BASE_NAME_CATEGORY: dict[str, str] = {
    # Optics / scopes
    "1x Digital Threat": "optic",
    "1x HCOG Classic": "optic",
    "1x Holo": "optic",
    "1x-2x Variable Holo": "optic",
    "2x HCOG Bruiser": "optic",
    "2x-4x Variable AOG": "optic",
    "3x HCOG Ranger": "optic",
    "4x-10x Digital Sniper Threat": "optic",
    "4x-8x Variable Sniper": "optic",
    "6x Sniper": "optic",
    # Barrel / laser — shown as one combined category, mirroring how the
    # in-game HUD itself only ever shows one attachment icon in that slot.
    "Barrel Stabilizer": "barrel_laser",
    "Standard Laser sight": "barrel_laser",
    # Stock-equivalent (rifle/sniper stock, or a shotgun's bolt slot)
    "Sniper Stock": "stock",
    "Standard Stock": "stock",
    "Shotgun Bolt": "stock",
    # Mag
    "Extended Energy Mag": "mag",
    "Extended Heavy Mag": "mag",
    "Extended Light Mag": "mag",
    "Extended Sniper Mag": "mag",
    # Named legendary hop-ups
    "Anvil Receiver": "hopup",
    "Deadeye's Tempo": "hopup",
    "Double Tap Trigger": "hopup",
    "Dual Shell": "hopup",
    "Kinetic Feeder": "hopup",
    "Precision Choke": "hopup",
    "Quickdraw Holster": "hopup",
    "Selectfire Receiver": "hopup",
    "Shatter Caps": "hopup",
    "Skullpiercer Rifling": "hopup",
    # Turbo
    "Turbocharger": "turbo",
    # Leaner 25-class barrel-only variants (no laser class in that model)
    "BlueBarrel": "barrel_laser",
    "PurpleBarrel": "barrel_laser",
    "WhiteBarrel": "barrel_laser",
    "TurboCharger": "turbo",
}

# Presentation order + display labels for the categories above, shared by
# any caller building a fixed-layout status panel.
CATEGORY_ORDER: tuple[str, ...] = (
    "weapon", "fire_mode", "barrel_laser", "optic", "turbo", "stock", "mag", "hopup",
)
CATEGORY_LABELS: dict[str, str] = {
    "weapon": "Weapon",
    "fire_mode": "Fire Mode",
    "barrel_laser": "Barrel/Laser",
    "optic": "Optic",
    "turbo": "Turbo",
    "stock": "Stock",
    "mag": "Mag",
    "hopup": "Hop-Up",
}

_RARITIES: tuple[str, ...] = ("common", "rare", "epic", "legendary")

_RARITY_RE = re.compile(r"^(.*?)\s*\((Common|Rare|Epic|Legendary)\)$")
_RESULT_LINE_RE = re.compile(r"^(.*): (\d+)%$")


def classify_hud_name(name: str) -> tuple[str, str | None, str]:
    """Classify one raw HUD class name.

    Returns (category, rarity, display_value):
      - category: one of CATEGORY_ORDER, or "unknown" if not recognized.
      - rarity: one of "common"/"rare"/"epic"/"legendary", or None (weapons,
        fire mode, and the generic hop-up flag carry no rarity).
      - display_value: the name with any rarity suffix stripped (or the
        fire-mode's own short value, e.g. "Auto").
    """
    if name in _WEAPON_NAMES:
        return "weapon", None, name
    if name in _FIRE_MODE_VALUES:
        return "fire_mode", None, _FIRE_MODE_VALUES[name]
    if name == _HOPUP_GENERIC:
        return "hopup", None, "Yes"

    m = _RARITY_RE.match(name)
    if m:
        base, rarity = m.group(1), m.group(2).lower()
        category = _BASE_NAME_CATEGORY.get(base)
        if category:
            return category, rarity, base

    # A handful of leaner-model names carry no rarity suffix at all.
    category = _BASE_NAME_CATEGORY.get(name)
    if category:
        return category, None, name

    return "unknown", None, name


def parse_hud_status(lines: list[str]) -> dict[str, dict]:
    """Parse hud_inference.get_hud_results()-style lines into a per-category
    status dict.

    Each input line is expected in the "{name}: {score}%" format that
    hud_inference._postprocess() already produces. Lines are assumed already
    sorted by score descending (as _postprocess() sorts them), so only the
    *first* line seen for a given category is kept — the highest-confidence
    detection wins when more than one candidate for the same category (e.g.
    two barrel-tier anchors both firing) passes threshold.

    Returns {category: {"value": str, "rarity": str | None, "score": int}}.
    A category with no current detection is simply absent from the result —
    callers render that as "None"/"—", never an explicit empty entry.
    Lines that don't match the expected format, or whose parsed name isn't
    in the classification table (an "unknown" category, or a below-threshold
    "[below threshold] best: ..." hint line), are skipped.
    """
    status: dict[str, dict] = {}
    for line in lines:
        m = _RESULT_LINE_RE.match(line)
        if not m:
            continue
        name, score_str = m.group(1), m.group(2)
        category, rarity, value = classify_hud_name(name)
        if category == "unknown" or category in status:
            continue
        status[category] = {"value": value, "rarity": rarity, "score": int(score_str)}
    return status
