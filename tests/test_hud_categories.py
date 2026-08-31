"""Tests for core.hud_categories — pure-Python HUD class-name categorization."""

from core.hud_categories import classify_hud_name, parse_hud_status, CATEGORY_ORDER


class TestClassifyHudName:
    def test_weapon_name_has_no_rarity(self):
        category, rarity, value = classify_hud_name("R-301 Carbine")
        assert category == "weapon"
        assert rarity is None
        assert value == "R-301 Carbine"

    def test_leaner_model_short_weapon_name(self):
        category, rarity, value = classify_hud_name("R301")
        assert category == "weapon"
        assert rarity is None
        assert value == "R301"

    def test_fire_mode_auto(self):
        category, rarity, value = classify_hud_name("Fire-mode Auto")
        assert category == "fire_mode"
        assert rarity is None
        assert value == "Auto"

    def test_fire_mode_2(self):
        category, rarity, value = classify_hud_name("Fire-mode 2")
        assert value == "2"

    def test_generic_hopup_flag_has_no_rarity(self):
        category, rarity, value = classify_hud_name("Hop-Ups (Yes)")
        assert category == "hopup"
        assert rarity is None
        assert value == "Yes"

    def test_named_legendary_hopup(self):
        category, rarity, value = classify_hud_name("Selectfire Receiver (Legendary)")
        assert category == "hopup"
        assert rarity == "legendary"
        assert value == "Selectfire Receiver"

    def test_optic_with_rarity(self):
        category, rarity, value = classify_hud_name("2x HCOG Bruiser (Rare)")
        assert category == "optic"
        assert rarity == "rare"
        assert value == "2x HCOG Bruiser"

    def test_barrel_and_laser_share_one_category(self):
        barrel = classify_hud_name("Barrel Stabilizer (Epic)")
        laser = classify_hud_name("Standard Laser sight (Common)")
        assert barrel[0] == "barrel_laser"
        assert laser[0] == "barrel_laser"

    def test_leaner_model_barrel_variant_no_rarity(self):
        category, rarity, value = classify_hud_name("PurpleBarrel")
        assert category == "barrel_laser"
        assert rarity is None
        assert value == "PurpleBarrel"

    def test_stock_covers_sniper_standard_and_shotgun_bolt(self):
        assert classify_hud_name("Sniper Stock (Rare)")[0] == "stock"
        assert classify_hud_name("Standard Stock (Common)")[0] == "stock"
        assert classify_hud_name("Shotgun Bolt (Legendary)")[0] == "stock"

    def test_mag_variants(self):
        assert classify_hud_name("Extended Light Mag (Epic)")[0] == "mag"
        assert classify_hud_name("Extended Sniper Mag (Legendary)")[0] == "mag"

    def test_turbo(self):
        category, rarity, value = classify_hud_name("Turbocharger (Legendary)")
        assert category == "turbo"
        assert rarity == "legendary"
        assert value == "Turbocharger"

    def test_leaner_model_turbo_variant(self):
        assert classify_hud_name("TurboCharger")[0] == "turbo"

    def test_unrecognized_name_is_unknown(self):
        category, rarity, value = classify_hud_name("Not A Real Class")
        assert category == "unknown"
        assert rarity is None
        assert value == "Not A Real Class"

    def test_every_declared_category_is_in_category_order(self):
        # Guards against a category constant existing in the mapping table
        # but forgotten in the presentation order tuple.
        seen = set()
        for name in ("R-301 Carbine", "Fire-mode Auto", "Barrel Stabilizer (Rare)",
                     "2x HCOG Bruiser (Rare)", "Turbocharger (Legendary)",
                     "Sniper Stock (Common)", "Extended Light Mag (Epic)",
                     "Hop-Ups (Yes)"):
            seen.add(classify_hud_name(name)[0])
        assert seen == set(CATEGORY_ORDER)


class TestParseHudStatus:
    def test_empty_lines_yields_empty_status(self):
        assert parse_hud_status([]) == {}

    def test_single_weapon_line(self):
        status = parse_hud_status(["R301: 92%"])
        assert status == {"weapon": {"value": "R301", "rarity": None, "score": 92}}

    def test_multiple_categories_all_captured(self):
        lines = [
            "Wingman: 91%",
            "Fire-mode Auto: 80%",
            "2x HCOG Bruiser (Rare): 70%",
            "Turbocharger (Legendary): 65%",
        ]
        status = parse_hud_status(lines)
        assert status["weapon"]["value"] == "Wingman"
        assert status["fire_mode"]["value"] == "Auto"
        assert status["optic"] == {"value": "2x HCOG Bruiser", "rarity": "rare", "score": 70}
        assert status["turbo"]["rarity"] == "legendary"

    def test_first_line_wins_when_two_hit_the_same_category(self):
        # _postprocess() sorts by score descending, so the first line for a
        # category is always the highest-confidence one for that category.
        lines = ["Barrel Stabilizer (Epic): 88%", "Standard Laser sight (Rare): 60%"]
        status = parse_hud_status(lines)
        assert status["barrel_laser"] == {"value": "Barrel Stabilizer", "rarity": "epic", "score": 88}

    def test_below_threshold_hint_line_is_skipped(self):
        lines = ["[below threshold] best: Wingman 45%  (threshold=50%)"]
        assert parse_hud_status(lines) == {}

    def test_unknown_name_is_skipped(self):
        assert parse_hud_status(["Not A Real Class: 99%"]) == {}

    def test_malformed_line_is_skipped_not_raised(self):
        assert parse_hud_status(["garbage line with no percent"]) == {}
