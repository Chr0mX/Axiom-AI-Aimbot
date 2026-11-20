"""One rule for "Move Method also sets the click method", shared by Qt and Web Control."""
from types import SimpleNamespace as NS

import pytest

from core import web_control_settings as wcs
from core.aim_coupling import CLICK_FOLLOWS_MOVE, click_method_for_move


@pytest.mark.parametrize("move", CLICK_FOLLOWS_MOVE)
def test_device_and_network_backends_take_the_click_with_them(move):
    assert click_method_for_move(move, "mouse_event") == move


@pytest.mark.parametrize("move", ["sendinput", "mouse_event", "ddxoft", "arduino", "xbox"])
def test_other_move_methods_leave_the_click_choice_alone(move):
    assert click_method_for_move(move, "sendinput") == "sendinput"
    assert click_method_for_move(move, "makcu") == "makcu"


class TestWebWritePath:
    @staticmethod
    def cfg(**kw):
        c = NS(mouse_move_method="sendinput", mouse_click_method="mouse_event")
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    @pytest.mark.parametrize("move", ["makcu", "vigem_network"])
    def test_one_post_sets_both_like_the_qt_page(self, move):
        c = self.cfg()
        assert wcs.apply_tab_settings(c, "aim", {"mouse_move_method": move})["ok"] is True
        assert (c.mouse_move_method, c.mouse_click_method) == (move, move)

    def test_other_methods_keep_the_current_click(self):
        c = self.cfg(mouse_click_method="makcu")
        wcs.apply_tab_settings(c, "aim", {"mouse_move_method": "sendinput"})
        assert c.mouse_click_method == "makcu"

    def test_an_explicit_click_method_in_the_same_body_wins(self):
        c = self.cfg()
        wcs.apply_tab_settings(c, "aim", {"mouse_move_method": "makcu", "mouse_click_method": "sendinput"})
        assert (c.mouse_move_method, c.mouse_click_method) == ("makcu", "sendinput")

    def test_trigger_tab_offers_every_click_backend_the_aim_page_can_set(self):
        choices = wcs._SCHEMA["trigger"]["mouse_click_method"]["choices"]
        assert set(CLICK_FOLLOWS_MOVE) <= set(choices)
