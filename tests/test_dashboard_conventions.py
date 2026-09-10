"""Source conventions for the dashboard JS that pytest can enforce.

There is no JS test runner in this repo (CI only does `node --check`), so
the few frontend invariants that have bitten us live here.
"""

import os
import re


def _static_js(name: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "screenmind", "api", "static", "js", name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_radio_pill_helper_compares_stringified_values():
    """_rp must not compare the setting to the radio value with bare ===.

    A radio `value` is always a string; the setting behind it may be a
    number. retention_days is — so `0 === '0'` was false, no pill rendered
    active or checked, and the group came up with nothing selected.
    """
    src = _static_js("settings.js")
    line = next((l for l in src.splitlines() if l.strip().startswith("function _rp(")), None)
    assert line, "_rp helper not found in settings.js"
    assert "String(cur)" in line and "String(val)" in line, (
        "_rp compares `cur` to `val` without normalizing types. `val` is a "
        "string attribute, `cur` comes from JSON and may be a number — see "
        "retention_days. Compare String(cur) === String(val)."
    )


def test_retention_days_is_never_sent_as_a_guess():
    """saveSettings must not invent a retention_days when none is selected.

    The old `retention ? parseInt(retention.value) : 7` fallback rewrote a
    saved "Forever" (0) to 7 days on *any* settings save, and the startup
    cleanup in main.py then permanently deleted everything older than that.
    Omitting the key leaves the stored value alone (see
    TestRuntimeOverrideMerge in test_config.py).
    """
    src = _static_js("settings.js")
    assignments = re.findall(r"retention_days\s*:\s*([^,\n]+)", src)
    for expr in assignments:
        assert "?" not in expr, (
            f"retention_days is assigned from a conditional ({expr.strip()!r}). "
            "A fallback here silently overwrites the user's retention setting "
            "and causes data loss on the next startup cleanup."
        )
