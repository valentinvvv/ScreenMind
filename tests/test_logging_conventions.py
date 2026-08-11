"""Tests for logging conventions — enforces migration integrity."""

import os
import re


# All source files that should use logging (excludes tests, overlay, setup scripts)
def _source_files():
    """Yield all .py source files in the project."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in (
            '__pycache__', '.git', 'node_modules', 'venv', '.venv', 'tests', '.agents',
            'dist', 'dist_build', 'build', '.eggs',
        )]
        for fn in files:
            if not fn.endswith('.py'):
                continue
            if fn.startswith('test_') or fn in ('conftest.py',):
                continue
            yield os.path.join(root, fn)


def test_no_bare_print_in_source():
    """No bare print() calls in source (all output via logging)."""
    # Allowed files: overlay.py (subprocess GUI), setup scripts
    EXCLUDED = {'overlay.py', 'setup_llama.py', 'setup_wizard.py'}
    bare_prints = []
    for filepath in _source_files():
        basename = os.path.basename(filepath)
        if basename in EXCLUDED:
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith('print(') and '# noqa' not in line:
                    rel = os.path.relpath(filepath, os.path.dirname(os.path.dirname(__file__)))
                    bare_prints.append(f"{rel}:{lineno}: {stripped[:80]}")

    assert bare_prints == [], (
        f"Found {len(bare_prints)} bare print() calls (use logger instead):\n"
        + "\n".join(bare_prints)
    )


def test_no_emoji_in_logger_calls():
    """Emoji in logger messages crash on Windows cp1252 terminals."""
    emoji_re = re.compile(r'[\U0001F300-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\uFE0F]')
    EXCLUDED = {'overlay.py'}
    hits = []
    for filepath in _source_files():
        if os.path.basename(filepath) in EXCLUDED:
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                if 'logger.' in line and emoji_re.search(line):
                    rel = os.path.relpath(filepath, os.path.dirname(os.path.dirname(__file__)))
                    safe = emoji_re.sub('[EMOJI]', line.strip()[:80])
                    hits.append(f"{rel}:{lineno}: {safe}")

    assert hits == [], (
        f"Found {len(hits)} emoji in logger calls (crashes cp1252 terminals):\n"
        + "\n".join(hits)
    )


def test_logger_uses_screenmind_namespace():
    """All logger declarations must use screenmind.* namespace."""
    bad = []
    for filepath in _source_files():
        with open(filepath, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                if 'logging.getLogger(' in line and 'logger' in line:
                    # Extract the logger name
                    match = re.search(r'getLogger\(["\']([^"\']+)["\']\)', line)
                    if match:
                        name = match.group(1)
                        if not name.startswith('screenmind'):
                            rel = os.path.relpath(filepath, os.path.dirname(os.path.dirname(__file__)))
                            bad.append(f"{rel}:{lineno}: logger name '{name}' (must start with 'screenmind.')")

    assert bad == [], (
        f"Found {len(bad)} loggers outside screenmind namespace:\n"
        + "\n".join(bad)
    )
