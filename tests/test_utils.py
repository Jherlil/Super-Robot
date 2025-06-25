import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import load_config


def test_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("email: a\npassword: b")
    monkeypatch.setenv("BOT_EMAIL", "x")
    monkeypatch.setenv("BOT_PASSWORD", "y")
    c = load_config(str(cfg))
    assert c["email"] == "x"
    assert c["password"] == "y"