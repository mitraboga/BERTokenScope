from pathlib import Path


def test_smoke_scripts_exist():
    assert Path("scripts/smoke_api.ps1").exists()
    assert Path("scripts/load_probe.py").exists()
