from pathlib import Path


def test_cicd_workflows_exist():
    assert Path(".github/workflows/ci.yml").exists()
    assert Path(".github/workflows/release.yml").exists()
    assert Path(".github/workflows/deploy-staging.yml").exists()


def test_dev_requirements_and_local_ci_script_exist():
    assert Path("requirements-dev.txt").exists()
    assert Path("scripts/ci_local.ps1").exists()
