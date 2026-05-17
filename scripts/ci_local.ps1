param(
    [switch]$SkipOptional
)

pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pytest -m contract
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pytest -m regression
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m compileall api app attention ber_tokenscope embeddings explainability financial_nlp model_comparison model_serving observability security deployment persistence
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose config --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipOptional) {
    if (Get-Command ruff -ErrorAction SilentlyContinue) {
        ruff check .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        ruff format --check .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    if (Get-Command mypy -ErrorAction SilentlyContinue) {
        mypy api app attention ber_tokenscope embeddings explainability financial_nlp model_comparison model_serving observability security deployment persistence
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

"Local CI checks passed."
