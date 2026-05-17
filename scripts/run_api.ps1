param(
    [int]$Port = 8788
)

$env:BERTSCOPE_API_PORT="$Port"
python -m uvicorn api.main:app --host 127.0.0.1 --port $Port --reload
