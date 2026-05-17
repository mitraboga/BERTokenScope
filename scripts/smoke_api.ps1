param(
    [string]$BaseUrl = "http://127.0.0.1:8788",
    [string]$ApiKey = $env:BERTSCOPE_API_KEY
)

if (-not $ApiKey) {
    throw "Set BERTSCOPE_API_KEY or pass -ApiKey before running the smoke test."
}

$headers = @{ "X-API-Key" = $ApiKey; "X-Request-ID" = "smoke-api" }

$health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
if ($health.status -ne "ok") {
    throw "Health check failed."
}

$body = @{ text = "Revenue growth was strong, but inflation risk remains." } | ConvertTo-Json
$analysis = Invoke-RestMethod -Uri "$BaseUrl/api/v1/financial-analysis" -Method Post -Headers $headers -Body $body -ContentType "application/json"
if (-not $analysis.sentiment.label) {
    throw "Financial analysis smoke check failed."
}

$metrics = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/metrics"
if ($metrics.Content -notmatch "bertscope_requests_total") {
    throw "Metrics smoke check failed."
}

"BERTScope API smoke test passed."
