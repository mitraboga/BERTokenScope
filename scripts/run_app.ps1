param(
    [int]$Port = 8501
)

$env:BERTSCOPE_DASHBOARD_PORT="$Port"
python -m streamlit run app/streamlit_app.py --server.port $Port --server.headless true
