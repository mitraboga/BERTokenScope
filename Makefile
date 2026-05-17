.PHONY: test compile lint format typecheck security-audit ci app api

test:
	pytest

compile:
	python -m compileall api app attention ber_tokenscope embeddings explainability financial_nlp model_comparison model_serving observability security deployment persistence

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy api app attention ber_tokenscope embeddings explainability financial_nlp model_comparison model_serving observability security deployment persistence

security-audit:
	bandit -q -r api app attention ber_tokenscope embeddings explainability financial_nlp model_comparison model_serving observability security deployment persistence
	pip-audit

ci:
	pytest
	pytest -m contract
	pytest -m regression
	python -m compileall api app attention ber_tokenscope embeddings explainability financial_nlp model_comparison model_serving observability security deployment persistence
	docker compose config --quiet

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn api.main:app --host 127.0.0.1 --port 8788 --reload
