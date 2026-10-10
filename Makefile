.PHONY: results check-publish install lint format test ci train api dashboard simulate clean
install:
	pip install -r requirements-dev.txt
lint:
	ruff check . && ruff format --check .
format:
	ruff check --fix . && ruff format .
test:
	pytest
ci: lint test
train:
	python scripts/download_data.py && python -m src.train
api:
	uvicorn app.api:app --reload
dashboard:
	streamlit run app/dashboard.py
simulate:
	python scripts/simulate_sensor_stream.py
clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
check-publish:
	python scripts/pre_publish_check.py
results:
	python scripts/generate_results_table.py --update-readme
