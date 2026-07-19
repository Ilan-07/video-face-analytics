# Face-analytics + video-understanding pipeline.
# One-step bootstrap and the common workflows, so a fresh clone is reproducible
# without remembering the incantations. Uses the project venv at .venv.

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.DEFAULT_GOAL := help
.PHONY: help bootstrap check-bins venv install pipeline transcribe app test smoke lock clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

bootstrap: venv install check-bins ## Full setup: venv + deps + binary checks
	@echo "bootstrap complete. Run 'make pipeline' (needs data), or 'make test'."

venv: ## Create the virtualenv if absent
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv ## Install pinned Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

check-bins: ## Warn if required system binaries are missing
	@command -v tesseract >/dev/null 2>&1 \
	  || echo "WARN: tesseract not found (OCR). macOS: brew install tesseract"
	@command -v ffmpeg >/dev/null 2>&1 \
	  || echo "WARN: ffmpeg not found (audio transcription). macOS: brew install ffmpeg"

pipeline: ## Run the full end-to-end pipeline (idempotent)
	$(PY) run_pipeline.py

transcribe: ## Run only the speech-transcription stage
	$(PY) transcribe.py

app: ## Launch the Streamlit application
	$(VENV)/bin/streamlit run search_app.py

test: ## Run the fast test suite
	$(PY) -m pytest -q

smoke: ## Run tests including the slow real-model integration checks
	$(PY) -m pytest -q --run-slow

lock: ## Freeze the current environment into requirements.lock.txt
	$(PIP) freeze > requirements.lock.txt

clean: ## Remove caches (keeps data/ and models)
	rm -rf .pytest_cache __pycache__ */__pycache__
