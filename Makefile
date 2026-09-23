.PHONY: install test test-integration lint format typecheck boundaries security workflows serve clean audit audit-clean

install:
	uv sync --all-extras

# ci.yml calls these targets; spelling a command out there again is how the two drifted apart
test:
	uv run pytest tests/unit/ -v --cov=ai_agent --cov-report=xml

test-integration:
	uv run pytest tests/integration/ -m integration -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run pyright

boundaries:
	uv run lint-imports

# pip-audit reads the synced environment: `-r` makes it build a venv of its own, which
# ensurepip cannot always create. `uv sync --locked` first, so what it reads is the lock.
security:
	uvx semgrep scan --metrics=off --error --config p/python src
	uvx bandit -q -r -ll src
	uv run --with pip-audit pip-audit --progress-spinner off --skip-editable

workflows:
	uvx zizmor --offline .github/workflows

serve:
	uv run uvicorn ai_agent.app:create_app --factory --host 0.0.0.0 --port 8484 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Reproducible security/quality scanner suite. Missing tools warn-skip so this
# target runs on any host with whatever subset is installed. Every tool dumps
# raw JSON to $(AUDIT_DIR) so findings can be inspected without an LLM in the
# loop. Run `make audit` after every change touching deps, network/auth, or
# request handling.
AUDIT_DIR ?= audit-out

audit:
	@mkdir -p $(AUDIT_DIR)
	@echo "==> ruff (lint)"
	@uv run ruff check --output-format json src/ tests/ > $(AUDIT_DIR)/ruff.json 2>&1 || true
	@echo "==> bandit (Python AST security)"
	@if uv run --with bandit bandit --version >/dev/null 2>&1; then \
		uv run --with bandit bandit -r src -f json -o $(AUDIT_DIR)/bandit.json -q || true; \
	else \
		echo "  SKIP: bandit unavailable"; \
	fi
	@echo "==> pip-audit (dep CVEs)"
	@if command -v pip-audit >/dev/null 2>&1; then \
		uv export --format requirements-txt --no-hashes > $(AUDIT_DIR)/.requirements.txt 2>/dev/null && \
		pip-audit -r $(AUDIT_DIR)/.requirements.txt -f json -o $(AUDIT_DIR)/pip-audit.json || true; \
		rm -f $(AUDIT_DIR)/.requirements.txt; \
	else \
		echo "  SKIP: pip-audit not installed (pipx install pip-audit)"; \
	fi
	@echo "==> trivy fs (vuln + misconfig)"
	@if command -v trivy >/dev/null 2>&1; then \
		trivy fs --scanners vuln,misconfig --severity HIGH,CRITICAL --format json --output $(AUDIT_DIR)/trivy.json . 2>/dev/null || true; \
	else \
		echo "  SKIP: trivy not installed (brew install trivy)"; \
	fi
	@echo "==> gitleaks (committed secrets)"
	@if command -v gitleaks >/dev/null 2>&1; then \
		gitleaks detect --source . --no-banner --report-format json --report-path $(AUDIT_DIR)/gitleaks.json 2>/dev/null || true; \
	else \
		echo "  SKIP: gitleaks not installed (brew install gitleaks)"; \
	fi
	@echo ""
	@echo "Audit output: $(AUDIT_DIR)/"
	@ls -la $(AUDIT_DIR)/ 2>/dev/null || true

audit-clean:
	rm -rf $(AUDIT_DIR)
