PYTHON = .venv/bin/python
ULTIMO_MES ?= 2026-07
EXPLORATORIOS = $(PYTHON) -m triagem_baixo_risco.experimentos.exploratorios

.PHONY: ambiente dados exploratorios figuras notebooks testes lint docs

ambiente:
	uv venv --python 3.12 .venv
	uv pip install --python $(PYTHON) -r requirements.txt
	uv pip install --python $(PYTHON) -e ".[dev]" -c requirements.txt

dados:
	$(PYTHON) -m triagem_baixo_risco.dataset --uf MG --years 2019 2026 --by mov --last-month $(ULTIMO_MES)

exploratorios:
	$(EXPLORATORIOS) --experimento todos --desfecho morte --sensibilidade nenhuma
	for s in s1 s2 s3 s4 s5 s6; do \
		$(EXPLORATORIOS) --experimento x3 --desfecho morte --sensibilidade $$s || exit 1; \
	done
	$(EXPLORATORIOS) --experimento x3 --desfecho perm_prolongada --sensibilidade nenhuma
	$(EXPLORATORIOS) --experimento x4 --desfecho perm_prolongada --sensibilidade nenhuma

figuras:
	$(PYTHON) -m triagem_baixo_risco.plots --resultados reports/exploratorios/morte/nenhuma
	$(PYTHON) -m triagem_baixo_risco.plots --resultados reports/exploratorios/perm_prolongada/nenhuma

# Chama o nbconvert direto: "jupyter nbconvert" procura o subcomando no PATH e pode achar o de outra instalação. Com
# JUPYTER_PATH, o kernel python3 é o do .venv, mesmo que haja outro na pasta Jupyter do usuário; record_timing=False
# evita que cada célula guarde a hora em que rodou.
notebooks:
	JUPYTER_PATH=$(CURDIR)/.venv/share/jupyter $(PYTHON) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.record_timing=False notebooks/*.ipynb

testes:
	$(PYTHON) -m pytest -q -W error::RuntimeWarning

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m ruff format --check .

docs:
	$(PYTHON) -m mkdocs build --strict
