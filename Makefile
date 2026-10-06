# One command reproduces every number: `make all`.
PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: all data panels integrity geography fragility report test clean-derived

all: data panels integrity geography fragility report

test:
	$(PY) -m unittest discover -s tests

data:
	$(PY) -m fairexams.download
	$(PY) -m fairexams.verify

panels:
	$(PY) -m fairexams.panels
	$(PY) -m fairexams.districts

integrity:
	$(PY) -m fairexams.screen
	$(PY) -m fairexams.verify --flagged
	$(PY) -m fairexams.serial
	$(PY) -m fairexams.bulk
	$(PY) -m fairexams.stability
	$(PY) -m fairexams.medium
	$(PY) -m fairexams.inject
	$(PY) -m fairexams.validate

geography:
	$(PY) -m fairexams.geography

fragility:
	$(PY) -m fairexams.fragility

report:
	$(PY) -m fairexams.numbers
	$(PY) -m fairexams.report_tables
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null

clean-derived:
	rm -f results/tables/*.csv results/figures/*.png results/numbers.json report/numbers.tex
