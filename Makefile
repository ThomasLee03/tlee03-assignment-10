# Makefile for Hybrid Query App

VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(PYTHON) -m pip

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) app.py

clean:
	rm -rf $(VENV)

reset: clean setup
