# Shortcuts for the common tasks. Everything here runs on a laptop CPU from the
# committed results: no GPU, no model download, no fMRI data.

PY ?= python

.PHONY: check figures hero

check:  ## every number-producing check that needs no GPU
	cd crosslingual_agreement && $(PY) experiment.py --sandbox
	cd crosslingual_agreement && $(PY) robustness.py
	cd crosslingual_agreement && $(PY) data_volume_check.py
	cd crosslingual_agreement && $(PY) range_restriction.py
	cd crosslingual_agreement && $(PY) language_bootstrap.py
	cd brain_alignment && $(PY) cka_decomposition.py --results-dir results
	cd brain_alignment && $(PY) cka_floor.py --selftest

figures:  ## both paper figures, written to figures/
	REPO_ROOT=. FIG_OUT=figures $(PY) make_figures.py

hero:  ## the animated header in assets/
	$(PY) tools/make_hero.py
