.PHONY: docs

default: install

all: install build


h help:
	@grep '^[a-z]' Makefile


install:
	pip install pip --upgrade
	pip install -r requirements-docs.txt

upgrade:
	pip install pip --upgrade
	pip install -r requirements-docs.txt --upgrade


toc:
	python gen_page_tocs.py

s serve: toc
	zensical serve


b build: toc
	zensical build --clean --strict

# Zensical has no gh-deploy; build, then force-push site/ to the gh-pages branch
# (-n adds .nojekyll, -f forces, -p pushes to origin).
d deploy: build
	ghp-import -n -p -f -m "Deployed $$(git rev-parse --short HEAD) with Zensical $$(zensical --version)" site
