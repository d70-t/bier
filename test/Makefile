BASEDIR = $(shell pwd)

NOSETESTS = export PYTHONPATH=$(BASEDIR):$(BASEDIR)/..:$(PYTHONPATH) && python -m nose

test:
	@-rm .coverage
	$(NOSETESTS) --with-coverage --cover-package=bier --cover-html --cover-branches -v

test-fast:
	$(NOSETESTS) -v

.PHONY: test test-fast
