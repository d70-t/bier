SOURCES := $(wildcard ../tasks/*.yaml)
TARGETS := $(patsubst ../tasks/%.yaml,%.pdf,$(SOURCES))

.PHONY: all
all: $(TARGETS)

.PRECIOUS: %.pdf %.npy %.prop.yaml
%.prop.yaml: %.npy
	@true

%.npy: ../tasks/%.yaml
	python ../bier/runFit.py $< $@

%.pdf: %.npy %.prop.yaml ../bier/createOutputPreviews.py
	python ../bier/createOutputPreviews.py $<

.PHONY: clean-all clean-plots
clean-all: clean-plots
	@- rm *.npy *.prop.yaml

clean-plots:
	@- rm *.pdf
