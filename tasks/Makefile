SIM_SOURCES=$(wildcard ../../pyradtran/out/rayleigh_only*.raw)
TARGET_SPECS=$(patsubst ../../pyradtran/out/%.raw, %.auto.yaml, $(SIM_SOURCES))

all: $(TARGET_SPECS)

%.auto.yaml: ../../pyradtran/out/%.raw
	echo "input: $<" > $@
	cat "rayleigh_autogen.yaml.tpl" >> $@

clean:
	@- rm *.auto.yaml
