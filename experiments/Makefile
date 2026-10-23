# experiments/Makefile — build the ANE verification probes.
#
# All binaries land in build/ so the source tree stays flat and clean.
# `make test` is offline; `make validate` / `make validate-bench` are the
# explicit hardware entry points.
CC = xcrun clang
CFLAGS = -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl
CFLAGS_ACC = -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate -ldl

BUILD := build
PYTHON ?= python3

COMMON_M := inmem_peak_spatial sustained compile_limit int8_decompose io_dtype_bench \
            api_surface op_probe
COMMON_BINS := $(addprefix $(BUILD)/,$(COMMON_M))

all: $(COMMON_BINS) \
     $(BUILD)/cpu_gemm_bench $(BUILD)/cpuhog $(BUILD)/devinfo \
     $(BUILD)/dispatch_latency $(BUILD)/mapping_overhead \
     $(BUILD)/static_shape_compile $(BUILD)/surface_capacity \
     $(BUILD)/multi_input $(BUILD)/multi_io \
     $(BUILD)/int8_quant_probe $(BUILD)/constexpr_scale_probe

$(BUILD):
	mkdir -p $(BUILD)

# Static pattern rule for the probes with identical flags.
$(COMMON_BINS): $(BUILD)/%: %.m ane_probe_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD)/cpu_gemm_bench: cpu_gemm_bench.m | $(BUILD)
	$(CC) $(CFLAGS_ACC) -o $@ $<
$(BUILD)/cpuhog: cpuhog.c | $(BUILD)
	xcrun clang -O2 -o $@ $<
$(BUILD)/devinfo: devinfo.m | $(BUILD)
	xcrun clang -O2 -fobjc-arc -framework Foundation -ldl -o $@ $<
$(BUILD)/dispatch_latency: dispatch_latency.m ane_probe_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<
$(BUILD)/mapping_overhead: mapping_overhead.m ane_probe_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<
$(BUILD)/static_shape_compile: static_shape_compile.m ane_probe_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<
$(BUILD)/surface_capacity: surface_capacity.m ane_probe_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<
$(BUILD)/multi_input: multi_input.m ane_probe_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<
$(BUILD)/multi_io: multi_io.m ane_probe_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<
$(BUILD)/int8_quant_probe: int8_quant_probe.m ane_quant_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<
$(BUILD)/constexpr_scale_probe: constexpr_scale_probe.m ane_quant_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

# Offline shared-header self test.
$(BUILD)/probe_common_selftest: probe_common_selftest.m ane_probe_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# Case generation and offline verification
# ---------------------------------------------------------------------------
# mil_gen.py is the only generator entry point; weighted_cases.py is delegated
# to it and its BLOB weights use the standard v2 container.
gen-cases: mil_gen.py weighted_cases.py
	$(PYTHON) mil_gen.py

# Regenerate into a temporary directory and compare byte-for-byte against the
# committed opcases/ fixtures (76 unique cases, _index.txt included).
verify-cases:
	$(PYTHON) verify_cases.py

# Offline only: protocol tests, generator determinism, manifest integrity,
# runner provenance and the shared-header self test.  No ANE call is made.
test: $(BUILD)/probe_common_selftest $(BUILD)/int8_quant_probe $(BUILD)/constexpr_scale_probe
	$(PYTHON) -m unittest discover -s . -p 'test_*.py' -v
	./$(BUILD)/probe_common_selftest op_probe.m
	./$(BUILD)/int8_quant_probe --selftest
	./$(BUILD)/constexpr_scale_probe --selftest

# ---------------------------------------------------------------------------
# Hardware opt-in (never reached from `all` or `test`)
# ---------------------------------------------------------------------------
validate:
	$(PYTHON) run_validation.py $(VALIDATE_ARGS)

validate-bench:
	$(PYTHON) run_validation.py --bench $(VALIDATE_ARGS)

# Op acceptance survey:
#   make all && make gen-cases && ./run_survey.sh [results/DIR | out.csv]
survey: $(BUILD)/op_probe gen-cases
	./run_survey.sh $(SURVEY_ARGS)

clean:
	rm -rf $(BUILD)

.PHONY: all clean gen-cases verify-cases test validate validate-bench survey
