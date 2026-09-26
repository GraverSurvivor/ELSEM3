.PHONY: install baseline validate promote-reference quantize search swap-spike repair-state headline test

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Stage 1 (Person A): full-precision eval on BBQ ambiguous, all 5 categories. Writes USR/SRS.
# Defaults to fp16 (see CLAUDE.md > Precision) -- override with `make baseline DTYPE=bf16` on an
# Ampere+ GPU. Never mix dtypes across conditions in one comparison.
DTYPE ?= fp16
BATCH_SIZE ?= 16
baseline:
	python -m src.eval.run_baseline --dtype $(DTYPE) --batch-size $(BATCH_SIZE)

# Promote a finished run to results/reference/, the one results path git tracks. Validates
# first -- a truncated or duplicated run promoted as "the baseline" would silently redefine
# what every later stage compares against. Pass RUN_ID=<id> NAME=<name>.
promote-reference:
	python -m src.experiments.promote_reference $(RUN_ID) --name $(NAME)

# Confidence check on the scorer itself, via BBQ's disambiguated condition. Run this before
# trusting any USR/SRS number from a new dtype, model, or prompt format.
validate:
	python -m src.eval.validate_scorer --dtype $(DTYPE) --batch-size $(BATCH_SIZE)

# Stage 2 (Person A): HQQ 3-bit eval on BBQ ambiguous, all 5 categories. Always fp16 compute, to
# match the baseline -- the dtype is fixed inside the script, not a flag. Needs a GPU.
# Then: make promote-reference RUN_ID=<printed id> NAME=hqq_3bit, and build the flipped set with
# python -m src.data.safe_set (see its --help).
quantize:
	python -m src.eval.run_quantized --batch-size $(BATCH_SIZE)

# Stage 4 (Person B): greedy attribution search on the fixed flipped-set sample. Needs a GPU.
# Draw the sample first (CPU): python -m src.attribution.search_sample
SAMPLE ?= results/search_samples/hqq_3bit_n300_seed0.json
THRESHOLD ?= 0.01
search:
	python -m src.attribution.run_search --sample $(SAMPLE) --threshold $(THRESHOLD) --batch-size $(BATCH_SIZE)

swap-spike:
	python -m src.quant.verify_module_swap

# Stage 4 groundwork (Person B): checks lazy checkpoint loading is bit-exact, that a trialled
# promotion reverts exactly, and prints the real memory numbers for the report.
repair-state:
	python -m src.quant.verify_repair_state

# Stages 1-5 x3 failure windows (Person C): full detect -> localize -> repair loop.
# NOT BUILT YET -- Week 6 deliverable.
headline:
	@echo 'Headline experiment not written yet (Week 6).' && exit 1

test:
	pytest tests/ -v
