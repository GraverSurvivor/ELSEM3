.PHONY: install baseline validate quantize swap-spike repair-state headline test

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

# Confidence check on the scorer itself, via BBQ's disambiguated condition. Run this before
# trusting any USR/SRS number from a new dtype, model, or prompt format.
validate:
	python -m src.eval.validate_scorer --dtype $(DTYPE) --batch-size $(BATCH_SIZE)

# Stage 2-3 (Person B/A): HQQ 3-bit compress, flipped-set + statistical detection.
# NOT BUILT YET -- the working entry point today is the module-swap spike below.
quantize:
	@echo 'Stage 2-3 driver not written yet. Available now: make swap-spike' && exit 1

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
