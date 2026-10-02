.PHONY: install baseline validate promote-reference quantize search repair disambig quant-error baselines controls sensitivity groups fairgate-check fairgate-report swap-spike repair-state figures headline headline-report test

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

# Stage 5 (Person B): verify any patch on the full benchmark. GPU. CONTEXT=disambig for the control set.
PATCH ?= results/patches/search_top5.json
CONTEXT ?= ambig
repair:
	python -m src.attribution.run_repair --patch $(PATCH) --context $(CONTEXT) --batch-size $(BATCH_SIZE)

# --- Revised plan: fairness controls (Improvement #1) -------------------------------------------
# Disambiguated control runs, one per model (GPU). Promote each afterwards as *_disambig.
disambig:
	python -m src.eval.run_baseline --context disambig --batch-size $(BATCH_SIZE)
	python -m src.eval.run_quantized --context disambig --batch-size $(BATCH_SIZE)
	python -m src.attribution.run_repair --patch results/patches/search_top5.json --context disambig --batch-size $(BATCH_SIZE)

# Per-module quantization error (GPU, a few minutes), then the matched-budget baseline patches (CPU).
quant-error:
	python -m src.quant.quant_error
baselines:
	python -m src.attribution.baselines $(if $(wildcard results/quant_error/hqq_3bit_g64.json),--quant-error results/quant_error/hqq_3bit_g64.json)

# Hedging check + search vs baselines -> results/stats/controls.json (CPU).
controls:
	python -m src.stats.controls

# Other analyses on committed runs (CPU).
sensitivity:
	python -m src.stats.sensitivity
groups:
	python -m src.stats.groups

# --- Revised plan: FairGate ------------------------------------------------------------------------
fairgate-check:
	python -m src.product.fairgate check
fairgate-report: controls groups
	python -m src.product.fairgate report

swap-spike:
	python -m src.quant.verify_module_swap

# Stage 4 groundwork (Person B): checks lazy checkpoint loading is bit-exact, that a trialled
# promotion reverts exactly, and prints the real memory numbers for the report.
repair-state:
	python -m src.quant.verify_repair_state

# Person C: every figure and results table the committed reference runs support. CPU only.
# Add a recovery curve with: python -m src.plotting.make_figures --search-summary results/search_<id>/summary.json
figures:
	python -m src.plotting.make_figures

# Stages 3-5 x3 failure windows (Person C): detect -> localize -> repair on the Age-heavy,
# SES-heavy and balanced windows. Needs a GPU, ~7-8 h on a T4 in total -- run one window per
# session with WINDOWS=age_heavy, and continue with RESUME=headline_<id>. Check the plan first
# with `make headline DRY_RUN=1`.
WINDOWS ?= age_heavy ses_heavy balanced
headline:
	python -m src.experiments.headline --windows $(WINDOWS) --threshold $(THRESHOLD) --batch-size $(BATCH_SIZE) 		$(if $(RESUME),--resume $(RESUME)) $(if $(DRY_RUN),--dry-run)

# Headline tables, overlap verdict and figures from a finished (or partly finished) experiment. CPU only.
headline-report:
	python -m src.experiments.headline_report $(HEADLINE_ID)

test:
	pytest tests/ -v
