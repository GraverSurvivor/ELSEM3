.PHONY: install baseline quantize headline test

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Stage 1 (Person A): BF16 eval on BBQ ambiguous, all 5 categories. Writes safe-set + USR/SRS.
baseline:
	python -m src.eval.run_baseline

# Stage 2-3 (Person B/A): HQQ 3-bit compress, flipped-set + statistical detection.
quantize:
	python -m src.experiments.run_quantize

# Stages 1-5 x3 failure windows (Person C): full detect -> localize -> repair loop.
headline:
	python -m src.experiments.run_headline

test:
	pytest tests/ -v
