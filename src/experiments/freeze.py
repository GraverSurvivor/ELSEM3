"""Week 6: freeze the results for v1.0.

Checks that every run the final numbers depend on is committed and complete, then writes
results/final/manifest.json: a SHA-256 for every tracked result file. Once that file is committed
and tagged, `--verify` proves later that no frozen number changed.

    python -m src.experiments.freeze             # checklist + manifest (exit 1 if a required run is missing)
    python -m src.experiments.freeze --verify    # compare the files against the committed manifest

Required runs are the ones the main claim rests on. Optional ones (the stretch goals) are listed but
don't block the freeze. Tagging is left to a person, after the manifest is committed:

    git tag -a v1.0 -m "Frozen results"  &&  git push origin v1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("results")
REFERENCE_DIR = RESULTS / "reference"
MANIFEST = RESULTS / "final" / "manifest.json"
# Tracked result folders (see .gitignore), hashed into the manifest.
TRACKED = ("reference", "flipped", "search_samples", "patches", "quant_error", "final")

# (folder under results/reference, what it is, required for v1.0)
EXPECTED = [
    ("baseline_fp16", "Stage 1: original model", True),
    ("hqq_3bit", "Stage 2: 3-bit model", True),
    ("hqq_repaired", "Stage 5: 3-bit + search patch", True),
    ("baseline_fp16_disambig", "Hedging check: original", True),
    ("hqq_3bit_disambig", "Hedging check: 3-bit", True),
    ("hqq_repaired_disambig", "Hedging check: repaired", True),
    ("baseline_fp16_confidence", "Monitoring: original with confidence", True),
    *[(f"repair_random_s{k}", f"Baseline: random set {k}", True) for k in range(5)],
    ("repair_positional", "Baseline: positional", True),
    ("repair_largest_error", "Baseline: largest quantization error", True),
    *[(f"repair_search_top5_top{k}", f"Recovery by modules: first {k}", True) for k in range(1, 5)],
    ("search_capability", "General-ability search", False),
    ("repair_capability", "General-ability patch, ambiguous", False),
    ("repair_capability_disambig", "General-ability patch, disambiguated", False),
    ("search_mlp_only", "MLP-only search", False),
    ("headline", "Headline: three windows", False),
    ("hqq_4bit", "4-bit model", False),
    ("repair_4bit", "4-bit + search patch", False),
    ("crows_fp16", "CrowS-Pairs: original", False),
    ("crows_hqq_3bit", "CrowS-Pairs: 3-bit", False),
    ("crows_hqq_repaired", "CrowS-Pairs: repaired", False),
    ("open_ended_fp16", "Open-ended: original", False),
    ("open_ended_hqq_3bit", "Open-ended: 3-bit", False),
    ("open_ended_hqq_repaired", "Open-ended: repaired", False),
    ("llamacpp_f16", "llama.cpp: f16", False),
    ("llamacpp_q3km", "llama.cpp: Q3_K_M", False),
    ("llamacpp_q3km_fairgate", "llama.cpp: Q3_K_M + patch", False),
]


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def check_run(reference_dir: Path, name: str) -> tuple[bool, str]:
    """Present and complete: a benchmark run must score as many questions as its reference run
    (the ambiguous or the disambiguated baseline); other folders just need to be non-empty."""
    folder = Path(reference_dir) / name
    if not folder.is_dir() or not any(folder.iterdir()):
        return False, "missing"
    results = folder / "question_results.jsonl"
    if not results.exists():
        return True, "present"
    n = _count_lines(results)
    base = Path(reference_dir) / ("baseline_fp16_disambig" if name.endswith("_disambig") else "baseline_fp16")
    base_results = base / "question_results.jsonl"
    if name.startswith("llamacpp_") or not base_results.exists() or folder == base:
        return True, f"{n:,} questions"
    expected = _count_lines(base_results)
    if n != expected:
        return False, f"incomplete: {n:,} of {expected:,} questions"
    return True, f"{n:,} questions"


def checklist(reference_dir: Path = REFERENCE_DIR) -> list[dict]:
    return [{"name": name, "what": what, "required": required, **dict(zip(("ok", "detail"), check_run(reference_dir, name)))}
            for name, what, required in EXPECTED]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_results(results_dir: Path = RESULTS, exclude: Path | None = MANIFEST) -> dict[str, str]:
    files = {}
    for folder in TRACKED:
        for path in sorted((Path(results_dir) / folder).rglob("*")):
            if path.is_file() and (exclude is None or path.resolve() != Path(exclude).resolve()):
                files[path.relative_to(results_dir).as_posix()] = sha256(path)
    return files


def verify(manifest: dict, results_dir: Path = RESULTS, exclude: Path | None = MANIFEST) -> dict:
    current = hash_results(results_dir, exclude)
    frozen = manifest["files"]
    return {
        "changed": sorted(f for f in frozen if f in current and current[f] != frozen[f]),
        "missing": sorted(f for f in frozen if f not in current),
        "added": sorted(f for f in current if f not in frozen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="check the files against the committed manifest")
    args = parser.parse_args()

    if args.verify:
        if not MANIFEST.exists():
            raise SystemExit(f"No manifest at {MANIFEST}; run without --verify first.")
        diff = verify(json.loads(MANIFEST.read_text(encoding="utf-8")))
        for kind, files in diff.items():
            for f in files:
                print(f"  {kind}: {f}")
        if diff["changed"] or diff["missing"]:
            raise SystemExit("Frozen results have changed.")
        print("All frozen result files match the manifest." + (" (new files were added since)" if diff["added"] else ""))
        return

    rows = checklist()
    print(f"{'':4}{'run':<30}{'what':<42}status")
    for row in rows:
        mark = "OK" if row["ok"] else ("!!" if row["required"] else "--")
        print(f"{mark:<4}{row['name']:<30}{row['what']:<42}{row['detail']}")
    blocking = [r for r in rows if r["required"] and not r["ok"]]
    lock = Path("requirements.lock.txt")
    print(f"\nrequirements.lock.txt: {'present' if lock.exists() else 'MISSING'}")
    if blocking:
        raise SystemExit(f"\n{len(blocking)} required run(s) missing or incomplete -- not freezing. See RUNBOOK.md.")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": rows,
        "requirements_lock_sha256": sha256(lock) if lock.exists() else None,
        "files": hash_results(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest written: {MANIFEST} ({len(manifest['files'])} files).")
    print("Commit it through a PR, then tag the merge commit:")
    print('  git tag -a v1.0 -m "Frozen results" && git push origin v1.0')


if __name__ == "__main__":
    main()
