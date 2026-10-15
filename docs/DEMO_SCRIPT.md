# Demo script: FairGate Fast Repair (about 5 minutes)

Fill the **[brackets]** from Person 4's real runs (`results/reference/fast_repair_qwen_*/report.json`)
before rehearsal. Never quote a number from a stand-in run.

## Before the session
1. Start Studio on Colab in **Bring your own model** mode (notebook `fairgate_studio.ipynb`, APP =
   "Bring your own model"), at least 10 minutes early.
2. Have the backup ready: `python -m src.product.studio --byom --byom-replay results/reference/fast_repair_qwen_fast`
   replays the recorded real run with no GPU, and the screen says it is a replay.

## 1. The problem, on our model (1 min) — tabs 1-3
"We compressed Phi-3.5 to 3-bit. It got 4.5x smaller, and it started answering by stereotype."
Show one Ask example, swap the two people, then Repair: 5 restored parts take the 50 test questions
from 50 to 8 stereotyped answers.

## 2. The question a company asks (30 s)
"That's our model. Would this work on mine?" Open **5 · Your model**.

## 3. Diagnose a different model (2 min)
Type `Qwen/Qwen2.5-3B-Instruct`, choose **Fast**, press **Diagnose**. While it runs, explain the stages:
- *Ask the original / compressed model*: the same [1,000 + 1,000] questions to both.
- *Find the damage*: "[N] of [M] fair answers flipped to the stereotype."
- *Scan the parts*: "Qwen has [252] parts. We screen all of them, then run our search on the best
  [12]. Our replay of the 7-hour Phi-3.5 search showed this keeps the same parts."
- *Check the repair*: on [~500 + ~500] questions the scan never saw.

If the live run won't finish in the slot, switch to the recorded run (it says so on screen).

## 4. The result (1 min)
- Cards: "can't tell" [orig] → [3-bit] → [repaired]; stereotyped [..]; hedging check [..].
- "Fairness recovered **[x]%** ([lo]–[hi]%), with [k] parts at 8-bit, **+[MB] MB**, in **[min] min**."
- The verdict, honestly: [PASS / FAIL with its reason].
- Show the grid (which parts), then the patch file and the llama.cpp line: "this is what you ship."

## 5. Limits, said out loud (30 s)
Up to ~4B parameters on a free T4; fast mode is "very likely the same parts", exact mode takes
~45 min; BBQ only; the repair needs the original weights (open models).
