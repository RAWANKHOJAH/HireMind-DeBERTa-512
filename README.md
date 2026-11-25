# HireMind: DeBERTa-v3-small at 512 tokens

DeBERTa-v3-small taken through the supervisor's plan steps for the HireMind CV–job matching model (No Fit / Potential Fit / Good Fit), with the full 512-token input: Knowledge v1 with B4 evidence selection (E2) on the original labels (D0) and the reviewed labels (D1), 8 seeds each, and Knowledge v1 only (E0, E1), 3 seeds each. 22 trainings in total, on GPU, 5 epochs.

**Start with `DeBERTa_512_results.pdf`** (4 pages: what was done, all results, what did better and why, conclusion and recommendation).

## Result in one paragraph
**DeBERTa-v3-small + Knowledge v1 + B4 evidence, trained on D1 at 512 tokens, is the first configuration that meets the preset selection rule on the mean over 8 seeds:** accuracy 57.23 vs 56.79 for the BERT-Tiny + Knowledge baseline, Good Fit recall 39.45 vs 36.72, and higher Good Fit precision (48.55 vs 43.68), macro-F1 (49.73 vs 48.61) and Potential recall (32.61 vs 29.68). The margin is small and within the seed-to-seed spread, so it is a candidate to confirm on data not used for any choice. If adopted, seed 44 (closest to the mean) is the one to freeze.

| File | Content |
|---|---|
| `DeBERTa_512_results.pdf` | The report |
| `deberta_512_all_runs.csv` | Every training (22): configuration, seed, best epoch, the five metrics, and the preset-rule check |
| `code/c_stage2_deberta_gpu.py` | The training script |
| `code/run_queue.py` | Runs all 22 trainings on the GPUs of one machine |

**Rules followed.** Validation split only (1,162 pairs); the test set was never used. The selection rule (beat the baseline on accuracy and Good Fit recall, no drop of more than 2 points in Good Fit precision, macro-F1 or Potential recall) was fixed before the experiments. No CV or job text is stored here.

**To re-run** (one machine with a GPU, e.g. Kaggle T4 x2), next to this repository:
```
git clone -q https://github.com/HIREMINDPROJECT/HireMind-Plan.git && cd HireMind-Plan
git clone -q https://github.com/Hams770/HireMind-DL.git _dl && git -C _dl checkout -q 1f22f34462c400f884bb0a00f061dbea0bd687d5
git clone -q https://github.com/Hams770/Senior-2.git _s2 && git -C _s2 checkout -q ec28dfc85e618a737cb192cbf959714cfee89aa5
cd .. && pip install transformers==4.51.3 huggingface-hub==0.30.2 sentencepiece protobuf
python HireMind-DeBERTa-512/code/run_queue.py out_512
```
