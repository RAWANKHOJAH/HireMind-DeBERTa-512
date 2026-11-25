"""Run the DeBERTa 512-token trainings on the GPUs of this machine (one worker per GPU), in priority order.
usage: [SHARD=k/n] python gpu/run_queue.py <out_dir>   (queue = QUEUE below; finished trainings are skipped, so a re-run continues)"""
import os, subprocess, sys, threading, queue, json, time
from pathlib import Path
import pandas as pd
import torch
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
D = ("microsoft/deberta-v3-small", "M7_deberta")
QUEUE = []
for arm in ("kev_D0", "kev_D1"):            # Knowledge + evidence (E2), 8 seeds
    QUEUE += [(D, arm, s) for s in range(42, 50)]
for m in (D,):                              # Knowledge only (E0 on D0, E1 on D1), seeds 42-44
    for arm in ("k_D0", "k_D1"):
        QUEUE += [(m, arm, s) for s in (42, 43, 44)]
here = Path(os.environ.get("PLAN", Path(__file__).resolve().parents[1] / "HireMind-Plan"))   # a clone of HIREMINDPROJECT/HireMind-Plan, with _dl and _s2 cloned inside it
E2 = here / "code/m5-distilbert-plan/phase_c2_m5/e2mods"
_sh = os.environ.get("SHARD", "0/1"); _ks, _n = _sh.split("/"); _ks = {int(x) for x in _ks.split(",")}; _n = int(_n)
QUEUE = [it for i, it in enumerate(QUEUE) if i % _n in _ks]  # SHARD=k/n: this machine takes every n-th training
if os.environ.get("ITEMS"):  # ITEMS=name:arm:seed,... runs exactly these, in this order
    _m = {"M7_deberta": D}
    QUEUE = [(_m[a], b, int(c)) for a, b, c in (x.split(":") for x in os.environ["ITEMS"].split(","))]
q = queue.Queue()
for it in QUEUE: q.put(it)
lock = threading.Lock()
def work(gpu):
    while True:
        try: (model, name), arm, seed = q.get_nowait()
        except queue.Empty: return
        d = OUT / f"{name}_{arm}_s{seed}"
        if (d / "C_stage2_m7_validation.csv").exists(): continue
        d.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), M7_MODEL=model, M7_NAME=name, M7_EPOCHS="5", M7_MICRO="16", M7_BUDGET_H="100",
                   M5_SEED=str(seed), M5_KNOWLEDGE_DIR=str(here / "_s2/evaluation/knowledge"), M5_E2_DIR=str(E2))
        env["M5_EVIDENCE" if arm.startswith("kev") else "M5_KNOWLEDGE"] = "1"
        if arm.endswith("D1"): env["M5_LABELS"] = str(here / "A5_label_overlay_D1/train_labels_v1.csv")
        t = time.time()
        with open(d / "full_log.txt", "w") as log:
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve().parent / "c_stage2_deberta_gpu.py"), str(here / "_dl"), str(d), "512"], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(here))
        with lock:
            if rc == 0 and (d / "C_stage2_m7_validation.csv").exists():
                r = pd.read_csv(d / "C_stage2_m7_validation.csv").iloc[0]
                print(f"DONE {name} {arm} s{seed} best_epoch {int(r.best_epoch)} acc {r.accuracy_pct:.2f} mf1 {r.macro_f1_pct:.2f} gfr {r.good_fit_recall_pct:.2f} gfp {r.good_fit_precision_pct:.2f} pr {r.potential_fit_recall_pct:.2f} ({(time.time()-t)/60:.0f} min, gpu {gpu})", flush=True)
            else:
                print(f"FAILED {name} {arm} s{seed} rc {rc}: " + " | ".join((d / "full_log.txt").read_text().splitlines()[-3:])[:400], flush=True)
n = max(1, torch.cuda.device_count())
print("GPUs:", n, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "| trainings:", len(QUEUE), flush=True)
th = [threading.Thread(target=work, args=(i,)) for i in range(n)]
for x in th: x.start()
for x in th: x.join()
print("ALL DONE", flush=True)
