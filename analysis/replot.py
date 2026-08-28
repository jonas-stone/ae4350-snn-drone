"""regenerate figures from saved csvs (no retraining)."""
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotstyle; plotstyle.apply()


def replot_multiseed(path, out, title):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ep   = np.array([int(r["episode"]) for r in rows])
    mean = np.array([float(r["mean_success"]) for r in rows]) * 100
    std  = np.array([float(r["std_success"]) for r in rows]) * 100
    seedcols = [k for k in rows[0] if k.startswith("seed")]
    plt.figure(figsize=(7, 4.5))
    for sc in seedcols:
        plt.plot(ep, [float(r[sc]) * 100 for r in rows], color="gray", alpha=0.3, lw=1)
    plt.plot(ep, mean, color="tab:blue", lw=2, label="mean")
    plt.fill_between(ep, mean - std, mean + std, color="tab:blue", alpha=0.2, label="$\\pm$ 1 std")
    plt.xlabel("episode"); plt.ylabel("success rate (%)  @ T=0.3")
    plt.title(title); plt.legend()
    plt.savefig(out, dpi=120); plt.close(); print("saved", out)


def replot_success(path, out, title):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ep   = [int(r["episode"]) for r in rows]
    succ = [float(r["success_T0.3"]) * 100 for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, succ, marker="o", color="tab:blue")
    plt.xlabel("episode"); plt.ylabel("success rate (%)  @ T=0.3")
    plt.title(title)
    plt.savefig(out, dpi=120); plt.close(); print("saved", out)


def replot_csweep(path, out):
    import re
    Cs, rates = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            m = re.match(r"SNN C=(\d+)", row["model"])
            if m:
                Cs.append(int(m.group(1)))
                key = [k for k in row if k.startswith("success")][0]
                rates.append(float(row[key]) * 100)
    order = np.argsort(Cs)
    Cs = np.array(Cs)[order]; rates = np.array(rates)[order]
    x = np.arange(len(Cs))                       # even spacing for 2, 4, 8, 16
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, rates, marker="o", ms=9, color="tab:blue")
    plt.xticks(x, Cs)
    plt.xlabel("integration cycles $C$"); plt.ylabel("success rate (%), 500 goals @ T=0.3")
    plt.title("SNN performance vs integration cycles")
    plt.savefig(out, dpi=120); plt.close(); print("saved", out)


replot_csweep("data/eval_final.csv", "figures/csweep.pdf")
replot_multiseed("data/multiseed_mlp_5seeds.csv", "figures/multiseed_mlp_5seeds.pdf",
                 "MLP: success over 5 seeds (mean $\\pm$ 1 std)")
replot_multiseed("data/multiseed_snn_3seeds.csv", "figures/multiseed_snn_3seeds.pdf",
                 "SNN: success over 3 seeds (mean $\\pm$ 1 std)")
replot_success("data/mlp_success_history.csv", "figures/mlp_learning_curve.pdf",
               "MLP training: success vs episode (single run)")
replot_success("data/snn_success_history.csv", "figures/snn_learning_curve.pdf",
               "SNN training: success vs episode (single run)")
