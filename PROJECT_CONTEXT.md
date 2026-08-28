# AE4350 Project — Full Context & State

## Person
- **Jonás Stone Álvarez** (student number 6532217), TU Delft MSc.
- Email: jonasstonealv@gmail.com.
- **Learning is the #1 priority, explicitly over end-result quality** ("even if the end result is weaker, it is the most important goal"). Wants to be taught from basic concepts through implementation and to write the code himself. Teaching order followed: setup → NN → RL → SNN.
- Communicates directly and bluntly; gets frustrated when things don't work or when explanations are too dense/verbose or over-hedged. Wants plain, non-"AI-flavoured" prose in the report (called it "watermarking"). Responds well to honest diagnosis and being told when a previous approach was wrong.
- Prefers to make his own edits; asked that report `.tex` files not be edited unless he explicitly says so — tell him what to change instead (this was later relaxed for specific requested edits).
- Wants terminal test commands briefly explained.

## Assignment
- Course: AE4350 Bio-inspired Intelligence and Learning for Aerospace Applications (TU Delft). Professors: Erik-Jan van Kampen and **G.C.H.E. de Croon**.
- **de Croon (author of the SNN reference paper) is grading it** — engage his paper accurately, be honest about limitations, don't inflate.
- Deliverable: report **max 10 pages** in TU Delft report style (template linked in brief; using it), + GitHub link (public or invite teachers). Filename must include his name. Hand in on Brightspace. Deadline "August 31" (brief says 2025; system clock is 2026 — confirm year).
- Rubric: 7 equally-weighted categories — complexity of method, environment/application complexity, scientific reporting, method description, results description (incl. **statistics on multiple runs / uncertainty**), sensitivity analysis (vary **multiple** parameters), analysis of the found solution. **No success-rate category** (raw performance not directly graded).
- **Grade target: 8–8.5** (not aiming 9+). Strategy: the 2 complexity categories are task-capped (~6.5–7, discrete/low-dim); push the other 5 to ~9. It is now a report-execution problem, not research.

## Task / environment (`drone_env.py`)
- 2D point drone (radius 0) in a 10×10 arena, navigating start→goal through a **fixed hand-designed 9-obstacle layout**.
- Start fixed at centre [5,5]; goal randomised each episode in free space ≥4 units away.
- **Observation**: 10 continuous values in [0,1] = 8 ray-cast distances (radial, every 45°, ray 0 = North/forward, clockwise) + normalised (dx,dy) to goal. State space is continuous.
- **Actions**: `Discrete(4)` cardinal moves (up/down/left/right), fixed step 0.2, position clipped to bounds. Deliberately simple/low-dim (the complexity ceiling).
- **Reward**: `progress = prev_path_dist − new_path_dist` (BFS obstacle-aware path distance on a 0.2 grid) − 0.01 time penalty; +10 goal; −1 collision (terminates); −0.05 wall-push (borders are soft walls: rays see them, small penalty, no termination). γ=0.99.
- Rays via analytic slab method (vectorized, ~15× faster than marching), clipped to [0, max_ray_length]; no inf leaks into observations.

## Networks (`networks.py`)
- **MLPPolicy**: 10→64→64→4, tanh; outputs logits → softmax → Categorical; `act(obs, greedy, temperature)`.
- **SNNPolicy**: same 10→64→64→4 skeleton, `snn.Leaky` LIF layers (snntorch) replacing tanh; **β=0.9**, **threshold=0.3** (tuned; default 1.0 gave dead layers ~0.2% firing → 0% success), **arctan surrogate gradient**, soft reset. **Encoding** = current injection ("cycling", de Croon's term): hold the obs constant, inject every cycle for **C=8** cycles. **Decoding** = accumulate the linear output layer (fc3, no LIF) over C cycles and average → 4 logits. Two clocks: outer = env steps (drone moves), inner = C cycles per decision (membranes reset each decision).
- **ValueNet** (critic): 10→64→64→1, tanh.

## RL (`train.py`)
- Hand-built REINFORCE → **actor-critic (REINFORCE-with-baseline)**. Adam lr=1e-3, γ=0.99, 6000 episodes, save-best by periodic eval.
- Advantage `A_t = G_t − V(s_t)` (critic detached when forming actor loss; advantage normalised per episode); critic loss = MSE to raw discounted returns; two optimizers.
- `train(train_which='MLP'|'SNN', num_episodes, seed=...)`; seeding makes runs reproducible (torch + np + env goal stream). Logs `data/{tag}_reward_history.csv`, `data/{tag}_success_history.csv`.
- Eval is **seeded** (`env.reset(seed=ep)`) so MLP and SNN face identical goals.

## Key findings / results (all at deployment temperature T=0.3 unless noted)
- **Greedy (argmax) deployment is trapped by limit cycles** (drone flickers between two states until timeout). Proven: on seed 138 greedy times out at 400 steps, the same policy reaches in 38 steps when sampled. It is an action-selection artefact of the discrete space, NOT a bad policy.
- **Temperature-scaled sampling** (logits/T) fixes it. Sweet spot T≈0.3–0.5. Actor-critic MLP temp sweep: greedy 22% → T=1.0 62% → T=0.5 75% → T=0.3 74% → T=0.2 66% → T=0.1 50%.
- **Actor-critic beats plain REINFORCE** but warms up slower and trains unstably (sharp drops; save-best essential). The instability is exactly what PPO's clipping prevents (why de Croon used PPO).
- **Multi-seed (converged = mean over last 25% of checkpoints):** MLP **53.8% ± 2.5%** (5 seeds), SNN **64.7% ± 2.2%** (3 seeds). **Intervals do not overlap → the SNN genuinely outperforms the matched MLP** (matches de Croon's SNN-beats-ANN finding). NOTE: figure endpoints (episode 6000) are higher — MLP 59.4±7.6, SNN 74.7±5.2 — because curves are still rising; text should label the reported number as a final-window average (pending fix in §3.4).
- **Best single models on 500 seeded goals:** MLP 61%, SNN 75% (`data/eval_final.csv`).
- **Firing sparsity:** ~22% mean firing rate, ~226 spikes/decision vs a dense ANN's 128 activations/pass, ~15% dead neurons, 0% saturated. The efficiency argument.
- **C-sweep (6000-ep, single seed, `data/csweep_final.csv`):** roughly 57–70% across C∈{2,4,8,16}, no monotonic trend; even C=2 (3 rate levels) works — task doesn't need fine rate resolution. Single-seed, so differences within run variance (same limitation de Croon acknowledges). C=8 kept as resolution/compute compromise.
- **Interpretability:** `action_response` figure (drone marches into a wall) shows firing rate rising as the wall nears (encoding) and P(up)≈0 / P(down) rising (avoidance). Honest caveat: controlled OOD probe; report the trend, not the leftward bias. Per-neuron tuning/heatmaps were tried and DROPPED (mixed selectivity, messy; de Croon's paper does no per-neuron analysis either).
- **Continuous actions were tried and ABANDONED** (Gaussian policy, tanh mean, learnable log_std): vanilla REINFORCE+baseline could not learn them (cos-alignment to goal 0.13 ≈ random, ~10%). Confirms continuous control needs PPO. **Do NOT mention this in the report** (Jonas's explicit instruction). Delivered system is discrete.
- **Reward-shaping ablation:** a √-funnel reward FAILED (starved far-field signal, 43%→10–22%), reverted to linear. Good negative-result material for an appendix.

## Code structure & conventions
- Architecture: **gen → CSV → plot** (data generation separated from plotting, so figures redraw from CSV without re-running models). Plot scripts have editable font/size knobs at the top.
- **Folder structure (reorganised for de Croon): root holds ONLY the core `drone_env.py`, `networks.py`, `train.py`.** Everything else lives in `analysis/`: `plotstyle.py` (shared SANS-serif style matching the template, ensures `figures/`+`data/` exist), the `gen_*`/`plot_*` pairs (trajectories, action_response, firing_stats, threshold_sweep), `plot_csweep.py`, `csweep.py` (6000-ep trainer), `multiseed.py`, `multiseed_overlay.py`, `replot.py` (regenerates multiseed + learning-curve figures from CSVs), `eval_final.py`, `temp_sweep.py`, `compare.py` (interactive watcher), `plot_world.py`, `plot_reward_field.py`. Scripts that import core modules carry a `sys.path` shim; **all analysis scripts must be run FROM ROOT** (e.g. `.\.venv\Scripts\python.exe analysis\plot_world.py`) so `data/`, `figures/`, and `*.pt` paths resolve.
- All script comments cleaned to minimal lowercase. Figures are **vector `.pdf`, sans-serif**, uniform font scheme, widths proportional to embed (single-panel figwidth ~7 @ `0.7\textwidth`; multi-panel figwidth ~10 @ `\textwidth`; world is 5×5 @ `0.5\textwidth`).
- Outputs: figures → `figures/` (`.pdf`), data → `data/`, models `*.pt` stay in root (loaded by name).
- Models: `mlp_policy_best.pt`, `snn_policy_best.pt`, `snn_C{2,4,8,16}_best.pt`.
- Env: Windows, PowerShell primary; Python 3.12 venv at `.venv` (run via `.\.venv\Scripts\python.exe`). Deps: numpy, matplotlib, gymnasium, torch (CPU), snntorch.
- **Git**: branch `main` = actor-critic (discrete, delivered). Branch `single-network` = REINFORCE-only baseline (~62%). `main` and `single-network` pushed to origin; `continuous` branch created then deleted.

## Report state (`report/`)
- Two folders: `current report/` (his working article-class version) and `TU Delft report template/` (the converted version to submit — section-based, fits <10 pages).
- **Template conversion done**: `tudelft-report` class, `oneside`, section-based (top-level sections numbered 1–7, appendices A–C), IEEE biblatex with `refs.bib`, packages (amsmath/mathtools/amssymb, float, gensymb, enumitem, subcaption), `\whiteline` defined, `\graphicspath{{./Images/}}`. Custom title page (`title.tex`) mirrors his original (logo `Images/tudelft_logo.png`, title "SNN + RL on a 2D obstacle course", course, professors, name+number, date). `main.tex` = `report.tex` (both master, either compiles). Book cover/preface/example chapters not included.
- **Content parts** (his words, condensed to his lean style; equations/numbers/figures/labels preserved):
  - **Part 1 (Introduction, Problem definition)**: intro, related work (cites decroon2025snn, ferede2024endtoend, suttonbarto2018, neftci2019), problem def, nav task, MDP 5-tuple with reward eq. `\autoref{fig:reward_field}` is a dangling ref (figure not placed).
  - **Part 2 (Method)**: MLP, SNN (LIF eq, β, surrogate gradient, encoding/decoding, two clocks), REINFORCE→actor-critic (eqs), temperature (eq), eval protocol, hyperparameter table. Soft-reset justification sits COMMENTED (needs un-commenting). `[GitHub link]` placeholder. References `sec:csweep`, `sec:firing`.
  - **Part 3 (Results)** — WRITTEN: 3.1 baseline variance, 3.2 greedy limit cycles + temperature (tab:temp), 3.3 actor-critic, 3.4 multi-seed (fig:multiseed, fig:overlay; numbers need the final-window-average labelling fix), 3.5 MLP vs SNN (tab:headtohead, fig:traj), 3.6 sensitivity (temperature + LIF threshold + C; **C-sweep figure was removed during condensing — needs re-adding now that the 6000-ep sweep exists**, with `\label{sec:csweep}`).
  - **Part 4 (Analysis, Discussion, Conclusion)** — SKELETON ONLY, not written.
  - **Appendices** — skeleton (Reproducibility/GitHub, Additional figures, Reward-shaping ablation negative result).
- Template `Images/` has: mlp_learning_curve, multiseed_mlp_5seeds, multiseed_overlay, snn_learning_curve, trajectory_compare, world, csweep, tudelft_logo, "TU delft logo". **Missing for Part 4**: action_response.png, firing_stats.png, reward_field.png; csweep.png needs refreshing from the 6000-ep sweep.
- Only 4 citation keys used (decroon2025snn, ferede2024endtoend, neftci2019, suttonbarto2018); `refs.bib` also holds unused entries (schulman2017ppo, williams1992, eshraghian2023snntorch, mnih2016a3c, towers2023gymnasium).

## Reference paper (de Croon et al., IMAV 2025, `AE4350...`/`17.pdf`)
- "Spiking Neural Networks for High-Speed Continuous Quadcopter Control Using PPO." Fully-spiking actor-critic, PPO via Stable-Baselines3, 3×64 LIF layers, "cycling" encoding (= our hold-input-constant), spike-rate decoding, **C=10** as compute/quality compromise, arctan surrogate. SNN BEAT the matched ANN (real-world R 70.63 vs 59.77). Reports single-seed runs, explicitly noting multi-seed averaging was infeasible on compute (same limitation we cite). ANN baseline follows Ferede et al. Reports "average of last 10% of timesteps" (final-window metric), not the endpoint. Does NO per-neuron interpretability.

## Report framing (agreed)
Narrative arc, findings told in the order the method was built: problem → simple REINFORCE → high variance + greedy flickering → temperature → actor-critic → multi-seed uncertainty → fair MLP-vs-SNN → sensitivity → analysis (encoding/behaviour, sparsity, strategy) → discussion (limitations pointing at PPO/continuous, i.e. de Croon's own choices) → conclusion. Cite Sutton & Barto for actor-critic (not de Croon's PPO). Metric hierarchy: loss = ignore, reward = proxy, **success rate = the true metric**. Every section makes one of two honest statements: a rigorous finding, or a real limitation whose known fix is what de Croon did.

## Outstanding before submission
Write Part 4 (+ `\label{sec:firing}`); fix §3.4 numbers; re-add §3.6 C-sweep figure+text; un-comment soft-reset sentence; add GitHub link; resolve dangling refs (reward_field, sec:firing); copy missing figures (action_response, firing_stats, reward_field, refreshed csweep) into template `Images/`; verify temperature table; name the file with his name; make GitHub repo public / invite teachers; commit+push final code; confirm deadline year; submit on Brightspace.
