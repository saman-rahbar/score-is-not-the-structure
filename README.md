# Most of a Correspondence Score Survives Removing the Correspondence

Reproduction code for the two studies in the paper. Both audit a widely used
correspondence measure by asking what it reads when the structure it is meant to
detect is removed, or cannot be measured reliably in the first place, and then
ask whether what survives the audit does any functional work.

The audit has three parts. A **content ablation** rebuilds the reference with its
correspondence destroyed but every nuisance property intact, so the score can be
split into content and form. An **instrument check** asks whether the score is
estimated with comparable reliability everywhere it is compared. An **inference
check** counts the independent units rather than the observations. Study 1 finds
the ablation decisive (geometry-broken targets recover most of a brain-alignment
score); Study 2 finds the instrument decisive (probe reliability declines along
the same axis as the predictor) and the unit count consequential.

```
study1_brain_alignment/        # brain-geometry alignment -> syntactic generalization
  experiment.py                # 4-condition fine-tune + BLiMP + CKA content ablation
                               #   + lambda sweep + layer ablation + TOST (multi-scale)
  cka_decomposition.py         # brain-specific increment vs the broken-target controls
  cka_floor.py                 # what CKA reads between targets with no correspondence
  build_pereira.py             # build the fMRI alignment-target cache (login node)
  build_pereira_noiseceiling.py# subject-split noise ceiling for the target
  pretrained_blimp.py          # no-fine-tuning BLiMP anchor
  run_slurm.sh                 # example SLURM launcher (edit the placeholders)
study2_crosslingual_agreement/ # cross-lingual steering -> subject-verb agreement
  experiment.py                # probe transfer + steering + confound-corrected leg C
  robustness.py                # language-level (Mantel) nulls + probe-validity subset
  data_volume_check.py         # rules out training-set size behind the failed probes
  run_slurm.sh
requirements.txt
```

## Study 1 — brain-geometry alignment

Fine-tunes Pythia (160M/410M/1.4B) on the Pereira et al. (2018) stimulus text with
an auxiliary soft linear-CKA loss pulling a hidden layer toward the fMRI
language-network target, under four matched conditions (LM-only, brain-aligned,
shuffled, rank-matched random). Reports BLiMP, the end-of-training CKA
content ablation, a subject-split noise ceiling, a lambda sweep, a layer
ablation, and TOST equivalence bounds.

```bash
# 1) one-time, on a machine WITH internet: build the fMRI target + noise ceiling
pip install brainscore_language        # needs mpi4py (load it as a cluster module)
python build_pereira.py --out cache/pereira_cache.npz
python build_pereira_noiseceiling.py --out cache/noise_ceiling.json

# 2) run an experiment (env vars point at the caches; offline-safe)
export PEREIRA_NPZ=cache/pereira_cache.npz
export NOISE_CEILING_JSON=cache/noise_ceiling.json
export EXP_MODEL=EleutherAI/pythia-160m EXP_SEEDS=20   # headline
python experiment.py                                    # writes results_<model>.json
# scale-generality: EXP_MODEL=EleutherAI/pythia-410m EXP_SEEDS=8 EXP_MAIN_ONLY=1
python pretrained_blimp.py                              # no-fine-tuning anchor

# 3) separate alignment-in-general from brain-specific structure (no GPU)
python cka_decomposition.py --results-dir results
```

The CKA the loss achieves is not by itself brain-specific, because the
geometry-broken controls raise CKA toward the same target. `cka_decomposition.py`
reports the increment over those controls with a paired sign-flip test, and the
share of the brain-aligned CKA the broken targets already recover.

## Study 2 — cross-lingual steering

Fits a linear subject-verb-agreement probe on the critical-verb residual stream
of a multilingual model (default XGLM-1.7B) using MultiBLiMP, for the languages
in the intersection of MultiBLiMP and the model's support. Regresses cross-lingual
probe transfer and activation-steering benefit on URIEL/lang2vec syntactic
distance, with a per-target random-direction baseline (confound) and a partial
correlation controlling for probe transfer (the decisive control).

```bash
pip install lang2vec scikit-learn      # lang2vec bundles URIEL (offline)
python experiment.py                   # writes results.json + figures/
python experiment.py --sandbox         # fast synthetic self-test (no model/GPU)
python robustness.py                   # language-level nulls + probe subset (no GPU)
```

The 272 ordered pairs come from 17 languages, each appearing in 32 of them, and
URIEL distances are phylogenetically structured, so a pair-level permutation
treats as independent what is not. `robustness.py` re-scores every leg against a
language-label permutation instead, and repeats both legs on the languages whose
within-language probe is above chance. It refuses to report anything until it
re-derives the published point estimates from results.json.

## Data

- **Pereira et al. (2018)** fMRI responses via the Brain-Score language package
  (`brainscore_language`), language-network voxels; pooled across participants and
  reduced to rank 128 by PCA. `build_pereira.py --inspect` prints the assembly
  structure if coordinate names differ.
- **BLiMP** (`nyu-mll/blimp`) and **MultiBLiMP** (`jumelet/multiblimp`) via the
  HuggingFace `datasets` library.
- **URIEL/lang2vec** syntactic features (`syntax_knn`), cosine distance.

Compute-cluster note: compute nodes are assumed offline. Prefetch all models and
datasets on a login node first; the scripts set HuggingFace offline flags and
fail with an error if a required cache is missing.

## Reproducibility

All models load in fp32 for numerical stability. Every reported statistic is
non-finite-guarded; a diverged run is surfaced as NaN, never coerced to 0. Random
seeds are fixed and reported. See the paper appendices for exact hyperparameters.
