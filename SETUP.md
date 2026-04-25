# Start-to-End Replication Guide

Run notebooks in order. Pre-saved results in `/results/` let you skip expensive compute steps and verify findings directly.

---

## Setup

```bash
pip install numpy==1.26.4
pip install transformer_lens datasets scikit-learn pandas matplotlib
```

---

## 01 — Behavioral Baseline (~10 min GPT-2 only, ~60 min all models)

**Notebook:** `01_behavioral_baseline.ipynb`  
**Hardware:** M4 for GPT-2 and Pythia-1.4B. A100 for larger models.

Edit the `MODELS` list in Cell 2 to select which models to run. For a quick check, keep only GPT-2 and Pythia-1.4B.

**Expected:** GPT-2 selects A=100%. Pythia-1.4B selects B=77%. Mistral-Instruct: A=31%, B=33%, C=36%.

---

## 02 — GPT-2 Race/Ethnicity Probing and Valence (~15 min)

**Notebook:** `02_gpt2_probing_valence.ipynb`  
**Hardware:** M4 or any GPU.

Run all cells. Produces `white_encoding_results.pkl` (required for notebooks 10 and 11).

**Expected:** 100.0% ± 0.0% at all layers for all three pairwise comparisons.

---

## 03 — Gender Identity Probing (~10 min GPT-2, ~20 min Pythia)

**Notebook:** `03_gender_probing.ipynb`  
**Hardware:** M4 for Cell 4 (GPT-2). A100 for Cell 5 (Pythia-1.4B).

Uses exact token matching — see notebook header for why this matters.

**Expected:** 100.0% ± 0.0% at all layers for both models.

---

## 04 — GPT-2 Causal Patching (~30 min)

**Notebook:** `04_causal_patching_gpt2.ipynb`  
**Hardware:** M4 or any GPU.

**Expected:** Layer 3-6: KL = 0.017-0.018. Layer 11: KL = 0.000.

---

## 05 — Pythia-1.4B Causal Patching (~30 min)

**Notebook:** `05_causal_patching_pythia.ipynb`  
**Hardware:** A100. Skip Cell 10 (diagnostic).

**Expected:** Layer 6-12: KL = 0.0015-0.0017. Layer 23: KL = 0.000.  
Pre-saved: `results/causal_patching_pythia_results.csv`

---

## 06 — Mistral-7B-Instruct Causal Patching + Permutation (~4 hours)

**Notebook:** `06_causal_patching_mistral.ipynb`  
**Hardware:** A100. Run the keep-alive cell after starting the permutation test.

**Expected:** p ≥ 0.292 n.s. at every layer (network-wide causal inertia).  
Pre-saved: `results/patching_permutation_test_mistral_instruct.csv`

---

## 07 — GPT-2 Permutation Test (~25 min)

**Notebook:** `07_permutation_test_gpt2.ipynb`  
**Hardware:** A100.

**Expected:** p ≤ 0.007 at layers 3-9. p = 0.512 n.s. at layer 11.  
Pre-saved: `results/patching_permutation_test_gpt2.csv`

---

## 08 — Pythia-1.4B Permutation Test (~40 min)

**Notebook:** `08_permutation_test_pythia.ipynb`  
**Hardware:** A100.

**Expected:** p ≤ 0.005 at layers 6-12. p = 1.000 n.s. at layer 23.  
Pre-saved: `results/patching_permutation_test_pythia14b.csv`

---

## 09 — Large Model Probing (~60 min)

**Notebook:** `09_large_model_probing.ipynb`  
**Hardware:** A100.

Runs Pythia-6.9B, Pythia-12B, and Mistral-7B-v0.1 probing.

**Expected:** 99-100% probe accuracy at all layers.

---

## 10 — Hispanic-Black Valence Across Layers (~3 min)

**Notebook:** `10_valence_layer_by_layer.ipynb`  
**Hardware:** M4. Requires `white_encoding_results.pkl` from notebook 02.

Produces Figure 8.

---

## 11 — White Encoding Valence at Layer 6 (~3 min)

**Notebook:** `11_valence_white_encoding.ipynb`  
**Hardware:** M4. Requires `white_encoding_results.pkl` from notebook 02.

Produces Figure 7.

---

## 12 — Mistral-Instruct Full Pipeline (~60 min)

**Notebook:** `12_mistral_instruct_probing.ipynb`  
**Hardware:** A100.

Behavioral baseline, rotation tests, and probing for Mistral-7B-Instruct-v0.1.

**Expected:** A=31%, B=33%, C=36% for race_ethnicity. 100.0% probe accuracy.

---

## Verifying without rerunning

```python
import pandas as pd

# GPT-2 permutation results (Table 2)
print(pd.read_csv('results/patching_permutation_test_gpt2.csv'))

# Mistral permutation results (Table 2)
print(pd.read_csv('results/patching_permutation_test_mistral_instruct.csv'))

# Probe accuracies (Table 1)
print(pd.read_csv('results/white_encoding_probe_results.csv'))
```
