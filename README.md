# Beyond Behavioral Benchmarks: Mechanistic Evidence of Demographic Encoding Dissociation in Language Models

> Under double-blind review. Author and institution information withheld.

This repository contains all code and saved results for the paper "Beyond Behavioral Benchmarks: Mechanistic Evidence of Demographic Encoding Dissociation in Language Models."

---

## Overview

A four-phase mechanistic evaluation pipeline for auditing demographic encoding in language models:

- **Phase 1** — Behavioral baseline with rotation correction
- **Phase 2** — Token-level activation probing (exact token matching)
- **Phase 3** — Permutation-validated causal activation patching
- **Phase 4** — Exploratory full-vocabulary semantic characterization

All results in the paper are reproducible from the notebooks in this repository. Pre-saved results in `/results/` let you verify findings without rerunning compute-intensive experiments.

---

## Installation

```bash
git clone https://github.com/[username]/identity-under-the-hood.git
cd identity-under-the-hood
```

Install in this order — numpy must precede transformer_lens:
```bash
pip install numpy==1.26.4
pip install transformer_lens datasets scikit-learn pandas matplotlib
```

---

## Data

All experiments use the [BBQ benchmark](https://huggingface.co/datasets/Elfsong/BBQ) loaded automatically from HuggingFace — no download required.

---

## Notebooks

All code is in numbered Colab notebooks. Run them in order.

| Notebook | Phase | Hardware | Runtime |
|---|---|---|---|
| `01_behavioral_baseline.ipynb` | 1 | M4 / A100 | ~10-60 min |
| `02_gpt2_probing_valence.ipynb` | 2+4 | M4 | ~15 min |
| `03_gender_probing.ipynb` | 2 | M4 / A100 | ~30 min |
| `04_causal_patching_gpt2.ipynb` | 3 | M4 | ~30 min |
| `05_causal_patching_pythia.ipynb` | 3 | A100 | ~30 min |
| `06_causal_patching_mistral.ipynb` | 3 | A100 | ~4 hours |
| `07_permutation_test_gpt2.ipynb` | 3 | A100 | ~25 min |
| `08_permutation_test_pythia.ipynb` | 3 | A100 | ~40 min |
| `09_large_model_probing.ipynb` | 2 | A100 | ~60 min |
| `10_valence_layer_by_layer.ipynb` | 4 | M4 | ~3 min |
| `11_valence_white_encoding.ipynb` | 4 | M4 | ~3 min |
| `12_mistral_instruct_probing.ipynb` | 1+2 | A100 | ~60 min |

See `SETUP.md` for step-by-step instructions with expected outputs for each notebook.

---

## Results Verification

Every number in the paper can be verified from pre-saved files in `/results/`:

| Paper claim | File | Column |
|---|---|---|
| GPT-2 p ≤ 0.007 at layers 3-6 | `patching_permutation_test_gpt2.csv` | `p_value` |
| Pythia p ≤ 0.005 at layers 6-12 | `patching_permutation_test_pythia14b.csv` | `p_value` |
| Mistral p ≥ 0.292 n.s. all layers | `patching_permutation_test_mistral_instruct.csv` | `p_value` |
| Pythia KL 0.0015-0.0017 | `causal_patching_pythia_results.csv` | `mean_kl` |
| Mistral KL 0.0011 at layer 6 | `causal_patching_mistral_instruct_results.csv` | `mean_kl` |
| Probe accuracy 100% | `white_encoding_probe_results.csv` | `accuracy` |
| Valence token projections | `white_encoding_results.pkl` | `valence_results` |

---

## Notes on Replication Coverage

**Models fully covered by notebooks:** GPT-2, Pythia-1.4B, Pythia-6.9B, Pythia-12B, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1.

**Models with results but no standalone notebook:** Pythia-1.4B-deduped and Qwen2-1.5B-Instruct probing (Table 1) was conducted in interactive sessions not preserved as standalone notebooks. Both produced 100.0% ± 0.0% probe accuracy consistent with all other models. `01_behavioral_baseline.ipynb` covers behavioral baseline for all eight models including these two.

**Behavioral JSONs for large models:** `pythia-6.9b_behavioral.json`, `pythia-12b_behavioral.json`, and `mistral-7b_behavioral.json` are saved to Google Drive. Download and place in `results/behavioral/`. See `results/behavioral/README.md` for expected values.

---

## Known Issues and Workarounds

**numpy binary incompatibility in Colab:** Always install `numpy==1.26.4` before `transformer_lens`. If you see `ValueError: numpy.dtype size changed`, restart the runtime and reinstall in the correct order.

**Colab session timeouts:** For `06_causal_patching_mistral.ipynb` (~4 hours), use the keep-alive cell at the end of the notebook and leave the browser tab active.

**Google Drive paths:** Notebooks save to `/content/drive/MyDrive/identity-under-the-hood/`. Update paths if your Drive structure differs.

**Exact token matching:** `03_gender_probing.ipynb` uses exact token matching to prevent `'man'` from matching inside `'woman'`. Substring matching produces ~60% accuracy. See the notebook header for details.

**Mistral tokenization:** Mistral-7B-Instruct-v0.1 tokenizes "Hispanic" as two subwords ("His" + "panic"). All notebooks extract at the first subword position.

---


## License

MIT License.
