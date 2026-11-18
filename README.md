# Salon Experiment & Causal Analysis

Data Science with Python Analyst Internship (YuvaIntern, Coimbatore). Author: Soni Prachi Kishanbhai.

A fictional Coimbatore salon and spa chain launched a change last quarter and claims it helped. This project (1) designs a proper A/B test for a new WhatsApp reminder, (2) analyses an experiment dataset, (3) estimates the impact of last quarter's change with difference-in-differences, (4) segments the effects, and (5) applies CUPED / regression adjustment.

## Headline results (simulated data)

| Question | Result | Decision |
| WhatsApp reminder, no-show rate | 11.9% to 9.3%, -2.6 pp (95% CI -3.7 to -1.6); all guardrails pass | GO |
| Last quarter's change, revenue per visit (DiD vs Tiruppur/Erode) | +3.0% (95% CI +0.3% to +5.8%) | GO, with monitoring |

## Data note

The brief describes the data (22 outlets, about 300,000 rows, no-shows stored as cancellations, service names that differ by outlet, walk-ins without client ID) but no file was attached. `generate_data.py` builds a simulated dataset with that structure and a fixed seed. To use real files, put them in `data/` using the column names in `DATA_DICTIONARY.md` and re-run the notebook.

## Files

| File | Purpose |
|---|---|
| `Salon_Experiment_Causal_Analysis.ipynb` | Executed analysis notebook (all parts, with outputs) |
| `analysis_src.py` | Same analysis as a plain script (the notebook is built from it) |
| `generate_data.py` | Simulated data generator |
| `build_notebook.py`, `build_docs.py` | Build the notebook and the Word documents |
| `deliverables/` | Project report, experiment design document, one-page decision memo (Word) |
| `data/`, `figures/`, `results.json` | Data, charts, every key number |

## Run

```bash
pip install -r requirements.txt
python generate_data.py          # creates data/
python build_notebook.py         # executes the notebook, writes figures/ and results.json
python build_docs.py             # writes the Word documents to deliverables/
```

## Method in one paragraph

Cleaning recovers no-shows from the 'Cancelled' status and harmonises service names. The A/B test is sized by power analysis (80% power, 5% significance, 1.5 pp minimum detectable effect), checked for sample ratio mismatch and covariate balance, and analysed with Wald and bootstrap confidence intervals and pre-defined guardrails. The quarterly change is evaluated with two-way fixed-effects difference-in-differences on an outlet-week panel with outlet-clustered errors, t critical values, a stratified cluster bootstrap and a permutation test, plus a pre-trend test, a placebo launch and an event study.
