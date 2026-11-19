# Evaluating Dataset Watermark Robustness Under Data Curation

AG News semester project: paired dirty-label watermarks, MiniLM embeddings,
semantic deduplication, kNN outlier removal, matched random controls, removal
metrics, and single-condition DistilBERT training.

The active protocol is **tokenized-v2**. Old unversioned artifacts and smoke
models remain historical and must not be used for corrected experiments.
No corrected full classifier has been trained, and no held-out verification
predictions or experimental verification p-values have been produced.

## Setup

Use Python 3.11+ and PowerShell from the repository root. Skip venv creation
when `.venv` already exists. Use this interpreter for every command:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --no-deps "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -c constraints-gpu.txt -e ".[curation,training,dev]"
.venv/Scripts/python.exe -m pip check
$env:HF_HOME = Join-Path (Get-Location) ".cache/huggingface"
```

The constraints preserve the working Windows/NVIDIA environment, including the
Datasets-compatible fsspec version. Torchvision is unnecessary for this project.
Other machines should choose their appropriate Torch build. First downloads
need internet. Cached runs can set `$env:HF_HUB_OFFLINE = "1"` and
`$env:HF_DATASETS_OFFLINE = "1"`.

The laptop has an RTX 4070 Laptop GPU with approximately 8 GiB VRAM and 16 GiB RAM.
Torch 2.11.0+cu128 / CUDA 12.8 works with the installed driver 576.52. Close other
memory-heavy applications and run one classifier at a time. The prior synthetic
batch-16, length-256 check used about 1.6 GiB allocated VRAM; full-run host RAM,
runtime, and GPU recovery have not been benchmarked. Colab is optional.

```powershell
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.ones(8, device='cuda'); print((x*x).sum().item())"
```

## Corrected data and verification units

| Full partition | Records | Labels: World / Sports / Business / Sci-Tech |
| --- | ---: | --- |
| Training | 104,954 | 26,997 / 25,989 / 25,987 / 25,981 |
| Clean validation | 12,000 | 3,000 / 3,000 / 3,000 / 3,000 |
| Calibration probes | 1,500 | 0 / 500 / 500 / 500 |
| Evaluation probes | 1,500 | 0 / 500 / 500 / 500 |
| Official test | 7,600 | 1,900 / 1,900 / 1,900 / 1,900 |

The pinned source is `fancyzhx/ag_news` revision
`eb185aade064a813bc0b7f42de02595523103ca4`. The official test's text, labels,
membership, and order are unchanged. Stable IDs identify original source rows.

Preparation uses the same pinned uncased DistilBERT tokenizer, special tokens,
right truncation, and maximum length 256 as training. Model-input fields are
hashed before dynamic padding. Forty-six development records are quarantined:
10 test collisions, 28 validation collisions, four calibration collisions,
three evaluation collisions, and one repeated evaluation input. Two existing
training records refill held-out vacancies in the same original classes.
Existing validation membership is unchanged. Training-internal duplicates remain
available for the deduplication study.

Both variants inject **1,049 records**, `floor(104954 * 0.01)`, on identical
non-target source IDs. Their actual injection rate is approximately 0.99949%.
Only selected training labels change to World (0). Triggers are
`cobalt lantern meadow` and
`This report provides a brief update on the situation.`

Both variants have 1,047 completely visible training triggers; two appended
triggers are truncated away. The 1% rule refers to inserted records, not an
extra visibility-screened sample. Training records report this distinction.

`verification_units.parquet` fixes a common panel where both complete triggers
are visible: calibration **1,499** (500 / 500 / 499 non-target classes), evaluation
**1,497** (500 / 498 / 499). Panel selection uses text structure only. Probe
inputs are unique and all clean/triggered views are isolated across partitions.
Distinct inputs can still describe related news stories; semantic independence
is an explicit assumption, not a guarantee.

Smoke is a separate protocol instance: 1,680 training, 200 validation, two probe
pools of 60, 16 injections per variant, and the unchanged 7,600-row test. Its
classifier caps are 128 training and 64 validation examples. Never warm-start
full runs from smoke checkpoints or combine their probe predictions.

## Generated artifacts and rebuild commands

| Active location | Contents |
| --- | --- |
| `data/processed/tokenized-v2/{full,smoke}` | Partitions, quarantine, input identities, manifests |
| `data/watermarked/tokenized-v2/{full,smoke}` | Paired variants, selected IDs, probe views and common panel |
| `data/embeddings/tokenized-v2/{full,smoke}` | Fresh pinned MiniLM vectors and current training mappings |
| `data/curated/tokenized-v2/{full,smoke}` | Scores, conditions, removal IDs and matched random controls |
| `artifacts/metrics/tokenized-v2/{full,smoke}` | Removal metrics and descriptive random comparisons |
| `artifacts/models/tokenized-v2/full/<condition>/seed_<seed>` | Selected model, validation outputs and manifest |

Rebuild in this order, substituting `smoke.toml` for a small pipeline check:

```powershell
.venv/Scripts/python.exe -m watermark_robustness prepare --config configs/full.toml
.venv/Scripts/python.exe -m watermark_robustness watermark --config configs/full.toml
.venv/Scripts/python.exe -m watermark_robustness embed --config configs/full.toml
.venv/Scripts/python.exe -m watermark_robustness curate --config configs/full.toml
.venv/Scripts/python.exe -m watermark_robustness metrics --config configs/full.toml
.venv/Scripts/python.exe -m watermark_robustness audit --config configs/full.toml
```

Original readiness validation, November 18, 2026: all 11 permanent tests passed,
including the full artifact replay. The smoke artifact audit, Ruff checks,
format checks, and dependency consistency check also pass.

Completed directories are never overwritten. Do not rerun completed stages.
TOML paths resolve relative to the config; CLI overrides resolve from the current
working directory. Version and protocol hashes, tokenizer settings, source
fingerprints, and parent-manifest checksums reject incompatible artifacts.
Use a new protocol/path for methodological changes; do not relabel old artifacts.

Full conditions: 3 uncurated + 18 treatments + 54 random controls = **75**.
Smoke has 39 conditions because it uses one random-removal seed.
For each of three corpora, two methods and three budgets each have one treatment
and three random-removal seeds (42, 43, 44). Nominal budgets are 1%, 5%, 10%;
deduplication caps unavailable removals, and random controls match actual counts.
See `conditions.parquet` for exact capped counts and IDs. Do not count identical
retained-dataset aliases as independent evidence. Removal rates are descriptive,
not learned-watermark verification results; zero denominators are undefined.

Corrected full removals are 1,049 at 1%, 5,247 at 5%, and 10,495 at 10% for
kNN. Deduplication reaches the first two counts; its nominal 10% caps are
9,841 clean, 9,769 phrase, and 9,836 sentence. Each random control uses its
associated actual count. The 75 named conditions represent 57 distinct retained
datasets: 225 logical seed/condition results can potentially reuse 171 fits.
Reuse is not automated, and aliases do not add independent replications.

## Permanent tests, source freeze, and first training runs

Tests use the pinned tokenizer cache populated by preparation. They include
split isolation, deterministic repair, dirty-label insertion, protocol/source
mismatch rejection, random controls, and interrupted-training equivalence with
a tiny dropout-enabled CPU model. The saved-full-artifact integration audit
runs when generated full artifacts are present; otherwise it is explicitly skipped.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m ruff check src/watermark_robustness tests
.venv/Scripts/python.exe -m watermark_robustness audit --config configs/full.toml
```

The implementation-only `verification-v1` amendment adds verification execution
and permanent tests. It preserves both configurations and the statistical plan.
See [the amendment record](docs/IMPLEMENTATION_AMENDMENT.md).
Amendment validation: **32 permanent tests passed with no skips**, including
the full artifact replay, both panel input audits without inference, exact
statistics, source revision enforcement, and synthetic CPU prediction execution.

The original 28-file `protocols/tokenized-v2/source-lock.json` and original ZIP
remain immutable. `source-lock-verification-v1.json` is the new versioned lock;
`active-source.json` selects it with a checksum. Training and recovery use its
source hash, while existing data/curation protocol hashes remain valid unchanged.
Inspect the enforced revision with:

```powershell
.venv/Scripts/python.exe -c "from watermark_robustness.protocol import require_frozen_source; print(require_frozen_source())"
```

Source archives under `artifacts/source/` exclude generated datasets/models.
Training refuses unreviewed source changes. A future reviewed implementation
amendment must use a NEW name with `freeze-source --revision <name>`; neither
existing locks nor archives can be overwritten. Methodology changes require a
new experimental protocol. Keep or commit the small source files and locks;
large artifacts remain ignored. Source freezing does not create a Git commit.

Train these baselines sequentially first; stop and inspect the clean run before
continuing. No full grid is launched automatically:

```powershell
.venv/Scripts/python.exe -m watermark_robustness train --config configs/full.toml --condition clean/uncurated --seed 42
.venv/Scripts/python.exe -m watermark_robustness train --config configs/full.toml --condition unrelated_phrase/uncurated --seed 42
.venv/Scripts/python.exe -m watermark_robustness train --config configs/full.toml --condition natural_sentence/uncurated --seed 42
```

Review the three seed-42 baselines before authorizing further seeds or curated
conditions. The eventual design retains seeds 42, 43, and 44. DistilBERT uses FP32, train/eval
batches 16/32, length 256, three epochs, AdamW learning rate 2e-5, weight decay
0.01 excluding bias/LayerNorm, linear decay with 10% warmup, and clipping at 1.0.
Select highest clean validation accuracy, then lowest loss, then earliest epoch.
No test/probe predictions are used for training or checkpoint selection.

## Interrupted runs and verification

Recovery is saved initially, every 500 optimizer steps, and after every epoch.
It includes current model, AdamW and scheduler states, batch order/cursor,
partial-epoch loss, global step, best model/history/validation predictions,
Python/NumPy/Torch/CUDA RNG states, and sampler RNG. Resume uses the exact same
configuration, condition, seed, source freeze, source artifacts, software, and
device. It may replay up to 499 optimizer steps since the last published state.

```powershell
.venv/Scripts/python.exe -m watermark_robustness train --config configs/full.toml --condition clean/uncurated --seed 42 --resume
```

An unfinished run stays in the adjacent `.seed_42.in-progress` directory; the
completed model directory is published only on success. Checksummed state
files are published through an atomic pointer and the previous generation is
retained. Corruption fails closed. Only load trusted local recovery files.
Filesystem/power-loss durability depends on the host, and cross-platform or
cross-driver bitwise reproducibility is not guaranteed. Mid-epoch recovery has
been regression-tested on a tiny CPU model, not a full GPU run.

The frozen [verification plan](docs/VERIFICATION_PLAN.md) specifies paired sign
tests against clean-trained controls, explicit assumptions, effects and ties,
and one Holm-corrected family of 150 named hypotheses. Curated comparisons
estimate the whole pipeline effect because policy-matched clean models can have
different retained IDs. Verification execution is implemented; no real probe
predictions or experimental p-values have been generated during this amendment.

After the three baselines are trained and reviewed, a calibration-only report
can be produced with this command (not run during the amendment):

```powershell
.venv/Scripts/python.exe -m watermark_robustness verify --config configs/full.toml --pool calibration_probe --output-dir artifacts/verification/tokenized-v2/full/calibration-baselines-seed42
```

Only completed models from the active source revision are accepted. Predictions
use the saved eligible common panel and strict record/unit-ID pairing, with one
model loaded at a time. The command saves predictions, article pairs, effects
overall/by class, trigger responses, calibration clean-control diagnostics, a
150-row Holm table, and a checksummed provenance manifest. No official-test
inference or training occurs. Existing report directories cannot be overwritten.

Ties are excluded from the sign-test nonzero count; all ties give p=1 and an
undefined positive fraction. The four target rates, mean A_W/A_C/D, and sign
counts accompany every comparison. Exact rational binomial tails and Holm
arithmetic avoid underflow: use `p_value_text` / `adjusted_p_value_text` when
reporting extremely small values; float columns carry explicit underflow flags.
Unrun hypotheses get p=1; aliases remain separate named hypotheses. Class-specific
p-values and clean-control diagnostics are descriptive. Calibration never makes
confirmatory rejection decisions. Evaluation requires explicitly selecting
`evaluation_probe`; a final decision additionally requires `--final` and all
150 watermarked models plus their clean references. No intermediate report
licenses an early rejection claim or changes to the frozen analysis.

New tests cover analytical binomial examples, agreement with SciPy, ties and
ordinary trigger effects, exact Holm arithmetic and underflow, strict pairing,
source/control provenance, final-look gates, immutable source revisions, and
prediction execution on a tiny random CPU model with synthetic text. Actual
stored probe panels are audited without model inference. After calibration
review, remaining work is the authorized training grid, held-out verification,
separate clean official-test evaluation, plots, and the semester report.

The package CLI uses `src/watermark_robustness/`. Legacy scripts and the notebook
are prototypes with historical sizes and do not define this experiment.
