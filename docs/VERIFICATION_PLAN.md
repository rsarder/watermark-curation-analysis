# Frozen verification plan: tokenized-v2 / paired-sign-v1

This plan is fixed before held-out evaluation predictions. No such predictions
have been generated. The source lock covers this file and both configurations.
Changing a confirmatory rule requires a new protocol version and an explicit
disclosure of any outcomes already observed.

## Population, units, and fixed panels

Target class is World (0). Calibration and evaluation come only from the three
other original classes. They have separate source IDs and classifier input
identities. The official test set remains unchanged and is used only for clean
classification accuracy after model selection.

Preparation quarantines development inputs identical to the official test,
then resolves other split conflicts in priority order: evaluation, calibration,
validation, training. Repeated tokenized probe inputs retain only the first
source ID. Vacancies are filled deterministically from the same training class.
Training-internal duplicates remain part of the curation question.

`verification_units.parquet` fixes one source record per tokenized clean input
and a common panel where BOTH entire appended triggers survive tokenization.
The panel is selected from text structure alone, without model predictions.
Clean and both triggered views are checked for cross-partition collisions.
Use exactly the stored eligible IDs for both variants and all model conditions.
Do not resample panels by model, condition, seed, or observed prediction.

Exact-input uniqueness does not prove independent news stories. The primary
analysis assumes the retained article units are independent conditional on the
fixed models and source class. Unknown semantic/story dependence remains a
limitation. Do not count clean/triggered views, aliases, seeds, or repeated use
of the same panel as extra independent article observations. A later story-
cluster analysis is exploratory unless frozen under a new protocol before
evaluation. No post-outcome clustering or exclusions are allowed here.

## Models, references, and estimand

Each W is a watermarked model selected using clean validation only. Reference C
is clean-trained using the SAME training seed and corresponding named curation
policy, budget, and random-removal seed (uncurated maps to clean/uncurated).
Policies may yield different retained IDs or capped counts across corpora.
Therefore curated comparisons estimate the whole watermark-plus-curation
pipeline effect, not an injection-only causal effect at fixed retained IDs.
All models start from the pinned base model; never warm-start from smoke models.

For article i, clean input x_i and triggered input t(x_i), define

    A_M(i) = 1[M(t(x_i)) = 0] - 1[M(x_i) = 0]
    D_i = A_W(i) - A_C(i), taking values -2, -1, 0, 1, 2.

This subtracts the ordinary trigger response of the clean-trained control.
Report four target-prediction rates (W/C, clean/triggered), mean A_W, mean A_C,
mean D, positive/negative/tie counts, and B/(B+F) where defined, also by original
source class. Mean D is an effect size; the sign test does not test its mean.
Report clean official-test accuracy separately, for every completed model.

## Primary null, alternative, and p-values

Conditional on nonzero D and fixed trained models, the composite null is:
for EACH eligible source class, P(D > 0 | D != 0, class) <= 1/2.
The alternative is excess positive signs, implying a violation in at least one
class. This is a directional sign-dominance test, not a test of equal accuracy,
uniform class predictions, or perfect watermark removal.

Let B = count(D > 0), F = count(D < 0), T = count(D = 0), n = B + F.
Ignore zero ties in the test, but always report them. For n = 0, set p = 1
and leave B/n undefined. Otherwise use the one-sided tail

    p = sum(comb(n, j) * 0.5**n for j in range(B, n+1))
      = scipy.stats.binomtest(B, n, p=0.5, alternative="greater").pvalue

Under independent article signs and the classwise null bounds, this tail is
conservative; it is exact at the fair-sign boundary. No symmetry of nonzero
magnitudes is assumed. Do not use p0 = 0.25 or estimate a null success rate from
three random-removal replicates. Models are conditioned on; three training seeds
describe training variability and do not establish a population-level seed effect.

## Multiplicity and stopping

One confirmatory family contains 2 variants x 25 conditions x 3 model seeds =
150 named hypotheses. Keep the family size 150 even when retained-data aliases
share a fitted model and identical p-value. Apply Holm step-down correction at
family-wise alpha = 0.05, valid without independence between hypotheses.
For sorted raw p-values p_(j), adjusted p_(j) is
min(1, max_{k<=j} ((150-k+1) * p_(k))). For an interim or incomplete report,
assign p = 1 to unrun hypotheses; recompute the complete family for the final
report. Do not treat repeated interim looks as new opportunities for rejection:
confirmatory rejection decisions are made once at the prespecified final look.
Baseline and calibration reports may be descriptive, without an early rejection
claim. No additional curation-versus-random significance test is prespecified.

## Calibration and negative controls

Use calibration only to check execution, establish whether uncurated watermarked
models show useful detectability, and report ordinary trigger behavior in clean
models. Do not tune trigger strings, target class, panel exclusions, hyperparameters,
test direction, or correction after this freeze. If calibration is inconclusive,
report it and pause the grid rather than quietly changing the procedure.

Clean controls must be queried on the same paired views. Report their trigger
effects, including class-specific bias. Additional diagnostics compare clean models
across the fixed seed pairs (42,43), (42,44), (43,44); label these calibration
diagnostics, not extra confirmatory evidence. A systematic control anomaly is a
reason to investigate before evaluation; a single nonsignificant control does not
validate specificity or prove a 5% false-positive rate. Identical-control self-
comparisons give D = 0 and p = 1 as a software sanity check only.

Random-removal controls answer whether curation differs descriptively from removing
the same number of records uniformly. They are not a verification null distribution.
Nonsignificance means insufficient evidence under this test; it does not prove
watermark absence. No verification results or p-values are claimed at this stage.

References: [SciPy binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html),
[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
