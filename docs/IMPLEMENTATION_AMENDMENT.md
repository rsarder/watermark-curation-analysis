# tokenized-v2 implementation amendment: verification-v1

Authorized scope: implement the existing paired-sign-v1 plan, add permanent tests,
and append an immutable source revision. The experimental protocol is unchanged.
Neither configuration, the constraints, nor `VERIFICATION_PLAN.md` is edited.
Dataset membership, watermark IDs and wording, probe panels, curation selections,
training settings/seeds, and statistical choices remain exactly as frozen.

The original 28-file `protocols/tokenized-v2/source-lock.json` and its original
source ZIP remain byte-identical. `source-lock-verification-v1.json` records the
new complete source inventory, parent/original hashes, and revision name.
`active-source.json` selects it using the lock's file checksum and source hash.
The training entry point still calls `require_frozen_source`; that guard now
checks the selected revision, original/parent locks, and every current source
file. Removing the pointer does not permit amended code to use the original lock.
Recovery states retain their existing source-hash binding. No enforcement is
bypassed, and existing experimental artifacts keep their original protocol hashes.

Verification reads completed full runs under the configured training directory.
It rejects incompatible protocol/settings/source revisions, incomplete checkpoints,
smoke runs, missing same-seed policy-matched clean references, and inconsistent
clean-validation checkpoint selection. It predicts only on the stored common
eligible panel for the requested pool, pairing by stable record and unit IDs.
One model is loaded at a time. Official-test inference and training are separate.

The implementation directly evaluates the plan's finite binomial sum using exact
integer/rational arithmetic. Ties are reported and omitted from the nonzero count;
all ties give p=1. Effect sizes and four target rates are reported overall and by
source class. Rational probabilities retain nonzero scientific notation even
when binary floats underflow. Holm ordering, adjustment, and threshold decisions
preserve exact probabilities and all 150 named hypotheses, including aliases.
Unrun hypotheses get p=1. Class-specific p-values and clean-control diagnostics
are descriptive and do not add confirmatory hypotheses.

Calibration is the default CLI pool. It never produces confirmatory decisions.
Evaluation requires an explicit pool selection. Even a complete evaluation report
remains descriptive unless `--final` is supplied; a final run requires all 150
watermarked models and their clean references before prediction starts. This flag
implements the frozen final-look rule without changing the test or hypothesis
family. The present task executes neither real calibration/evaluation predictions
nor full classifier training. Tests use synthetic predictions and a tiny random
CPU model with synthetic text; the existing full artifact audit is read-only.

Outputs are new, non-overwriting artifact directories: predictions, article-level
pairs, effects, trigger responses, clean-control diagnostics, full-family Holm
table, and a checksummed manifest with source/model/plan/panel provenance.
Prior reports are not pooled across probe pools or treated as new independent
observations. Story-level dependence remains the limitation stated in the plan.
