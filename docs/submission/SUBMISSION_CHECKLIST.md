# Submission checklist

## Completed locally

- [x] Preserve the existing package and architecture; avoid neural/UI rewrites.
- [x] Investigate actual low-resolution enlargement and real Dreamy artifacts.
- [x] Compare genuine 20/60-step outputs at 128/256/384 processing limits.
- [x] Preserve source detail in Hybrid while retaining real neural influence.
- [x] Keep Preview lightweight and distinguish Grade/Neural/Hybrid truthfully.
- [x] Run ordinary tests, separate cached-model tests, Ruff, pip and Git checks.
- [x] Exercise all 15 mode/preset combinations through the actual Gradio API.
- [x] Verify negative inputs, repeated generation, download isolation and caching.
- [x] Inspect real dark UI captures and record browser coverage accurately.
- [x] Include actual selected screenshots and report exact experiment settings.
- [x] Prepare academic report, source attribution, QA summary and timed demo script.
- [x] Keep weights, caches, secrets and temporary/full-size outputs out of Git.
- [x] Make no automatic commit, push, identity, history or remote changes.

## Before handing in or presenting

- [ ] Fill institution-specific student names, IDs, supervisor and required cover.
- [ ] Confirm permission to distribute the investigated landscape and derivative
      screenshots; otherwise rerun with a photograph you own and update evidence.
- [ ] Review the report and explain the code/contributions in your own words.
- [ ] Check local `outputs/final_qa/` evidence if submitting full-size originals and
      results separately; these files are intentionally absent from Git checkout.
- [ ] Rehearse the 5–7 minute demo on the classroom machine with the chosen image.
- [ ] Confirm Python 3.12, dependencies and the official cached model before going
      offline. Do not include pretrained weights in the source submission.
- [ ] Save desired PNG downloads before temporary cache expiry/server shutdown.
- [ ] Apply the university's formatting, citation and disclosure requirements.
- [ ] Review `git diff` and manually decide whether to commit.

Earlier milestone suggested commit:
`fix: preserve hybrid scene detail and prepare final QA submission`

Launch from the repository root:

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) '.cache\torch'
.venv\Scripts\python -m moodshift.ui.app --port 7860
```

Open http://127.0.0.1:7860. Keep Offline enabled for the existing cached model.
Further research and deployment remain outside this submission.

## Milestone 7E handoff checks

- [x] Full suite: 255 passed, 3 optional skips; all three opt-in tests passed separately.
- [x] Verify the pinned trained guided checkpoint and 55 optimizer steps.
- [x] Check the actual five-epoch records, training curve and held-out outputs.
- [x] Exercise local Studio modes and separate guided generation/download by HTTP.
- [x] Confirm local image/gallery configuration excludes the Spaces Share action.
- [ ] Before distribution, confirm rights for every supplied photograph, comparison
      image, and screenshot; otherwise replace or omit the affected evidence.
- [ ] Fill the institution-specific cover, student identity and course requirements.
- [ ] On the presentation laptop, install Python 3.12/dependencies, restore the
      separate trusted decoder package, and obtain the official cached encoder.
- [ ] Independently review the documentation, provenance and academic disclosures.

The experimental decoder's 96×96 result retains structure but transfers style only
subtly. It is not evidence of strong mood transfer or broad generalization.

## Milestone 7F.1 package

The revised source package excludes the local photographic comparisons and
screenshots because permissions are not verified. The original local files are
preserved. It retains recorded numerical losses and the project-generated loss
curve, not the raw training dataset. The trained decoder ZIP is separate and its
training-data and weight redistribution rights remain unconfirmed. Review
`THIRD_PARTY_NOTICES.md` and `SUBMISSION_READINESS_AUDIT.md`; add an appropriate
project license only after the project owner selects one.

- [ ] Complete every field on the academic cover.
- [ ] Confirm rights for training data and any trained-weight distribution.
- [ ] Confirm project and dependency/model terms before distributing.
- [ ] Test a clean installation on the destination machine and obtain the official
      pretrained encoder independently if it is not cached.
