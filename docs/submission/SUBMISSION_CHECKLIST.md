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

Suggested conventional commit:
`fix: preserve hybrid scene detail and prepare final QA submission`

Launch from the repository root:

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) '.cache\torch'
.venv\Scripts\python -m moodshift.ui.app --port 7860
```

Open http://127.0.0.1:7860. Keep Offline enabled for the existing cached model.
Future research and deployment work are outside this final QA task.
