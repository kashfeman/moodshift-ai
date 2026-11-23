# Final submission readiness audit

Audit date: 24 November 2026. The archives were inspected without changing their
contents. "Unconfirmed" means no permission or applicable license evidence was
found in the project or archive; it is not a legal conclusion that use is prohibited.
Do not distribute unconfirmed images or their derivatives until their provenance and
permission for the intended university submission have been verified.

## Image inventory

All paths below are entries in `MoodShift_AI_University_Submission_7E.zip`.
`docs/submission/evidence/` is abbreviated as `evidence/` and
`checkpoints/experiment_7c2/` as `experiment_7c2/`.

| File | Documented source | License / permission evidence found | Distribution assessment | Manual confirmation |
| --- | --- | --- | --- | --- |
| `evidence/dreamy-browser-comparison.png` | Local QA screenshot containing the landscape and derivatives; report records source hash, not author/source. | None for underlying landscape. | Unconfirmed; exclude until cleared. | Yes: original photo and derivatives. |
| `evidence/final-modes.png` | Local comparison using landscape and processed outputs. | None for underlying landscape. | Unconfirmed; exclude until cleared. | Yes: original photo and derivatives. |
| `evidence/ui-laptop.png` | Local screenshot of the empty application interface; no photograph visible. | No screenshot license or owner confirmation found. | No unknown photograph visible; distribution permission still unconfirmed. | Yes: screenshot ownership and UI/dependency terms. |
| `experiment_7c2/loss_curve.png` | Project-generated plot of recorded training losses. | No separate project license notice found; no photograph visible. | No photographic rights issue identified; project ownership/license not verified. | Confirm project ownership / release terms. |
| `experiment_7c2/val_1_comparison.png` | Local validation panel containing content, style and generated images. | None for underlying photographs. | Unconfirmed; exclude until cleared. | Yes, including both inputs and derivatives. |
| `experiment_7c2/val_2_comparison.png` | Local validation panel containing content, style and generated images. | None for underlying photographs. | Unconfirmed; exclude until cleared. | Yes, including both inputs and derivatives. |
| `experiment_7c2/val_1_content.png` | Cropped local held-out content photograph. | Author, source and license not documented. | Unconfirmed; exclude until cleared. | Yes. |
| `experiment_7c2/val_1_style.png` | Cropped local held-out style photograph. | Author, source and license not documented. | Unconfirmed; exclude until cleared. | Yes. |
| `experiment_7c2/val_1_guided.png` | Guided output derived from validation content and style. | No source-photo permissions documented. | Unconfirmed; exclude until cleared. | Yes: source images and derivative use. |
| `experiment_7c2/val_2_content.png` | Cropped local held-out content photograph. | Author, source and license not documented. | Unconfirmed; exclude until cleared. | Yes. |
| `experiment_7c2/val_2_style.png` | Cropped local held-out style photograph. | Author, source and license not documented. | Unconfirmed; exclude until cleared. | Yes. |
| `experiment_7c2/val_2_guided.png` | Guided output derived from validation content and style. | No source-photo permissions documented. | Unconfirmed; exclude until cleared. | Yes: source images and derivative use. |
| `experiment_7c2/val_1_swapped_style.png` | Guided output using a swapped held-out style image. | No source-photo permissions documented. | Unconfirmed; exclude until cleared. | Yes: source images and derivative use. |
| `experiment_7c2/val_2_swapped_style.png` | Guided output using a swapped held-out style image. | No source-photo permissions documented. | Unconfirmed; exclude until cleared. | Yes: source images and derivative use. |

The source report states that the earlier landscape's authorship/license was not
established (`PROJECT_REPORT.md`, section 14). The hash in section 11 identifies
bytes but is not evidence of ownership. Generated results can depend on underlying
photo rights; do not treat image transformations as permission. The original
training images are excluded from both ZIPs, but 96×96 validation inputs and
photo-derived outputs are present in the source ZIP. The decoder ZIP contains no
images; the training-data terms and any conditions on sharing trained weights still
need review. The original archives remain unchanged by this audit.

## Academic details

There are **no pre-existing labeled fields** for names, roll numbers, group members,
instructor, university or course in the report. Add the university-required cover
before the title at `docs/submission/PROJECT_REPORT.md:1`, including those details
and any other fields required by the institution. Do not infer or invent them.
`docs/submission/SUBMISSION_CHECKLIST.md:21` and `:56` explicitly flag the missing
cover and identity information. `README.md:1` and `docs/submission/DEMO_SCRIPT.md:1`
have no labeled personal-information placeholders; add an attribution only if the
university requires one there. Check the institution's required format, submission
method, disclosures, and naming conventions.

## Archive and installation checks

- Source ZIP: 98 entries, including 45 source Python files, 22 test Python files,
  `pyproject.toml`, `README.md`, setup instructions, actual loss CSV, chart and
  validation evidence. The original dataset directory is absent. No environment,
  cache, `.git`, key/certificate or checkpoint files were found by filename scan;
  this is not proof that all contents are free of sensitive data.
- Decoder ZIP: `checkpoints/experiment_7c2/checkpoint.pt` and a short README.
  Checkpoint SHA-256:
  `59594434fb728daf2f9dc5cc27ec0e911ce4755819fcb05f54c0cf83b4c5a485`.
  It loads with `weights_only=True`, reports `guided-v1`, five epochs, eight decoder
  tensors and eight Adam states at step 55. Recorded epoch training and validation
  losses match the source ZIP's CSV. These checks corroborate saved training
  records; they do not independently reconstruct or prove the training dataset.
- The official SqueezeNet encoder is not bundled. The setup instructions describe
  obtaining the official weights separately; Internet and upstream availability
  are needed if the weights are not already cached. No top-level project LICENSE
  file or third-party NOTICE bundle was found in either ZIP. Confirm project and
  dependency/model terms appropriate to the intended distribution.
- Tested during Milestone 7E: archive extraction, restoring the decoder, an editable
  install of extracted source into a separate environment using dependencies from
  the existing workstation environment, and local experimental Gradio inference.
  Full test and UI results are in `TEST_SUMMARY.md`.
- Not tested: a genuinely clean machine or environment downloading/installing all
  declared dependencies and the official encoder from the Internet. The current
  workstation's Python launcher did not find a registered Python 3.12 installation;
  its existing virtual environment worked. Confirm Python 3.12 availability and
  the `py -3.12` launcher command on the target Windows laptop, or adapt the command
  to the installed Python 3.12 executable. Test the exact setup instructions there.

The audit file is a new workspace document and is **not included in the existing
source ZIP**. A revised submission archive is needed after resolving permissions,
adding required academic information and deciding whether to include this audit.
