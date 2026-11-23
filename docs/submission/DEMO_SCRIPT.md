# Classroom demonstration (approximately 6 minutes)

## Prepare before class

Use the existing Python 3.12 environment. From the repository root:

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
$env:TORCH_HOME = Join-Path (Get-Location) '.cache\torch'
.venv\Scripts\python -m moodshift.ui.app --port 7860
```

Open http://127.0.0.1:7860. Keep Offline enabled when the official checkpoint
already exists in that cache. Installation may require Internet; cached inference
does not. Verify the checkpoint before class, rather than attempting downloads
during the demonstration. Use a photograph you have permission to present and
have a second contrasting reference ready. Save outputs before temporary downloads
expire. Keep any separately cleared evidence screenshots open as backup if class hardware
runs slowly; clearly label saved evidence rather than pretending it is live.

## 0:00–0:40 — Purpose and architecture

“MoodShift AI explores photographic color processing and reference-guided neural
styling. Grading changes tone and color directly; neural transfer optimizes image
pixels through frozen pretrained features. We did not train SqueezeNet ourselves.”
Show the three modes. Explain the separation of imaging, engine, service and UI.

## 0:40–1:35 — Cinematic and Sunset Grade

Upload the content photograph. Show the validated RGB preview. Select Color grade,
Cinematic, intensity 70 and Generate. Move the before/after slider and download
the PNG. Show metadata: `numpy-opencv`, no neural steps, no reference applied.
Switch to Sunset and regenerate. Explain cool/warm tonal chroma differences and
LAB luminance/chroma separation. Do not call either result neural inference.

## 1:35–2:50 — Genuine Neural Transfer

Select Neural transfer and upload the contrasting reference. The preset disables
because the external reference is used. Choose Preview, neural strength 0.5,
content preservation 0.7, CPU and Offline. Generate and show actual metadata:
`squeezenet1_1`, `cpu`, reference `user`, steps and internal grid.
“Only image pixels are optimized. Content MSE preserves feature structure;
Gram matrices compare channel correlations; total variation discourages noise.”
Explain that texture artifacts are visible and this is not guaranteed photo-realism.

## 2:50–4:10 — Dreamy Hybrid and quality investigation

Remove the external reference, select Hybrid and Dreamy. Keep intensity 70,
strength 0.5, Preview 20 steps, content preservation 0.7 and no output limit.
Generate. Show `synthetic-preset` and `content-guided-residual` in metadata.
Explain: “The real neural result supplies a color residual. A guided filter uses
the original scene to preserve detail, then Dreamy grading and final blending run.”
If rights have been cleared, open the local genuine 20/60-step browser comparison. Point to cloud artifacts
in the previous enlargement and preserved trees in the new images. The saved
experiment used the same 978×1365 source and CPU settings listed in the report.

## 4:10–5:10 — Quality controls and honest limitations

Enable custom steps and choose 60 while keeping Preview resolution for a fair
step-only comparison. Generate if runtime permits; otherwise refer to the recorded real
60-step metrics and state that they were measured during QA. Explain that Balanced
also increases processing resolution to 256; High is 120/384. The controlled
experiment found no universal benefit from more steps. Hybrid deliberately
reduces new style texture to preserve source structure. Zero intensity bypasses
effects; zero neural strength makes Hybrid grade-only with truthful metadata.

## 5:10–6:00 — Verification and closing

Show the test summary: ordinary correctness tests, separately enabled cached-model
tests, and 15 actual Gradio mode/preset combinations. Show the report's references
and credit the pretrained model, Gatys style-transfer principle and guided-filter
mathematics. Mention 8-bit RGB, CPU runtime, temporary downloads, no automatic mood
recognition, and limited photographic evaluation. Stop the server with Ctrl+C.

If asked whether quality is proved generally: “We observed improvement on the
reported landscape, supported by real crops and detail diagnostics. We have not
conducted a broad user study or established an aesthetic benchmark.”

## Added training demonstration (Milestone 7E)

In **Training & Experiments**, show the checkpoint-backed five-epoch loss table,
loss curve and, if locally available and cleared, held-out comparisons. Explain the 250,691-parameter
content-guided decoder was trained with Adam on local photographs, while the
official ImageNet encoder stayed frozen. The checkpoint history does not itself
record the historical dataset counts; they come from the experiment session.
Select the trusted guided checkpoint, upload a permitted content photograph and
style reference, and generate the separate experimental CPU output. Compare its
96×96 crop and download the PNG. State that its style influence is subtle and that
its output is not equivalent to the Studio's per-image optimization. If required
weights or evidence are missing, point to the displayed explanation instead of
claiming a live result. The trained checkpoint package and official cached encoder
must be installed before class, following the README.
