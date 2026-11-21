# Changelog

Project: **Silent Speech Recognition Using Facial Motion Signals**
Location: `/Users/apple/SILENT_SPEECH_RECOGNITION`
Local-only project. No audio, no cloud, no committed model weights.

All entries below were verified by running the code in this repository. Nothing in this
file is projected or estimated. **The project is not functional yet**: there is no
recorded dataset, no trained model, no live recogniser and no dashboard. The Phase 2
recorder is implemented but **has never been run against a camera**; real acceptance
testing is still pending (see [Pending work](#pending-work)).

Format: newest phase first within each section. Dates are ISO-8601 (`YYYY-MM-DD`).

---

## Phase 0 — Safety baseline and recovery investigation

Completed on 2026-11-12. No source code was changed in this phase.

### Backup

| Item | Value |
| --- | --- |
| Archive | `/Users/apple/silent_speech_backups/ssr_phase0_20261112_231805.tar.gz` (3.99 MB) |
| Checksum file | `ssr_phase0_20261112_231805.sha256` |
| Integrity re-checked | `shasum -a 256 -c` → **OK** |

### Added

| File | Purpose |
| --- | --- |
| `.gitignore` | Excludes `.venv/`, `.venv-*`, `__pycache__/`, `.DS_Store`, nested `backups/`, `*.tar.gz`, `*.log`, and the regenerable `dataset/processed/*.npy` / `*.pkl`. Explicitly documents what must **not** be ignored, including the irreplaceable `dataset/raw/**/*.csv`. |
| `requirements.txt` | Runtime dependencies. |
| `requirements-lock.txt` | Pinned versions actually present in `.venv`. |

### Repository state

- `git init` on branch `main`. **0 commits** at the time of writing.
- Git identity and configuration were deliberately left untouched.
- Untracked, not staged: 54 files (53 project files + `.gitignore`).

### Recovery investigation — conclusion: the old dataset is unrecoverable

- `dataset/raw/` contains **0 files**. No recordings exist anywhere in the project.
- The 22 files in `outputs/` are derived analysis artefacts of a single 1142-frame clip
  (`multi_lip_landmarks.csv`, `normalized_lip_landmarks.csv`, graphs, reports). They are
  not training recordings and cannot be turned into a multi-word dataset.
- A leftover aborted virtual environment (`.venv-1`) was identified and is ignored, not
  deleted.

**Consequence:** every sample used for training must be recorded again in Phase 2. No
accuracy figure from any earlier session may be reused or quoted.

---

## Phase 1 — Core architecture and shared feature extraction (in progress)

Objective: make the training path and the live path use **one** implementation of
frame → feature, and remove the duplicated constants that caused the earlier silent
mismatch. Five files are affected in this phase.

### `config/config.py` — **new** (155 lines)

Single source of truth for every constant that was previously copy-pasted (landmark
index list was duplicated in 6 files, word labels in 6, working-directory-relative paths
in 14).

- Paths (`RAW_DIR`, `PROCESSED_DIR`, `OUTPUTS_DIR`, `MODELS_DIR`, `X_PATH`, `Y_PATH`,
  `METADATA_PATH`, `SKIPPED_PATH`, `MODEL_PATH`) resolved from `__file__`, so scripts no
  longer depend on the current working directory.
- `LANDMARK_IDS` (10 landmarks) and `LANDMARK_NAMES`.
- Feature contract: `SEQUENCE_LENGTH = 60`, `FEATURES_PER_FRAME = 21`,
  `FLAT_FEATURE_COUNT = 1260`, `MIN_FRAMES = 25`.
- Camera and MediaPipe options, `SMOOTHING` parameters, `RECORDING` parameters.
- Inert module: no I/O, no camera, no `cv2`/`mediapipe` import.

### `src/utils/lip_features.py` — **new** (622 lines)

The single shared feature engine. Both training and live prediction must call it.

- `rows_from_dataframe` / `landmarks_to_row` — CSV row ↔ 21 features, always 21 columns.
- `frame_size_from_dataframe` — returns per-frame `frame_w`/`frame_h`, or `None`.
- **Isotropic mode** — both axes multiplied by the *same* dimension
  (`SCALE_AXIS = "width"`), which makes features invariant to `frame_w:frame_h`.
- **Legacy pixel mode** — anisotropic (`x*W`, `y*H`), kept only so old files can be read;
  it is documented as deprecated.
- Mode is decided **only** by whether `frame_w`/`frame_h` were passed. It is never
  inferred from the number of CSV columns.
- `NORMALISED_COORDINATE_LIMIT = 2.0` — legacy mode refuses input that looks
  MediaPipe-normalised instead of silently producing meaningless features.
- `resample_sequence` — whole-clip linear time normalisation to 60 points.
- `build_sequence` — the full pipeline, taking the frame count **after** de-duplication.

### `tests/test_lip_features.py` — **new** (581 lines)

29 permanent plain-assert tests, no `pytest` (it is not installed and was not installed).
Run with `./.venv/bin/python tests/test_lip_features.py`; exits 0 on success, 1 on failure.

Coverage includes: real 1142-row CSV, `(60, 21)` shape, `float32`, finiteness, aspect
invariance across 4:3 / 16:9 / HD, duplicate timestamps, unsorted input, ragged rows,
invalid inputs, legacy regression, and a test that asserts the two `mouth_opening`
definitions are **different** so a future "fix" to match the analysis CSV fails loudly.

### `src/model/prepare_dataset.py` — **modified** (244 → 400 → 448 lines)

Was a self-contained script that duplicated the feature maths and ran at import time.
Now it is a thin, importable layer over `lip_features.py`.

- Importing it has **no side effects** (no `mkdir`, no scanning) — verified.
- Entry point guarded: `if __name__ == "__main__": main()`.
- `PreparedRecording` NamedTuple; `collect()`, `save()`, `report()`, `main()` split apart.
- **Validity mask** covers finite coordinates, finite `frame_w`/`frame_h`, and
  `frame_w > 0`, `frame_h > 0`. The *same* mask is applied to the coordinates **and** the
  frame-size arrays; all dropped frames are counted.
- `MIN_FRAMES` is checked after cleaning, with the drop count in the error message.
- `EXCLUDED_FILES` (silent, hard-coded set) replaced by `KNOWN_BAD_RECORDINGS`, a
  documented `{(word, filename): reason}` map. Matches are printed and written to
  `skipped_recordings.csv` with `status="excluded"` — never silent.
- `collect()` catches only `EXPECTED_RECORDING_ERRORS` (`ValueError`,
  `FileNotFoundError`, `UnicodeDecodeError`, pandas parse errors). Any other exception
  is treated as a programming bug and propagates.
- `save()` refuses to write anything unless: the three collections are aligned, every
  sequence is `(60, 21)`, every label is in `{0, 1, 2}`, `X` is `(N, 60, 21)` `float32`
  and `y` is `(N,)` `int64`. All checks run **before** `mkdir`, so a rejected run leaves
  no partial files.
- `save()` also refuses any dataset containing a legacy recording (see decisions).
- **Phase 2** added session-id propagation (`PreparedRecording.session_id` and an eighth
  `metadata.csv` column). Details in the Phase 2 section below.

### `src/model/train_model.py` — **modified** (148 → 480 lines)

Was a script that ran at import time, hard-coded five word names plus
`labels=[0, 1, 2, 3, 4]`, used one 80/20 split, created directories on import, and saved
the model **before** looking at any metric. Now:

- Importing it has **no side effects**; no training happens on import (verified: 0 bytes
  of output, 0 files created).
- Vocabulary, labels, dimensions and paths all come from `config/config.py`
  (`ACTIVE_WORDS`, `LABEL_WORDS`, `NUM_CLASSES`, `FLAT_FEATURE_COUNT`, `X_PATH`, `Y_PATH`,
  `METADATA_PATH`, `MODEL_PATH`). No word name, label range or shape is hard-coded.
- **No feature extraction.** It consumes `X.npy` produced by `prepare_dataset.py` and
  asserts `flat width == cfg.FLAT_FEATURE_COUNT`, so a silent feature-shape change cannot
  reach the classifier.
- Dataset validation: 3-D `(N, 60, 21)`, `len(X) == len(y)`, finite values, every label in
  `{0, 1, 2}`, all three classes present, at least 2 samples per class, and metadata row
  count matching the sample count.
- **Out-of-fold evaluation.** `StandardScaler` sits inside each fold's `Pipeline`, so it is
  never fitted on the data a prediction is later compared against. Accuracy, per-class
  precision/recall/F1, macro and weighted averages, and the confusion matrix all come
  from out-of-fold predictions.
- **Session-grouped CV when the metadata allows it.** `StratifiedGroupKFold` is used when
  `metadata.csv` has a `session_id` / `session` / `group` column; otherwise
  `StratifiedKFold` is used, a multi-line warning is printed, and `"grouped": false` with
  the reason is recorded in `model_evaluation.json`. Every fold is checked to contain all
  three classes.
- Model selection compares three scikit-learn SVC configurations (`rbf` C=1, `rbf` C=10,
  `linear` C=1) by macro-F1 on the same folds; the winner is refit on all samples.
- Artefacts are written **only after** cross-validation completes and the confusion matrix
  is verified to cover every sample: `models/silent_speech_svm.pkl`,
  `outputs/classification_report.csv`, `outputs/confusion_matrix.png`, and
  `outputs/model_evaluation.json` (includes a model card with shape, classes and sklearn
  version).
- Exit codes: `0` trained, `2` no dataset, `3` invalid dataset. Both failure paths state
  that nothing was trained and nothing was written.
- `matplotlib` uses the `Agg` backend; no window opens and `plt.show()` is gone.

### Decisions recorded in Phase 1

| # | Decision | Rationale |
| --- | --- | --- |
| 1 | **Three initial classes: HELLO, YES, NO** (`HELLO=0`, `YES=1`, `NO=2`) | The vocabulary the project actually aims to demonstrate. `MOHIT`, `DILJAAN`, `HII`, `SHIVAM` are listed as *planned* only; no code path pretends they are supported. |
| 2 | **Isotropic coordinate mode is compulsory** | MediaPipe normalises `x` by frame width and `y` by frame height. Multiplying back by the raw pixel dimensions rescales the two axes differently, so the same mouth shape yields different features at different aspect ratios. Both axes are now scaled by one dimension. |
| 3 | **Legacy recordings are rejected, not converted** | A dataset that is entirely legacy is anisotropic; a dataset mixing legacy and isotropic is not numerically comparable. Both are refused with an error naming the files to re-record. The previous behaviour produced a 0.12 same-word correlation on the surviving clip. |
| 4 | **One shared feature extraction engine** | Training and live prediction previously had separate copies of the maths. Any drift is invisible at training time and fatal at inference time. `src/utils/lip_features.py` is now the only implementation. |
| 5 | **No TensorFlow and no PyTorch** | Intel Mac, 8 GB RAM, CPU only. Planned classifier is scikit-learn (`1.7.2` is already installed). No deep model, no GPU dependency. |
| 6 | **Streamlit runs as a separate process** from the OpenCV camera window | Avoids two libraries competing for the camera and the GUI thread. |
| 7 | **`__main__` guards only on runtime scripts** | Import-time camera/loop execution was a latent hazard; analysis scripts are left visually unchanged where renaming was unnecessary. |
| 8 | **No Git baseline commit yet** | Deferred deliberately (placeholder identity, no commit without approval). |
| 9 | **Evaluate out-of-fold, never on a single split** | One 80/20 split on ~90 samples is unstable, and scoring a model that was fitted on all data would be meaningless. |
| 10 | **Session-grouped validation whenever the metadata allows it** | Recordings from one session share lighting, pose and camera distance, so a random split leaks session identity. Falls back to stratified CV with an explicit warning while `session_id` is not propagated. |
| 11 | **Write artefacts only after validation succeeds** | The previous script dumped the model before computing a single metric. A broken dataset now leaves no model file behind. |

### Verification results (actual output)

Feature-engine suite, run 2026-11-12:

```
$ ./.venv/bin/python tests/test_lip_features.py
...
29/29 passed          exit 0
```

`src/model/prepare_dataset.py` verification, all executed on synthetic data in the system
temp directory. `dataset/`, `outputs/` and `models/` were hashed before and after.

| Check | Result |
| --- | --- |
| **V1** existing feature tests | **29/29 passed**, exit 0 — engine unchanged |
| **V2** `import src.model.prepare_dataset` | exit 0, 0 bytes stdout, 0 bytes stderr, 0 files created |
| **V3** `collect()` on 6 isotropic recordings | 6 sequences of `(60, 21)`, labels `[0, 0, 1, 1, 2, 2]`, all `isotropic` |
| **V4** flattened width | `(6, 1260)` |
| **V5** legacy-only dataset | **rejected**: `0 isotropic, 3 legacy`, all three `WORD/sample_101.csv` named, `X.npy` not created |
| **V6** mixed dataset (3 isotropic + 1 legacy) | **rejected**: `3 isotropic, 1 legacy`, `YES/sample_101.csv` named, nothing written |
| **V7** validity mask covers coordinates **and** frame sizes | 4 rows dropped (NaN coordinate, NaN `frame_w`, `frame_w = 0`, `frame_h = inf`), 37 kept, sequence `(60, 21)` |
| **V8** fewer than `MIN_FRAMES` usable frames | rejected: `not enough usable frames: 24 < 25 (16 row(s) dropped)` |
| **V9** missing column / 8 frames / empty file / binary file | 4 × `status="failed"` with reasons; the loop continued |
| **V10** unexpected exception (injected `RuntimeError`) | **propagated** out of `collect()` |
| **V11** known-bad files | 1 recording accepted, 2 `status="excluded"` with the reason `clip contains several repetitions of the word` |
| **V12a–c** unaligned lists / `(60, 20)` sequence / label `5` | 3 × `RuntimeError`, `X.npy` never created |
| **V12d** valid save | `X` `(6, 60, 21)` `float32`, `y` `(6,)` `int64`, metadata CSV with **8 columns** (the 7 designed here plus `session_id`, added in Phase 2) |
| **V13** numeric equivalence with the original algorithm on the real 1142-row CSV | max absolute difference **1.486e-08** (float32 rounding only) |
| **V14** data integrity | **22/22** files in `dataset/`, `outputs/`, `models/` byte-identical; `dataset/` + `models/` still contain 0 files; git 0 commits / 0 staged / 54 untracked |

Frame-size alignment was also demonstrated directly: the pre-fix call
`build_sequence(rows[mask], frame_w=unmasked)` raises
`ValueError: frame_w has 41 entries but there are 40 frames`, while the masked call
returns `(60, 21)`.

`src/model/train_model.py` verification (2026-11-13), again on synthetic data in the
system temp directory, with `dataset/`, `outputs/` and `models/` hashed before and after:

| Check | Result |
| --- | --- |
| **V1** existing feature tests | **29/29 passed**, exit 0 |
| **V2** `import src.model.train_model` | exit 0, 0 bytes stdout, 0 bytes stderr, 0 files created |
| **V3** real run with no dataset | exit **2**, prints `No prepared dataset found … Nothing was trained and no files were written`, nothing created |
| **V4** synthetic 3-class dataset | exit **0**, four artefacts written only inside the temp scratch directory |
| **V4b** saved artefact reloaded | `Pipeline`, `predict` → `(66,)`, `predict_proba` → `(66, 3)` |
| **V5** out-of-fold integrity | `[[22,0,0],[0,22,0],[0,0,22]]`, sum `66 == N`, `"predictions": "out-of-fold"`, `flat_feature_count == 1260` |
| **V6** scaler discipline | `StandardScaler` inside `Pipeline`, `NotFittedError` before `fit`, `probability=True` |
| **V7** with a `session_id` column | strategy `stratified_group`, 3 folds, held-out sessions `[s1] [s2] [s3]`, **no** session shared between train and test |
| **V8** without a session column | strategy `stratified`, `"grouped": false`, reason `metadata.csv has no session column`, warning printed |
| **V9** invalid datasets: missing class, label `7`, NaN, `(60, 20)` frames, metadata row mismatch | all exit **3**, no model and no outputs written |
| **V10** integrity | **22/22** data files byte-identical; `models/` 0, `outputs/` 22, `dataset/processed/` 0; git 0 commits / 0 staged / 55 untracked |

That run reported `accuracy: 100.00%` and `macro F1: 1.0000`. The data was three
well-separated Gaussian clusters (22 samples per class, class offset `+1.2` against noise
`σ = 0.25`), so perfect separation is expected. **It validates the pipeline and is not a
project accuracy figure.** Every artefact from that run stayed in the temp directory.

### Recorded limitations and known issues

- **Two conflicting `mouth_opening` definitions exist.** The training path computes
  `‖upper_lip − lower_lip‖ / mouth_width`; `outputs/normalized_lip_landmarks.csv` divides
  the tracker's `×frame_height` column by the width. The training definition is
  authoritative; the difference is documented in two docstrings and locked by a test.
  `prepare_dataset.py` reproduces the original training maths exactly (V13).
- **No real dataset exists.** `dataset/raw/` and `dataset/processed/` are both empty.
  `prepare_dataset.py` exits with `No valid recordings found` and `train_model.py` exits
  with code `2` and `No prepared dataset found`. Both refuse to invent data.
- **No real model has been trained.** `models/` contains **0 files**. The only model ever
  produced was fitted to synthetic noise inside a temp directory during verification and
  was deliberately left there.
- **The synthetic 100% accuracy is a pipeline test, not project accuracy.** It must never
  be quoted as the system's performance. Only a run on real recordings —
  `prepare_dataset.py` then `train_model.py` — can produce a number worth reporting, and
  that number does not exist yet.
- **Grouped validation only works when every recording carries a session id.** As of
  Phase 2 `collect_dataset.py` writes one and `prepare_dataset.py` propagates it into
  `metadata.csv`, but **no real recording exists yet**, so grouped cross-validation has
  only been exercised on synthetic files. If any recording lacks a `session_id`,
  `train_model.py` falls back to `StratifiedKFold`, prints a warning and records
  `"grouped": false` — a decision for the whole dataset, never a silent partial grouping.
- **`src/model/live_prediction.py` still uses the old five-word vocabulary.** It is not on
  the current training path and is not yet aligned with `config.ACTIVE_WORDS`.
- **The Phase 2 recorder has never been run against a camera.** The section below states
  exactly what was verified on synthetic files and what remains untested.

---

## Phase 2 — Session-aware dataset collection

Implemented on 2026-11-13. **The recorder has not been run against a camera yet.** Every
result below comes from synthetic CSVs written to a temp directory: no real recording
exists, and no model has been trained on real data.

Objective: record HELLO/YES/NO clips that this pipeline can actually consume — isotropic
coordinates, frame size, and a session id without which session-grouped cross-validation
is impossible.

### `src/lip_tracking/collect_dataset.py` — **rewritten** (183 → 479 lines)

The old recorder was unusable by design, not merely untidy:

| Old behaviour | Consequence |
| --- | --- |
| `cv2.VideoCapture(0)` and the capture loop at module level | importing the module opened the webcam |
| `WORDS` list of five names incl. `MOHIT_SINGH`, `MRITYUNJAY` | disagreed with `config.ACTIVE_WORDS` |
| own `LANDMARKS` dict, `refine_landmarks=True` | duplicated `config`, wasted CPU |
| wrote `point.x * width`, `point.y * height` | **pixel coordinates → anisotropic**; `prepare_dataset.py` now rejects exactly these files as legacy |
| `sample_number = len(existing) + 1` then `open(..., "w")` | **silently overwrote data**: with `001` and `003` present it wrote `003` |
| manual SPACE start / SPACE stop, no session, no checks | no fixed window, no provenance, no quality gate |

Now:

- **Recording state machine** `IDLE → COUNTDOWN → RECORDING → REVIEW → IDLE`. Keys: `SPACE`
  record, `N`/`P` change word, `Q` quit. The overlay shows word, session, `n/30`, status,
  the countdown, a progress bar during the window, and the 10 landmark dots.
- **2.5-second automatic recording window.** `countdown_s = 1.0` and `duration_s = 2.5`
  come from `config.RECORDING`; nothing is hard-coded. `--duration` can override the window
  for experiments and `--per-word` the target count.
- **Session id generation and `--session`.** By default one id per run in the form
  `sYYYYMMDD-HHMMSS`, written into every row. `--session s2` supplies it explicitly, e.g. to
  resume a session. Both are validated against `^[A-Za-z0-9_-]{1,40}$`; an unusable value
  exits with argparse code `2` **before** the camera is opened.
- **Three-class vocabulary from config.** `words = list(config.ACTIVE_WORDS)` — the recorder
  contains no word list of its own, so HELLO/YES/NO is the only possibility.
- **Isotropic coordinates and frame metadata.** Each row is built by
  `lip_features.landmarks_to_row` (MediaPipe-normalised 0–1, config landmark order), then
  `frame_w`, `frame_h` and `session_id` are appended. The row keys are asserted to equal
  `config.CSV_COLUMNS` before writing, so a layout drift fails at record time instead of
  producing files the pipeline cannot read. Pixel values are never written.
- **No-overwrite protection.** The next index is `max(existing) + 1`, so gaps are skipped
  rather than recycled, and the file is created with `open(path, "x")`, retrying on
  `FileExistsError`. Two writers cannot claim the same name; no file is ever reopened `"w"`.
- **Recording quality checks and rejection behaviour.** Before a clip is written: at least
  `MIN_FRAMES` frames, a face in at least 80 % of the window frames, and a non-zero mouth
  width. A failing clip is printed with each reason and **not saved**. Lips that barely move
  produce a warning but are still kept, because only the speaker can judge that.
  `FACE_DETECTION_MIN` and `MOVEMENT_WARNING` are documented recorder-local heuristics, not
  validated thresholds.
- **Progress reporting.** Per-word `n/30` on screen, measured quality lines per clip, and a
  session summary on exit (session id, clips written, per-word totals, next command).
- Import-safe: `cv2` and `mediapipe` are imported **inside** `main()`, so importing the
  module for tests cannot touch the camera.

### `src/model/prepare_dataset.py` — **session propagation** (400 → 448 lines)

- `PreparedRecording` gains `session_id`; `metadata.csv` gains an eighth column.
- `_session_id()` reads the single non-empty `session_id` from the CSV. A file that mixes two
  sessions is **rejected** (`ValueError`) instead of being grouped arbitrarily; a file with
  no such column yields `""`.
- The value is read **before** the validity mask, so a mixed file is refused even when some
  of its rows would have been dropped.
- `report()` prints `Sessions : s1=45, s2=45` and warns when any recording carries no session
  id — precisely the condition under which grouped validation silently stops working.

**Metadata schema: 8 columns** — `filename`, `word`, `label`, `frames`, `duration`,
`dropped_rows`, `coordinate_mode`, `session_id`. The first seven are unchanged; `session_id`
was added as the last column, so existing readers of the leading columns still work.

### `tests/test_collect_dataset.py` — **new** (271 lines, 14 tests)

Plain-assert, no `pytest`, no camera, all files under a `tempfile.mkdtemp()` tree.

| Test | Covers |
| --- | --- |
| T01 | session id format is sortable and deterministic |
| T02 | empty folder → `sample_001.csv` |
| T03 | `001` + `003` present → next is `004`; the gap is not recycled |
| T04 | `create_sample_file` never overwrites; first file byte-identical after the second |
| T05 | row keys equal `config.CSV_COLUMNS`; frame size and session present |
| T06 | coordinates stay in `0..1` — never pixels |
| T07 | header and row count written; a second write gets a different name |
| T08 | healthy 40-frame clip passes the gate |
| T09 | 10 frames rejected, message names `MIN_FRAMES` |
| T10 | face found in 30 of 100 frames rejected |
| T11 | zero mouth width rejected |
| T12 | recorder CSV → `prepare_recording` yields `isotropic`, `(60, 21)`, session preserved |
| T13 | 12 clips / 2 sessions → 8-column `metadata.csv`, 10 rows after the 2 policy exclusions, `resolve_groups` → `stratified_group` |
| T14 | file without `session_id` → `""`, and `resolve_groups` falls back to `stratified` |

### Decisions recorded in Phase 2

| # | Decision | Rationale |
| --- | --- | --- |
| 12 | **One session id per recording file** | Grouping is only meaningful if a clip belongs to exactly one session. Writing it per row also keeps the CSV self-describing when files are moved. |
| 13 | **A file mixing sessions is rejected, not split** | Splitting would require guessing which frames belong where; refusing keeps grouped validation sound. |
| 14 | **Never overwrite; skip gaps instead of reusing them** | The old `len(existing) + 1` scheme destroyed an earlier recording. `open(..., "x")` makes reuse impossible even across processes. |
| 15 | **Reject obviously broken clips at capture time** | A clip with 8 frames or no face cannot be repaired later; telling the speaker immediately is cheaper than discovering it after 90 recordings. |

### Verification results (actual output)

Recorder and session test suite, run 2026-11-13:

```
$ ./.venv/bin/python tests/test_collect_dataset.py
...
14/14 passed          exit 0
```

No camera was used. `dataset/`, `outputs/` and `models/` were hashed before and after.

| Check | Result |
| --- | --- |
| **V1** existing feature tests | **29/29 passed**, exit 0 — engine untouched |
| **V2** new recorder/session tests | **14/14 passed** |
| **V3** `import src.lip_tracking.collect_dataset` | `'cv2' in sys.modules` → **False**, mediapipe → **False**; importing cannot open a camera |
| **V4** `collect_dataset.py --help` | usage printed, exit **0**, camera never opened |
| **V5** `--session "bad/id"` | `error: --session may only contain letters, digits, '-' and '_' (max 40)`, exit **2**, nothing created |
| **V6** recorder CSV → `prepare_recording` | `isotropic`, `(60, 21)`, `session_id` preserved, `dropped_rows = 0` |
| **V7** `collect()` + `save()` on a 2-session tree | `metadata.csv` header ends `…,coordinate_mode,session_id`; report `Sessions : s1=7, s2=9` (16 recordings) |
| **V8** `StratifiedGroupKFold` with two synthetic sessions | `{"strategy": "stratified_group", "reason": "grouped by 'session_id' (2 groups)", "folds": 2, "grouped": true}` — no session shared between train and test |
| **V9** recording without a session column | `session_id = ""` → strategy `stratified`, reason `column 'session_id' contains empty values` |
| **V10** overwrite protection | `001` + `003` present → next file `004`; two consecutive writes → two files, the first byte-identical afterwards |
| **V11** data integrity | **22/22** files in `dataset/`, `outputs/`, `models/` byte-identical; `dataset/raw` 0, `dataset/processed` 0, `models` 0 |
| **V12** git state | 0 commits / 0 staged / **56** untracked (was 55; +1 is the new test file) |

Regression check: the earlier `prepare_dataset.py` suite was re-run and gave **12/12** after
two of its assertions were updated from 7 to 8 metadata columns. No production behaviour
changed — only the added column.

That demo run also printed `accuracy 0.38` on its synthetic dataset. The features were
random noise, so the figure is meaningless and is **not** a performance result; it is
recorded here only to show that no number is being selected or hidden.

### What is not verified

- **Webcam and live recording have NOT been tested.** The capture loop, overlay, countdown
  timing, `cv2.waitKey` handling and real MediaPipe behaviour on this machine are all
  unproven. Only the pure functions (row building, naming, quality gate, writing) were run.
- **Synthetic recording results are not real-world performance.** They prove the data path
  works, nothing about recognition.
- **No actual recordings and no trained model exist.** `dataset/raw/` is still empty and
  `models/` still contains 0 files.
- **Real camera acceptance testing is pending.** Manual checklist: SPACE → countdown →
  ~2.5 s window → result line; a clip with the face turned away is discarded, not saved;
  the CSV contains `frame_w`, `frame_h`, `session_id` and coordinates in `0..1`; a second
  run cannot overwrite the first; `prepare_dataset.py` reports the session and
  `train_model.py` then reports `stratified_group` instead of the leakage warning.

---

## Pending work

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Safety baseline, backup, recovery investigation | **Done** |
| 1 | Config, shared feature engine, tests, dataset-preparation and training rewrite | **In progress** — `live_prediction.py` still to align |
| 2 | Dataset collection (countdown + 2.5 s automatic recording, HELLO/YES/NO) | **Recorder implemented** — camera acceptance testing and real recordings pending |
| 3 | Signal processing / feature engineering on the recorded data | Pending |
| 4 | Model training and evaluation (scikit-learn) | Pending |
| 5 | Live recognition, including **rejection of silence and unknown words** | Pending |
| 6 | Streamlit dashboard (separate process) | Pending |
| 7 | Testing and optimisation for the 8 GB Intel hardware | Pending |
| 8 | Documentation | Pending |

---

## Environment (verified 2026-11-12)

| Component | Version |
| --- | --- |
| Python | 3.10.0 (`./.venv/bin/python`) |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 (used by `train_model.py`; no project model exists yet) |
| OpenCV | 4.11.0 |
| MediaPipe | 0.10.21 |
| OS | macOS 15.8.1, Intel, 8 GB RAM |

Notes: there is no `python` on `PATH`; always use `./.venv/bin/python`. `pytest` is not
installed and the test file is a standalone script. `src/` uses implicit namespace
packages — there are no `__init__.py` files, and modules bootstrap `sys.path` when run
directly.
