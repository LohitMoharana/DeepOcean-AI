# DeepOcean-AI 🌊

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C?style=flat)

>**Underwater object detection and tracking pipeline — YOLOv8-based mine/UUV/diver classifier with a two-stage BoT-SORT tracking layer and a safety-interlock design for human divers.**

Built as an iterative, evidence-driven engineering project: every design decision below was arrived at by testing against real underwater footage, catching and fixing real failure modes (camouflage blindness, class confusion, dataset leakage), and re-validating rather than trusting a single training run.

---

## 🎯 What it detects

4-class taxonomy, chosen deliberately rather than inherited from a generic dataset:

| Class | ID | Notes |
|---|---|---|
| `sea_mine` | 0 | Static benthic targets, including heavily camouflaged/encrusted mines |
| `uuv` | 1 | Unmanned underwater vehicles / ROVs |
| `diver` | 2 | Human divers — treated as a safety-interlock class, not just another detection target |
| `misc` | 3 | Catch-all background class (torpedoes, distractors, clutter) |

**Why `misc` absorbs torpedoes and explosives:** fast-moving ordnance isn't realistically caught by a frame-based optical pipeline before impact — by the time a camera registers it, the event is over. Rather than waste model capacity on a target class the system can't meaningfully act on, those objects are folded into `misc` so the network isn't penalized for failing to distinguish them from background.

---

## 🧠 Model Architecture & Training

- **Base:** YOLOv8s, fine-tuned from a manually-curated dataset checkpoint (not a fresh COCO backbone — see *Design Decisions* below).
- **Synthetic data generation:** a custom physics-aware compositor (`dataset_R26_clean` generator) that pastes asset PNGs onto real underwater backgrounds using a Beer-Lambert light-attenuation model — the pasted asset's color is blended toward the ambient light of the specific background patch it's placed on, rather than just being color-shifted generically. This was the single biggest driver of the model's ability to detect low-contrast, camouflage-blended mines.
- **Active learning:** the model was iteratively hardened against real, observed failure modes — not just synthetic augmentation. Confirmed false-positive triggers (scuba tanks, dive flashlights, black ROVs misclassified as divers) were extracted directly from test footage, labeled, and injected back into training as targeted corrections.
- **Backbone freezing:** fine-tuning runs use `freeze=10` to lock the pretrained backbone's low-level visual features (edges, textures, underwater color response) while letting the neck/head adapt to the 4-class taxonomy — this consistently outperformed retraining from a fresh backbone, which required far more data to relearn basic underwater optical physics.

---

## 📊 Performance — Held-Out Validation

**These are the true held-out validation numbers**, confirmed after catching and fixing a train/val split leak in an earlier evaluation run (an intermediate script pointed the val split at cached training images, which inflated all metrics — e.g. reported mAP50 of 0.89 vs. the real 0.668). Only numbers verified against the genuine, unseen 934-image validation split are quoted here.

| Metric | Value |
|---|---|
| mAP50 (all classes) | **0.668** |
| mAP50-95 (all classes) | **0.423** |
| Sea Mine mAP50 | 0.819 |
| UUV mAP50 | 0.798 |
| Diver Recall | 0.334 |

### Known limitation: diver detection

Diver recall (33.4%) is meaningfully lower than sea_mine/UUV performance, and the model shows a real bias toward false-triggering on ambient water turbulence, bubbles, and light refraction as `diver` in empty-water frames. This is a genuine, acknowledged limitation, not glossed over:

- Human silhouettes in turbid water lack the hard geometric edges that make rigid, man-made objects (mines, UUVs) easier to separate from background.
- The design response to this limitation was deliberate, not accidental: the inference pipeline runs the diver class threshold **lower** than other classes specifically because a missed diver is a worse failure mode than a false alarm — see *Safety Interlock* below.

---

## 🎥 Tracking & Inference — Two-Stage Threshold Design

Raw per-frame YOLO detections are noisy — a single low-confidence flicker on a scuba tank or a light-refraction artifact can otherwise trigger a spurious detection. Rather than accept that noise or bluntly raise the global confidence threshold (which would also suppress real detections of camouflaged mines), the tracker uses a **two-stage confidence system** on top of BoT-SORT:

- **Stage 1 — Initialization threshold:** a track is only created if a detection clears a *high* confidence bar (e.g. 65-70% for sea_mine/UUV/diver). This filters out one-off noise before it ever gets an ID.
- **Stage 2 — Hold threshold:** once a track is established, it can persist through *lower* confidence frames (e.g. down to 20-30%) — this lets the system keep tracking a mine through camouflage-heavy frames or a diver through partial occlusion, without needing every single frame to independently clear a high bar.

This is combined with:
- **Persistence/streak filtering** — a detection needs to survive a minimum number of consecutive frames before it renders, further suppressing single-frame ghosts.
- **Species locking** — once a track is confidently identified as a machine (mine/UUV/misc) or a human (diver), the system resists flipping that classification on a brief contradictory detection, correcting for the class-bleed that occurs when e.g. a black ROV is momentarily misclassified as a diver.

### Safety Interlock

The diver class is treated differently throughout the pipeline, by design:
- Lower initialization threshold than machine classes (missing a diver is a worse outcome than a false alarm).
- Shorter persistence requirement (2 frames vs. 3) so a genuine diver detection renders faster.
- Detections render with an explicit `⚠️ SAFETY LOCK` label rather than a generic ID tag.

This asymmetry is intentional: the system is tuned to over-flag potential divers rather than risk under-flagging a real one, even at the cost of the false-positive rate documented above.

---

## 🎬 Sample Outputs

Annotated tracking output on real underwater test footage.

<p align="center">
  <video src="docs/gifs/test_video_1_R26_verified.mp4" autoplay loop muted playsinline></video>
  <video src="docs/gifs/test_video_8_R26_verified.mp4" autoplay loop muted playsinline></video>
</p>
<p align="center">
  <b>Left:</b> Sea Mine Detection (camouflaged) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Right:</b> Diver Safety-Lock Trigger
</p>

<br>

<p align="center">
  <video src="docs/gifs/test_video_2_R26_verified.mp4" autoplay loop muted playsinline></video>
  <video src="docs/gifs/test_video_3_R26_verified.mp4" autoplay loop muted playsinline></video>
</p>
<p align="center">
  <b>Left:</b> UUV Tracking Through Occlusion &nbsp;&nbsp;|&nbsp;&nbsp; <b>Right:</b> False-Positive Suppression
</p>


---

## 🔭 Limitations & Next Steps

This project is a portfolio/research piece, not a production system — the limitations below are stated plainly rather than glossed over, and reflect an active, ongoing engineering process:

- **Diver recall (33.4%) is the primary open problem.** Human silhouettes in turbid water lack the rigid geometric edges that make mines/UUVs easier to separate from background — closing this gap needs more diverse real-world diver footage across poses, occlusion levels, and lighting, not just more synthetic augmentation.
- **Background false-positive rate on the diver class** needs targeted hard-negative mining specifically around water turbulence, bubbles, and light refraction — the exact failure mode the confusion matrix surfaced.
- **Domain scope is intentionally narrow.** The model is tuned for underwater footage specifically; it has no negative examples from other domains (dry land, air) and isn't expected to generalize there — this is a deliberate scoping choice, not an oversight.
- **Next phase — neuromorphic conversion:** this vision pipeline is the front-end for [MarineSpike](https://github.com/LohitMoharana/MarineSpike), a related project converting the trained detector into a Spiking Neural Network (SNN) for low-power edge deployment. Early attempts at direct ANN→SNN conversion hit a documented failure mode (spike degradation in dense detection heads — see [SpikeYOLO, ECCV 2024](https://github.com/BICLab/SpikeYOLO) for the architectural reasons why), and the current approach is shifting toward training a spike-native detection architecture rather than post-hoc conversion.

---

## 🛠️ Engineering Process Notes

This project went through several iterations that are worth being transparent about, since they shaped every design choice above:

- An earlier attempt to train from a fresh COCO backbone required ~10x more data/epochs to relearn basic underwater optical physics that the manually-curated checkpoint already had — abandoned in favor of anchored fine-tuning.
- An experiment forcing full "colorblind" augmentation (aggressive grayscale/hue randomization, on the hypothesis that the model over-relied on color) was tested and **empirically rejected** — real-world video performance got worse, not better, so the change was reverted. Colorblindness wasn't a free win; color is a legitimate signal in this domain.
- A train/val evaluation leak (validation accidentally pointed at cached training data) inflated reported metrics until it was caught by re-running validation against the genuine held-out split — see *Performance* section above for the corrected numbers.

---

## Usage

```bash
python inference/validate_using_botsort.py --source path/to/video.mp4 --weights weights/deepocean_final.pt
```

See `training/` for the dataset generation, active-learning patch, and fine-tuning scripts.
