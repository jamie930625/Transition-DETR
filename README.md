# Transition-DETR (early prototype)

Cue-point detection for DJ transitions with a DETR-style detector on mel-spectrograms.

Given a clip of the outgoing track (Track A) and a clip of the incoming track (Track B), the model predicts where Track A should be mixed out (cue-out) and where Track B should be mixed in (cue-in).

> **Status:** this repository contains the inference code from an early stage of the project (NTU Deep Learning for Music Analysis and Generation, final project). The current version of Transition-DETR, which detects variable-length transition regions on full-track spectrograms, is being prepared for publication and is not included here.

## How it works

- Each track is rendered as a mel-spectrogram image (sr = 22050 Hz, hop length = 512, so one pixel ≈ 23 ms).
- The training data are pairs of clips (outgoing and incoming track) cut around real DJ transitions, with cue positions taken from track-to-mix alignments of real DJ mixes.
- A DETR detector, fine-tuned from [CUE-DETR](https://github.com/ETH-DISCO/cue-detr), predicts cue points as narrow boxes with two classes: cue-in and cue-out.
- At inference, a 355-pixel window (about 8 s) slides over each clip, and the most confident cue-out on Track A and cue-in on Track B are reported.

## Repository structure

```text
predict_transition.py      # demo: predict cue-out / cue-in on a pair of clips
examples/                  # one example pair of spectrogram clips
model/
  cue_detr_model.py        # LightningModule wrapping DetrForObjectDetection
  cue_detr_data.py         # datasets and data module
  cue_detr_train.py        # training script
  cue_detr_pred.py         # batch prediction helper
  cue_detr_utils.py        # slicing, box conversion, plotting
  LICENSE-CUE-DETR         # license of the upstream CUE-DETR code
```

## Setup

```bash
pip install -r requirements.txt
```

Download the checkpoint `last.ckpt` from [Google Drive](https://drive.google.com/file/d/17DKCkQ9pK9i2O29O4VpIQgwyuz4R0dkA/view?usp=share_link) and place it at:

```text
checkpoints/exp_afternoon_demo_0/last.ckpt
```

## Run the demo

```bash
python predict_transition.py
```

Use your own clips:

```bash
python predict_transition.py --prev path/to/trackA.png --next path/to/trackB.png --ckpt path/to/last.ckpt
```

The script prints the predicted cue-out (Track A) and cue-in (Track B) in seconds from the start of each clip, with the detector's confidence.

## Acknowledgements

The model and training code build on [CUE-DETR](https://github.com/ETH-DISCO/cue-detr) (MIT License, Copyright (c) 2024 ETH DISCO), adapted here for two-class cue detection on track pairs.
