"""
Demo: predict DJ cue points on a pair of spectrogram clips.

Given the clip of the outgoing track (Track A) and the clip of the incoming
track (Track B), the detector slides a 355-pixel window over each clip and
reports the most confident cue-out position on Track A and cue-in position
on Track B, in seconds from the start of each clip.

Usage:
    python predict_transition.py
    python predict_transition.py --prev examples/mix0000-00_prev.png \
                                 --next examples/mix0000-00_next.png \
                                 --ckpt checkpoints/exp_afternoon_demo_0/last.ckpt
"""
import argparse
import sys

import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor

sys.path.append('./model')
from cue_detr_model import CuePointDetr  # noqa: E402

# Spectrogram settings used to render the clips (sr=22050, hop_length=512).
SEC_PER_PIXEL = 512 / 22050
WINDOW_PX = 355
STRIDE_PX = WINDOW_PX // 2
LABELS = {0: 'cue-in', 1: 'cue-out'}


def load_model(ckpt_path, device):
    model = CuePointDetr.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    return model.to(device)


def window_starts(width):
    """Left edges of overlapping windows that cover the whole clip."""
    if width <= WINDOW_PX:
        return [0]
    starts = list(range(0, width - WINDOW_PX + 1, STRIDE_PX))
    if starts[-1] != width - WINDOW_PX:
        starts.append(width - WINDOW_PX)
    return starts


@torch.no_grad()
def detect(model, processor, image_path, device):
    """Returns {label: (seconds_in_clip, score)} with the best detection per class."""
    image = np.asarray(Image.open(image_path).convert('RGB'))
    width = image.shape[1]
    if width < WINDOW_PX:
        pad = np.zeros((image.shape[0], WINDOW_PX - width, 3), dtype=image.dtype)
        image = np.concatenate([image, pad], axis=1)

    best = {}
    for left in window_starts(image.shape[1]):
        window = image[:, left:left + WINDOW_PX]
        pixel_values = processor.preprocess(window, return_tensors='pt')['pixel_values'].to(device)
        outputs = model.model(pixel_values=pixel_values)

        probs = outputs.logits.softmax(-1)[0, :, :-1]  # drop the "no object" class
        boxes = outputs.pred_boxes[0]                   # (center_x, center_y, w, h), normalized
        scores, classes = probs.max(-1)
        for score, cls, box in zip(scores.tolist(), classes.tolist(), boxes.tolist()):
            center_px = left + box[0] * WINDOW_PX
            if center_px >= width:
                continue
            label = LABELS[cls]
            if label not in best or score > best[label][1]:
                best[label] = (center_px * SEC_PER_PIXEL, score)
    return best


def fmt(seconds):
    return f'{seconds:6.2f} s ({int(seconds // 60):02d}:{seconds % 60:05.2f})'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prev', default='examples/mix0000-00_prev.png', help='spectrogram clip of the outgoing track')
    parser.add_argument('--next', default='examples/mix0000-00_next.png', help='spectrogram clip of the incoming track')
    parser.add_argument('--ckpt', default='checkpoints/exp_afternoon_demo_0/last.ckpt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)
    processor = DetrImageProcessor(do_resize=False, do_pad=False)

    prev = detect(model, processor, args.prev, device)
    nxt = detect(model, processor, args.next, device)

    print('Predicted transition (seconds from the start of each clip)')
    if 'cue-out' in prev:
        t, s = prev['cue-out']
        print(f'  Track A cue-out: {fmt(t)}  score {s:.2f}  [{args.prev}]')
    if 'cue-in' in nxt:
        t, s = nxt['cue-in']
        print(f'  Track B cue-in:  {fmt(t)}  score {s:.2f}  [{args.next}]')


if __name__ == '__main__':
    main()
