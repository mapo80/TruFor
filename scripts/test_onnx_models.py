#!/usr/bin/env python3
import os
import argparse
import glob
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort

from TruFor_train_test.lib.models.DnCNN import make_net
from TruFor_train_test.lib.models.cmx.encoders.dual_segformer import mit_b2

NOISEPRINT_ONNX = os.path.join('onnx_models', 'noiseprint_pp.onnx')
MIT_B2_ONNX = os.path.join('onnx_models', 'mit_b2.onnx')
NOISEPRINT_WEIGHTS = os.path.join('TruFor_train_test', 'pretrained_models', 'noiseprint++', 'noiseprint++.th')
SEGFORMER_WEIGHTS = os.path.join('TruFor_train_test', 'pretrained_models', 'segformers', 'mit_b2.pth')
DATASET_DIR = os.path.join('sample_dataset')


def parse_args():
    parser = argparse.ArgumentParser(description='Compare ONNX and PyTorch models')
    parser.add_argument('--max-images', type=int, default=None,
                        help='limit number of images for quicker tests')
    return parser.parse_args()


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def build_noiseprint_model():
    num_levels = 17
    out_channel = 1
    model = make_net(
        3,
        kernels=[3] * num_levels,
        features=[64] * (num_levels - 1) + [out_channel],
        bns=[False] + [True] * (num_levels - 2) + [False],
        acts=['relu'] * (num_levels - 1) + ['linear'],
        dilats=[1] * num_levels,
        bn_momentum=0.1,
        padding=1,
    )
    state = torch.load(NOISEPRINT_WEIGHTS, map_location='cpu')['network']
    model.load_state_dict(state)
    model.eval()
    return model


def build_mit_b2_model():
    model = mit_b2()
    model.init_weights(pretrained=SEGFORMER_WEIGHTS)
    model.eval()
    return model


def main():
    args = parse_args()
    # Load models
    torch_np = build_noiseprint_model()
    ort_np = ort.InferenceSession(NOISEPRINT_ONNX)

    torch_seg = build_mit_b2_model()
    ort_seg = ort.InferenceSession(MIT_B2_ONNX)

    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, 'real', '*.png')) +
                       glob.glob(os.path.join(DATASET_DIR, 'fake', '*.png')))
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    np_diffs = []
    seg_diffs = []

    for p in img_paths:
        inp = load_image(p)
        with torch.no_grad():
            np_torch = torch_np(inp).numpy()
            np_rgb = np.repeat(np_torch, 3, axis=1)
            seg_torch = torch_seg(inp, torch.from_numpy(np_rgb))
            seg_torch = [s.detach().numpy() for s in seg_torch]

        np_onnx = ort_np.run(None, {'image': inp.numpy()})[0]
        np_rgb_onnx = np.repeat(np_torch, 3, axis=1)
        seg_onnx = ort_seg.run(None, {'rgb': inp.numpy(), 'np': np_rgb_onnx})

        np_diffs.append(np.abs(np_torch - np_onnx).mean())
        seg_img_diff = [np.abs(a - b).mean() for a, b in zip(seg_torch, seg_onnx)]
        seg_diffs.append(np.mean(seg_img_diff))

    print('image\tnoiseprint_mae\tmit_b2_mae')
    for path, np_err, seg_err in zip(img_paths, np_diffs, seg_diffs):
        print(f"{os.path.basename(path)}\t{np_err:.6e}\t{seg_err:.6e}")

    np_diffs = np.array(np_diffs)
    seg_diffs = np.array(seg_diffs)
    print()  # blank line before summary
    print(
        f"Noiseprint++ MAE: {np_diffs.mean():.6e} (std {np_diffs.std():.2e}, max {np_diffs.max():.2e})"
    )
    print(
        f"mit_b2 MAE: {seg_diffs.mean():.6e} (std {seg_diffs.std():.2e}, max {seg_diffs.max():.2e})"
    )


if __name__ == '__main__':
    main()
