#!/usr/bin/env python3
import os
import argparse
import torch
from TruFor_train_test.lib.models.DnCNN import make_net
from TruFor_train_test.lib.models.cmx.encoders.dual_segformer import mit_b2

NOISEPRINT_WEIGHTS = os.path.join('TruFor_train_test', 'pretrained_models', 'noiseprint++', 'noiseprint++.th')
SEGFORMER_WEIGHTS = os.path.join('TruFor_train_test', 'pretrained_models', 'segformers', 'mit_b2.pth')


def export_noiseprint(output_path: str) -> None:
    """Export Noiseprint++ model to ONNX."""
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

    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['image'],
        output_names=['noiseprint'],
        dynamic_axes={
            'image': {0: 'batch', 2: 'height', 3: 'width'},
            'noiseprint': {0: 'batch', 2: 'height', 3: 'width'},
        },
        opset_version=11,
    )
    print(f"Noiseprint++ exported to {output_path}")


def export_mit_b2(output_path: str) -> None:
    """Export SegFormer mit_b2 backbone to ONNX."""
    model = mit_b2()
    model.init_weights(pretrained=SEGFORMER_WEIGHTS)
    model.eval()

    rgb = torch.randn(1, 3, 256, 256)
    np_feat = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        (rgb, np_feat),
        output_path,
        input_names=['rgb', 'np'],
        output_names=[f'fused_{i}' for i in range(4)],
        dynamic_axes={
            'rgb': {0: 'batch', 2: 'height', 3: 'width'},
            'np': {0: 'batch', 2: 'height', 3: 'width'},
            'fused_0': {0: 'batch', 2: 'h1', 3: 'w1'},
            'fused_1': {0: 'batch', 2: 'h2', 3: 'w2'},
            'fused_2': {0: 'batch', 2: 'h3', 3: 'w3'},
            'fused_3': {0: 'batch', 2: 'h4', 3: 'w4'},
        },
        opset_version=11,
    )
    print(f"mit_b2 exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export TruFor models to ONNX")
    parser.add_argument('--output', default='onnx_models', help='Output directory')
    parser.add_argument('--models', nargs='+', default=['noiseprint', 'mit_b2'],
                        choices=['noiseprint', 'mit_b2'], help='Models to export')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if 'noiseprint' in args.models:
        export_noiseprint(os.path.join(args.output, 'noiseprint_pp.onnx'))
    if 'mit_b2' in args.models:
        export_mit_b2(os.path.join(args.output, 'mit_b2.onnx'))


if __name__ == '__main__':
    main()
