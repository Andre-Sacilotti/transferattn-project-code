"""
Extract frame-level features using STAM (16-frame clips, 7+8 temporal padding).
Requires the 'stam' package (create_model).
"""
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from argparse import Namespace
from torch.autograd import Variable

from common import get_parser, build_paths, run_main

# -----------------------------------------------------------------------------
# STAM: 16-frame clips, 7 frames before + 8 after (zero-padded)
# -----------------------------------------------------------------------------


def _stam_clip_batch(batch_tensor):
    """Build 16-frame clips (7 before, 8 after). Return (N, 16, 3, H, W) for STAM."""
    n, c, h, w = batch_tensor.shape
    device = batch_tensor.device
    x1 = torch.zeros(7, c, h, w, device=device, dtype=batch_tensor.dtype)
    x2 = torch.zeros(8, c, h, w, device=device, dtype=batch_tensor.dtype)
    padded = torch.cat([x1, batch_tensor, x2], dim=0)
    clips = []
    for b in range(7, batch_tensor.size(0) + 7):
        clip = padded[b - 7 : b + 9]
        clip = clip.transpose(0, 1)
        clips.append(clip)
    return torch.stack(clips)


def main():
    parser = get_parser(description="Dataset preparation – STAM feature extraction")
    args = parser.parse_args()
    if not args.pretrain_weight:
        raise SystemExit("--pretrain_weight is required for STAM")

    path_input, path_output = build_paths(args)
    os.makedirs(path_output, exist_ok=True)

    from stam.models import create_model

    stam_args = Namespace(
        val_dir=None,
        model_path=args.pretrain_weight,
        model_name="stam_16",
        num_classes=0,
        input_size=224,
        val_zoom_factor=0.875,
        batch_size=128,
        num_workers=8,
        frames_per_clip=16,
        frame_rate=1.6,
        step_between_clips=1000,
    )
    model = create_model(stam_args).cuda()
    state = torch.load(args.pretrain_weight, map_location="cpu").get("model", {})
    if state:
        model.load_state_dict(state, strict=False)
    model.eval()
    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 16

    def extract_batch(frames_tensor):
        with torch.no_grad():
            batch_tensor = torch.stack(frames_tensor)
            clips = _stam_clip_batch(batch_tensor)
            clips = Variable(clips)
            parts = []
            for chunk in clips.split(batch_size):
                out = chunk.cuda()
                feats = model(out)
                parts.append(feats)
            features = torch.cat(parts, dim=0)
        return features.view(features.size(0), -1).cpu()

    run_main(args, transform, extract_batch, max_frames=None)


if __name__ == "__main__":
    main()
