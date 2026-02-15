"""
Extract frame-level features using I3D (clip-based, STAM-style temporal padding).
"""
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

from common import get_parser, build_paths, run_main

# -----------------------------------------------------------------------------
# I3D clip setup
# -----------------------------------------------------------------------------
I3D_CLIP_SIZE = 16
DEFAULT_WEIGHT_PATH = "./i3d/models/rgb_imagenet.pt"


def _stam_tensor_batch(batch_tensor):
    """Build per-frame 16-frame clips with 7 frames before, 8 after (zero-padded)."""
    n, c, h, w = batch_tensor.shape
    device = batch_tensor.device
    x1 = torch.zeros(7, c, h, w, device=device, dtype=batch_tensor.dtype)
    x2 = torch.zeros(8, c, h, w, device=device, dtype=batch_tensor.dtype)
    padded = torch.cat([x1, batch_tensor, x2], dim=0)
    clips = []
    for b in range(7, batch_tensor.size(0) + 7):
        clip = padded[b - 7 : b + 9]   # 16 frames
        clip = clip.transpose(0, 1).unsqueeze(0)  # (1, 3, 16, H, W)
        clips.append(clip)
    return torch.cat(clips, dim=0)


def main():
    parser = get_parser(description="Dataset preparation – I3D feature extraction")
    parser.add_argument("--pretrain_weight", type=str, default=DEFAULT_WEIGHT_PATH, help="I3D weight file")
    args = parser.parse_args()

    path_input, path_output = build_paths(args)
    os.makedirs(path_output, exist_ok=True)

    from i3d.pytorch_i3d import InceptionI3d

    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load(args.pretrain_weight, map_location="cpu"))
    model.eval()
    model.to("cuda")
    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    batch_size = 16

    def extract_batch(frames_tensor):
        with torch.no_grad():
            batch_tensor = torch.stack(frames_tensor[:1000])
            clips = _stam_tensor_batch(batch_tensor)
            clips = Variable(clips)
            parts = []
            for chunk in clips.split(batch_size):
                out = chunk.cuda()
                feats = model.extract_features(out).detach().cpu()
                feats = feats.squeeze(-1).squeeze(-1).squeeze(-1)
                parts.append(feats)
            features = torch.cat(parts, dim=0)
        return features.view(features.size(0), -1)

    run_main(args, transform, extract_batch, max_frames=900)


if __name__ == "__main__":
    main()
