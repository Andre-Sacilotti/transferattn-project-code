"""
Extract frame-level features using TSN with TRN (Temporal Relation Network) consensus.
"""
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

from common import get_parser, build_paths, run_main
from models import TSN

# -----------------------------------------------------------------------------
# TRN clip: 8 frames per clip (3 before + 5 after padding)
# -----------------------------------------------------------------------------
DEFAULT_WEIGHT_PATH = "pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth"


def _trn_clip_batch(batch_tensor):
    """Build 8-frame clips with 3 zero frames before, 5 after."""
    n, c, h, w = batch_tensor.shape
    device = batch_tensor.device
    x1 = torch.zeros(3, c, h, w, device=device, dtype=batch_tensor.dtype)
    x2 = torch.zeros(5, c, h, w, device=device, dtype=batch_tensor.dtype)
    padded = torch.cat([x1, batch_tensor, x2], dim=0)
    clips = []
    for b in range(3, batch_tensor.size(0) + 3):
        clip = padded[b - 3 : b + 5]
        clip = clip.transpose(0, 1).unsqueeze(0)
        clips.append(clip)
    return torch.cat(clips, dim=0)


def main():
    parser = get_parser(description="Dataset preparation – TRN feature extraction")
    parser.add_argument("--pretrain_weight", type=str, default=DEFAULT_WEIGHT_PATH, help="TRN checkpoint")
    args = parser.parse_args()

    path_input, path_output = build_paths(args)
    os.makedirs(path_output, exist_ok=True)

    net = TSN(
        30, 8, "RGB",
        base_model="InceptionV3",
        consensus_type="TRNmultiscale",
        img_feature_dim=224,
        print_spec=False,
    )
    net.load_state_dict(torch.load(args.pretrain_weight, map_location="cpu"))
    net.eval()
    net.to("cuda")
    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 8

    def extract_batch(frames_tensor):
        with torch.no_grad():
            batch_tensor = torch.stack(frames_tensor)
            clips = _trn_clip_batch(batch_tensor)
            clips = Variable(clips)
            parts = []
            for chunk in clips.split(batch_size):
                out = chunk.cuda()
                feats = net(out).detach().cpu()
                feats = feats.squeeze(-1).squeeze(-1).squeeze(-1) if feats.dim() > 2 else feats
                parts.append(feats)
            features = torch.cat(parts, dim=0)
        return features.view(features.size(0), -1)

    run_main(args, transform, extract_batch, max_frames=None)


if __name__ == "__main__":
    main()
