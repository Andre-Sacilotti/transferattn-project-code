"""
Shared logic for video-to-feature extraction scripts.
"""
import argparse
import os
import time
from multiprocessing.dummy import Pool as ThreadPool

import imageio
import numpy as np
import torch
from colorama import init, Fore, Back
from PIL import Image

init(autoreset=True)

MAX_THREAD = 8
FEATURE_EXT = ".t7"


def get_parser(description="Dataset preparation – feature extraction"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_path", type=str, default="", help="Root path of the dataset")
    parser.add_argument("--video_in", type=str, default="", help="Input folder name under data_path (e.g. RGB)")
    parser.add_argument("--feature_in", type=str, default="RGB-feature", help="Output feature folder name prefix")
    parser.add_argument("--input_type", type=str, default="video", choices=["video", "frames"], help="Input: video files or frame folders")
    parser.add_argument("--structure", type=str, default="tsn", choices=["tsn", "imagenet"], help="Output layout: tsn (per-video dirs) or imagenet (per-class dirs)")
    parser.add_argument("--base_model", type=str, default="resnet101", help="Model name (for paths/logging)")
    parser.add_argument("--pretrain_weight", type=str, default="", help="Path to model weights")
    parser.add_argument("--num_thread", type=int, default=-1, help="Number of worker threads (default: %d)" % MAX_THREAD)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for extraction")
    parser.add_argument("--start_class", type=int, default=1, help="First class index (1-based)")
    parser.add_argument("--end_class", type=int, default=-1, help="Last class index (-1 = all)")
    parser.add_argument("--class_file", type=str, default="class.txt", help="File with class names to process (or 'none' for unlabeled)")
    return parser


def build_paths(args):
    """Return (path_input, path_output). Paths use trailing slash."""
    path_input = os.path.join(args.data_path, (args.video_in or "").strip(), "")
    if args.structure == "tsn":
        path_output = os.path.join(args.data_path, "%s_%s" % (args.feature_in, args.base_model), "")
    else:
        path_output = os.path.join(args.data_path, "%s-%s" % (args.feature_in, args.structure), "")
    return path_input, path_output


def load_class_names(args):
    """Return set of class names to process."""
    if args.class_file == "none":
        return {"unlabeled"}
    names = []
    with open(args.class_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            names.append(parts[1] if len(parts) > 1 else parts[0])
    return set(names)


def load_frames(path_input, class_name, video_file, input_type, transform, input_type_frames_is_dir=True):
    """
    Load frames for one video. Return list of tensors.
    transform: callable (PIL or numpy image -> tensor).
    For input_type=='frames', video_file is either a dir name under class or a file path; if input_type_frames_is_dir
    we list path_input/class_name/video_file/ as frame images.
    """
    class_dir = os.path.join(path_input, class_name, "")
    frames_tensor = []
    last_id_frame = 0

    def to_tensor(im):
        if not hasattr(im, "shape"):
            im = np.array(im)
        if np.sum(im.shape) == 0:
            return None
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        pil = Image.fromarray(im)
        return transform(pil)

    if input_type == "video":
        reader = imageio.get_reader(os.path.join(class_dir, video_file))
        try:
            for t, im in enumerate(reader):
                t_im = to_tensor(im)
                if t_im is not None:
                    last_id_frame = t + 1
                    frames_tensor.append(t_im)
        except RuntimeError as e:
            print(Back.RED + "Could not read frame %s from %s: %s" % (last_id_frame + 1, video_file, e))
    else:
        frame_dir = os.path.join(class_dir, video_file)
        if not os.path.isdir(frame_dir):
            return frames_tensor
        list_frames = sorted(os.listdir(frame_dir))
        try:
            for t, name in enumerate(list_frames):
                im = imageio.imread(os.path.join(frame_dir, name))
                t_im = to_tensor(im)
                if t_im is not None:
                    last_id_frame = t + 1
                    frames_tensor.append(t_im)
        except RuntimeError as e:
            print(Back.RED + "Could not read frame %s from %s: %s" % (last_id_frame + 1, video_file, e))

    return frames_tensor


def get_num_exist_files(structure, path_output, video_name, class_name):
    """Number of existing feature files for this video (for skip-if-done)."""
    if structure == "tsn":
        video_dir = os.path.join(path_output, video_name)
        if not os.path.isdir(video_dir):
            return 0
        return len([f for f in os.listdir(video_dir) if f.endswith(FEATURE_EXT)])
    else:
        class_dir = os.path.join(path_output, class_name)
        if not os.path.isdir(class_dir):
            return 0
        prefix = video_name + "_"
        return len([f for f in os.listdir(class_dir) if f.startswith(prefix) and f.endswith(FEATURE_EXT)])


def save_features(features, path_output, video_name, class_name, structure):
    """Save feature tensor to per-frame .t7 files."""
    for t in range(features.size(0)):
        id_frame_name = str(t + 1).zfill(5)
        if structure == "tsn":
            filename = os.path.join(path_output, video_name, "img_%s%s" % (id_frame_name, FEATURE_EXT))
        else:
            filename = os.path.join(path_output, class_name, "%s_%s%s" % (video_name, id_frame_name, FEATURE_EXT))
        if not os.path.exists(filename):
            torch.save(features[t].clone(), filename)


def run_main(args, transform, extract_batch_fn, max_frames=None):
    """
    Main loop: for each class in data_path, for each video, load frames, extract features, save.
    - transform: torchvision-style transform (PIL -> tensor).
    - extract_batch_fn: (list of tensors) -> 1D tensor of shape (N, D).
    - max_frames: if set, cap number of frames per video (default None = no cap).
    """
    path_input, path_output = build_paths(args)
    os.makedirs(path_output, exist_ok=True)

    num_thread = args.num_thread if 0 < args.num_thread <= MAX_THREAD else MAX_THREAD
    print(Fore.CYAN + "thread #: %s" % num_thread)
    pool = ThreadPool(num_thread)

    class_names_proc = load_class_names(args)
    list_class = sorted([d for d in os.listdir(path_input) if os.path.isdir(os.path.join(path_input, d))])
    id_start = max(0, args.start_class - 1)
    id_end = len(list_class) if args.end_class <= 0 else min(args.end_class, len(list_class))

    def extract_one(item):
        class_name, video_file = item
        video_name = os.path.splitext(video_file)[0]

        if args.structure == "tsn":
            video_dir = os.path.join(path_output, video_name)
            os.makedirs(video_dir, exist_ok=True)
        num_exist = get_num_exist_files(args.structure, path_output, video_name, class_name)

        frames_tensor = load_frames(path_input, class_name, video_file, args.input_type, transform)
        num_frames = len(frames_tensor)
        if num_frames == 0:
            return
        if num_frames == num_exist:
            return

        if max_frames is not None:
            frames_tensor = frames_tensor[:max_frames]
            num_frames = len(frames_tensor)

        features = extract_batch_fn(frames_tensor)
        features = features[:num_frames]
        save_features(features, path_output, video_name, class_name, args.structure)

    total_start = time.time()
    for i in range(id_start, id_end):
        class_name = list_class[i]
        if class_name not in class_names_proc:
            print(Fore.RED + "%s is not selected" % class_name)
            continue

        print(Fore.YELLOW + "class %s: %s" % (i + 1, class_name))
        if args.structure == "imagenet":
            os.makedirs(os.path.join(path_output, class_name), exist_ok=True)

        class_dir = os.path.join(path_input, class_name)
        if args.input_type == "video":
            list_video = sorted([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
        else:
            list_video = sorted([f for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))])
        work = [(class_name, v) for v in list_video]
        pool.map(extract_one, work, chunksize=1)

        class_elapsed = time.time() - total_start  # approximate
        print("Elapsed for %s: %.1fs" % (class_name, class_elapsed))

    total_elapsed = time.time() - total_start
    print("Total elapsed: %.1fs" % total_elapsed)
    print(Fore.GREEN + "Features generated for %s" % (args.data_path or path_input))
