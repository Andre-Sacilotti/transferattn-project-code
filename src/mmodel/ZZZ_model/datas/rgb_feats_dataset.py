# modified from https://github.com/cmhungsteve/TA3N/blob/master/dataset.py

import itertools
import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Sampler

# from colorama import init
# from colorama import Fore, Back, Style

# init(autoreset=True)


def get_data_list(dataset, source, target):
    list_fmt = "src/mmodel/ZZZ_model/datas/{dset}/{subset}_{p}.txt"
    return (
        list_fmt.format(dset=dataset, subset=source, p="train"),
        list_fmt.format(dset=dataset, subset=target, p="train"),
        list_fmt.format(dset=dataset, subset=target, p="eval"),
    )


class VideoData(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(
        self, list_file, sampled_frame, image_tmpl="img_{:05d}.t7", sample_number=5
    ):
        self.list_file = list_file
        self.sample_num = sample_number
        self.sampled_frame = sampled_frame
        self.image_tmpl = image_tmpl
        self.video_list = [
            VideoData(x.strip().split(" ")) for x in open(self.list_file)
        ]
        self.sampler = BalancedSampler([v.label for v in self.video_list])
        # labs = [v.label for v in self.video_list]
        # labs = np.array(labs)
        # _, c = np.unique(labs, return_counts=True)
        # print(c)
        # assert False
        self.classes = np.unique([v.label for v in self.video_list])
        # frames = [v.num_frames for v in self.video_list]
        # self.frame_length_range = [min(frames), max(frames)]
        # print(self.frame_length_range)

    def _load_frame_feature(self, directory, idx):
        feat_path = os.path.join(directory, self.image_tmpl.format(idx))
        feat = [torch.load(feat_path)]
        return feat

    def _get_indices(self, data):
        sampled_frame = self.sampled_frame
        data_frame = data.num_frames
        if data_frame >= sampled_frame:
            tick = float(data_frame) / float(sampled_frame)
            offsets = np.array(
                [
                    int(random.uniform(0, tick) + tick * float(x))
                    for x in range(self.sampled_frame)
                ]
            )  # pick the central frame in each segment
            offsets = offsets + 1
            for i in range(self.sample_num - 1):
                offsets_1 = np.array(
                    [
                        int(random.uniform(0, tick) + tick * float(x))
                        for x in range(self.sampled_frame)
                    ]
                )
                offsets = np.concatenate([offsets, offsets_1 + 1])
            return offsets
        else:  # the video clip is too short --> duplicate the last frame
            id_select = np.array([x for x in range(data_frame)])
            # expand to the length of self.num_segments with the last element
            temp = int(sampled_frame - data_frame + 1)
            offsets = np.array([x for x in range(data_frame)])
            # print()
            for i in range(1, temp):
                offsets = np.insert(
                    offsets,
                    int(data_frame / temp * i),
                    id_select[int(data_frame / temp * i)],
                )
            # print(offsets.size)
            offsets = offsets + 1
            """
            if self.list_file[-8:] == "eval.txt":
                return offsets
            """

            for i in range(self.sample_num - 1):
                offsets = np.concatenate([offsets, offsets])
            return offsets

    def __getitem__(self, index):
        data = self.video_list[index]

        segment = self._get_indices(data)
        # print(segment.shape)
        temp = []
        frames = list()
        for idx in segment:
            p = int(idx)
            seg_feats = self._load_frame_feature(data.path, p)
            frames.extend(seg_feats)
            """
            if self.list_file[-8:] == "eval.txt":
            for i in range(1):
                process_data = torch.stack(frames[i * self.sampled_frame:(i + 1) * self.sampled_frame])
                temp.append(process_data)
            return temp[0], data.label 
            """

        for i in range(self.sample_num):
            process_data = torch.stack(
                frames[i * self.sampled_frame : (i + 1) * self.sampled_frame]
            )
            temp.append(process_data)
        """
          temp[0],   temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8], temp[9],
        """
        if self.sample_num == 1:
            return temp[0], data.label
        elif self.sample_num == 2:
            return temp[0], temp[1], data.label
        elif self.sample_num == 3:
            return temp[0], temp[1], temp[2], data.label
        elif self.sample_num == 4:
            return temp[0], temp[1], temp[2], temp[3], data.label
        else:
            return temp[0], temp[1], temp[2], temp[3], temp[4], data.label

    def __len__(self):
        return len(self.video_list)


class BalancedSampler(Sampler):
    """sampler for balanced sampling"""

    def __init__(self, targets, max_per_cls=None):
        self.targets = targets
        self.max_per_cls = max_per_cls

    def generate_idxs_list(self, targets):
        targets = torch.tensor(targets)
        all_classes = torch.unique(targets)
        cls_idxs = list()
        for curr_cls in all_classes:
            sample_indexes = [i for i in range(len(targets)) if targets[i] == curr_cls]

            if self.max_per_cls:
                random.shuffle(sample_indexes)
                sample_indexes = sample_indexes[0 : self.max_per_cls]

            cls_idxs.append(sample_indexes)

        cls_num = len(cls_idxs)

        self.total = cls_num * min([len(i) for i in cls_idxs])

        def shuffle_with_return(l):
            random.shuffle(l)
            return l

        cls_idxs = list(map(shuffle_with_return, cls_idxs))
        cls_idxs = list(zip(*cls_idxs))

        sample_idx = list(itertools.chain(*cls_idxs))
        return sample_idx

    def __iter__(self):
        idx = self.generate_idxs_list(self.targets)
        return iter(idx)

    def __len__(self):
        return self.total


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST

    trans = transforms.Compose([transforms.ToTensor()])

    minist = MNIST(root="./DATASET/MNIST", train=True, download=True, transform=trans)
    a = minist.targets.numpy()
    pa = PartialSampler(a, [1, 2, 3, 4, 5, 6, 7])
    balanced = BalancedSampler(pa)
    data = DataLoader(
        minist, batch_size=120, shuffle=False, sampler=balanced, drop_last=True
    )

    for batch_idx, samples in enumerate(data):
        d, t = samples

    for batch_idx, samples in enumerate(data):
        d, t = samples
        print(t)
