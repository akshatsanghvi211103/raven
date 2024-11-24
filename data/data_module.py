import os

from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)

from .dataset import AVDataset
from .samplers import (
    # ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from .transforms import AdaptiveLengthTimeMask

import numpy as np
import random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) < 3:
        collated_batch = collated_batch.unsqueeze(1)
    else:
        collated_batch = collated_batch.permute(
            (0, 4, 1, 2, 3)
        )  # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        if data_type == 'idx':
            batch_out['ids'] = torch.tensor(
                [s[data_type] for s in batch]
            )
            continue
        if data_type == 'video_path':
            # print(f"The data type is {data_type = }")
            batch_out['video_paths'] = [s[data_type] for s in batch]
            continue
        pad_val = -1 if data_type == "label" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type] = c_batch
        batch_out[data_type + "_lengths"] = sample_lengths

    return batch_out


class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        # self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        # print("total gpus:", self.total_gpus)

    def _video_transform(self, mode):
        args = self.cfg.data
        transform = [
            Lambda(lambda x: x / 255.0),
        ] + (
            [
                RandomCrop(args.crop_type.random_crop_dim),
                RandomHorizontalFlip(0.5),
            ]
            if mode == "train"
            else [CenterCrop(args.crop_type.random_crop_dim)]
        )
        if self.cfg.data.channel.in_video_channels == 1:
            transform.extend(
                [
                    Lambda(lambda x: x.transpose(0, 1)),
                    Grayscale(),
                    Lambda(lambda x: x.transpose(0, 1)),
                ]
            )
        transform.append(instantiate(args.channel.obj))

        if mode == "train" and args.use_masking:
            transform.append(
                AdaptiveLengthTimeMask(
                    window=int(args.timemask_window * 25),
                    stride=int(args.timemask_stride * 25),
                    replace_with_zero=True,
                )
            )

        return Compose(transform)

    def _audio_transform(self, mode):
        args = self.cfg.data
        transform = [Lambda(lambda x: x)]

        if mode == "train" and args.use_masking:
            transform.append(
                AdaptiveLengthTimeMask(
                    window=int(args.timemask_window * 16000),
                    stride=int(args.timemask_stride * 16000),
                    replace_with_zero=True,
                )
            )

        return Compose(transform)

    def _dataloader(self, ds, collate_fn):
        g = torch.Generator()
        g.manual_seed(0)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
            # batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        ds_args = self.cfg.data.dataset
        transform_video = self._video_transform(mode="train")
        train_ds = AVDataset(
            data_path=ds_args.train_file,
            transforms={'video': transform_video},
            modality=self.cfg.data.modality
        )
        return self._dataloader(train_ds, collate_pad)

    def test_dataloader(self):
        ds_args = self.cfg.data.dataset
        print(ds_args, "hi there")
        # print(self.cfg)

        transform_video = self._video_transform(mode="val")
        transform_audio = self._audio_transform(mode="val")

        parent_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)
        )

        test_ds = AVDataset(
            # data_path=os.path.join(parent_path, "data_paths", ds_args.test_csv),
            # data_path=os.path.join(parent_path, "output.txt"),
            data_path=ds_args.train_file,
            # video_path_prefix_lrs2=ds_args.paths.root_lrs2_video,
            # audio_path_prefix_lrs2=ds_args.paths.root_lrs2_audio,
            # video_path_prefix_lrs3=ds_args.paths.root_lrs3_video,
            # audio_path_prefix_lrs3=ds_args.paths.root_lrs3_audio,
            transforms={"video": transform_video, "audio": transform_audio},
            modality=self.cfg.data.modality,
        )
        # sampler = ByFrameCountSampler(
        #     test_ds, self.cfg.data.frames_per_gpu_val, shuffle=False
        # )
        # if self.total_gpus > 1:
            # sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        # return self._dataloader(test_ds, sampler, collate_pad)
        return self._dataloader(test_ds, collate_pad)

    def val_dataloader(self):
        ds_args = self.cfg.data.dataset

        val_dataloaders = []
        transform_video = self._video_transform(mode="val")

        if ds_args.test_file is not None:
            dataset = AVDataset(
                data_path=ds_args.test_file,
                transforms={"video": transform_video},
                modality=self.cfg.data.modality
            )
            dataloader = self._dataloader(dataset, collate_pad)
            val_dataloaders.append(dataloader)

        if ds_args.val_file is not None:
            dataset = AVDataset(
                data_path=ds_args.val_file,
                transforms={"video": transform_video},
                modality=self.cfg.data.modality
            )
            dataloader = self._dataloader(dataset, collate_pad)
            val_dataloaders.append(dataloader)

        return val_dataloaders