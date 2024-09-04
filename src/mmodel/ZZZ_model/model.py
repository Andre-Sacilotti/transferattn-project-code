from functools import partial

import einops
import numpy as np
import torch
import torch.distributed as dist
from src.mmodel.ZZZ_model.networks.gradient_reverse_layer import GradReverseLayer
from torch import nn
from torch.utils.data import DataLoader

from .loss import HLoss
from .networks.network import Discriminator

from src.transformer_utils import IBHead, TemporalModelling

from src.mtrain.measurer import AcuuMeasurer, CWAcuuMeasurer
from src.mtrain.partial_lr_scheduler import pStepLR
from src.mtrain.partial_optimzer import pAdam
from src.transformer_utils import compute_ib_loss

from ..basic_model import TrainableModel


class ZZZ_model(TrainableModel):
    def __init__(
        self,
        dropout=0.2,
        num_heads=8,
        num_layers=4,
        hidden_size=512,
        use_ibhead=True,
        MDTA=True,
        cls_loss=True,
        h_loss=True,
        adv_loss=True,
        ib_loss=True,
        params=None
    ):
        # Transformer Parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Dataset Parameters
        self.sample_number = 2

        # Losses to be used
        self.cls_loss = cls_loss
        self.h_loss = h_loss
        self.adv_loss = adv_loss
        self.ib_loss = ib_loss

        self.numF = params.sampled_frame

        # Calling the Super Class
        super().__init__(params)

        

    def prepare_dataloaders(self):
        from .datas.rgb_feats_dataset import VideoDataset, get_data_list

        ls_s_train, ls_t_train, ls_t_eval = get_data_list(
            self.cfg.dataset, self.cfg.source, self.cfg.target
        )

        _VideoDataset = partial(VideoDataset, sampled_frame=self.cfg.sampled_frame)
        source_dset = _VideoDataset(ls_s_train, sample_number=self.sample_number)
        target_dset = _VideoDataset(ls_t_train, sample_number=self.sample_number)
        eval_dset = _VideoDataset(ls_t_eval, sample_number=self.sample_number)

        _DataLoader = partial(
            DataLoader,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )

        train_loader = {
            "source": _DataLoader(
                source_dset, drop_last=True, sampler=source_dset.sampler
            ),
            "target": _DataLoader(target_dset, drop_last=True, shuffle=True),
        }
        eval_loader = _DataLoader(eval_dset, drop_last=False)

        measurer = [AcuuMeasurer(), CWAcuuMeasurer()]
        return train_loader, eval_loader, measurer

    # Taken from https://github.com/vturrisi/UDAVT. Thanks for it
    @torch.no_grad()
    def dequeue_and_enqueue(self, z_s, z_t, y_s, y_t):
        z_s = gather(z_s)
        y_s = gather(y_s)
        z_t = gather(z_t)
        y_t = gather(y_t)

        batch_size = z_s.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.source_queue[ptr : ptr + batch_size, :] = z_s
        self.source_queue_y[ptr : ptr + batch_size] = y_s
        self.target_queue[ptr : ptr + batch_size, :] = z_t
        self.target_queue_y[ptr : ptr + batch_size] = y_t
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def regist_networks(self):
        self.queue_size = self.cfg.queue_size
        self.NewAttn = self.cfg.use_MDTA
        self.use_ib_loss = self.cfg.use_ib_loss
        self.use_ibhead = True if self.cfg.use_ib_loss else False
        self.use_adv_loss = self.cfg.use_adv_loss

        if self.use_ib_loss is False:
            self.ib_weight = 0
        else:
            self.ib_weight = self.cfg.IB_weight

        # Defining the Patch embedding for the transformer
        if self.cfg.dataset == "K_NEC":
            FEATURE_DIM = 768
        elif self.cfg.dataset == "UCF_HMDB_TRN":
            FEATURE_DIM = 256
        elif self.cfg.dataset == "UCF_HMDB_I3D":
            FEATURE_DIM = 1024
        elif self.cfg.dataset == "UCF_HMDB_STAM":
            FEATURE_DIM = 768
        else:
            FEATURE_DIM = 2048

        loss = nn.BCELoss()
        m = nn.Sigmoid()

        if self.cfg.dataset == "K_NEC":
            to_patch_embedding = nn.Sequential(
                nn.Linear(FEATURE_DIM, 2048),
                nn.Linear(2048, self.hidden_size),
            )
            dims = 1024
        elif self.cfg.dataset == "UCF_HMDB_TRN":
            to_patch_embedding = nn.Sequential(
                nn.Linear(FEATURE_DIM, 2048),
                nn.Linear(2048, self.hidden_size),
            )
            dims = 1024
        elif self.cfg.dataset == "UCF_HMDB_I3D":
            to_patch_embedding = nn.Sequential(
                nn.Linear(FEATURE_DIM, 2048),
                nn.Linear(2048, self.hidden_size),
            )
            dims = 1024
        elif self.cfg.dataset == "UCF_HMDB_STAM":
            to_patch_embedding = nn.Sequential(
                nn.Linear(FEATURE_DIM, 2048),
                nn.Linear(2048, self.hidden_size),
            )
            dims = 1024
        else:
            to_patch_embedding = nn.Sequential(
                nn.Linear(2048, self.hidden_size),
            )

        # Defining the GRL to be used inside transformer and before classifier
        pre = nn.Sequential(
            GradReverseLayer(self.coeff),
        ).to("cuda")

        # Transformer structure
        temporalModelling = TemporalModelling(
            width=self.hidden_size,
            layers=self.num_layers,
            heads=self.num_heads,
            dropout=self.dropout,
            pre=pre,
            NewAttn=self.NewAttn,
        )

        # Transformer embeddding
        temporalEmbedding = torch.nn.Embedding(self.numF, self.hidden_size)
        nn.init.normal_(temporalEmbedding.weight, std=0.01)

        # Domain Discriminator
        mlp_head = Discriminator(self.hidden_size)

        # IB Head to be used inside transformer
        ibhead = IBHead(self.hidden_size, dims, dims)

        # Queue to be used in IB loss
        self.source_queue = torch.randn(self.queue_size, 1024).cuda()
        self.target_queue = torch.randn(self.queue_size, 1024).cuda()
        self.source_queue_y = -torch.ones(self.queue_size, dtype=torch.long).cuda()
        self.target_queue_y = -torch.ones(self.queue_size, dtype=torch.long).cuda()
        self.queue_ptr = torch.zeros(1, dtype=torch.long).cuda()

        # Parts of the network that will be trainned
        self.loss_opts = [
            "temporalModelling",
            "temporalEmbedding",
            "pre",
            "mlp_head",
            "to_patch_embedding",
        ]

        action_classifier = nn.Linear(self.hidden_size, self.cls_num)
        soft_entropy_criterion = HLoss()
        cross_entropy_criterion = torch.nn.CrossEntropyLoss()

        # Returning all the parts of the network.
        return {
            "CEL": cross_entropy_criterion,
            "HL": soft_entropy_criterion,
            "BCEt": loss,
            "m": m,
            "pre": pre,
            "temporalModelling": temporalModelling,
            "temporalEmbedding": temporalEmbedding,
            "mlp_head": mlp_head,
            "to_patch_embedding": to_patch_embedding,
            "C_adapt": action_classifier,
            "ibhead": ibhead,
        }

    def calculate_loss(self, sf, tf, strg):
        loss_cls_at_adapt = 0
        loss_entropy_at_adapt = 0
        loss_ib = 0

        b = sf.shape[0]
        b2 = tf.shape[0]
        fs = sf.shape[1]

        xs = einops.rearrange(sf.float(), "b t c -> t b c", t=self.numF)
        xt = einops.rearrange(tf.float(), "b t c -> t b c", t=self.numF)

        tempEmbedding = einops.repeat(
            self.temporalEmbedding(torch.arange(self.numF).to(sf.device)),
            "t c -> t b c",
            b=xs.size(1),
        )
        xs = xs + tempEmbedding.to(sf.device)

        tempEmbedding = einops.repeat(
            self.temporalEmbedding(torch.arange(self.numF).to(sf.device)),
            "t c -> t b c",
            b=xt.size(1),
        )
        xt = xt + tempEmbedding.to(sf.device)

        xs, _, _, _ = self.temporalModelling(xs, domain="source")
        xt, _, _, _ = self.temporalModelling(xt, domain="target")

        xs = xs.mean(dim=0)
        xt = xt.mean(dim=0)

        xs = (xs / xs.norm(dim=-1, keepdim=True)) / 0.07
        xt = (xt / xt.norm(dim=-1, keepdim=True)) / 0.07

        cls_s = self.C_adapt(xs)
        cls_t = self.C_adapt(xt)

        l_aux1 = self.CEL(cls_s, strg)
        l_aux2 = self.HL(cls_t)

        loss_cls_at_adapt += l_aux1
        loss_entropy_at_adapt += l_aux2

        if self.use_ibhead:
            projs = self.ibhead(xs)
            projt = self.ibhead(xt)
            pseudo_y = cls_t.detach().argmax(dim=1)

            loss, npairs = compute_ib_loss(
                projs,
                projt,
                strg,
                pseudo_y,
                self.source_queue,
                self.source_queue_y,
                loss=self.ib_weight,
            )
            loss_ib += loss

            self.dequeue_and_enqueue(projs, projt, strg, pseudo_y)

        xs = self.pre(xs)
        xt = self.pre(xt)
        xs = self.mlp_head(xs)
        xt = self.mlp_head(xt)
        xs = self.m(xs)
        xt = self.m(xt)

        loss_h1_s = self.BCEt(xs, torch.zeros((b, 1)).to("cuda"))
        loss_h1_t = self.BCEt(xt, torch.ones((b2, 1)).to("cuda"))

        return (loss_h1_s, loss_h1_t, loss_cls_at_adapt, loss_entropy_at_adapt, loss_ib)

    def train_process(self, data, ctx, epoch=None):
        loss_entropy_ = 0
        loss_cls_ = 0
        loss_ib_ = 0
        loss_adv_tgt_ = 0
        loss_adv_src_ = 0

        sfeat = data["source"][: self.sample_number]
        y_src = data["source"][self.sample_number]
        tfeat = data["target"][: self.sample_number]

        sfeat_0 = [self.to_patch_embedding(i) for i in sfeat]
        tfeat_0 = [self.to_patch_embedding(i) for i in tfeat]

        # Iterate over N clips
        for x_src, x_tgt in zip(sfeat_0, tfeat_0):
            (
                loss_adv_src,
                loss_adv_tgt,
                loss_cls,
                loss_entropy,
                loss_ib,
            ) = self.calculate_loss(x_src, x_tgt, y_src)

            loss_adv_src_ += loss_adv_src
            loss_adv_tgt_ += loss_adv_tgt
            loss_ib_ += loss_ib
            loss_cls_ += loss_cls
            loss_entropy_ += loss_entropy

        self.record_metric("loss_adv_src", loss_adv_src_)
        self.record_metric("loss_adv_tgt", loss_adv_tgt_)
        self.record_metric("loss_entropy", loss_entropy_)
        self.record_metric("loss_cls", loss_cls_)
        self.record_metric("loss_ib", loss_ib_)

        if self.use_adv_loss is False:
            loss_adv_src_ = loss_adv_tgt_ = 0

        L = (
            loss_adv_src_ + loss_adv_tgt_ + loss_entropy_ + loss_cls_ + loss_ib_
        ) / self.sample_number

        with self.optimize_config(
            optimer=pAdam(lr=self.cfg.lr, weight_decay=0.0005),
            lr_scheduler=pStepLR(
                step_size=self.cfg.lr_decay_epoch, gamma=self.cfg.lr_gamma
            ),
        ):
            self.optimize_loss("global_loss", L, self.loss_opts)

    def eval_process(self, data, ctx, number=""):
        feats, trg = data[0 : self.sample_number], data[self.sample_number]
        batch_size = feats[0].shape[0]

        # Patch embedding every backbone feature
        feats = [self.to_patch_embedding(feat) for feat in feats]

        # Forward the first clip
        xs = feats[0]
        xs = einops.rearrange(xs.float(), "b t c -> t b c", t=self.numF)
        tempEmbedding = einops.repeat(
            self.temporalEmbedding(torch.arange(self.numF).to(feats[0].device)),
            "t c -> t b c",
            b=xs.size(1),
        )
        xs = xs + tempEmbedding.to(feats[0].device)
        xs, _, _, _ = self.temporalModelling(xs, posi_emb=None, domain="target")
        xs = xs.mean(dim=0)
        xs = (xs / xs.norm(dim=-1, keepdim=True)) / 0.07
        pred = self.C_adapt(xs)

        # Forward the remaining clip
        for i in range(1, self.sample_number):
            xs = feats[i]
            xs = einops.rearrange(xs.float(), "b t c -> t b c", t=self.numF)
            tempEmbedding = einops.repeat(
                self.temporalEmbedding(torch.arange(self.numF).to(feats[i].device)),
                "t c -> t b c",
                b=xs.size(1),
            )
            xs = xs + tempEmbedding.to(feats[i].device)
            xs, _, _, _ = self.temporalModelling(xs, posi_emb=None, domain="target")
            xs = xs.mean(dim=0)
            xs = (xs / xs.norm(dim=-1, keepdim=True)) / 0.07

            pred += self.C_adapt(xs)

        pred /= self.sample_number
        return pred, trg

    def coeff(self):
        alpha = 10
        high = self.cfg.adv_coeff
        low = 0
        bias = 0
        epoch = max(self.current_epoch - bias, 0)
        epoch = int(epoch)
        p = epoch / (self.cfg.epoch - bias)
        return float(
            2.0 * (high - low) / (1.0 + np.exp(-alpha * p)) - (high - low) + low
        )


# Taken from https://github.com/vturrisi/UDAVT, thanks
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
            dist.all_gather(output, input)
        else:
            output = [input]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        if dist.is_available() and dist.is_initialized():
            grad_out = torch.zeros_like(input)
            grad_out[:] = grads[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    return torch.cat(GatherLayer.apply(X), dim=dim)
