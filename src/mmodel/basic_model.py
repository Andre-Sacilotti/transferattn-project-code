import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from src.mtrain import Logger, ObservableTensor, Trainer
from src.mtrain.logger_writer import WandBWriter
from src.mtrain.utility import anpai


def inf_iter(iterable):
    while True:
        for i in iterable:
            yield i


class NoUpdateDict(dict):
    def __setitem__(self, key, value):
        if key in self.keys():
            raise Exception("{} aleady exist".format(key))
        else:
            super().__setitem__(key, value)


class Context(dict):
    def __getitem__(self, key):
        if key not in self.keys():
            super().__setitem__(key, list())
        return super().__getitem__(key)


class Flag:
    TRAIN = "train"
    EVAL = "eval"
    END_EPOCH = "end_epoch"
    EVAL_ACCU = "eval_accuracy"


class optimize_ops:
    def __init__(
        self, target, optimer, lr_scheduler=None, scheduler_step_signal="epoch"
    ):
        if not isinstance(optimer, partial):
            raise Exception("Use partial for optimer")
        if lr_scheduler is not None and not isinstance(lr_scheduler, partial):
            raise Exception("Use partial for optimer")
        if scheduler_step_signal not in ["epoch", "step"]:
            raise Exception("Must in epoch or step")
        self.optimer_fn = optimer
        self.lr_scheduler_fn = lr_scheduler
        self.scheduler_step_signal = scheduler_step_signal
        self.target = target

    def __enter__(self):
        self.target.optimer_fn = self.optimer_fn
        self.target.lr_scheduler_fn = self.lr_scheduler_fn
        self.target.scheduler_step_signal = self.scheduler_step_signal

    def __exit__(self, *args):
        self.target.optimer_fn = None
        self.target.lr_scheduler_fn = None
        self.target.lr_scheduler_signal = None


class TrainableModel(ABC):
    def __init__(self, cfg):
        super(TrainableModel, self).__init__()
        writer = WandBWriter(cfg.project_name, cfg.enable_log, cfg)
        writer.log_params(cfg)
        self.cfg = cfg
        self.writer = writer
        # self.max_rate = torch.FloatTensor(0).cuda()

        # get class infos and dataloaders
        train_dset, eval_dset, eval_measures = self.prepare_dataloaders()
        is_dict_train_dest = isinstance(train_dset, dict)
        if is_dict_train_dest:
            dset_items = list(train_dset.items())
            train_dset_key, train_dset = dset_items[-1]
            other_dset_its = {k: inf_iter(d) for k, d in dset_items[0:-1]}
        self.cls_num = len(train_dset.dataset.classes)

        is_dict_eval_dset = isinstance(eval_dset, dict)
        if not is_dict_eval_dset:
            eval_dset = {"dataset": eval_dset}

        # data feeding function
        def iter_dsets(mode):
            if mode == Flag.TRAIN:
                self._current_dset = "train"
                for data in iter(train_dset):
                    if is_dict_train_dest:
                        _data = dict()
                        _data[train_dset_key] = anpai(data)
                        for key in other_dset_its:
                            _data[key] = anpai(next(other_dset_its[key]))
                        data = _data
                    else:
                        data = anpai(data)
                    yield data
            elif mode == Flag.EVAL:
                for key in eval_dset:
                    self._current_dset = key
                    yield key, eval_dset[key]

        self.iter_dsets = iter_dsets

        eval_measures_dict = {}
        if isinstance(eval_measures, dict):
            for key in eval_measures:
                assert key in eval_dset
                measures = eval_measures[key]
                if not isinstance(measures, (list, tuple)):
                    measures = [measures]
                eval_measures_dict[key] = measures
        else:
            if not isinstance(eval_measures, (list, tuple)):
                eval_measures = [eval_measures]
            for key in eval_dset:
                eval_measures_dict[key] = eval_measures
        self.eval_measures = eval_measures_dict

        # get all networks and send networks to gup
        networks = self.regist_networks()

        assert isinstance(networks, dict)
        networks = {i: anpai(j) for i, j in networks.items()}
        # make network to be class attrs
        for i, j in networks.items():
            self.__setattr__(i, j)
        self.networks = networks
        self.current_step = 0
        self.current_epoch = 0
        self.epoch_steps = len(train_dset)
        self.tensors = NoUpdateDict()
        self.trainer = NoUpdateDict()
        self.loggers = NoUpdateDict()
        self.optimize_config = partial(optimize_ops, target=self)

        # testing
        self._current_mode = ""
        self.greatestacc = 0

    @abstractmethod
    def prepare_dataloaders(self):
        pass

    @abstractmethod
    def regist_networks(self):
        pass

    @abstractmethod
    def train_process(self, data, ctx):
        pass

    @abstractmethod
    def eval_process(self, data, ctx):
        pass

    def train_model(self):
        self._current_mode = Flag.TRAIN
        yes_or_no = True
        self.max_i = 0
        print(self.cfg.epoch, self.epoch_steps)
        self.all_steps = self.cfg.epoch * self.epoch_steps
        self.stepsp = 0
        for i in tqdm.tqdm(range(self.cfg.epoch), desc="EPOCHS"):
            self._current_mode = Flag.TRAIN
            for nnn, j in self.networks.items():
                j.train(True)

            ctx = Context()
            for datas in self.iter_dsets(mode=Flag.TRAIN):
                self.train_process(datas, ctx, epoch=i)
                for loss_name in self.trainer:
                    lr = self.trainer[loss_name].get_lr()
                    self.record_metric("lr@" + loss_name, lr)
                self.current_step += 1
                self.current_epoch = self.current_step / self.epoch_steps
                self.stepsp += 1
            assert self.current_epoch == int(self.current_epoch)

            # begin eval if nedded
            if self.current_epoch % self.cfg.eval_epoch_interval == 0:
                temp = self.eval_model(i)
                if yes_or_no:
                    self.max_rate = temp
                    yes_or_no = False
                elif temp > self.max_rate:
                    self.max_rate = temp
                    self.max_i = i
                self.record_metric("best_acc", self.max_rate)

        print(self.max_rate, self.max_i)
        self.record_metric("MELHORACC", self.max_rate)

    def plot_embedding(self, data, label, name):
        # print("test1", data, label)
        x_min, x_max = np.min(data), np.max(data)
        data = (data - x_min) / (x_max - x_min)
        # print("test2", data, label)
        plt.axis("off")
        for i in range(0, 360):
            if label[i] < 4:
                plt.scatter(
                    data[i][0],
                    data[i][1],
                    marker="x",
                    color=plt.cm.Set1(label[i] / 4.0),
                )
            elif label[i] < 8:
                plt.scatter(
                    data[i][0],
                    data[i][1],
                    marker="x",
                    color=plt.cm.Set2((label[i] - 4) / 4.0),
                )
            else:
                plt.scatter(
                    data[i][0],
                    data[i][1],
                    marker="x",
                    color=plt.cm.Set3((label[i] - 8) / 4.0),
                )
        for i in range(360, data.shape[0]):
            if label[i] < 4:
                plt.scatter(
                    data[i][0],
                    data[i][1],
                    marker=".",
                    color=plt.cm.Set1(label[i] / 4.0),
                )
            elif label[i] < 8:
                plt.scatter(
                    data[i][0],
                    data[i][1],
                    marker=".",
                    color=plt.cm.Set2((label[i] - 4) / 4.0),
                )
            else:
                plt.scatter(
                    data[i][0],
                    data[i][1],
                    marker=".",
                    color=plt.cm.Set3((label[i] - 8) / 4.0),
                )
        plt.savefig("images/" + name + ".png")
        plt.cla()

    def fig_tsne(self, feats, trg, name):
        # tsne = TSNE(n_components=2, init='pca', random_state=0)
        feats = feats.data.cpu().numpy()
        trg = trg.data.cpu().numpy()
        # result = tsne.fit_transform(feats)

        with open(f"./pca/{name}_feats.npy", "wb") as f:
            np.save(f, feats)

        with open(f"./pca/{name}_trg.npy", "wb") as f:
            np.save(f, trg)

        # self.plot_embedding(result, trg, name)

    def eval_model(self, number, **kwargs):
        # set all networks to eval mode
        self._current_mode = Flag.EVAL
        for _, i in self.networks.items():
            i.eval()

        # iter every eval dataset
        with torch.no_grad():
            sum_preds = list()
            sum_targs = list()
            for key, dset in self.iter_dsets(mode=Flag.EVAL):
                ctx = Context()
                ctx.dataset = key
                preds = list()
                targs = list()
                for data in iter(dset):
                    data = anpai(data)
                    pred, targ = self.eval_process(data, ctx)
                    preds.append(pred)
                    targs.append(targ)
                    sum_preds.append(pred)
                    sum_targs.append(targ)
                ctx = {k: torch.cat(ctx[k], dim=0) for k in ctx}
                preds = torch.cat(preds, dim=0)
                targs = torch.cat(targs, dim=0)
                # self.fig_tsne(preds, targs, str(number))
                measures = self.eval_measures.get(key, list())
                # print(measures)
                for met in measures:
                    tag = met.tag + "@eval_" + key
                    val = met.cal(preds, targs, ctx)
                    if val > self.greatestacc:
                        self.greatestacc = val
                    print(self.current_epoch, tag, val)
                    self.record_metric(tag, val, met.type)

                ctx = Context()
                for datas in self.iter_dsets(mode=Flag.TRAIN):
                    pred, targ = self.eval_process(datas["source"], ctx)
                    sum_preds.append(pred)
                    sum_targs.append(targ)

                sum_preds = torch.cat(sum_preds, dim=0)
                sum_targs = torch.cat(sum_targs, dim=0)
                # self.fig_tsne(sum_preds, sum_targs, str(number))
            return val

    def optimize_loss(self, name, value, networks, lr_mult=1):
        if name not in self.tensors:
            assert isinstance(networks, (tuple, list))
            networks = {k: self.networks[k] for k in networks}

            optimer_fn = self.optimer_fn
            lr_scheduler_fn = self.lr_scheduler_fn
            scheduler_step_signal = self.scheduler_step_signal
            if optimer_fn is None:
                raise Exception("need to set optiemr with 'with'.")

            loss = ObservableTensor(name)
            trainer = Trainer(
                networks, optimer_fn, lr_scheduler_fn, scheduler_step_signal, lr_mult
            )
            logger = Logger(name, self.cfg.log_step_interval, self.writer)
            loss.add_listener([trainer, logger])

            self.tensors[name] = loss
            self.trainer[name] = trainer
            self.loggers[name] = logger

        self.tensors[name].update(value, self.current_step, self.current_epoch)

    def record_metric(self, name, value, ttype="scalar"):
        if name not in self.tensors:
            tensor = ObservableTensor(name)
            log_interval = (
                1 if self._current_mode == Flag.EVAL else self.cfg.log_step_interval
            )
            logger = Logger(name, log_interval, self.writer, ttype=ttype)
            tensor.add_listener(logger)
            self.tensors[name] = tensor
            self.loggers[name] = logger
        self.tensors[name].update(value, self.current_step, self.current_epoch)

    def save_model(self, key):
        net = self.networks[key]
        path = Path(os.environ["SVAE_PATH"])
        path = path.joinpath("{}_{}.pth".format(key, self.current_epoch))
        torch.save(net.state_dict(), path)
