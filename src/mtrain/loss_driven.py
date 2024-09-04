from abc import ABC, abstractmethod
from functools import partial

import torch
from torch import optim as optim

from .logger_writer import LoggerWriter


class LossChangeListener(ABC):
    @abstractmethod
    def before_change(self, step, epoch):
        pass

    @abstractmethod
    def in_change(self, value):
        pass

    @abstractmethod
    def after_change(self):
        pass


class ObservableTensor(object):
    def __init__(self, tag):
        self.tag = tag
        self.value = None
        self.listeners = list()

    def add_listener(self, lsn):
        lsn = (
            [
                lsn,
            ]
            if not isinstance(lsn, (tuple, list))
            else lsn
        )
        for l in lsn:
            if not isinstance(l, LossChangeListener):
                raise Exception("needs LossChangeListener")
            self.listeners.append(l)

    def update(self, value, step, epoch=None):
        for i in self.listeners:
            i.before_change(step, epoch)

        for i in self.listeners:
            i.in_change(value)

        for i in self.listeners:
            i.after_change()


class Trainer(LossChangeListener):
    """this is a tool class for helping training a network"""

    def __init__(
        self, networks, optimer_fn, lr_scheduler_fn, scheduler_step_signal, lr_mult=1
    ):
        super(Trainer, self).__init__()

        assert isinstance(networks, dict)
        assert isinstance(optimer_fn, partial)
        assert lr_scheduler_fn is None or isinstance(lr_scheduler_fn, partial)

        if isinstance(lr_mult, (int, float)):
            lr_mult_default = lr_mult
            lr_mult_dcit = dict()
        elif isinstance(lr_mult, dict):
            lr_mult_default = 1
            lr_mult_dcit = lr_mult
        else:
            raise Exception("lr_mult type error")

        # get all parameters in network list
        base_lr = optimer_fn.keywords["lr"]
        param_groups = list()
        # param_groups += [{'params': torch.zeros(1), 'lr':base_lr}]
        for k, n in networks.items():
            if isinstance(n, torch.nn.DataParallel):
                n = n.module
            lr_mult = lr_mult_dcit.get(k, lr_mult_default)
            param_info = [
                {
                    "params": n.parameters(),
                    "lr": lr_mult * base_lr,
                },
            ]
            param_groups += param_info

        # init optimer base on type and args
        optimer = optimer_fn(param_groups)

        # init optimer decay option
        lr_scheduler = None
        if lr_scheduler_fn is not None:
            lr_scheduler = lr_scheduler_fn(optimer)

        self.optimer = optimer
        self.lr_scheduler = lr_scheduler
        self.scheduler_step_signal = scheduler_step_signal

        self.cepoch = -1

    def before_change(self, step, epoch):
        if self.lr_scheduler is not None:
            if self.scheduler_step_signal == "epoch" and int(self.cepoch) != int(epoch):
                self.lr_scheduler.step()
            elif self.scheduler_step_signal == "step" and self.cepoch != epoch:
                self.lr_scheduler.step()
        self.cepoch = epoch
        self.optimer.zero_grad()

    def in_change(self, value):
        value.backward(retain_graph=True)
        self.optimer.step()

    def after_change(self):
        return

    def get_lr(self):
        # return self.optimer.param_groups[0]['lr']
        for param_group in self.optimer.param_groups:
            lr = param_group["lr"]
            return torch.tensor(lr)


class Logger(LossChangeListener):
    def __init__(self, tag, log_interval, writer: LoggerWriter, ttype="scalar"):
        assert ttype in writer.support_type
        self.accumulative = ttype == "scalar"
        # the writer
        if ttype == "scalar":
            write_fn = writer.log_metric
        elif ttype == "matrix":
            write_fn = writer.log_confusion_matrix
        self.write = write_fn
        self.tag = tag
        # vars for step and epoch infos
        self.current_step = 0
        self.current_epoch = 0
        # vars for accumulative mode
        self.range_value = 0.0
        self.range_step = 0.0
        self.log_interval = log_interval

    def before_change(self, step, epoch):
        self.current_step = step
        self.current_epoch = epoch

    def in_change(self, value):
        value = self._apply(value, lambda x: x.item())
        if self.accumulative:
            self.range_value += value
            self.range_step += 1
        else:
            self.value = value

    def after_change(self):
        if self.current_step % self.log_interval == 0:
            if self.accumulative:
                value = self.range_value / self.range_step
                self.range_value = 0.0
                self.range_step = 0.0
            else:
                value = self.value
            self.write(self.tag, value, self.current_epoch)

    def _apply(self, value, fn):
        if isinstance(value, (int, float)):
            value = value
        elif isinstance(value, torch.Tensor):
            value = fn(value)
        elif isinstance(value, (list, tuple)):
            value = [fn(value) for v in value]
        else:
            raise Exception("tensor type not support")
        return value
