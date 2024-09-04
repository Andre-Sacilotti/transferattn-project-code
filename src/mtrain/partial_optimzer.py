from functools import partial as P

from torch import optim


def pSGD(lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    return P(
        optim.SGD,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )


def pAdam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
    return P(
        optim.Adam,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )


def pAdam3(lr=1e-3, betas=(0.5, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
    return P(
        optim.Adam,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )


def pAdam2(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
    return P(
        optim.Adam,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        foreach=False,
    )
