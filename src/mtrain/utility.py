import torch


def anpai(targets):
    if targets is None:
        return None

    targets = (
        targets
        if isinstance(targets, (list, tuple))
        else [
            targets,
        ]
    )

    handle = list()

    device = torch.device("cuda:0")
    for i in targets:
        if isinstance(i, torch.nn.Module) and torch.cuda.device_count() > 1:
            i = torch.nn.DataParallel(i)
            _i = i.to(device)
        else:
            _i = i.to(device)
        handle.append(_i)

    return handle[0] if len(handle) == 1 else handle
