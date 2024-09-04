import torch.nn as nn


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        DIM = 1024
        self.classifer = nn.Sequential(
            nn.Linear(in_dim, DIM),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(DIM, 1),
        )
        self.apply(init_weights)

    def forward(self, feat):
        logit = self.classifer(feat)
        return logit
