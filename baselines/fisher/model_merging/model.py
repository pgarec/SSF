import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()

        self.image_shape = cfg.data.image_shape
        self.num_classes = cfg.data.n_classes
        self.hidden_dim = cfg.train.hidden_dim

        self.feature_map = nn.Sequential(
            nn.Linear(cfg.data.image_shape, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )

        self.clf = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_classes, bias=False),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, training=True):
        if not training:
            with torch.no_grad():
                x = self.feature_map(x)

        else:
            x = self.feature_map(x)

        return self.clf(x)
