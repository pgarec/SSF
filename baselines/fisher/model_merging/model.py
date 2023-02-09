import torch
import torch.nn as nn
import torch.nn.functional as F


def get_featuremap_and_clf(model):
    feature = model.feature_map
    clf = model.clf

    return feature, clf


def get_feaeturemap(model):
    return get_featuremap_and_clf(model)[0]


def get_mergeable_featuremap_variables(model):
    # TODO
    return model.get_trainable_parameters()


def get_mergeable_variables(model):
    return model.get_trainable_parameters()


def clone_model(model, cfg):
    cloned = model.__class__(cfg=cfg)
    cloned.load_state_dict(model.state_dict().copy())
        
    return cloned


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
    
    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]
    
    def get_featuremap_trainable_parameters(self):
        return [param for param in self.feature_map.parameters() if param.requires_grad]
    
    def get_clf_trainable_parameters(self):
        return [param for param in self.clf.parameters() if param.requires_grad]
    

