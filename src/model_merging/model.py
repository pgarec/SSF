import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def get_featuremap_and_clf(model):
    feature = model.feature_map
    clf = model.clf

    return feature, clf


def get_featuremap(model):
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
    def __init__(self, cfg, normal=False):
        super(MLP, self).__init__()

        if cfg.train.torch_seed > -1:
            torch.manual_seed(cfg.train.torch_seed)

        self.image_shape = cfg.data.image_shape
        self.num_classes = cfg.data.n_classes
        self.hidden_dim = cfg.train.hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.image_shape, self.hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            # nn.BatchNorm1d(self.hidden_dim),  
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            # nn.BatchNorm1d(self.hidden_dim),  
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
        )

        self.clf = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_classes, bias=False),
        )
        if normal:
            self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data, mean=0, std=1)

    def forward(self, x):
        x = self.model(x)
        
        return self.clf(x)
    
    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]
  
    def get_featuremap_trainable_parameters(self):
        return [param for param in self.feature_map.parameters() if param.requires_grad]
    
    def get_clf_trainable_parameters(self):
        return [param for param in self.clf.parameters() if param.requires_grad]
    

class MLP_regression(nn.Module):
    def __init__(self, cfg, normal=False):
        super(MLP_regression, self).__init__()

        if cfg.train.torch_seed > -1:
            torch.manual_seed(cfg.train.torch_seed)

        self.hidden_dim = cfg.train.hidden_dim
        self.model = nn.Sequential(
            nn.Linear(cfg.data.dimensions, self.hidden_dim),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        if normal:
            self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data, mean=0, std=1)

    def forward(self, x):
        
        return self.model(x)
    
    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]
    
    def get_regressor_trainable_parameters(self):
        return [param for param in self.regressor.parameters() if param.requires_grad]
    
