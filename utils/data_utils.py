import numpy as np
import torch
from torchvision import datasets, transforms


class MNISTDataset:
    def __init__(self, data_path, digits=None):
        self.data_path = data_path
        self.digits = digits
        self.data = np.load(self.data_path)
        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data()

    def split_data(self):
        X_train, y_train, X_test, y_test = self.data['X_train'], self.data['y_train'], self.data['X_test'], self.data['y_test']
        if self.digits:
            mask = np.isin(y_train, self.digits)
            X_train, y_train = X_train[mask], y_train[mask]
            mask = np.isin(y_test, self.digits)
            X_test, y_test = X_test[mask], y_test[mask]
            
        return X_train, y_train, X_test, y_test



def get_mnist_loaders(data_path, batch_size=512, model_class='MLP',
                      train_batch_size=128, val_size=2000, download=False, device='cpu'):
    """
    Source: https://github.com/runame/laplace-refinement/blob/308a3ff1f16b69dcb5bcea6ea302cf986e07350b/utils/data_utils.py#L605
    """
    if 'MLP' in model_class:
        tforms = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    else:
        tforms = transforms.ToTensor()

    train_set = datasets.MNIST(data_path, train=True, transform=tforms,
                               download=download)
    val_test_set = datasets.MNIST(data_path, train=False, transform=tforms,
                                  download=download)

    Xys = [train_set[i] for i in range(len(train_set))]
    Xs = torch.stack([e[0] for e in Xys]).to(device)
    ys = torch.stack([torch.tensor(e[1]) for e in Xys]).to(device)
    train_loader = FastTensorDataLoader(Xs, ys, batch_size=train_batch_size, shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader



class FastTensorDataLoader:
    """
    Source: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    and https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset = tensors[0]

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches