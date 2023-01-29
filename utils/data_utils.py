import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils


def get_mnist_loaders(
    data_path,
    digits,
    batch_size=512,
    model_class="MLP",
    train_batch_size=128,
    val_size=2000,
    download=False,
    device="cpu",
):
    """
    Source: https://github.com/runame/laplace-refinement/blob/308a3ff1f16b69dcb5bcea6ea302cf986e07350b/utils/data_utils.py#L605
    """
    if "MLP" in model_class:
        tforms = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    else:
        tforms = transforms.ToTensor()

    train_set = datasets.MNIST(data_path, train=True, transform=tforms, download=download)
    val_test_set = datasets.MNIST(data_path, train=False, transform=tforms, download=download)

    Xys = [train_set[i] for i in range(len(train_set))]
    Xs = torch.stack([e[0] for e in Xys]).to(device)  # images
    ys = torch.stack([torch.tensor(e[1]) for e in Xys]).to(device)  # label
    train_loader = FastTensorDataLoader(Xs, ys, batch_size=train_batch_size, shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set, batch_size=batch_size, val_size=val_size)

    return train_loader, val_loader, test_loader


def val_test_split(dataset, val_size=5000, batch_size=512, num_workers=0, pin_memory=False):
    # Split into val and test sets
    test_size = len(dataset) - val_size
    dataset_val, dataset_test = data_utils.random_split(
        dataset,
        (val_size, test_size),
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = data_utils.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = data_utils.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return val_loader, test_loader


# https://discuss.pytorch.org/t/missing-reshape-in-torchvision/9452/6
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class FastTensorDataLoader:
    """
    Source: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    and https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset = tensors[0]
        self.image_shape = tensors[0].shape[1:]

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
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
