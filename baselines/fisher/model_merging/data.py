import torch
import hydra
import torchvision
import pickle


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class MNIST:
    def __init__(self, cfg):
        self.cfg = cfg.data

    def load_mnist(self):
        # Load the MNIST dataset
        self.digits = self.cfg.digits
        dataset = torchvision.datasets.MNIST("./data/", train=True, download=True,
            transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        ReshapeTransform((-1,))
                    ]
                ))
        test_dataset = torchvision.datasets.MNIST("./data/", train=False, download=True,
            transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        ReshapeTransform((-1,))
                    ]
                ))

        # Filter the dataset to only include the desired digits
        filtered_dataset = [(x, y) for x, y in dataset if y in self.digits]
        filtered_test_dataset = [(x, y) for x, y in test_dataset if y in self.digits]

        # Create the train and test loaders
        self.train_loader = torch.utils.data.DataLoader(
            filtered_dataset,
            batch_size=self.cfg.batch_size_train,
            shuffle=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            filtered_test_dataset,
            batch_size=self.cfg.batch_size_test,
            shuffle=True
        )
    
    def create_dataloaders(self):
        self.load_mnist()
        return self.train_loader, self.test_loader


def store_fisher(fisher, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(fisher, f)


def load_fisher(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)