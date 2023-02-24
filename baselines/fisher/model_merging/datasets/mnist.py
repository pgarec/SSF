import torch
import torchvision
import torchvision.transforms as transforms


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class MNIST:
    def __init__(self, cfg):
        self.cfg = cfg.data
        # transformation for the unbalanced classes
        self.transform_unbalanced = transforms.Compose([
            ReshapeTransform((28,28,1)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            ReshapeTransform((-1,)),
        ])
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ReshapeTransform((-1,))]
        )

    def create_inference_dataloader(self):
        self.classes = self.cfg.classes
        test_dataset = torchvision.datasets.MNIST(
            "./data/",
            train=False,
            download=True,
            transform=self.tranform
        )
        filtered_test_dataset = [(x, y) for x, y in test_dataset if y in self.classes]
        test_loader = torch.utils.data.DataLoader(
            filtered_test_dataset, batch_size=1, shuffle=True
        )

        return test_loader

    def load_mnist(self, unbalanced_classes=[]):
        # Load the MNIST dataset
        self.classes = self.cfg.classes
        dataset = torchvision.datasets.MNIST(
            "./data/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ReshapeTransform((-1,)),
                ]
            ),
        )
        test_dataset = torchvision.datasets.MNIST(
            "./data/",
            train=False,
            download=True,
            transform=self.transform
        )

        # Filter the dataset to only include the desired classes
        filtered_dataset = [(x, y) for x, y in dataset if y in self.classes]
        filtered_test_dataset = [(x, y) for x, y in test_dataset if y in self.classes]

        # Check if we need to add more data to specific classes
        if unbalanced_classes != []:
            # Keep track of the new filtered dataset
            new_filtered_dataset = []

            for digit in self.classes:
                if digit in unbalanced_classes:
                    # Load double data for this digit
                    additional_data = [
                        (x, y)
                        for x, y in filtered_dataset
                        if y == digit
                    ]
                    additional_data_transformed = [
                        (self.transform_unbalanced(x), y)
                        for x, y in filtered_dataset
                        if y == digit
                    ]
                    new_filtered_dataset.extend(additional_data) 
                    new_filtered_dataset.extend(additional_data_transformed) 
                else:
                    # Use the half of the data for this digit
                    l = [(x, y) for x, y in filtered_dataset if y == digit]
                    # new_filtered_dataset.extend([])
                    new_filtered_dataset.extend(l[:(int(len(l)/2))])

            filtered_dataset = new_filtered_dataset
            
        # Create the train and test loaders
        self.train_loader = torch.utils.data.DataLoader(
            filtered_dataset,
            batch_size=self.cfg.batch_size_train,
            shuffle=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            filtered_test_dataset, batch_size=self.cfg.batch_size_test, shuffle=True
        )

    def create_dataloaders(self, unbalanced=[]):
        self.load_mnist(unbalanced)

        return self.train_loader, self.test_loader
    