import torchvision

def load_data():
    train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_data = train_dataset.data.numpy()
    train_label = train_dataset.targets.numpy()
    test_data = test_dataset.data.numpy()
    test_label = test_dataset.targets.numpy()
    labels = train_dataset.classes
    return train_data, train_label, test_data, test_label, labels