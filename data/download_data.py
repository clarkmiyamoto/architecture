from torchvision import datasets, transforms

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./', train=False, download=True, transform=transform)