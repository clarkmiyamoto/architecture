import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

from tqdm import tqdm
import json
import argparse

from model import MLP

path_MNIST = '../../data'


def train_mnist_model(model,
                      epochs=10,
                      batch_size=64,
                      learning_rate=0.001,
                      patience = 5,
                      tolerance = 1e-4 ,
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      seed=123):
    """
    Train a PyTorch model on the MNIST dataset for classification.

    Parameters:
    - model: PyTorch model to be trained.
    - epochs: Number of epochs to train for.
    - batch_size: Batch size for the DataLoader.
    - learning_rate: Learning rate for the optimizer.
    - patience: Number of epochs to wait for improvement.
    - tolerance: Minimum improvement in loss to reset the patience.
    - device: Device to train on ('cuda' or 'cpu').

    Returns:
    - model: Trained PyTorch model.
    """
    torch.manual_seed(seed)

    # Set the device
    model = model.to(device)

    # Define transformations for the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with mean and std of MNIST dataset
        lambda x: x.squeeze(0)
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root=path_MNIST, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=path_MNIST, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping 
    best_val_loss = float('inf')
    counter = 0
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average epoch loss
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")

        ### Early Stopping Logic
        model.eval()

        # Validation loss
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

        # Early stopping logic with tolerance
        improvement = best_val_loss - val_loss
        if improvement > tolerance:  # Improvement must exceed tolerance
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Validation loss improved by {improvement:.6f}. Model saved.")
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience} (Improvement: {improvement:.6f})")
            if counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    ### Final Evaluation
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy_train = 100 * correct / total
    print(f"Accuracy on train set: {accuracy_train:.2f}%")

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy_test = 100 * correct / total
    print(f"Accuracy on test set: {accuracy_test:.2f}%")

    return model, accuracy_train, accuracy_test

def parse_args():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--nodes', type=int, required=True, help="Num of nodes per layer (uniform)")
    parser.add_argument('--layers', type=int, required=True, help="Num of layers")

    args = parser.parse_args()
    return args

def generate_layers(nodes, layers):
    return [nodes for i in range(layers)]

if __name__ == '__main__':
    print('Parsing args')
    args = parse_args()
    nodes = args.nodes
    layers = args.layers
    print('Finish parsing args')
    
    # Parameters
    layer_width = generate_layers(nodes, layers)
    epochs = 50
    batch_size = 8
    learning_rate = 1e-3
    patience = 3
    tolerance = 1e-2 # Should be an order higher than learning rate

    seed = 123
    
    # Run simulation
    print('Initalizing model')
    model = MLP(layer_width)
    print('Finish initalizing model')

    print('Training')
    model_trained, accuracy_train, accuracy_test = train_mnist_model(
                      model = model,
                      epochs = epochs,
                      batch_size = batch_size,
                      learning_rate = learning_rate,
                      patience = patience,
                      tolerance = tolerance,
                      seed = seed
        )
    print('Finished training')

    print('Saving')
    save_path = './data/'
    name = f'_nodes{nodes}_layers_{layers}'
    torch.save(model_trained.state_dict(), save_path + "model" + name + ".pth")

    data = {
        "result" : {
            "accuracy_train_pct" : accuracy_train, 
            "accuracy_test_pct": accuracy_test,
        },
        "model" : {
            "nodes" : nodes,
            "layers" : layers
        },
        "parameters" : {
            "epochs" :  epochs,
            "batch_size" : batch_size,
            "learning_rate" : learning_rate,
            "patience" : patience,
            "tolerance" : tolerance,
            "seed" : seed
        },
    }
    with open(save_path + "data" + name + ".json", "w") as f:
        json.dump(data, f)
    print('Done!')

