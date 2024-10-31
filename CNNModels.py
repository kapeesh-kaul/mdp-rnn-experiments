import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a simple single-layer CNN
class SingleLayerCNN(nn.Module):
    def __init__(self):
        super(SingleLayerCNN, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1)  # 5x5 convolution
        self.fc = nn.Linear(10 * 24 * 24, 10)  # Adjusted output shape for 28x28 input
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return F.softmax(x, dim=1)  # Softmax output layer
    
class TwoLayerCNNWithMaxPooling(nn.Module):
    def __init__(self):
        super(TwoLayerCNNWithMaxPooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)  # First convolution
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)  # Second convolution
        self.pool = nn.MaxPool2d(4, stride=4)  # Max pooling layer with 4x4 kernel and stride 4
        self.fc = nn.Linear(20 * 5 * 5, 10)  # Adjusted for output after pooling

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return F.softmax(x, dim=1)  # Softmax output layer
    
def plot_comparison(output1, output2, title):
    # Convert tensors to numpy arrays for plotting if necessary
    if isinstance(output1, torch.Tensor):
        output1 = output1.detach().numpy().flatten()
    if isinstance(output2, torch.Tensor):
        output2 = output2.detach().numpy().flatten()
    
    # Plot the outputs as bar graphs
    indices = range(len(output1))
    width = 0.35  # Width of the bars

    plt.figure(figsize=(10, 5))
    plt.bar(indices, output1, width=width, label='Original Image Output')
    plt.bar([i + width for i in indices], output2, width=width, label='Translated Image Output')
    plt.xlabel('Output Neuron Index')
    plt.ylabel('Activation')
    # plt.ylim(0,1)
    plt.title(title)
    plt.legend()
    plt.show()