import torch
import torch.nn.functional as F

def leaky_relu(x):
    alpha = 0.1
    x_pos = F.relu(x)
    x_neg = F.relu(-x)
    return x_pos - alpha * x_neg


def relu(x):
    return F.relu(x)


def sigmoid(x):
    return torch.sigmoid(x) - 0.5

