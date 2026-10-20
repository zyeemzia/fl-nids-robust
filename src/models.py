import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class MLP(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


def get_vec(m):
    return parameters_to_vector(m.parameters()).detach().clone()


def set_vec(m, v):
    vector_to_parameters(v.clone(), m.parameters())


def evaluate(m, X, y):
    m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(X)).argmax(1).numpy()
    return dict(accuracy=float(accuracy_score(y, pred)),
                f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)))
