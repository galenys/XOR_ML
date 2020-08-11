import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

# TRAINING DATA
train_set = [] # (embed_vector_word, context_vector_word, similarity classification)
for i in range(10_000):
    x = random.choice([1, 0])
    y = random.choice([1, 0])
    train_set.append(
            ((float(x), float(y)),
                float(x ^ y))
            )

# MODEL
class XOR(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        return x
model = XOR()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

# LEARN
learning_rate = 1e-5
for epoch in range(10):
    print(f"Epoch {epoch + 1} in progress.")
    total_loss = 0
    for (i, ((x, y), xor)) in enumerate(train_set):
        loss = (xor - model(torch.tensor([x, y]))) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        if i%999==0:
            print(f"Average Loss: {total_loss/i}")

print("")
print("0 and 0 => ", round( model(torch.tensor([0.0, 0.0])).item() ))
print("0 and 1 => ", round( model(torch.tensor([0.0, 1.0])).item() ))
print("1 and 0 => ", round( model(torch.tensor([1.0, 0.0])).item() ))
print("1 and 1 => ", round( model(torch.tensor([1.0, 1.0])).item() ))

