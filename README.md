# MicroTorch

> A lightweight, zero-dependency deep learning framework built entirely from scratch in pure Python.

MicroTorch mirrors PyTorch's architectural design—`Module`, `Sequential`, `Linear`, `ReLU`, `Sigmoid`, `BCELoss`, `Adam`, and `DataLoader`—without external numerical or autograd libraries. Everything (forward execution, backpropagation gradients, and parameter updates) is implemented with pure Python primitives.

---

## Features

- **Pure Python**: Zero external dependencies (no `numpy`, no `torch`).
- **Explicit Backpropagation**: Full manual analytical gradient computation per module.
- **PyTorch-Compatible Paradigm**: 1-to-1 mapping with PyTorch concepts (`forward()`, `backward()`, `step()`, `zero_grad()`).
- **Modular Components**:
  - **Layers**: `Linear`, `Dropout`, `BatchNorm`
  - **Activations**: `ReLU`, `Sigmoid`, `Tanh`
  - **Containers**: `Sequential`
  - **Loss Functions**: `MSELoss`, `BCELoss`
  - **Optimizers**: `SGD`, `Adam`
  - **Data Handling**: Mini-batch `DataLoader` with shuffling

---

## Architecture Overview

```
                        +-------------------+
                        |      Module       |
                        +---------+---------+
                                  |
         +------------------------+------------------------+
         |                        |                        |
+--------v-------+       +--------v-------+       +--------v-------+
|  Linear Layer  |       |   Activations  |       |   Sequential   |
| (Weights, Grad)|       | (ReLU/Sigmoid) |       |  (Chain Exec)  |
+----------------+       +----------------+       +----------------+
```

---

## Quickstart & Usage

### 1. Installation

Clone the repository and install in editable mode:

```bash
git clone [https://github.com/mohamedMoutea/microtorch.git]
cd microtorch
pip install -e .
```

### 2. Basic Example

```python
from microtorch import Sequential, Linear, ReLU, Sigmoid, BCELoss, Adam

# Define a 4-layer MLP
model = Sequential(
    Linear(2, 16),
    ReLU(),
    Linear(16, 16),
    ReLU(),
    Linear(16, 8),
    ReLU(),
    Linear(8, 1),
    Sigmoid()
)

criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Forward pass
inputs = [0.5, -1.2]
target = [1.0]

pred = model.forward(inputs)
loss = criterion(pred, target)

# Backward pass & update
optimizer.zero_grad()
grad = criterion.backward()
model.backward(grad)
optimizer.step()

print(f"Loss: {loss:.6f}")
```

### 3. Run Circle Classification Demo

Run the non-linear classification script in the `examples/` directory:

```bash
python examples/circle_classification.py
```

---

## MicroTorch vs PyTorch Equivalence

| Component | MicroTorch | PyTorch |
| :--- | :--- | :--- |
| **Base Class** | `Module` | `torch.nn.Module` |
| **Linear Layer** | `Linear(fan_in, fan_out)` | `torch.nn.Linear(in_features, out_features)` |
| **Sequential Container** | `Sequential(*modules)` | `torch.nn.Sequential(*modules)` |
| **Loss** | `BCELoss()` | `torch.nn.BCELoss()` |
| **Optimizer** | `Adam(params, lr=0.01)` | `torch.optim.Adam(params, lr=0.01)` |
| **Execution Loop** | `zero_grad() -> backward() -> step()` | `zero_grad() -> backward() -> step()` |

---

## License

This project is open-source and available under the [MIT License](LICENSE).