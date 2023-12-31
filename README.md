<h1 align="center">
  <br>
    kittygrad <img src="extra/attachments/kitty.gif" width="25" height="25" />
  <br>
</h1>

<h4 align="center">A small NumPy-based deep learning library with PyTorch-like API</h4>

<p align="center">
    <img alt="Static Badge" src="https://img.shields.io/badge/version-1.0.0--alpha-orange">
    <img src="https://img.shields.io/github/languages/top/tejpaper/kittygrad">
</p>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a> •
  <a href="#details">Details</a> •
  <a href="#license">License</a>
</p>

## Description

kittygrad is a pet project that offers a familiar toolkit for training neural networks, written entirely in Python using [NumPy](https://numpy.org/). The project has the goal of building a minimal foundation for implementing basic popular deep learning techniques. Potentially, for those passing by, kittygrad could be useful for understanding some of the details in this area of machine learning. The concept of dynamic computational graph on tensors is taken as a basis. For simplicity the API is maximally close to [PyTorch](https://pytorch.org/).

## Features

At the moment the project is not finished and offers the following functionality:

-  Autograd engine (with 25+ operations)
-  Computational graph visualization 
-  Custom functions support
-  `no_grad` decorator and context-manager

## Installation

At the moment, the best way to install kittygrad is through this repository.

```bash
git clone https://github.com/tejpaper/kittygrad.git
cd kittygrad
pip install -r requirements.txt
```

### Running tests

Unit tests are run through [pytest](https://docs.pytest.org/en). To check the correctness of gradients, an element-wise comparison with tolerance to the results of the same operations in PyTorch is used (see ```Comparison``` class in ```test/conftest.py```).

As a consequence, these packages must be installed before running the tests:

```bash
pip install -r requirements/test.txt
```

Then simply:

```bash
pytest
```

## Examples

For a large number of examples, see the `examples` folder.

### Gradients

```python
import kittygrad as kitty

a = kitty.tensor([1, 2, 3], requires_grad=True)
b = a.std()
b.retain_grad()
c = b * kitty.dot(a, a)
print(c)  # tensor(14., grad_fn=<MulBackward>)

c.backward()
print(a.grad)  # tensor([-5.,  4., 13.])
print(b.grad)  # tensor(14.)
```

### Computational graph

```python
import kittygrad as kitty

a, b = kitty.randn(2, 2, 3)
a.requires_grad = True

with kitty.CompGraph() as dot:
    c = a + 42
    c **= 2
    _ = c * b
dot.view(cleanup=True)  # or just dot in case of jupyter cell
```
![graph example](extra/attachments/graph.svg)

The round shape of the node denotes an operation on tensors or gradient flow, the rectangular shape denotes the tensors themselves. Colors have the following interpretation:

- **yellow** means no relation to auto-differentiation,
- **green** color denotes tensors whose gradients we are interested in,
- **brown** denotes the part of forward computations for which the backward graph is created,
- all backward nodes are marked in **blue**.

Inspired from [this](https://www.youtube.com/watch?v=MswxJw-8PvE) YouTube video.

## Details

### Autograd

To properly handle edge cases and provide greater flexibility, kittygrad performs some preliminary steps before performing the operation on a tensor or calculating its gradient. Let's describe the whole process briefly.

#### Forward pass

Let's assume that we want to perform an operation on a tensor involving another tensor, a NumPy array, or a scalar. First, we call the frontend function (for example, this could be the dunder method `__mul__`). Next, if necessary, the two operands are cast into tensor form using the `autocast` decorator. If the operation is element-wise, broadcasting is also applied to align shapes. Then, within the frontend function, additional conditions are checked to determine the validity of the operation. If everything is OK, the backend function is called, passing an additional context variable to store intermediate values that may be needed in the backward pass. Subsequently, the necessary computation takes place, and depending on the `requires_grad` attribute of the resulting tensor, a backward node is created. The resulting tensor is returned with a reference to it.

The `autocast` decorator uses the `type` method and the `broadcast_tensors` function, both of which are also differentiable. Therefore, a single operation may generate multiple backward nodes simultaneously. More generally, this doesn't mean that complex operations are implemented using a set of simpler ones. For example, computing the variance does not create a **MeanBackward** node, even though the mean value is computed.

#### Backward pass

The computation of gradients begins with calling the `backward` method on the final tensor. Instead of using topological sorting and BFS, kittygrad traverses the backward graph using DFS and the concept of **locks**. The backward node of each computed tensor in the forward pass has a `_lock` attribute, indicating the number of operations in which this tensor participated. Thus, when an update to the gradient arrives at the node of this tensor, it accumulates until the lock value drops to zero. This allows simulating topological sorting without an additional graph traversal and, most importantly, enables using lock values to verify the correctness of the constructed computational graph and the `backward` method call itself. For example, kittygrad will promptly inform you if you initiate the process from the middle of the computational graph or if there are multiple final tensors awaiting the `backward` method call.

Backpropagation ends in the **AccumulateGrad** nodes, which transfer the accumulated gradients to the `grad` attribute of the tensor itself. If the tensor was not initialized by the user but was created during the forward pass, it won't receive its gradient. This can be bypassed by calling `retain_grad` method on it, which will force its backward node to also perform the **AccumulateGrad** functions.

#### Tensor version control

Another important aspect worth mentioning is the version control of tensors. In order for inplace operations on tensors to work correctly and without creating unnecessary copies of them, kittygrad assigns a zero version to each tensor, which increments if it is modified in-place (see `inplace` decorator). If such a tensor is retained in the context of a backward node and is needed for gradient computation, its version is compared to the one it had at the time of creating the backward node, and an exception is raised in case of an error.

Some operations do not create full-fledged tensors in computer memory but rather use another **view** on an existing data **storage**. In such cases, the resulting tensor shares its version with its predecessor. This is achieved by using the `share` decorator.

### To understand more

Above, I described the most crucial points that may not capture the entire picture. Therefore, for those interested, I provide the following links:

- [How it works in PyTorch](https://pytorch.org/docs/stable/notes/autograd.html)
- [micrograd](https://github.com/karpathy/micrograd/tree/master) and its awesome spelled-out [explanation](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)

## License

MIT