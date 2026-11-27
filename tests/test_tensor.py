import numpy as np
import torch
from astergrad.tensor import Tensor


def test_linear_layer_matches_pytorch():
    x_np = np.random.randn(3, 2)
    w_np = np.random.randn(2, 4)
    b_np = np.random.randn(4)

    x, w, b = Tensor(x_np), Tensor(w_np), Tensor(b_np)
    (x @ w + b).sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    wt = torch.tensor(w_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    (xt @ wt + bt).sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(w.grad, wt.grad.numpy())
    assert np.allclose(b.grad, bt.grad.numpy())


def test_log_matches_pytorch():
    x_np = np.random.rand(3, 4) + 0.1     # positive only: log needs x > 0

    x = Tensor(x_np)
    x.log().sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    xt.log().sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy())
    
def test_mul_broadcast_matches_pytorch():
    x_np = np.random.randn(3, 2)
    s_np = np.random.randn(2)

    x, s = Tensor(x_np), Tensor(s_np)
    (x * s).sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    st = torch.tensor(s_np, requires_grad=True)
    (xt * st).sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(s.grad, st.grad.numpy())


def test_relu_matches_pytorch():
    x_np = np.random.randn(3, 2)
    w_np = np.random.randn(2, 4)

    x, w = Tensor(x_np), Tensor(w_np)
    (x @ w).relu().sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    wt = torch.tensor(w_np, requires_grad=True)
    (xt @ wt).relu().sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(w.grad, wt.grad.numpy())


def test_exp_matches_pytorch():
    x_np = np.random.randn(3, 4)

    x = Tensor(x_np)
    x.exp().sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    xt.exp().sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy())


def test_reshape_matches_pytorch():
    x_np = np.random.randn(2, 3)

    x = Tensor(x_np)
    (x.reshape((6,)) * 2).sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    (xt.reshape((6,)) * 2).sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy())


def test_max_matches_pytorch():
    x_np = np.random.randn(3, 4)

    x = Tensor(x_np)
    x.max(dim=1).sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    xt.max(dim=1).values.sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy())
    
def test_mean_matches_pytorch():
    x_np = np.random.randn(3, 4)

    x = Tensor(x_np)
    (x.mean(dim=1) ** 2).sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    (xt.mean(dim=1) ** 2).sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy())