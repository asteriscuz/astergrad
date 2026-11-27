import pytest
import torch
from astergrad.value import Value


def t(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


def test_add_mul_matches_pytorch():
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    L = (a * b + c) * f
    L.backward()

    at, bt, ct, ft = t(2.0), t(-3.0), t(10.0), t(-2.0)
    Lt = (at * bt + ct) * ft
    Lt.backward()

    assert L.data == pytest.approx(Lt.item())
    for v, vt in [(a, at), (b, bt), (c, ct), (f, ft)]:
        assert v.grad == pytest.approx(vt.grad.item())


def test_sub_div_relu_matches_pytorch():
    a, b, c, f = Value(3.0), Value(5.0), Value(6.0), Value(8.0)
    L = (((2 - a) + b) / c).relu() * f
    L.backward()

    at, bt, ct, ft = t(3.0), t(5.0), t(6.0), t(8.0)
    Lt = (((2 - at) + bt) / ct).relu() * ft
    Lt.backward()

    assert L.data == pytest.approx(Lt.item())
    for v, vt in [(a, at), (b, bt), (c, ct), (f, ft)]:
        assert v.grad == pytest.approx(vt.grad.item())


def test_tanh_exp_pow_matches_pytorch():
    x = Value(0.5)
    y = x.tanh() + x.exp() + x ** 3
    y.backward()

    xt = t(0.5)
    yt = xt.tanh() + xt.exp() + xt ** 3
    yt.backward()

    assert y.data == pytest.approx(yt.item())
    assert x.grad == pytest.approx(xt.grad.item())
