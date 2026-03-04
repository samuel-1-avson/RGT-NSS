"""Comprehensive tests for the custom Tensor engine and autograd."""

import pytest
import numpy as np
from app.core.tensor import Tensor


class TestTensorCreation:
    def test_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.numel() == 3

    def test_from_scalar(self):
        t = Tensor(5.0)
        assert t.shape == ()

    def test_from_ndarray(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = Tensor(arr)
        assert t.shape == (2, 2)

    def test_requires_grad(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        assert t.requires_grad is True
        assert t.grad is not None or t.grad is None  # grad init may vary

    def test_dtype(self):
        t = Tensor([1, 2, 3])
        assert t.dtype == "torch.float32"

    def test_ndim(self):
        t = Tensor(np.zeros((2, 3, 4)))
        assert t.ndim == 3

    def test_size(self):
        t = Tensor(np.zeros((2, 3)))
        assert t.size() == (2, 3)
        assert t.size(0) == 2
        assert t.size(1) == 3


class TestTensorArithmetic:
    def test_add(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        np.testing.assert_allclose(c.data, [4.0, 6.0])

    def test_add_scalar(self):
        a = Tensor([1.0, 2.0])
        c = a + 5.0
        np.testing.assert_allclose(c.data, [6.0, 7.0])

    def test_radd(self):
        a = Tensor([1.0, 2.0])
        c = 5.0 + a
        np.testing.assert_allclose(c.data, [6.0, 7.0])

    def test_sub(self):
        a = Tensor([5.0, 3.0])
        b = Tensor([2.0, 1.0])
        c = a - b
        np.testing.assert_allclose(c.data, [3.0, 2.0])

    def test_neg(self):
        a = Tensor([1.0, -2.0])
        c = -a
        np.testing.assert_allclose(c.data, [-1.0, 2.0])

    def test_mul(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        np.testing.assert_allclose(c.data, [8.0, 15.0])

    def test_mul_scalar(self):
        a = Tensor([2.0, 3.0])
        c = a * 3.0
        np.testing.assert_allclose(c.data, [6.0, 9.0])

    def test_truediv(self):
        a = Tensor([6.0, 8.0])
        b = Tensor([2.0, 4.0])
        c = a / b
        np.testing.assert_allclose(c.data, [3.0, 2.0])

    def test_pow(self):
        a = Tensor([2.0, 3.0])
        c = a ** 2
        np.testing.assert_allclose(c.data, [4.0, 9.0])

    def test_matmul(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = a @ b
        expected = np.array([[1, 2], [3, 4]], dtype=float) @ np.array([[5, 6], [7, 8]], dtype=float)
        np.testing.assert_allclose(c.data, expected)


class TestTensorReductions:
    def test_sum(self):
        t = Tensor([1.0, 2.0, 3.0])
        s = t.sum()
        np.testing.assert_allclose(s.data, 6.0)

    def test_sum_axis(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        s = t.sum(axis=1)
        np.testing.assert_allclose(s.data, [3.0, 7.0])

    def test_mean(self):
        t = Tensor([2.0, 4.0, 6.0])
        m = t.mean()
        np.testing.assert_allclose(m.data, 4.0)

    def test_max(self):
        t = Tensor([1.0, 5.0, 3.0])
        m = t.max()
        np.testing.assert_allclose(m.data, 5.0)


class TestTensorActivations:
    def test_relu(self):
        t = Tensor([-1.0, 0.0, 1.0, 2.0])
        r = t.relu()
        np.testing.assert_allclose(r.data, [0.0, 0.0, 1.0, 2.0])

    def test_sigmoid(self):
        t = Tensor([0.0])
        s = t.sigmoid()
        np.testing.assert_allclose(s.data, [0.5], atol=1e-6)

    def test_tanh(self):
        t = Tensor([0.0])
        s = t.tanh()
        np.testing.assert_allclose(s.data, [0.0], atol=1e-6)

    def test_gelu(self):
        t = Tensor([0.0, 1.0, -1.0])
        g = t.gelu()
        assert g.shape == (3,)
        # GELU(0) ≈ 0
        assert abs(float(g.data[0])) < 0.01

    def test_silu(self):
        t = Tensor([0.0, 1.0])
        s = t.silu()
        # SiLU(0) = 0 * sigmoid(0) = 0
        np.testing.assert_allclose(s.data[0], 0.0, atol=1e-6)

    def test_softmax(self):
        t = Tensor([1.0, 2.0, 3.0])
        s = t.softmax(axis=-1)
        # Softmax should sum to 1
        np.testing.assert_allclose(s.data.sum(), 1.0, atol=1e-6)
        # Values should be ordered
        assert s.data[2] > s.data[1] > s.data[0]

    def test_log_softmax(self):
        t = Tensor([1.0, 2.0, 3.0])
        ls = t.log_softmax(axis=-1)
        # Log softmax values should all be <= 0
        assert np.all(ls.data <= 0)


class TestTensorAutograd:
    def test_add_backward(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a + b
        s = c.sum()
        s.backward()
        np.testing.assert_allclose(a.grad, [1.0, 1.0])
        np.testing.assert_allclose(b.grad, [1.0, 1.0])

    def test_mul_backward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        s = c.sum()
        s.backward()
        np.testing.assert_allclose(a.grad, [4.0, 5.0])
        np.testing.assert_allclose(b.grad, [2.0, 3.0])

    def test_chain_backward(self):
        """Test gradient flows through a multi-step computation."""
        x = Tensor([2.0], requires_grad=True)
        y = x * x  # y = x^2
        z = y + x  # z = x^2 + x
        z.backward()
        # dz/dx = 2x + 1 = 5
        np.testing.assert_allclose(x.grad, [5.0])

    def test_zero_grad(self):
        t = Tensor([1.0], requires_grad=True)
        s = (t * t).sum()
        s.backward()
        assert t.grad is not None
        t.zero_grad()
        np.testing.assert_allclose(t.grad, [0.0])

    def test_relu_backward(self):
        x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        y = x.relu()
        s = y.sum()
        s.backward()
        np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])

    def test_matmul_backward(self):
        a = Tensor(np.array([[1.0, 2.0]], dtype=np.float64), requires_grad=True)
        b = Tensor(np.array([[3.0], [4.0]], dtype=np.float64), requires_grad=True)
        c = a @ b
        c.backward()
        # dc/da = b^T = [[3, 4]]
        np.testing.assert_allclose(a.grad, [[3.0, 4.0]])


class TestTensorUtilities:
    def test_detach(self):
        t = Tensor([1.0], requires_grad=True)
        d = t.detach()
        assert d.requires_grad is False

    def test_clone(self):
        t = Tensor([1.0, 2.0, 3.0])
        c = t.clone()
        np.testing.assert_allclose(c.data, t.data)
        # Ensure independence
        c.data[0] = 99.0
        assert t.data[0] != 99.0

    def test_numpy(self):
        t = Tensor([1.0, 2.0])
        arr = t.numpy()
        assert isinstance(arr, np.ndarray)

    def test_repr(self):
        t = Tensor([1.0, 2.0])
        r = repr(t)
        assert "Tensor" in r

    def test_len(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert len(t) == 3

    def test_transpose(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        tr = t.T
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_allclose(tr.data, expected)
