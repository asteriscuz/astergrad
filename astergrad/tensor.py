import numpy as np

def unbroadcast(grad, target_shape):
    if grad.shape == target_shape: # if nothing was stretched then do nothing
        return grad
    
    
    ndim_grad = grad.ndim
    ndim_shape = len(target_shape)
    
    # sum the extra leading axis
    new_grad = grad.sum(axis = tuple(range(ndim_grad - ndim_shape)), keepdims = False)
    
    # sum the axis that was stretched
    axes_to_sum = tuple((np.array(target_shape) != np.array(new_grad.shape)).nonzero()[0])
    new_grad = new_grad.sum(axis = axes_to_sum, keepdims = True)
    return new_grad


class Tensor:
    def __init__(self, data, parents = ()):
        self.data = np.array(data, dtype = float)
        self.ndim = self.data.ndim
        self.shape = self.data.shape
        
        # adding gradient tracking:
        self.grad = np.zeros(self.shape, dtype = float)
        self.grad_fn = lambda: None  #leaf nodes have no parents to pass the gradients to 
        self.parents = parents
        
        
    def __repr__(self):
        return f"Tensor: {self.data}"
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, parents = (self, other))
        
        def grad_fn():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)
        out.grad_fn = grad_fn
        return out
        
       
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * (-1)
        
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * other**(-1)
    
    def __rtruediv__(self, other):
        return other * self**(-1)
    
    def item(self): # returns a one number tensor as a plain number
        return self.data.item()
    
    def detach(self): # returns a copy of the tensor with no gradients
        return Tensor(self.data) 
    
    
    def sum(self, dim = None, keepdim = False):
        out = Tensor(self.data.sum(axis = dim, keepdims = keepdim), parents = (self,))
        
        def grad_fn():
            grad = out.grad
            if dim is not None and not keepdim:
                if isinstance(dim, int):
                    grad = np.expand_dims(grad, dim)
                else:
                    for ax in sorted(dim):
                        grad = np.expand_dims(grad, ax)
                        
            self.grad += np.broadcast_to(grad, self.shape)
        out.grad_fn = grad_fn
        return out
            
    def mean(self, dim = None, keepdim = False):
        if dim is None:
            self.data.size
        elif isinstance(dim, int):
            n = self.shape[dim]
        else:
            n = 1
            for d in dim:
                n *= self.shape[d]
        return self.sum(dim = dim, keepdim = keepdim) / n
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, parents = (self,other))
        
        def grad_fn():
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)
        out.grad_fn = grad_fn
        return out
    
    def __rmul__(self, other):
        return self * other
        
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, parents = (self, other))
        
        def grad_fn():
            self.grad += out.grad @ other.data.swapaxes(-2, -1)
            other.grad += self.data.swapaxes(-2,-1) @ out.grad
        out.grad_fn = grad_fn
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), parents = (self,)) 
        
        def grad_fn():
            self.grad += (out.data > 0) * out.grad
        out.grad_fn = grad_fn
        return out
        
    def __pow__(self, e):
        assert isinstance(e, (int, float))
        x = self.data
        out = Tensor(x ** e, parents = (self,))
        
        def grad_fn():
            self.grad += (e * x**(e-1)) * out.grad
        out.grad_fn = grad_fn
        return out
    
    def exp(self):
        x = self.data
        t = np.exp(x)
        out = Tensor(t, parents = (self,))
        
        def grad_fn():
            self.grad += out.grad * t
        out.grad_fn = grad_fn
        return out
    
    def log(self):
        x = self.data
        out = Tensor(np.log(x), parents = (self,))
        
        def grad_fn():
            self.grad += (1/x) * out.grad
        out.grad_fn = grad_fn
        return out
    
    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), parents = (self,))
        
        def grad_fn():
            self.grad += out.grad.reshape(self.shape)
        out.grad_fn = grad_fn
        return out
    
    
    def max(self, dim=None, keepdim=False):
        out = Tensor(self.data.max(axis=dim, keepdims=keepdim), parents=(self,))

        def grad_fn():
            # mask: 1 where the element is the max, 0 elsewhere
            max_vals = self.data.max(axis=dim, keepdims=True)    # keepdims so it lines up with self.data
            mask = (self.data == max_vals).astype(float)
            mask /= mask.sum(axis=dim, keepdims=True)            # if there's a tie, split the gradient equally

            # put the squashed axis back
            grad = out.grad
            if dim is not None and not keepdim:
                if isinstance(dim, int):
                    grad = np.expand_dims(grad, dim)
                else:
                    for ax in sorted(dim):
                        grad = np.expand_dims(grad, ax)

            self.grad += mask * grad
        out.grad_fn = grad_fn
        return out

    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for inp in v.parents:
                    build_topo(inp)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        
        for node in reversed(topo):
            node.grad_fn()