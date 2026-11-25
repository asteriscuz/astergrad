import math 

class Value:
    
    """ stores a simgle scalar value and its gradient """
    
    def __init__(self, data, _children = (), _op = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # leaf Values have no parents so nothing to pass back
        self._prev = set(_children)
        self._op = _op
        
        
    def __repr__(self):
        return f"Value: {self.data}"
    
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad # += so grads add up when the value is used more than once
            other.grad += out.grad
        out._backward = _backward
        return out
    
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self
    
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward    
        return out
    
    
    def __rmul__(self, other):
        return self * other
    
    
    def relu(self):
        x = self.data
        out = Value(max(0, x), (self, ), 'relu')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1-(t**2)) * out.grad
        out._backward = _backward
        return out
    
    
    def exp(self):
        x = self.data
        t = math.exp(x)
        out = Value(t, (self, ), 'exp')
        
        def _backward():
            self.grad += t * out.grad
        out._backward = _backward
        return out
    
    
    def __pow__(self, e):
        assert isinstance(e, (float, int))
        x = self.data
        out = Value(x ** e, (self, ), f"**{e}")
        
        def _backward():
            self.grad += (e * (x**(e-1))) * out.grad
        out._backward = _backward
        return out
    
    
    def __truediv__(self, other):
        return self * other**(-1)
        
    def __rtruediv__(self, other):
        return other * self**(-1)
      
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v): # topological sort
            if v not in visited:
                visited.add(v)
                for inp in v._prev:
                    build_topo(inp)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
