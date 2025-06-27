# This is my attempt at an MLP Autograd Network
# I followed Andrej Karpathy's "Zero to Hero" Neural Network's 1st video to build the structure
# I am trying to apply it to another dataset and see how it works.


import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


# This class represents a single number that remembers how it was created
# during calculations. It's the building block of our neural network and 
# helps us figure out how to improve the network by using gradients

class Value:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data # Stores the number
    self.grad = 0.0
    self._prev = set(_children) # keeps track of the numbers  used to calculate this one
    self._backward = lambda: None # A function to help move gradients backward through the graph
    self._op = _op # the type of operation that produced this number (like + or *)
    self.label = label # optional name for visualization
  
  # This allows to print the Value object nicely
  def __repr__(self):
    return f"Value(data={self.data})"
  
  # Allows for use of negative numbers
  def __neg__(self):
    return self * -1
  
  # Allows for teh addition of two Value numbers (x + y)
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
  
  # Allows to multiply two numbers (x * y)
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
  
  # Allows for the subtraction of two numbers (x - y)
  def __sub__(self, other):
    return self + (-other) 
  
  # Allows for raising to the power of numbers (x ** y)
  def __pow__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data ** other.data, (self, other), f'**')

    def _backward():
      self.grad += other.data * self.data ** (other.data - 1) * out.grad
    out._backward = _backward
    return out

  # Allows for the division of two numbers (x / y)
  def __truediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return self * other**-1

  # Allows for floor division (x // y) (drops the decimal)
  def __floordiv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data // other.data, (self, other), '//')
    return out  
  
  # Allows for the exponentiation (e^x)
  def exp(self):
    x = self.data
    out= Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out
  
  # This applies the tanh function, which helps squash numbers between -1 and 1
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t , (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out
  
  # The following allow for the reverse input of the previous operators
  def __rmul__(self, other):
    return self * other
  def __radd__(self, other):
    return self + other
  def __rsub__(self, other):
    return other + (-self)
  def __rtruediv__(self, other):
    return self / other
  def __rfloordiv__(self, other):
    return self // other
  
  # This function kicks off back propagation to figure out how each number contributed to the final result
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    topo

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()


# This helps to build out a visual map of the steps and computation needed 
# and how each value relates to the other
def trace(root):
  # This builds a set of all the nodes and edges in a graph
  nodes, edges = set(), set()

  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges


# draw_dot allows for the visualization of the neural network
def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)
  
  return dot

# A single artifical "neuron" that takes inputs, multiplies them by weights, adds the bias, and applies tanh activation
class Neuron:
    
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
# A layer of neurons. Each neuron looks at the same input and produces an output
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs 
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

# A full neural network made up of multiple layers (Hence the Multi Layer Perceptron name)
class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

if __name__ == "__main__":
   print("Micrograd MLP core loaded.")

