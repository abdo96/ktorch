# Ktorch: PyTorch Backend for Keras

##### Still WIP
------

## Internal topology library

### Imperative

```python
from ktorch import *
import numpy as np

a = Variable(np.zeros((2, 3, 4)))
b = Variable(np.ones((3, 4)))
c = a + 0.2 + b * 0.3
print c
'''
<ktorch.graph.tensor.Tensor object at 0x0000000003E82DA0>
'''
print c.value
'''
[[[ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]]

 [[ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]]]
'''

```

### Symbolic

```python
from ktorch import *
import numpy as np

a = Tensor()
b = Tensor()
c = a + 0.2 + b * 0.3
f = Function([a, b], c)

x = np.zeros((2, 3, 4))
y = np.ones((3, 4))

print f([x, y])[0]  # Function returns a list

'''
[[[ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]]

 [[ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]]]
'''
```


Note that evaluation is greedy. The value of a tensor is computed the instant all the information required to compute it is available. The value will be cached in the `.value` attribute of the tensor. You can explicitly set the value for an input tensor using the `.set_value()` method, and all the tensors in the graph depending on that input will be updated in real time.

```python
from ktorch import *
import numpy as np

a = Tensor()
b = Tensor()
c = Tensor()
d = a + b * c
print d.value
'''
AttributeError: 'Tensor' object has no attribute 'value'
'''
#Obviously, because we haven't set values for a, b and c
a.set_value(5)
b.set_value(3)
c.set_value(2)
print d.value
'''
11
'''
'''
Change the value for any of the inputs, and value of d will be automatically updated:
'''
c.set_value(4)
print d.value
'''
17
'''
```


## Working with Keras

* Use `ktorch` branch of my fork

```python
git clone http://www.github.com/farizrahman4u/keras.git
cd keras
git checkout ktorch
python setup.py install
```

* Make sure `import keras` prints `Using Torch backend`

* This allows using if statements, loops etc in custom layers (order of imports is important):

```
from keras.layers import *
from ktorch import dynamic_graph
dynamic_graph.initialize(globals())
```






