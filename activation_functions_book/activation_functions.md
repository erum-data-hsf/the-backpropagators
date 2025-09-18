---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
--- 

# Activation Functions

## sigmoid

Maps real values to (0, 1) with an S-shaped curve:  

$ \sigma(x)=\frac{1}{1+e^{-x}}$.

Useful for probabilities in binary classification. Can saturate for large $|x|$ which slows learning due to small gradients.

```{code-cell} ipython3
:tags: [hide-input] 

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-6, 6, 1000)
plt.figure()
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)
plt.show()
```


## relu

Rectified Linear Unit: $\text{ReLU}(x)=\max(0,x)$.
Simple, fast, and helps mitigate vanishing gradients for $x>0$. 

Downsides: “dying ReLUs” where neurons stuck at $x<0$ output 0.

```{code-cell} ipython3
:tags: [hide-input] 

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

plt.figure()
plt.plot(x, relu(x))
plt.title("ReLU")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid(True)
plt.show()
```

## tanh

Hyperbolic tangent squashes to $(-1,1)$: $\tanh(x)$.
Zero-centered (often trains better than sigmoid). 

Still saturates for large $|x|$ which can reduce gradients.

```{code-cell} ipython3
:tags: [hide-input] 

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.plot(x, np.tanh(x))
plt.title("tanh")
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.grid(True)
plt.show()
```


## step_function

Heaviside step with threshold at 0: outputs 0 for $x<0$, 1 for $x\ge 0$.
Historically used in perceptrons; not used for gradient-based learning because it’s non-differentiable and has zero gradient almost everywhere.


```{code-cell} ipython3
:tags: [hide-input] 

import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    y = np.zeros_like(x)
    y[x >= 0] = 1.0
    return y

plt.figure()
plt.plot(x, step_function(x))
plt.title("Step Function (threshold = 0)")
plt.xlabel("x")
plt.ylabel("H(x)")
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()
```



## Softmax

### Definition

Converts a tuple of K real numbers into a probability distribution. 

$ \sigma{(z)_{i}}=e^{z_{i}} / \sum_{j=1}^{K}e^{z_{j}}$

The softmax applies the standard exponential function to each element $z_{i}$ of the input tuple z and normalizes these values by dividing by the sum of all these exponentials.


```{code-cell} ipython3
import numpy as np
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

scores = [3.0, 1.0, 0.2]
print(scores)
print(softmax(scores))
```

### Use

Softmax is exclusively used in the output layer of a network in a multi-class classification problem. 

## ELU: Exponential Linear Unit

ELU is an advanced version of ReLU. But before understanding ELU it's important to recognize the shortcomings of ReLU and Leaky ReLU activation function

$f(x) = x$ if x > 0  
$f(x) = \alpha(e^{x}-1)$   if x < 0

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha):
    if x < 0:
        return alpha*(np.exp(x)-1)
    return x

X = np.linspace(-10,10,100)
alpha = 2
y = [elu(x, alpha) for x in X]
plt.figure()
plt.plot(X,y)
plt.show()
```

## SELU: Scaled Exponential Linear Unit
SELU helps keeping the output of each layer automatically normalized and thus stabilize the learning process.

$f(x) = \lambda\alpha(e^x-1)$ if x<0  
$f(x) = \lambda x$ if x>0

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def selu(x, alpha, mylambda):
    if x < 0:
        return mylambda*alpha*(np.exp(x)-1)
    return mylambda*x

X = np.linspace(-10,10,100)
alpha = 1.6
mylambda = 1.1
y = [selu(x, alpha, mylambda) for x in X]
plt.figure()
plt.plot(X,y)
plt.show()
```

Drawbacks of SELU

- Requires careful weights initialization and the inputs to be standardized.
- Works mainly with fully connected (dense) layers. 
- Special Dropout Required: alpha dropout must be used because regular dropout disrupts SELU’s normalization benefits. 



## Mish

Mish exhibits a "self-regularizing" behavior attributed to a term in its first derivative

$ f ( x ) = {\displaystyle f(x)=x\tanh {\big (} ln(1+e^x) {\big )}}$

