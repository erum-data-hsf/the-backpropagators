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

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
plt.figure()
plt.plot(x,x*x)
plt.show()
```


## relu


## tanh


## step_function


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
    return alpha*(np.exp(x)-1)

x = np.linspace(-10,10,100)
alpha = 2
plt.figure()
plt.plot(x,elu(x))
plt.show()
```

## SELU: Scaled Exponential Linear Unit
SELU helps keeping the output of each layer automatically normalized and thus stabilize the learning process.

$f(x) = \lambda\alpha(e^x-1)$ if x<0  
$f(x) = \lambda x$ if x>0

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
alpha = 1.6
lambda = 1.1
plt.figure()
plt.plot(x,alpha*np.exp(x)-1)
plt.show()
```

Drawbacks of SELU

- Requires careful weights initialization and the inputs to be standardized.
- Works mainly with fully connected (dense) layers. 
- Special Dropout Required: alpha dropout must be used because regular dropout disrupts SELU’s normalization benefits. 



## Mish

Mish exhibits a "self-regularizing" behavior attributed to a term in its first derivative

$ f ( x ) = {\displaystyle f(x)=x\tanh {\big (} ln(1+e^x) {\big )}}$

