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

# Mathematical Foundation
The activation functions are in general used for adding non-linearity to the neural networks.
However, before we dive into the non-linear functions, let's first understand what a linear function is.

A linear function is in general defined as $f(x) = ax + b$, where $a$ and $b$ are constants. The graph of a linear function is a straight line, and it exhibits a constant rate of change. Linear functions satisfy the properties of additivity and homogeneity, which means that the function's output is directly proportional to its input. In the context of neural networks, if we were to use only linear activation functions, the entire network would essentially behave like a single linear transformation, regardless of the number of layers. This would limit the network's ability to model complex relationships in the data. A simple example of a linear activation function is the line y = x, as shown in the figure below.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
y = x
plt.plot(x, y)
plt.title("Linear Activation Function: y = x")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()
```

A non-linear function, on the other hand, is a function that does not satisfy the properties of linearity.
A general non-linear function can be defined as $f(x) = ax^n + bx^{n-1} + ... + k$, where $n$ is a positive integer greater than 1, and $a, b, ..., k$ are constants.
Non-linear functions can take various forms, such as polynomial, exponential, logarithmic, and trigonometric functions, which can generally be expressed as polynomials of various orders with a taylor series expansion.
In neural networks, non-linear activation functions are crucial because they allow the network to learn complex patterns and relationships in the data.
Examples of non-linear activation functions are shown in the figure below.
```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 100)
y1 = 1 / (1 + np.exp(-x))  # Sigmoid
y2 = np.tanh(x)             # Tanh
y3 = np.maximum(0, x)      # ReLU
plt.plot(x, y1, label="Sigmoid")
plt.plot(x, y2, label="Tanh")
plt.plot(x, y3, label="ReLU")
plt.title("Non-linear Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.show()
```

Detailed descriptions of common non-linear activation functions used in neural networks are given in the next sections
