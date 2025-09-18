# Welcome to our book on Activation Functions


Check out the content pages bundled with this sample book to see more.

```{tableofcontents}
```


## Questions and Objectives
At the end of this chapter (or after working through this set of notebooks), you should be able to answer these questions and meet these objectives.

**Learning Objectives**
By the time you finish, you will be able to:
- Explain the mathematical basis of activation functions — what they are, where they come from, and why they are used in neural networks.
- Distinguish between different activation functions (e.g. ReLU, Sigmoid, Tanh, Softmax) along dimensions such as differentiability, computational cost, saturation, and vanishing gradient behavior.
- Implement activation functions from scratch; compute their forward pass and derivatives.
Visualize the behavior of activation functions under different input ranges; know how choices affect network training in practice.
- Apply activation functions in neural network frameworks; experiment with different activation choices and observe their effects on learning.

## Key Questions
To check understanding, consider the following questions as you go through the material:
1. **What criteria make a good activation function?**
   For example: smoothness, monotonicity, boundedness, computational simplicity, etc.
2. **How do activation functions influence gradient flow?**
What happens in the forward pass, and how do those properties affect backpropagation and training stability?**
3. **In what scenarios might one activation function be preferred over another?**
Think about classification vs regression, deep vs shallow networks, computational constraints.
4. **What is the relationship between activation functions and learning speed/convergence?**
Can activation functions speed up or slow down learning? Why?
5. **Are there trade-offs involved in choosing activation functions?**
For instance, introducing non-linearity to gain expressivity vs potential issues like vanishing gradients or dead neurons.