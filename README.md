# Perceptron Implementation from Scratch

### Overview
This project demonstrates a **from-scratch implementation of a binary classification Perceptron** using only **NumPy**.  
The goal is to deeply understand how a neural network works internally—without relying on high-level machine learning libraries.

Unlike framework-based implementations, this project manually handles:
- Weight and bias initialization
- Forward propagation
- Sigmoid activation
- Gradient-based learning
- Model evaluation

This repository is intended to showcase **core machine learning fundamentals**, clean code structure, and mathematical intuition.

---

### Key Features
- Fully custom Perceptron class
- Sigmoid activation function
- Stochastic Gradient Descent (SGD)
- Manual weight & bias updates
- Binary classification evaluation
- Lightweight and dependency-minimal

---

### Tech Stack
- **Python 3**
- **NumPy**
- **Jupyter Notebook**

---

### Model Architecture
The model is a **single-layer neural network** consisting of:
- Input layer
- Weighted summation
- Bias term
- Sigmoid activation
- Binary output (0 or 1)

Mathematically:

\[
z = w \cdot x + b
\]

\[
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

---

### Class Structure

#### `SigmoidPerceptron`
The perceptron is implemented as a reusable Python class with the following methods:

- `__init__`  
  Initializes weights and bias randomly based on input size.

- `sigmoid(z)`  
  Applies the sigmoid activation function.

- `predict(inputs)`  
  Computes the weighted sum and returns the activated output.

- `fit(inputs, targets, learning_rate, epochs)`  
  Trains the model using **stochastic gradient descent**.

- `evaluate(inputs, targets)`  
  Computes classification accuracy.

---

### Training Logic
- Loss is minimized using gradient descent
- Weights and bias are updated per sample (SGD)
- Uses sigmoid derivative for backpropagation
- Classification threshold set at `0.5`

---

### Example Usage
```python
model = SigmoidPerceptron(input_size=2)
model.fit(X_train, y_train, learning_rate=0.02, epochs=10)
accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
```

---

### Author
Name: Athul V Nair
Github: https://github.com/athul-v-nair