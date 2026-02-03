import numpy as np

"""
Creating a perceptron object

Member functions(methods)
*   init - initializes weight and bias
*   sigmoid - sigmoid function logic
*   predict - summation funcation. i.e multiply the input features with weights and add with bias
*   fit - update of weight and bais (using stocastic gradient descent)
*   evaluate - evaluating the model results
"""

class SigmoidPerceptron:
  def __init__(self, input_size):
    # input_size = numbers of columns in the dataset
    self.weights = np.random.randn(input_size) # create a random weight array according to input size
    self.bias = np.random.randn(1)

  # returns the sigmoid value according to the formula 1/(1+e^-z)
  def sigmoid(self,z):
    return 1/(1+np.exp(-z))

  # calculates the weighted sum: (w1x1 + w2x2 .... +wnxn) + b [w1x1 + w2x2 .... can be calulated by using dot product]
  def predict(self, inputs):
    z = np.dot(inputs, self.weights) + self.bias
    return self.sigmoid(z)

  # Update the weights and bais (using stochastic gradient descent)
  def fit(self, inputs, targets, learning_rate=0.02, epochs=1):
    num_inputs = inputs.shape[0]
    
    # covering pandas series to numpy array
    targets=targets.values

    for epoch in range(epochs):
      for i in range(num_inputs):
        input_vector = inputs[i]
        target_value = targets[i]

        prediction = self.predict(input_vector)
        error = target_value - prediction

        # updating the weight
        gradient_weight = error * prediction * (1 - prediction) * input_vector     # calculating gradient of the loss = (\frac{\partial L}{\partial w}\)
        self.weights += learning_rate * gradient_weight  # wnew = wold - learning rate * gradient of loss

        # updating the bias
        gradient_bias = error * prediction * (1 - prediction)      # calculating gradient of error wrt bias = (\frac{\partial E}{\partial b}\)
        self.bias += learning_rate * gradient_bias  # bnew = bold - learning rate * gradient of error wrt bias

  def evaluate(self, inputs, targets):
    num_inputs = inputs.shape[0]
    correct_prediction = 0

    # covering pandas series to numpy array
    targets=targets.values

    for i in range(num_inputs):
      input_vector = inputs[i]
      target_value = targets[i]

      prediction = self.predict(input_vector)

      if prediction >= 0.5:
        predicted_class = 1
      else:
        predicted_class = 0

      if predicted_class==target_value:
        correct_prediction += 1

    accuracy = correct_prediction / num_inputs  # accuracy = num of correct predictions / total num of predictions or inputs
    return accuracy