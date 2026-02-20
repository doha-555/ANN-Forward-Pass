 
git clone https://github.com/doha-555/ANN-Forward-Pass.git
Execute:

Bash
python ann_model.py

📈 Future ISimple ANN Forward Pass Implementation
This repository contains a Python implementation of a basic Feedforward Artificial Neural Network (ANN). 
The project focuses on the Forward Pass mechanism, demonstrating how data flows from input layers through a hidden layer to produce final outputs using matrix multiplication and activation functions.
 Project OverviewThe
 goal of this assignment was to build a simple neural network architecture from scratch using NumPy,
 focusing on precise weight initialization and the application of the Hyperbolic Tangent activation function.
 🛠️ Technical SpecificationsArchitecture:
 2 Input neurons, 1 Hidden layer (2 neurons), and 2 Output neurons.Activation Function:
 Hyperbolic Tangent (tanh) used for all layers to introduce non-linearity.Weight Initialization:
 Randomized within the interval [-0.5, 0.5] to ensure a balanced starting point for the network.Biases: Fixed bias values of $b_1 = 0.5$ for the hidden layer and b_2 = 0.7 for the output layer.
  Mathematical LogicThe network 
  follows the standard forward propagation formulas:Net Input Calculation:$$net = \sum (weight \times input) + bias$$Activation:$$out = \tanh(net)
  
  How to Run
Prerequisites: Ensure you have Python and NumPy installed.

Bash
pip install numpy
Clone the Repository:

Bashmprovements
While this version covers the Forward Pass, the theoretical foundation for the Backwards Pass (Backpropagation) is documented in the project's logic to eventually allow the network to minimize error using the Delta Rule and Gradient Descent.
  
