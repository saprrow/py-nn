#neural network class definition
import numpy as np
import scipy.special

class neuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
		self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
		self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
		
		hidden_inputs = np.dot(self.wih, inputs)
		self.activation_function = lambda x: scipy.special.expit(x)
		hidden_outputs = self.activation_function(hidden_inpits)
	pass

test =  neuralNetwork(3,3,3,0.5)
print(test.wih)
