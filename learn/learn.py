#neural network class definition
import numpy as np

class neuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
		self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
		self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
	pass

test =  neuralNetwork(3,3,3,0.5)
print(test.wih)
