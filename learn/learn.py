#neural network class definition
import numpy as np
import scipy.special

class neuralNetwork:
        #initialise the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
            # set number of nodes in each input, hidden, output layer
	    self.inodes = inputnodes
	    self.hnodes = hiddennodes
	    self.onodes = outputnodes
                
            #link weight matrices, wih and who
	    self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
	    self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
	        
            #set learning rate
	    self.lr = learningrate
	        
            #activation function	
	    self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    def train(self, input_lists, target_lists):
        #convert input array to 2d 
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
            
        #calculate signal into hidden layyer
        hidden_inputs = np.dot(self.wih, inputs)
	    hidden_outputs = self.activation_function(hidden_inpits)
        final_inputs = np.dot(slef.who, hidden_outputs)
        fianl_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
            
        #update matrices according to errors
        self.who += self.lr * np.dot((outpus_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    
    def query(self, input_list):
        inputs = np.array(input_list, ndmin-2).T
        hidden_outputs = numpy.dot(self.wih, inputs)
        final_inputs = numpy.dot(slef.who, hidden_outputs)

        return self.activation_function(final_inputs)
    pass
test =  neuralNetwork(3,3,3,0.5)
print(test.wih)
