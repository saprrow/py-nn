#neural network class definition
import numpy as np
import scipy.special
import matplotlib.pyplot as plot
import heapq

class neuralNetwork:
    #initialise the neural network
        def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
            #set number of nodes in each input, hidden, output layer
                self.inodes = inputnodes
                self.hnodes = hiddennodes
                self.onodes = outputnodes

                #link weight matrices, wih and who
                self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
                self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

                #set learning rate
                self.lr = learningrate

                #activation function
                self.activation_function = lambda x: scipy.special.expit(x)
                pass

        def train(self, input_lists, target_lists):
            #convert input array to 2d
                inputs = np.array(input_lists, ndmin=2).T
                targets = np.array(target_lists, ndmin=2).T

                #calculate signal into hidden layyer
                hidden_inputs = np.dot(self.wih, inputs)
                hidden_outputs = self.activation_function(hidden_inputs)
                final_inputs = np.dot(self.who, hidden_outputs)
                final_outputs = self.activation_function(final_inputs)

                output_errors = targets - final_outputs
                hidden_errors = np.dot(self.who.T, output_errors)

                #update matrices according to errors
                self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
                self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
                pass

        def query(self, input_list):
            inputs = np.array(input_list, ndmin=2).T #inputs' dim is 2, because it's a matrix. while list is a vector whose dim is 1, so use "ndmin =2" 
                hidden_outputs = np.dot(self.wih, inputs)
                final_inputs = np.dot(self.who, hidden_outputs)
                result = np.array(self.activation_function(final_inputs), ndmin=1).T
                result = result.tolist()[0]
                index_list = list(map(result.index, heapq.nlargest(3, result)))
                print(index_list)
                for i in index_list:
                    print("This number is {0} whose probablity is {1}.".format(i, round(result[i], 5)))

input_nodes = 28**2
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

#create instance of network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#load training data
with open("../database/mnist_train_100.txt",'r') as tdf:
    tdl = tdf.readlines()

#train the network
epochs = 5
#go through all records in the data training data set
for e in range(epochs):
    for record in tdl:
        all_values = record.split(',')
                #adjust the input
                inputs = np.asfarray(all_values[1:])/255.0*0.99 + 0.01
                #set target
                targets = np.zeros(output_nodes) + 0.01
                targets[int(all_values[0])] = 0.99
                n.train(inputs, targets)
                pass

test_file = open("../database/mnist_test_10.csv", 'r')
test_data = test_file.readlines()
test_file.close()

value = test_data[3].split(',')
print(value[0])
image_array	= np.asfarray(value[1:]).reshape((28,28))
plot.imshow(image_array, cmap='Greys', interpolation='None')
plot.show()

n.query((np.asfarray(value[1:]) / 255.0 * 0.99) + 0.01)






