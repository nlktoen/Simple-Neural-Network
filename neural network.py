import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input    = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y        = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find the derivation of the
        # loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) *
                     sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output)
                    * sigmoid_derivative(self.output), self.weights2.T) *
                      sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

#Create an 2D array to test our neural program:
if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)
    #An array 'times' is  no of times we feedforward and backprop the neural network
    times= [100,200,500,1000,1500]
    #An array 'accuracy' is the least squared error between prediction and actual values for plotting
    error=[]
    for i in times:
        for t in range(i):
            nn.feedforward()
            nn.backprop()
        error.append(np.sum(np.subtract(nn.output,y))**2)
        
    #Visualization the level of error while training neural network
    plt.plot(times,error)
    plt.ylabel('Least Squared Error')
    plt.xlabel('No of times')
    plt.show()



