import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set seeds. Do not touch
np.random.seed(77) 
torch.manual_seed(77)

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#sigmoid derivative function
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))  

class MLP():
    # Constructor
    def __init__(self, inputLsize, hiddenLsize, outputLsize, learning_rate):
        self.inputLsize = inputLsize
        self.hiddenLsize = hiddenLsize
        self.outputLsize = outputLsize
        self.learning_rate = learning_rate
        # initialize weights and biases
        self.w1 = 0.01 * np.random.randn(self.inputLsize, self.hiddenLsize)
        self.b1 = np.zeros((1, self.hiddenLsize))
        self.w2 = 0.01 * np.random.randn(self.hiddenLsize, self.outputLsize)
        self.b2 = np.zeros((1, self.outputLsize))
        # arrays to store loss and accuracy
        self.lossArr = []
        self.accuracyArr = []

    def forward(self, x):
        '''
        Forward pass.
        Set z1, a1, z2, a2
        '''
        self.a1 = np.dot(x, self.w1) + self.b1
        #self.f1 = sigmoid(self.a1) #sigmoid
        self.f1 = self.a1 * (self.a1 >= 0) #relu
        self.a2 = np.dot(self.f1, self.w2) + self.b2

    def backward(self, x, y_onehot):
        '''
        Backward pass.
        Implement backward pass for the 3-layer network. 
        Backpropagate Output layer --> Hidden layer and 
                      Hidden Layer --> Input Layer
        '''
        self.d_a2 = (self.a2 - y_onehot) / 100
        self.d_w2 = np.dot(self.f1.T, self.d_a2)
        self.d_f1 = np.dot(self.d_a2, self.w2.T)
        #self.d_a1 = self.d_f1 * sigmoid_deriv(self.f1) # sigmoid
        self.d_a1 = self.d_f1 * (self.a1 >= 0) # relu
        self.d_w1 = np.dot(x.T, self.d_a1)

    def step(self):
        '''
        Update weights and biases
        '''
        self.w1 -= self.learning_rate * self.d_w1
        self.b1 -= self.learning_rate * self.d_a1
        self.w2 -= self.learning_rate * self.d_w2
        self.b2 -= self.learning_rate * self.d_a2

################################################DO NOT TOUCH################################################  
# Do not run any additional function nor touch any of codes below this line.
    def train(self, epochs, train_loader):
        for epoch in range(epochs):
            loss_sum, corrects, accuracy = 0, 0, 0
            i = 0
            for data in tqdm(train_loader):
                x, y = data # get data x and label y
                x = x.reshape((1, -1)) # flatten the image 28x28 --> 1x784

                # convert labels into one-hot vectors
                y_onehot = np.zeros((1, 10))
                y_onehot[0,y] += 1

                # forward pass
                self.forward(x)

                # use Mean Squared Error for loss function
                loss = 0.1 * (self.a2 - y_onehot)**2

                # backward pass
                self.backward(x, y_onehot)

                # update weights
                self.step()

                # add loss
                loss_sum += np.sum(loss)
                #calculate accuracy 
                if np.argmax(self.a2) == y:
                    corrects += 1

            accuracy = corrects / 60000 * 100
            
            # print accuracy
            print("Epoch {}, Accuracy: {:.3f}%".format(epoch + 1, accuracy)) 
            
            # add loss to loss array and renew the loss
            self.lossArr.append(loss_sum)
            self.accuracyArr.append(accuracy)


def main():
    # Do not run any additional function nor touch any of these codes

    # define dataset and dataloader
    train_MNIST = datasets.MNIST("MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_MNIST, shuffle=True, drop_last=True)
    
    # set hyperparameters
    epochs = 5
    learning_rate = 0.1
    inputL_dim, hiddenL_dim, outputL_dim = 28*28, 128, 10
    
    # initialize model
    mlp = MLP(inputL_dim, hiddenL_dim, outputL_dim, learning_rate)

    # start training
    mlp.train(epochs, train_loader)
    
    plt.subplot(2, 1, 1)
    plt.plot(range(epochs), mlp.lossArr, label='loss')
    plt.title('Loss Graph')
    plt.xlabel('# Epoch')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(range(epochs), mlp.accuracyArr, label='accuracy')
    plt.title('Accuracy Graph')
    plt.xlabel('# Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('plots.png')

main()
################################################DO NOT TOUCH################################################