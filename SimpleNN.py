import numpy as np
from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import animation
import time

class SimpleNN:
    
    mini_batch = 32
    nb_epoch = 1
    W = []
    b = []
    train_losses = []
    test_losses = []
    
    
    def __init__(self):
        pass
    
    def load_data(self):
        digits = load_digits()
        
        X = digits.images
        X_norm, self.mu_X, self.sigma_X = normalize_data(X)
        
        labels = digits.target
        Y = np.zeros((X.shape[0], 10))

        for i, label in enumerate(labels):
            Y[i][label] = 1
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_norm, Y, test_size=0.2, shuffle=False)
        self.n_sample_train = self.X_train.shape[0]
        
    def initialize_weights(self):
        self.W = []
        self.b = []
        W_1 = np.sqrt(2./64)*(np.random.sample((128,64)))
        b_1 = np.zeros((1,128))
        W_2 = np.sqrt(2./128)*(np.random.sample((128,128)))
        b_2 = np.zeros((1,128))
        W_3 = np.sqrt(2./128)*(np.random.sample((10,128)))
        b_3 = np.zeros((1,10))
        
        self.W.append(0)
        self.W.append(W_1)
        self.W.append(W_2)
        self.W.append(W_3)
        self.b.append(0)
        self.b.append(b_1)
        self.b.append(b_2)
        self.b.append(b_3)
        
    def load_weights(self, path, epoch):
        self.W = []
        self.b = []
        W_1 = np.load(path + 'e' + epoch + '_w1.npz')['arr_0']
        W_2 = np.load(path + 'e' + epoch + '_w2.npz')['arr_0']
        W_3 = np.load(path + 'e' + epoch + '_w3.npz')['arr_0']
        b_1 = np.load(path + 'e' + epoch + '_b1.npz')['arr_0']
        b_2 = np.load(path + 'e' + epoch + '_b2.npz')['arr_0']
        b_3 = np.load(path + 'e' + epoch + '_b3.npz')['arr_0']  
        
        self.W.append(0)
        self.W.append(W_1)
        self.W.append(W_2)
        self.W.append(W_3)
        self.b.append(0)
        self.b.append(b_1)
        self.b.append(b_2)
        self.b.append(b_3)
        
    def train(self, l_rate=0.01, l_decay=0, mini_batch=64, nb_epoch=100, verbose=True, plot=False):
        self.mini_batch = mini_batch
        self.nb_epoch = nb_epoch
        number_mini_batch = self.n_sample_train/self.mini_batch + 1
        self.alpha = l_rate
        for e in range(0, nb_epoch):
            for i in range(0, number_mini_batch):
                if i!=number_mini_batch-1:
                    J = self.full(self.X_train[mini_batch*i:mini_batch*(i+1),:,:], 
                               self.Y_train[mini_batch*i:mini_batch*(i+1)], 
                               self.alpha/(l_decay*e+1))
                else:
                    J = self.full(self.X_train[mini_batch*i:,:,:], 
                               self.Y_train[mini_batch*i:], 
                               self.alpha/(l_decay*e+1))

            self.train_losses.append(J)
            test_score = self.run(self.X_test, self.Y_test)
            self.test_losses.append(test_score)

            if verbose and e%10 == 0:
                print("J: " + str(J))
                
        if plot:
            self.plot_losses()
                
    def simple_prediction(self, idx_in_test_set, plot=True):
        data = self.X_test[idx_in_test_set,:,:].reshape((1,8,8))
        image = unnormalize_data(data, self.mu_X, self.sigma_X)
        label = self.Y_test[idx_in_test_set,:].reshape((1,10))
        
        pred, score = fw_prop(data, label, self.W, self.b)
        print("Truth: " + str(np.argmax(label)))
        print("Predicted: " + str(np.argmax(pred)))
        
        if plot:
            plt.imshow(image[0])
            plt.show()
        
    def plot_losses(self):
        plt.plot(np.arange(self.nb_epoch), self.train_losses, np.arange(self.nb_epoch), self.test_losses)
        plt.show()
        
    def save_weights(self, path):
        np.savez_compressed('e' + str(self.nb_epoch) + '_w1.npz', self.W[1])
        np.savez_compressed('e' + str(self.nb_epoch) + '_w2.npz', self.W[2])
        np.savez_compressed('e' + str(self.nb_epoch) + '_w3.npz', self.W[3])
        np.savez_compressed('e' + str(self.nb_epoch) + '_b1.npz', self.b[1])
        np.savez_compressed('e' + str(self.nb_epoch) + '_b2.npz', self.b[2])
        np.savez_compressed('e' + str(self.nb_epoch) + '_b3.npz', self.b[3])
        
    def run(self, X, Y):
        m = X.shape[0]
        n_0 = X.shape[1]*X.shape[2]
        n_1 = self.W[1].shape[0]
        n_2 = self.W[2].shape[0]
        n_3 = self.W[3].shape[0]
        self.A_0 = X.reshape((m,n_0))

        self.Z_1 = np.dot(self.A_0.reshape((m,n_0)), self.W[1].T) + self.b[1]
        self.A_1 = np.vectorize(leaky_relu)(self.Z_1)

        self.Z_2 = np.dot(self.A_1.reshape((m,n_1)), self.W[2].T) + self.b[2]
        self.A_2 = np.vectorize(leaky_relu)(self.Z_2)

        self.Z_3 = np.dot(self.A_2.reshape((m,n_2)), self.W[3].T) + self.b[3]
        self.A_3 = softmax(self.Z_3)

        score = loss(self.A_3, Y)
        return score

    def full(self, X, Y, alpha):
        m = X.shape[0]
        n_0 = X.shape[1]*X.shape[2]
        n_1 = self.W[1].shape[0]
        n_2 = self.W[2].shape[0]
        n_3 = self.W[3].shape[0]
        self.A_0 = X.reshape((m,n_0))

        self.Z_1 = np.dot(self.A_0.reshape((m,n_0)), self.W[1].T) + self.b[1]
        self.A_1 = np.vectorize(leaky_relu)(self.Z_1)
        self.dA_1 = np.vectorize(d_leaky_relu)(self.Z_1)

        self.Z_2 = np.dot(self.A_1.reshape((m,n_1)), self.W[2].T) + self.b[2]
        self.A_2 = np.vectorize(leaky_relu)(self.Z_2)
        self.dA_2 = np.vectorize(d_leaky_relu)(self.Z_2)

        self.Z_3 = np.dot(self.A_2.reshape((m,n_2)), self.W[3].T) + self.b[3]
        self.A_3 = softmax(self.Z_3)

        score = loss(self.A_3, Y)

        self.dZ_3 = self.A_3 - Y
        self.dW_3 = (1./m)*np.dot(self.dZ_3.T, self.A_2)
        self.db_3 = (1./m)*np.sum(self.dZ_3, axis=0, keepdims=True)

        self.dZ_2 = np.dot(self.dZ_3, self.W[3])*self.dA_2
        self.dW_2 = (1./m)*np.dot(self.dZ_2.T, self.A_1)
        self.db_2 = (1./m)*np.sum(self.dZ_2, axis=0, keepdims=True)

        self.dZ_1 = np.dot(self.dZ_2, self.W[2])*self.dA_1
        self.dW_1 = (1./m)*np.dot(self.dZ_1.T, self.A_0)
        self.db_1 = (1./m)*np.sum(self.dZ_1, axis=0, keepdims=True)

        self.W[1]-=self.alpha*self.dW_1
        self.W[2]-=self.alpha*self.dW_2
        self.W[3]-=self.alpha*self.dW_3
        self.b[1]-=self.alpha*self.db_1
        self.b[2]-=self.alpha*self.db_2
        self.b[3]-=self.alpha*self.db_3

        return score


def fw_prop(X, Y, W, b):
    m = X.shape[0]
    n_0 = X.shape[1]*X.shape[2]
    n_1 = W[1].shape[0]
    n_2 = W[2].shape[0]
    n_3 = W[3].shape[0]
    A_0 = X.reshape((m,n_0))
    
    Z_1 = np.dot(A_0.reshape((m,n_0)), W[1].T) + b[1]
    A_1 = np.vectorize(leaky_relu)(Z_1)
    
    Z_2 = np.dot(A_1.reshape((m,n_1)), W[2].T) + b[2]
    A_2 = np.vectorize(leaky_relu)(Z_2)
    
    Z_3 = np.dot(A_2.reshape((m,n_2)), W[3].T) + b[3]
    A_3 = softmax(Z_3)
    
    score = loss(A_3, Y)
    return A_3, score

def fw_bk_prop(X, Y, W, b, alpha):
    m = X.shape[0]
    n_0 = X.shape[1]*X.shape[2]
    n_1 = W[1].shape[0]
    n_2 = W[2].shape[0]
    n_3 = W[3].shape[0]
    A_0 = X.reshape((m,n_0))
    
    Z_1 = np.dot(A_0.reshape((m,n_0)), W[1].T) + b[1]
    A_1 = np.vectorize(leaky_relu)(Z_1)
    dA_1 = np.vectorize(d_leaky_relu)(Z_1)
    
    Z_2 = np.dot(A_1.reshape((m,n_1)), W[2].T) + b[2]
    A_2 = np.vectorize(leaky_relu)(Z_2)
    dA_2 = np.vectorize(d_leaky_relu)(Z_2)
    
    Z_3 = np.dot(A_2.reshape((m,n_2)), W[3].T) + b[3]
    A_3 = softmax(Z_3)
    
    score = loss(A_3, Y)
    
    dZ_3 = A_3 - Y
    dW_3 = (1./m)*np.dot(dZ_3.T, A_2)
    db_3 = (1./m)*np.sum(dZ_3, axis=0, keepdims=True)
    
    dZ_2 = np.dot(dZ_3, W[3])*dA_2
    dW_2 = (1./m)*np.dot(dZ_2.T, A_1)
    db_2 = (1./m)*np.sum(dZ_2, axis=0, keepdims=True)
    
    dZ_1 = np.dot(dZ_2, W[2])*dA_1
    dW_1 = (1./m)*np.dot(dZ_1.T, A_0)
    db_1 = (1./m)*np.sum(dZ_1, axis=0, keepdims=True)
    
    W[1]-=alpha*dW_1
    W[2]-=alpha*dW_2
    W[3]-=alpha*dW_3
    b[1]-=alpha*db_1
    b[2]-=alpha*db_2
    b[3]-=alpha*db_3
    
    return W, b, score

def leaky_relu(x):
    if x >=0:
        return x
    else:
        return -0.01*x
    
def d_leaky_relu(x):
    if x >=0:
        return 1.0
    else:
        return -0.01
    
def softmax(x):
    x -= np.max(x, axis=1).reshape((x.shape[0],1))
    result = (np.exp(x))/(np.sum(np.exp(x), axis=1, keepdims=True))
    return result
    
def loss(pred, y):
    loss = -1.*np.sum(y*np.log(pred+0.001)/y.shape[0])
    return loss
    
def normalize_data(data, epsilon=10e-8):
    epsilon = 10e-8
    mu = np.mean(data, axis=0)
    sigma = np.var(data, axis=0)
    norm_data = (data-mu)/(sigma+epsilon)
    return norm_data, mu, sigma

def unnormalize_data(norm_data, mu, sigma, epsilon=10e-8):
    data = norm_data*(sigma+epsilon) + mu
    return data
        


def init():
    im.set_data(np.zeros((nx, ny)))
    
def animate(i):
    im.set_data(nn.X_test[i])
    return im
    


    
if __name__=='__main__':
    
    nn = SimpleNN()
    nn.load_data()
    nn.initialize_weights()
    #nn.load_weights("save/SimpleNN/", "500")
    nn.train(l_rate=0.01, l_decay=0, mini_batch=64, nb_epoch=500, verbose=True, plot=True)
    nn.simple_prediction(359)
    nn.save_weights('.')