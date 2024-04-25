import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import matplotlib.pyplot as plt # type: ignore
device = torch.device("mps")

class NeuralNetwork:
    def __init__(self, h0, h1, h2, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.h0 = h0
        self.h1 = h1
        self.h2 = h2
        self.W1, self.W2, self.b1, self.b2 = self.init_params()
        

    def init_params(self):
        W1 = torch.normal(0, torch.sqrt(torch.divide(2, (self.h0 + self.h1))), size=(self.h1, self.h0)).to(device)
        W2 = torch.normal(0, torch.sqrt(torch.divide(2, (self.h1 + self.h2))), size=(self.h2, self.h1)).to(device)
        b1 = torch.normal(0, torch.sqrt(torch.divide(2, (self.h0 + self.h1))), size=(self.h1, 1)).to(device)
        b2 = torch.normal(0, torch.sqrt(torch.divide(2, (self.h1 + self.h2))), size=(self.h2, 1)).to(device)
        return W1, W2, b1, b2

    def loss(self,Y,y_pred):
        return -torch.sum(Y * torch.log(y_pred)) / Y.shape[1]
    
#     def sigmoid(self,z):
#         return 1/( 1 + torch.exp(-z))
    
    def softmax(self, z):
        expo = torch.exp(z)
        return expo / torch.sum(expo, dim=0)

    def relu(self, z):
        return torch.maximum(torch.tensor(0.0).to(device), z)
    
    def d_relu(self, z):
        return torch.where(z <= 0, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

    
    def forward_pass(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.softmax(Z2)
        return A2, Z2, A1, Z1
   
    def backward_pass(self, X, Y, A1, A2, Z1, Z2 ):
#        = self.forward_pass(X)
        n = X.shape[1]
        
        dL_dZ2 = A2 - Y
        dZ2_dW2 = A1.T
        
        dW2 = (1/n) * torch.matmul(dL_dZ2, dZ2_dW2)
        db2 = (1/n) * torch.sum(dL_dZ2, dim=1, keepdim=True)
        
        dZ2_dA1 = self.W2
        dA1_dZ1 = self.d_relu(Z1)
        dZ1_dW1 = X.T
        
        dW1 = (1/n) * torch.matmul(torch.matmul(dZ2_dA1.T, dL_dZ2) * dA1_dZ1, dZ1_dW1)
        db1 = (1/n) * torch.sum(torch.matmul(dZ2_dA1.T, dL_dZ2) * dA1_dZ1, dim=1, keepdim=True)
        
        return dW1, dW2, db1, db2
     
    def predict(self,X):
        A2, Z2, A1, Z1 = self.forward_pass(X)
        return A2
    
    # def accuracy(self,y, y_pred):
    #     y = y.cpu().numpy()
    #     y_pred = y_pred.cpu().numpy()
    #     #y_pred = torch.tensor(A2)
    #     pred = torch.zeros_like(y_pred)
    #     L_max = torch.max(y_pred, axis=0)

    #     for i in range(y.shape[1]):
    #         pred[:,i] = (L_max[i] == y_pred[:,i]).astype(int)
    #     acc = torch.mean((y == pred).astype(int))*100
    #     return acc
    def accuracy(self, y, y_pred):
        pred = torch.zeros_like(y_pred)
        L_max, _ = torch.max(y_pred, dim=0)

        for i in range(y.shape[1]):
            pred[:, i] = (L_max[i] == y_pred[:, i]).int()
        acc = torch.mean((y == pred).float()) * 100
        return acc


    def update(self, dW1, dW2, db1, db2):
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

    def fit(self, X_train, Y_train, X_test, Y_test):
        train_loss = []
        test_loss = []
        
        for i in range(self.epochs):
            A2, Z2, A1, Z1 = self.forward_pass(X_train)
            #print(A2)
            dW1, dW2, db1, db2 = self.backward_pass(X_train, Y_train, A1, A2, Z1, Z2 )
            self.update(dW1, dW2, db1, db2)
            
            loss_train = self.loss(Y_train,A2)
            train_loss.append(loss_train)
            
            
            A22, Z22, A11, Z11 = self.forward_pass(X_test)
            loss_test = self.loss(Y_test,A22)
            test_loss.append(loss_test)
            
            if i % 200 == 0:
                print(f" Epoch {i}\nTrain loss:--- {loss_train:.4f}  Test loss:---{loss_test:.4f}")

        plt.plot(torch.tensor(train_loss).cpu(),label='Train')
        plt.plot(torch.tensor(test_loss).cpu(),label = 'Validation')
        plt.title('Plots of losses')
        plt.legend()
        plt.show()

        train_accuracy = self.accuracy(Y_train, A2 )
        print(f"train accuracy: {train_accuracy:.2f}%")

        y_pred = self.predict(X_test)
        test_accuracy = self.accuracy(Y_test, y_pred)
        print (f"test accuracy : {test_accuracy:.2f}%")


    