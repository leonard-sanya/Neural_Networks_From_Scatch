import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from random import shuffle
from torchvision import datasets, transforms 
import pandas as pd # type: ignore
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 

device = torch.device('mps')
import time
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import accuracy_score 
from MNIST_NN_1_HL import NeuralNetwork

transform = transforms.Compose([transforms.ToTensor(),])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, )

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


def retrive_data(loader):
    for images, labels in loader:
        images_pixels = images
        labels = labels
   
    images_flattened = images_pixels.reshape(images.shape[0], -1)
   
    data = pd.DataFrame(images_flattened)
    data['label'] = pd.Series(labels)

    return data


train_data = retrive_data(train_loader)
test_data = retrive_data(test_loader)


X_train = torch.tensor(train_data.drop(columns=['label']).values, dtype=torch.float32)
Y_train = torch.tensor(train_data['label'].values, dtype=torch.long).reshape(-1,1)
encod = OneHotEncoder()
Y_train_EN = torch.tensor(encod.fit_transform(Y_train).todense(), dtype=torch.float32)

X_test = torch.tensor(test_data.drop(columns=['label']).values, dtype=torch.float32)
Y_test = torch.tensor(test_data['label'].values, dtype=torch.long).reshape(-1,1)
Y_test_EN = torch.tensor(encod.fit_transform(Y_test).todense(), dtype=torch.float32)



X_train = X_train.T.to(device)
Y_train = Y_train_EN.T.to(device)
X_test = X_test.T.to(device)
Y_test = Y_test_EN.T.to(device)

import seaborn as sns # type: ignore

    
model = NeuralNetwork(784,500,10,0.01,3000)

def main():

    model.fit(X_train,Y_train,X_test, Y_test)

    def plot_CM(model, X_test, Y_test):
    # Make predictions on test data
        y_pred = model.predict(X_test)
        y_pred = torch.argmax(y_pred, dim=0).cpu().numpy()
        y_true = torch.argmax(Y_test, dim=0).cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
    plot_CM(model, X_test, Y_test)

         
if __name__ == "__main__":
   main()
