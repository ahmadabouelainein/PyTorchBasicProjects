import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# print(torch.__version__)
# !nvidia-smi

device = "cuda" if torch.cuda.is_available() else "cpu"

class DataSet:
    def __init__(self, n_samples = 1000):
        self.X, self.y = make_blobs(n_samples=n_samples,n_features=2,centers=4,cluster_std=1.5, random_state=42)
        self.X = torch.from_numpy(self.X).type(torch.float)
        self.y = torch.from_numpy(self.y).type(torch.LongTensor)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                self.y,
                                                test_size=0.2,
                                                random_state=0)
    def plot_data(self, X, y):
        plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
        plt.show()

class CircleModelV0(nn.Module):
    def __init__(self, input_features=2, output_features=4, hidden_units=8):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
    def accuracy_fn(self,y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100 
        return acc
    
    def training_loop(self, epochs, data):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1)
        data.X_train, data.X_test, data.y_train, data.y_test = data.X_train.to(device), data.X_test.to(device), data.y_train.to(device), data.y_test.to(device) 
        for epoch in range(epochs):
            self.train()

            y_logits = self(data.X_train)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            acc = self.accuracy_fn(y_true=data.y_train, y_pred=y_pred) 
            loss = loss_fn(y_logits, data.y_train) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Testing
            self.eval()
            with torch.inference_mode():
                test_logits = self(data.X_test)
                test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)
                test_loss = loss_fn(test_logits,
                                    data.y_test)
                test_acc = self.accuracy_fn(y_true=data.y_test,
                                    y_pred=test_pred)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

if __name__=="__main__":
    print(f"Running on: {device}")
    dataset= DataSet()
    model_0 = CircleModelV0().to(device=device)
    model_0.training_loop(epochs=100,data=dataset)
    dataset.plot_data(dataset.X, dataset.y)


    