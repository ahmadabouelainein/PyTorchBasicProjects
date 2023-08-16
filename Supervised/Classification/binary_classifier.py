import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
# print(torch.__version__)
# !nvidia-smi

device = "cpu"

class DataSet:
    def __init__(self, n_samples = 1000):
        self.X, self.y = make_circles(n_samples,
                noise=0.05, # a little bit of noise to the dots
                random_state=42, # keep random state so we get the same values
                factor=0.75)
        self.X = torch.from_numpy(self.X).type(torch.float).to(device)
        self.y = torch.from_numpy(self.y).type(torch.float).to(device)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                self.y,
                                                test_size=0.2,
                                                random_state=42)
    def plot_data(self, X, y):
        plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
        plt.show()

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2,50)
        self.layer_2 = nn.Linear(50,50)
        self.relu = nn.ReLU()
        self.layer_3 = nn.Linear(50,1)
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.layer_1(x))))
    
    def accuracy_fn(self,y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100 
        return acc
    
    def training_loop(self, epochs, data):
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1)

        for epoch in range(epochs):
            self.train()

            y_logits = self(data.X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
            y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
            loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                        data.y_train) 
            acc = self.accuracy_fn(y_true=data.y_train, 
                            y_pred=y_pred) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Testing
            self.eval()
            with torch.inference_mode():
                test_logits = self(data.X_test).squeeze() 
                test_pred = torch.round(torch.sigmoid(test_logits))
                test_loss = loss_fn(test_logits,
                                    data.y_test)
                test_acc = self.accuracy_fn(y_true=data.y_test,
                                    y_pred=test_pred)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

if __name__=="__main__":
    dataset= DataSet()
    model_0 = CircleModelV0()
    model_0.training_loop(epochs=1000,data=dataset)
    # dataset.plot_data(dataset.X, dataset.y)


    