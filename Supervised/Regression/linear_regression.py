import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# print(torch.__version__)
# !nvidia-smi

device = "cpu"

class DataSet:
    def __init__(self, start=0, end=1, step=0.02, weight=0.6, bias=0.4):
        torch.manual_seed(42)
        self.X = torch.arange(start, end, step)
        self.y = weight * self.X + bias
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                self.y,
                                                test_size=0.2,
                                                shuffle=False)
    def plot_data(self, predictions):
        # plt.figure()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        scats= [plt.scatter(self.X_train, self.y_train, marker="s", c="y", s=14, label="Training data"),  plt.scatter(self.X_test, self.y_test, marker="x", c="c", s=14, label="Testing data"), plt.scatter(self.X_test, predictions.detach().numpy(), c="m", s=14, label="Predictions")]
        plt.pause(0.05)
        for scat in scats: scat.remove()
        return scats
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float),
                                requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float),
                             requires_grad=True)

    def forward(self, x):
        return self.weights * x + self.bias 
  
    def training_loop(self, epochs, data):
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.SGD(params=self.parameters(), lr=0.002)

        for epoch in range(epochs):
            self.train()# put model in training mode

            y_pred = self(data.X_train)
            loss = loss_fn(y_pred, data.y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### Testing
            self.eval()
            with torch.inference_mode():
                test_pred = self(data.X_test)
                test_loss = loss_fn(test_pred,
                                    data.y_test)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")
                
                plt.legend(data.plot_data(self(data.X_test)), ["Training","Testing",f"Prediction after {epoch+10} loops"])
        plt.show()
        return self(data.X_test)

if __name__=="__main__":
    dataset= DataSet()
    model_0 = LinearRegressionModel()
    model_0.training_loop(epochs=501, data=dataset)



    