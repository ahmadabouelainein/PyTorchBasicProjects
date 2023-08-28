import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
BATCH_SIZE=32
device="cpu"
class DataSet:
    def __init__(self, batch_size=BATCH_SIZE):
        self.train_data = torchvision.datasets.FashionMNIST(root='data',train=True, download=True, transform=torchvision.transforms.ToTensor())
        self.test_data = torchvision.datasets.FashionMNIST(root='data',train=False, download=True, transform=torchvision.transforms.ToTensor())
        self.class_names = self.train_data.classes
        self.train_dataloader=torch.utils.data.DataLoader(dataset=self.train_data,batch_size=BATCH_SIZE, shuffle=True)
        self.test_dataloader=torch.utils.data.DataLoader(dataset=self.test_data,batch_size=BATCH_SIZE, shuffle=False)

    def plot_random_samples(self):
        torch.manual_seed(42)
        fig = plt.figure(figsize=(9, 9))
        rows, cols = 6, 8
        for i in range(1, rows * cols + 1):
            random_idx = torch.randint(0, len(self.train_data), size=[1]).item()
            img, label = self.train_data[random_idx]
            fig.add_subplot(rows, cols, i)
            plt.imshow(img.squeeze(), cmap="gray")
            plt.title(self.class_names[label])
            plt.axis(False);
        plt.show()


class FashionMNISTModelNN(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Flatten(), # neural networks like their inputs in vector form
            torch.nn.Linear(in_features=input_shape, out_features=hidden_units), 
            torch.nn.ReLU(), # does not massively increase acc ~2%)
            torch.nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)
    
    def accuracy_fn(self,y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100 
        return acc

    def training_loop(self, data, epochs):
        loss_fn = torch.nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
        optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1)
        # Create training and testing loop
        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n-------")
            ### Training
            train_loss = 0
            # Add a loop to loop through training batches
            for batch, (X, y) in enumerate(data.train_dataloader):
                self.train() 
                y_pred = self(X)
                loss = loss_fn(y_pred, y)
                train_loss += loss # accumulatively add up the loss per epoch 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print out how many samples have been seen
                if batch % 400 == 0:
                    print(f"Looked at {batch * len(X)}/{len(data.train_dataloader.dataset)} samples")

            # Divide total train loss by length of train dataloader (average loss per batch per epoch)
                train_loss /= len(data.train_dataloader)
            
    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 
        test_loss, test_acc = 0, 0 
        self.eval()
        with torch.inference_mode():
            for X, y in data.test_dataloader:
                # 1. Forward pass
                test_pred = self(X)
            
                # 2. Calculate loss (accumatively)
                test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
                test_acc += self.accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
                test_loss /= len(data.test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
            test_acc /= len(data.test_dataloader)

    ## Print out what's happening
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# Create a convolutional neural network 
class FashionMNISTModelV2(torch.nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.block_2(self.block_1(x)))
        
    


    
if __name__=="__main__":
    print(f"Running on: {device}")
    dataset= DataSet()
    model_1 = FashionMNISTModelNN(input_shape=784,
                                  hidden_units=10,
                                  output_shape=len(dataset.class_names)
                                  )
    model_1.training_loop(epochs=5,data=dataset)

    model_2 = FashionMNISTModelV2(input_shape=1, 
                                  hidden_units=10, 
                                  output_shape=len(dataset.class_names))
    
    # model_2.training_loop(epochs=5,data=dataset)
    # dataset.plot_random_samples()


    

