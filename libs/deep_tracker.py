import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True) 
        )
class DeepTrackerModel(nn.Module):
    
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(DeepTrackerModel, self).__init__()
        self.batchNorm = batchNorm
        # self.clip = par.clip
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=0)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=0)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=0)
        
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        # output = F.relu(self.bn4(self.conv4(output)))     
        # output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)
        return output
class DeepTrackerVO():
    def __init__(self):
        self.model = DeepTrackerModel()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

    def saveModel(self):
        path = "./trackerModel.pth"
        torch.save(self.model.state_dict(), path)
    def testAccuracy(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # run the model on the test set to predict labels
                outputs = model(images)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        
        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        return(accuracy)
    def train(self,num_epochs):
        best_accuracy = 0.0
        # Define your execution device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("The model will be running on", device, "device")
        # Convert model parameters and buffers to CPU or Cuda
        self.model.to(device)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            running_acc = 0.0

            for i, (images, labels) in enumerate(train_loader, 0):
                
                # get the inputs
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = self.model(images)
                # compute the loss based on model output and real labels
                loss = self.loss_fn(outputs, labels)
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                self.optimizer.step()

                # Let's print statistics for every 1,000 images
                running_loss += loss.item()     # extract the loss value
                if i % 1000 == 999:    
                    # print every 1000 (twice per epoch) 
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 1000))
                    # zero the loss
                    running_loss = 0.0

            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            accuracy = self.testAccuracy()
            print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
            
            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                self.saveModel()
                best_accuracy = accuracy
    # Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()