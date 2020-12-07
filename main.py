import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from utils import update_lr, plot_losses
from gen_synthetic import DataGen
from models import ConvNet3D

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def validate(model, criterion, no_classes, val_set, training_proc_avg, test_proc_avg):
    # Test the model (validation set)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        current_losses_test = []

        class_correct = list(0. for i in range(no_classes))
        class_total = list(0. for i in range(no_classes))
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            current_losses_test.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        test_proc_avg.append(mean(current_losses_test))

        for i in range(10):
            print('Total objects in class no. {} ({}): {}. Accuracy: {}'.format(i, classes[i],
            class_total[i], 100 * class_correct[i] / class_total[i]))

    # plot loss
    plot_losses(training_proc_avg, test_proc_avg)

def train():
    # trains model from scratch
    # Controlling source of randomness: pytorch RNG
    torch.manual_seed(0)

    # train constants
    num_epochs = 500 #5000 originally
    batch_size = 256
    learning_rate = 0.001

    # generate synthetic data for training
    # stores it all in memory
    # alternatively split it so the generator generates as needed
    # or save to a txt file then read as needed
    datagenerator = DataGen()
    num_classes = datagenerator.no_classes # (76) no of heart rates to classificate
    # if using regression then this is no longer needed
    gen_signals, gen_labels = datagenerator.gen_signal()

    # initiates model and loss 
    model = ConvNet3D(num_classes).to(device)
    criterion = nn.CrossEntropyLoss() # alternatively MSE if regression or PSNR/PSD or pearson correlation
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    summary(model, input_size=gen_signals.shape[1:])
    
    # Train the model
    total_step = len(gen_signals) // batch_size
    curr_lr = learning_rate

    print(gen_signals.shape, gen_labels.shape)
    training_proc_avg = []
    test_proc_avg = []

    for epoch in range(num_epochs):
        current_losses = []
        for i in range(total_step):
            if i<total_step-1:
                signals = torch.from_numpy(gen_signals[i*batch_size:(i+1)*batch_size]).to(device).float()
                labels = torch.from_numpy(gen_labels[i*batch_size:(i+1)*batch_size]).to(device).long()
            else:
                signals = torch.from_numpy(gen_signals[i*batch_size:-1]).to(device).float()
                labels = torch.from_numpy(gen_labels[i*batch_size:-1]).to(device).long()
            
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                    
                current_losses.append(loss.item()) # appends the current value of the loss into a list
        
        # Decay learning rate
        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

        training_proc_avg.append(mean(current_losses)) # calculates mean of losses for current epoch and appends to list of avgs
        

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

def main():
    train()

if __name__ == '__main__':
    main()