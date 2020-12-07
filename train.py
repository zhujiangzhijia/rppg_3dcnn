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

def validate(device, batch_size, classes,
model, criterion, no_classes, gen_signals_val, gen_labels_val, 
training_proc_avg, test_proc_avg, last=False):
    # Test the model (validation set)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        current_losses_test = []

        class_correct = list(0. for i in range(no_classes))
        class_total = list(0. for i in range(no_classes))
        
        total_step = len(gen_signals_val) // batch_size
        for i in range(total_step):
            if i<total_step-1:
                signals = torch.from_numpy(gen_signals_val[i*batch_size:(i+1)*batch_size]).to(device).float()
                labels = torch.from_numpy(gen_labels_val[i*batch_size:(i+1)*batch_size]).to(device).long()
            else:
                signals = torch.from_numpy(gen_signals_val[i*batch_size:-1]).to(device).float()
                labels = torch.from_numpy(gen_labels_val[i*batch_size:-1]).to(device).long()
            
            # Forward pass
            outputs = model(signals)
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

        if last==True:
            for i in range(no_classes):
                print('Total objects in class no. {} ({}): {}. Accuracy: {}'.format(i, classes[i],
                class_total[i], 100 * class_correct[i] / class_total[i]))

            # plot loss
            plot_losses(training_proc_avg, test_proc_avg)

def train():

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # trains model from scratch
    # Controlling source of randomness: pytorch RNG
    torch.manual_seed(0)

    # train constants
    no_epochs = 5000 #5000 originally
    no_videos_by_class = 200
    batch_size = 256
    learning_rate = 0.001

    # generate synthetic data for training
    # stores it all in memory
    # alternatively split it so the generator generates as needed
    # or save to a txt file then read as needed
    datagenerator_train = DataGen(no_videos_by_class=no_videos_by_class)
    
    # (74) no of heart rates to classificate + 1: None
    # originally in the tf implementation it was 75 + 1 since they used
    # linspace instead of arange so it includes the last value unlike arange
    # if using regression then this is no longer needed
    # in original tf implementation by bousefsaf,
    # they do the train generation in each epoch (probably to reduce overfitting)
    # could also do this
    gen_signals_train, gen_labels_train = datagenerator_train.gen_signal()

    # validation/test set
    datagenerator_val = DataGen(no_videos_by_class=no_videos_by_class//10)
    gen_signals_val, gen_labels_val = datagenerator_val.gen_signal()
    no_classes = datagenerator_val.no_classes + 1 

    # initiates model and loss 
    model = ConvNet3D(no_classes).to(device)
    criterion = nn.CrossEntropyLoss() # alternatively MSE if regression or PSNR/PSD or pearson correlation
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    summary(model, input_size=gen_signals_train.shape[1:])
    
    # Train the model
    total_step = len(gen_signals_train) // batch_size
    curr_lr = learning_rate

    print('Training data: ', gen_signals_train.shape, gen_labels_train.shape)
    print('Validation data: ', gen_signals_val.shape, gen_labels_val.shape)
    training_proc_avg = []
    test_proc_avg = []

    for epoch in range(no_epochs):
        current_losses = []
        for i in range(total_step):
            if i<total_step-1:
                signals = torch.from_numpy(gen_signals_train[i*batch_size:(i+1)*batch_size]).to(device).float()
                labels = torch.from_numpy(gen_labels_train[i*batch_size:(i+1)*batch_size]).to(device).long()
            else:
                signals = torch.from_numpy(gen_signals_train[i*batch_size:-1]).to(device).float()
                labels = torch.from_numpy(gen_labels_train[i*batch_size:-1]).to(device).long()
            
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}' 
                    .format(epoch+1, no_epochs, i+1, total_step, loss.item()))
                current_losses.append(loss.item()) # appends the current value of the loss into a list

                validate(device=device, batch_size=batch_size, 
                classes=np.concatenate((datagenerator_val.heart_rates, None), axis=None),
                model=model, criterion=criterion, no_classes=no_classes, 
                gen_signals_val=gen_signals_val, gen_labels_val=gen_labels_val, 
                training_proc_avg=training_proc_avg, test_proc_avg=test_proc_avg, last=False)

        # Decay learning rate
        if (epoch+1) % 500 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

        training_proc_avg.append(mean(current_losses)) # calculates mean of losses for current epoch and appends to list of avgs
      
    # validate on test set
    validate(device=device, batch_size=batch_size, 
    classes=np.concatenate((datagenerator_val.heart_rates, None), axis=None),
    model=model, criterion=criterion, no_classes=no_classes, 
    gen_signals_val=gen_signals_val, gen_labels_val=gen_labels_val, 
    training_proc_avg=training_proc_avg, test_proc_avg=test_proc_avg, last=True)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')