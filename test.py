# to hold tests
import cv2
import torch
import torch.nn as nn
from torchsummary import summary

from utils import visualize_vid, visualize_waveform, to_categorical
from gen_synthetic import DataGen
from models import ConvNet3D

def test_model():

    no_classes = 76
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001

    input = torch.randn(200, 1, 60, 25, 25, requires_grad=True).to(device)
    target = torch.empty(200, dtype=torch.long).random_(no_classes).to(device)

    # https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function
    # doesnt need one hot encoding if using categorical cross entropy loss
    # if want to use target as one hot vector then need to use nn.functional.log_softmax + nn.NLLLoss
    #target = to_categorical(target, no_classes)
    #target = torch.from_numpy(target).to(device)
    #print(input, target)
    print(target)
    #print(target[0])

    model = ConvNet3D(no_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    print(model)
    summary(model, input_size=(1, 60, 25, 25))
    
    for i in range(1000):
        #input.to(device)
        #target.to(device)
        # Forward pass
        outputs = model(input)
        #print(outputs, outputs.size)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100==0:
            print(loss)

def test_visualize_vid():
    cap = cv2.VideoCapture("/home2/edwin_ed520/databases/UBFC_og_14sub/S13.avi")
    vid = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False: break
        vid.append(frame)
    vid = np.array(vid)
    visualize_vid(vid)

def test_datagen():
    datagenerator = DataGen(heart_rate_high=90, no_videos_by_class=2)
    x, y = datagenerator.gen_signal()
    #print(x.shape, y.shape)
    #print(x[0])
    #print(y[0])
    visualize_vid(x)

#test_visualize_vid()
#test_datagen()
test_model()