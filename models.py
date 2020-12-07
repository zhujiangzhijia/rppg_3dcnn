import torch 
import torch.nn as nn

# 3D CNN
# this assumes input to be of shape
# no_samples_batch, img_channels, no_frames, img_width, img_height

# to accept different shape would need to adjust
# what comes before layer 1

# one option is to resize the last two dimensions (height/width)
# and then reduce/increase number of frames to 60 but that last option 
# seems bad since it kills the temporal consistency

# another more versatile option is to change all the 
# constant values such as 32, 58, 20 and so on for values
# calculated based on conv kernel formulas so they become variables depending
# on the initial input size

class ConvNet3D(nn.Module):
    def __init__(self, num_classes, fc_neurons=512):
        super(ConvNet3D, self).__init__()
        self.fc_neurons = fc_neurons
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(20, 20, 58), stride=1, padding=0),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.ReLU(),
            nn.Dropout(p=0.2)
            )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.fc_neurons, self.num_classes)
            )   
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.fc(out)
        return out