import numpy as np
#from utils import to_categorical

# Adapted from 
# 3D Convolutional Neural Networks for Remote Pulse Rate Measurement
#  and Mapping from Facial Video by Bousefsaf et al.

class DataGen():
    def __init__(self, heart_rate_low=55, heart_rate_high=240,
                 heart_rate_resolution=2.5, length_vid=60, img_width=25,
                 img_height=25, img_channels=1, sampling=1/30,
                 no_videos_by_class=300):
        # constants
        self.heart_rates = np.arange(
            heart_rate_low, heart_rate_high, heart_rate_resolution)
        self.length_vid = length_vid
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.sampling = sampling
        self.no_videos_by_class = no_videos_by_class

        self.no_classes = len(self.heart_rates)
        self.t = np.linspace(0, self.length_vid *
                             self.sampling - self.sampling, self.length_vid)
        self.freqs = self.heart_rates / 60

        # prepare labels and label categories
        
        # for categorical cross entropy doesnt need one hot vectors
        # pytorch expects long dtypes
        #labels = np.arange(0, self.no_classes + 1, 1, dtype='uint8')
        #self.labels_cat = to_categorical(labels, self.no_classes + 1)
        
        labels = np.arange(0, self.no_classes + 1, 1, dtype='long')
        self.labels_cat = labels
        
        # coefficients for the fitted-ppg method
        self.coeff = {
        'a0': 0.440240602542388,
        'a1': -0.334501803331783,
        'b1': -0.198990393984879,
        'a2': -0.050159136439220,
        'b2': 0.099347477830878,
        'w': 2 * np.pi     
        }

        # Tendencies (linear, 2nd order, 3rd order)
        self.tendencies_min = (-3, -1, -1)
        self.tendencies_max = (3, 1, 1)
        self.tendencies_order = (1, 2, 3)

    def gen_trend(self, length, order, min, max, offset):
        if (order == 1):   # linear
            tend = np.linspace(min, max, length)

        elif (order == 2):  # quadratic
            if (offset == 0):
                tend = np.linspace(0, 1, length)
                tend = tend*tend
                tend = tend-min
                tend = max*tend/np.max(tend)

            else:
                tend = tend = np.linspace(-0.5, 0.5, length)
                tend = tend*tend
                tend = tend-min
                tend = 0.5*max*tend/np.max(tend)

        elif (order == 3):  # cubic
            if (offset == 0):
                tend = np.linspace(0, 1, length)
                tend = tend*tend*tend
                tend = tend-min
                tend = max*tend/np.max(tend)

            else:
                tend = tend = np.linspace(-0.5, 0.5, length)
                tend = tend*tend*tend
                tend = tend-min
                tend = 0.5*max*tend/np.max(tend)
        return tend

    def gen_signal(self):
        # generate synthetic training data (and labels) in form of videos
        # of shape no_videos, frames, img_height, img_width, channels
        # can use this class for training
        x = np.zeros(shape=((self.no_classes + 1) * self.no_videos_by_class,
                            self.img_channels, self.img_height, self.img_height, self.length_vid))
        #x = np.zeros(shape=((self.no_classes + 1) * self.no_videos_by_class,
        #                    self.length_vid, self.img_height, self.img_height, self.img_channels))
        y = np.zeros(shape=((self.no_classes + 1) * self.no_videos_by_class), dtype='long')
        #y = np.zeros(shape=((self.no_classes + 1) * self.no_videos_by_class, self.no_classes + 1))
        
        c = 0
        # generates signals that resemble rppg waves
        for vids_per_freq in range(self.no_videos_by_class):
            for i, freq in enumerate(self.freqs):
                # generates baseline signal (fitted fourier series)
                # phase. 33 corresponds to a full phase shift for HR=55 bpm
                t2 = self.t + (np.random.randint(low=0, high=33) * self.sampling)
                signal = (self.coeff['a0'] + self.coeff['a1'] * np.cos(t2 * self.coeff['w'] * freq) 
                + self.coeff['b1'] * np.sin(t2 * self.coeff['w'] * freq) 
                + self.coeff['a2'] * np.cos(2 * t2 * self.coeff['w'] * freq) 
                + self.coeff['b2'] * np.sin(2 * t2 * self.coeff['w'] * freq) )
                signal = signal - np.min(signal)
                signal = signal / np.max(signal)

                # adds trends and noise
                r = np.random.randint(low=0, high=len(self.tendencies_max))
                trend = self.gen_trend(len(self.t), self.tendencies_order[r], 0, np.random.uniform(
                    low=self.tendencies_min[r], high=self.tendencies_max[r]), np.random.randint(low=0, high=2))

                signal = np.expand_dims(signal + trend, 1)
                signal = signal - np.min(signal)
                
                # replicates the signal value to make an image
                img = np.tile(signal, (self.img_height, 1, self.img_height))
                img = np.transpose(img, axes=(0, 2, 1))

                img = img / (self.img_height * self.img_height)

                amplitude = np.random.uniform(low=1.5, high=4)
                noise_energy = amplitude * 0.25 * \
                    np.random.uniform(low=1, high=10) / 100

                # puts the images together to make a video
                for j in range(0, self.length_vid):
                    temp = 255 * ((amplitude * img[:, :, j]) + np.random.normal(
                        size=(self.img_height, self.img_height), loc=0.5, scale=0.25) * noise_energy)
                    temp[temp < 0] = 0
                    x[c, 0, :, :, j] = temp.astype('uint8') / 255.0

                x[c] = x[c] - np.mean(x[c])
                y[c] = self.labels_cat[i]
                
                #print(c, y[c])
                c = c + 1
        
        for vids_per_freq in range(self.no_videos_by_class):
            # generates noise to act as the class for no heart rate?
            # constant image noise (gaussian distribution)
            r = np.random.randint(low=0, high=len(self.tendencies_max))
            trend = self.gen_trend(len(self.t), self.tendencies_order[r], 0, np.random.uniform(
                low=self.tendencies_min[r], high=self.tendencies_max[r]), np.random.randint(low=0, high=2))

            # add a tendancy on noise
            signal = np.expand_dims(trend, 1)
            img = np.tile(signal, (self.img_height, 1, self.img_height)) / \
                (self.img_height * self.img_height)
            img = np.expand_dims(np.transpose(img, axes=(0, 2, 1)), 3)
            temp = np.expand_dims(np.random.normal(
                size=(self.img_height, self.img_height, self.length_vid)) / 50, 3) + img
            x[c] = np.reshape(temp, (1, self.img_height, self.img_height, self.length_vid))
            x[c] = x[c] - np.mean(x[c])
            y[c] = self.labels_cat[self.no_classes]
            
            #print(c, y[c])
            c = c + 1

        return x, y
