############################################################
# Definition of classes. These classes are imported in the
# other files, for training and evaluation.
# Mathilda Gustafsson
# 2025-05-29
############################################################

# Load packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset

############################################################
# Classes for datasets
############################################################

# dataset of sequences of images
class datasetSequentialImages(Dataset):
    def __init__(self, pipeline_spherical_bacteria, dataset_length):

        self.sequences = [pipeline_spherical_bacteria.update()() for _ in range(dataset_length)] # resolve to remove deeptrack object

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        
        images = self.sequences[idx][0]
        positions = self.sequences[idx][1]

        return images, positions


############################################################
# Models
############################################################

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.ReLU = nn.ReLU()
        self.leakyReLU = nn.LeakyReLU()

        # batch normalization
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.batch_norm_3 = nn.BatchNorm2d(32)
        self.batch_norm_4 = nn.BatchNorm2d(32)
        self.batch_norm_5 = nn.BatchNorm2d(64)
        self.batch_norm_6 = nn.BatchNorm2d(64)
        self.batch_norm_7 = nn.BatchNorm2d(128)
        self.batch_norm_8 = nn.BatchNorm2d(128)

        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 2, stride = 1, padding = 1)
        self.conv_2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 2, stride = 1, padding = 0)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 2, stride = 1 , padding = 1)
        self.conv_4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1, padding = 0)
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 1, padding = 1)
        self.conv_6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 1 , padding = 0)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
      
        self.conv_7 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 2, stride = 1, padding = 1)
        self.conv_8 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 2, stride = 1 , padding = 0)
        self.maxpool_4 = nn.MaxPool2d(kernel_size = 3, stride = 3)

        self.linear_1 = nn.Linear(in_features = 128*5*5, out_features = 256)
        self.linear_2 = nn.Linear(in_features = 256, out_features = 32)
        self.linear_3 = nn.Linear(in_features = 32, out_features = 32)
        self.linear_4 = nn.Linear(in_features = 32, out_features = 32)
        self.linear_5 = nn.Linear(in_features = 32, out_features = 16)
        self.output = nn.Linear(in_features = 16, out_features = 2)


    def forward(self, x):

        # max pooling and convolutional layers
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.leakyReLU(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.leakyReLU(x)

        x = self.maxpool_1(x)
       
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.leakyReLU(x)        
        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.leakyReLU(x)

        x = self.maxpool_2(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.leakyReLU(x)
        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.leakyReLU(x)

        x = self.maxpool_3(x)

        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.leakyReLU(x)
        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.leakyReLU(x)

        x = self.maxpool_4(x)
        x = torch.flatten(x, start_dim = 1, end_dim = 3)

        # linear layers
        x = self.linear_1(x)
        x = self.ReLU(x)

        x = self.linear_2(x)
        x = self.ReLU(x)

        x = self.linear_3(x)
        x = self.ReLU(x)

        x = self.linear_4(x)
        x = self.ReLU(x)

        x = self.linear_5(x)
        x = self.ReLU(x)

        x = self.output(x)

        return x
        

class RNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ReLU = nn.ReLU()
        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.hidden_state = []
        self.cell_state = []

        # batch normalization
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.batch_norm_3 = nn.BatchNorm2d(32)
        self.batch_norm_4 = nn.BatchNorm2d(32)
        self.batch_norm_5 = nn.BatchNorm2d(64)
        self.batch_norm_6 = nn.BatchNorm2d(64)
        self.batch_norm_7 = nn.BatchNorm2d(128)
        self.batch_norm_8 = nn.BatchNorm2d(128)

        # encoder layers
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 2, stride = 1, padding = 1)
        self.conv_2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 2, stride = 1, padding = 0)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 2, stride = 1 , padding = 1)
        self.conv_4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1, padding = 0)
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 1, padding = 1)
        self.conv_6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 1 , padding = 0)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
      
        self.conv_7 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 2, stride = 1, padding = 1)
        self.conv_8 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 2, stride = 1 , padding = 0)
        self.maxpool_4 = nn.MaxPool2d(kernel_size = 3, stride = 3)

        self.linear_1 = nn.Linear(in_features = 128*5*5, out_features = 256)

        # recurrent layers
        self.LSTM = nn.LSTM(input_size = 256, hidden_size = 32, num_layers = 3, bias = True, batch_first = True)

        # linear layers
        self.linear_2 = nn.Linear(in_features = 32, out_features = 16)
        self.output = nn.Linear(in_features = 16, out_features = 2)


    def encoder(self, x):

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.leakyReLU(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.leakyReLU(x)

        x = self.maxpool_1(x)
       
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.leakyReLU(x)        
        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.leakyReLU(x)

        x = self.maxpool_2(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.leakyReLU(x)
        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.leakyReLU(x)

        x = self.maxpool_3(x)

        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.leakyReLU(x)
        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.leakyReLU(x)

        x = self.maxpool_4(x)

        x = torch.flatten(x, start_dim = 1, end_dim = 3)

        x = self.linear_1(x)
        x = self.leakyReLU(x)

        return x

    def forward(self, x, is_initial_image):

        x = self.encoder(x)

        # add dimension for sequence length
        x = x.unsqueeze(1) 

        # recurrent layers
        if (is_initial_image):
            x, (h_n, c_n) = self.LSTM(x)
            self.hidden_state = h_n
            self.cell_state = c_n
        else:
            x, (h_n, c_n) = self.LSTM(x, (self.hidden_state, self.cell_state))
            self.hidden_state = h_n
            self.cell_state = c_n

        # remove extra dimension for sequence length
        x = x.squeeze(1)

        x = self.linear_2(x)
        x = self.ReLU(x)

        x = self.output(x)

        return x
