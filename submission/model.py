import torch
import torch.nn as nn

#########################################
#       Improve this basic model!       #
#########################################
'''

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(6924, 48)

    def forward(self, pv, hrv):
        x = torch.relu(self.pool(self.conv1(hrv)))
        x = torch.relu(self.pool(self.conv2(x)))
        x = torch.relu(self.pool(self.conv3(x)))
        x = torch.relu(self.pool(self.conv4(x)))

        x = self.flatten(x)
        x = torch.concat((x, pv), dim=-1)

        x = torch.sigmoid(self.linear1(x))

        return x

'''

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, num_pv_features, num_hrv_features, num_classes):
        super(Model, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_hrv_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Define the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 7 * 7 + num_pv_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pv, hrv):
        # Pass HRV data through convolutional layers
        x = self.relu(self.conv1(hrv))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        # Flatten the output of convolutional layers
        x = self.flatten(x)

        # Concatenate with PV data
        x = torch.cat((x, pv), dim=1)

        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x
