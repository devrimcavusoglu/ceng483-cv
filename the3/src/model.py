from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding="same")
        self.conv2 = nn.Conv2d(8, 8, 3, padding="same")
        self.conv3 = nn.Conv2d(8, 8, 3, padding="same")
        self.conv4 = nn.Conv2d(8, 3, 3, padding="same")
        self.relu = nn.ReLU()  # No trainable parameters, shared

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        x = self.relu(self.conv1(grayscale_image))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x
