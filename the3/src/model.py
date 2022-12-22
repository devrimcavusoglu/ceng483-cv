from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, padding=2)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        x = self.conv1(grayscale_image)
        return x
