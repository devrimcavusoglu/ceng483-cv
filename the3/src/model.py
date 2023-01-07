from torch import nn


class Net(nn.Module):
    kernel_size: int = 3

    def __init__(self, n_conv: int = 1, h_channels: int = 2):
        super(Net, self).__init__()

        self.relu = nn.ReLU()  # No trainable parameters, shared
        if n_conv == 1:
            self.h_convs = nn.ModuleList([nn.Conv2d(1, 3, self.kernel_size, padding="same")])
        else:
            h_convs = []
            for i in range(n_conv):
                if i == 0:
                    c_in, c_out = 1, h_channels
                elif i == n_conv - 1:
                    c_in, c_out = h_channels, 3
                else:
                    c_in, c_out = h_channels, h_channels
                h_convs.append(nn.Conv2d(c_in, c_out, self.kernel_size, padding="same"))
            self.h_convs = nn.ModuleList(h_convs)

    def forward(self, x):
        # apply your network's layers in the following lines:
        for i, conv_layer in enumerate(self.h_convs):
            if i < len(self.h_convs) - 1:
                x = self.relu(conv_layer(x))
            else:  # last layer
                x = conv_layer(x)
        return x
