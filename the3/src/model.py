from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, padding: str = "same", add_bn: bool = False):
        super(ConvBlock, self).__init__()

        self.block = nn.ModuleList([nn.Conv2d(c_in, c_out, kernel_size, padding=padding)])
        if add_bn:
            self.block.append(nn.BatchNorm2d(c_out))

    def forward(self, x):
        for component in self.block:
            x = component(x)
        return x


class Net(nn.Module):
    kernel_size: int = 3

    def __init__(self, n_conv: int = 1, h_channels: int = 2, add_bn: bool = False, tanh_last: bool = False):
        super(Net, self).__init__()

        self.tanh = nn.Tanh() if tanh_last else None
        self.relu = nn.ReLU()  # No trainable parameters, shared
        if n_conv == 1:
            self.h_convs = nn.ModuleList([ConvBlock(1, 3, self.kernel_size, add_bn=add_bn)])
        else:
            h_convs = []
            for i in range(n_conv):
                if i == 0:
                    c_in, c_out = 1, h_channels
                elif i == n_conv - 1:
                    c_in, c_out = h_channels, 3
                else:
                    c_in, c_out = h_channels, h_channels
                h_convs.append(ConvBlock(c_in, c_out, self.kernel_size, padding="same", add_bn=add_bn))
            self.h_convs = nn.ModuleList(h_convs)

    def forward(self, x):
        # apply your network's layers in the following lines:
        for i, conv_layer in enumerate(self.h_convs):
            if i < len(self.h_convs) - 1:
                x = self.relu(conv_layer(x))
            else:  # last layer
                x = conv_layer(x)
                if self.tanh is not None:
                    x = self.tanh(x)
        return x
