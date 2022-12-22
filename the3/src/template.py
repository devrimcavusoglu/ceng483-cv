"""
Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
This code is given as a starting point.

REFEFERENCES
The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

---- hyper-parameters ----
You should tune these hyper-parameters using:
(i) your reasoning and observations,
(ii) by tuning it on the validation set, using the techniques discussed in class.
You definitely can add more hyper-parameters here.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import hw3utils
from the3.src import LOG_DIR, DATA_ROOT
from the3.src.model import Net


def get_loaders(batch_size, device):
    train_set = hw3utils.HW3ImageFolder(root=DATA_ROOT / "train", device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=DATA_ROOT / "val", device=device)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader


def train(
        batch_size: int,
        max_epochs: int,
        lr: float,
        device: str = None,
        visualize: bool = False,
        load_checkpoint: bool = True
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('device: ' + str(device))
    net = Net().to(device=device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=hps['lr'])
    train_loader, val_loader = get_loaders(batch_size, device)

    if load_checkpoint:
        print('loading the model from the checkpoint')
        ckp_path = os.path.join(LOG_DIR, 'checkpoint.pt')
        net.load_state_dict(torch.load(ckp_path))

    print('training begins')
    for epoch in range(max_num_epoch):
        running_loss = 0.0 # training loss of the network
        for iteri, data in enumerate(train_loader, 0):
            inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.

            optimizer.zero_grad() # zero the parameter gradients

            # do forward, backward, SGD step
            preds = net(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            # print loss
            running_loss += loss.item()
            print_n = 100 # feel free to change this constant
            if iteri % print_n == (print_n-1):    # print every print_n mini-batches
                print('[%d, %5d] network-loss: %.3f' %
                      (epoch + 1, iteri + 1, running_loss / 100))
                running_loss = 0.0
                # note: you most probably want to track the progress on the validation set as well (needs to be implemented)

            if (iteri == 0) and visualize:
                hw3utils.visualize_batch(inputs, preds, targets)

        print('Saving the model, end of epoch %d' % (epoch+1))
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        torch.save(net.state_dict(), os.path.join(LOG_DIR,'checkpoint.pt'))
        export_path = LOG_DIR / 'example.png'
        hw3utils.visualize_batch(inputs, preds, targets, export_path)

    print('Finished Training')


if __name__ == "__main__":
    batch_size = 16
    max_num_epoch = 10
    hps = {'lr': 0.001}

    torch.multiprocessing.set_start_method('spawn', force=True)
    train(
            batch_size,
            max_num_epoch,
            hps['lr'],
            device=None,
            visualize=False,
            load_checkpoint=False
    )
