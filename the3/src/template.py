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

import json
import os
from pathlib import Path
from typing import Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import hw3utils
from the3.src import LOG_DIR, DATA_ROOT, TEST_IMAGES_PATH, PROJECT_ROOT
from the3.src.early_stopping import EarlyStopping
from the3.src.evaluate import main as evaluate
from the3.src.model import Net
from the3.src.utils import seed_all


def get_loaders(batch_size, device):
    train_set = hw3utils.HW3ImageFolder(root=DATA_ROOT / "train", device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=DATA_ROOT / "val", device=device)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader


def sample_from_dataset(a: Union[int, np.ndarray], mode: str = "val", device=None, seed: int = 42, export_names: bool = False):
    if mode == "val":
        dataset = hw3utils.HW3ImageFolder(root=DATA_ROOT / "val", device=device)
    elif mode == "test":
        dataset = hw3utils.HW3ImageFolder(root=DATA_ROOT / "test", device=device)
    else:
        raise ValueError("only 'val' and 'test' sets are supported.")

    if isinstance(a, int):
        seed_all(seed)
        indices = np.random.choice(2000, size=a, replace=False)
    else:
        indices = a
    comparison_inputs = []
    comparison_targets = []
    for i in indices:
        comparison_input, comparison_target = dataset[i]
        comparison_inputs.append(comparison_input)
        comparison_targets.append(comparison_target)
    if export_names:
        test_file = TEST_IMAGES_PATH.open("w")
        for idx in indices:
            test_file.write(f"{dataset.imgs[idx][0]}\n")
        test_file.close()
    return torch.stack(comparison_inputs), torch.stack(comparison_targets)


def get_estimations(experiment_dir: Union[str, Path], model: Net,  mode: str = "val", device: str = "cpu", export_names: bool = False):
    experiment_dir = Path(experiment_dir) if isinstance(experiment_dir, str) else experiment_dir
    sample_input, _ = sample_from_dataset(100, "val", device=torch.device(device), seed=147, export_names=export_names)
    # ckp_path = experiment_dir / "checkpoint.pt"
    output_path = experiment_dir / "estimations.npy"
    # model.load_state_dict(torch.load(ckp_path))
    model.eval()
    with torch.no_grad():  # this allows not having detach.
        net_out = model(sample_input)
    net_out = (net_out.cpu().numpy()/2 + 0.5)*255
    np.save(output_path, net_out)


def train(
        batch_size: int,
        min_epochs: int,
        max_epochs: int,
        lr: float,
        experiment_name: str,
        device: str = None,
        visualize: bool = False,
        load_checkpoint: bool = True,
        model_params: dict = None,
        early_stopping_params: dict = None,
        seed: int = 42
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_params = model_params or {}
    early_stopping_params = early_stopping_params or {}
    device = torch.device(device)
    print('device: ' + str(device))
    net = Net(**model_params).to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    train_loader, val_loader = get_loaders(batch_size, device)

    train_losses, val_losses = [], []
    seed_all(seed)

    if load_checkpoint:
        print('loading the model from the checkpoint')
        ckp_path = os.path.join(LOG_DIR, 'checkpoint.pt')
        net.load_state_dict(torch.load(ckp_path))

    experiment_dir = LOG_DIR / experiment_name
    # Create dirs if not exists
    (experiment_dir / "examples").mkdir(exist_ok=False, parents=True)
    early_stopping_params["path"] = experiment_dir / "checkpoint.pt"
    early_stopping_params.setdefault("delta", 5e-5)
    early_stopping_params.setdefault("patience", 3)
    early_stopper = EarlyStopping(**early_stopping_params)
    training_params = {
        "batch_size"     : batch_size,
        "min_epochs"     : min_epochs,
        "max_epochs"     : max_epochs,
        "lr"             : lr,
        "device"         : device,
        "visualize"      : visualize,
        "load_checkpoint": load_checkpoint,
        "model_params"   : model_params,
        "seed"           : seed
    }

    print('training begins')
    for epoch in range(max_epochs):
        running_loss = 0.0  # training loss of the network
        for iteri, data in enumerate(train_loader, 0):
            net.train()  # switch to train mode
            inputs, targets = data  # inputs: low-resolution images, targets: high-resolution images.

            optimizer.zero_grad()  # zero the parameter gradients

            # do forward, backward, SGD step
            preds = net(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            # print loss
            running_loss += loss.item()
            print_n = 100  # feel free to change this constant
            if iteri % print_n == (print_n-1):    # print every print_n mini-batches
                # note: you most probably want to track the progress on the validation set as well
                # (needs to be implemented)
                val_loss = 0
                net.eval()  # go into eval mode
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        y_pred = net(x_val)
                        loss = criterion(y_pred, y_val)
                        val_loss += loss.item()
                    val_loss = val_loss / len(val_loader)
                if len(val_losses) == 0 or (len(val_losses) > 0 and val_loss < val_losses[-1]):
                    # Check if we made progress
                    export_path = experiment_dir / f'examples/epoch_end-{epoch + 1}.png'

                val_losses.append(val_loss)
                train_loss = running_loss / 100
                train_losses.append(train_loss)
                pixelwise_acc = evaluate_model(experiment_dir, net)
                print('[%d, %5d] network-loss: %.5f | validation-loss: %.5f | pixelwise_acc: %.5f' %
                      (epoch + 1, iteri + 1, train_loss, val_loss, pixelwise_acc))
                running_loss = 0.0
                training_params["best_cp"] = {
                    "epoch": epoch + 1,
                    "iter": iteri+1,
                    "loss" : {
                        "train": train_losses[-1],
                        "val"  : val_losses[-1]
                    },
                    "test_images_acc": pixelwise_acc
                }
                if epoch >= min_epochs:
                    early_stopper(val_loss, net, training_params)
                if early_stopper.early_stop:
                    break

            if (iteri == 0) and visualize:
                hw3utils.visualize_batch(inputs, preds, targets)
        visualize_loss_plot(train_losses, val_losses, experiment_dir)
        hw3utils.visualize_batch(inputs, preds, targets, export_path)
        if early_stopper.early_stop:
            break

    comp_size = 12
    total_comps = 3
    for comp in range(total_comps):
        seed = 100 * (comp+1)
        comparison_vis_path = experiment_dir / f"comparison_val_{comp+1}.png"
        comparison_inputs, comparison_targets = sample_from_dataset(comp_size, mode="val", device=device, seed=seed)
        comparison_outs = net(comparison_inputs)
        hw3utils.visualize_batch(comparison_inputs, comparison_outs, comparison_targets, comparison_vis_path)
    print('Finished Training')


def evaluate_model(experiment_dir, model):
    get_estimations(experiment_dir, model, device="cuda")
    estimations_path = experiment_dir / "estimations.npy"
    test_images = PROJECT_ROOT / "test_images.txt"
    print("Avg. Pixelwise Accuracy:")
    return evaluate([estimations_path, test_images])


def visualize_loss_plot(train_losses, val_losses, experiment_dir):
    index_vals = list(range(len(train_losses)))
    plt.plot(index_vals, train_losses, label='Training Loss')
    plt.plot(index_vals, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(experiment_dir / "loss.png")


def check_early_stopping(val_losses, init_patience: int = 30,  improvement_rate: float = 1e-2) -> True:
    """
    Checks if training procedure should be stopped (i.e. if model converged). Unlike
    common implementations, I'll check for relative improvement w.r.t to percentage
    as it's more inherent (imho) when judging the desired improvement amount. Furthermore, I'll
    compare the most recent validation loss with the previous 4th, this corresponds to
    effectively comparing among epochs.

    Args:
        val_losses (list(int)): List of validation losses saved so far during training.
        init_patience (int): How long should the algorithm wait before comparing improvements.
            30 by default corresponding to 10 epochs. This is not the patience between epochs.
            This function does not implement commonly used patience to gracefully determine
            whether stopping training or not.
        improvement_rate (float): The desired improvement rate between epochs (effectively)
            to continue training. Must be in range (0,1).
    """
    assert 0 < improvement_rate < 1, "Improvement rate must be between 0 and 1."
    init_patience = 4 if init_patience < 4 else init_patience
    if len(val_losses) < init_patience:
        return False

    d = val_losses[-1] / val_losses[-4]
    if 1 - d < improvement_rate:
        return True
    return False


def grid_search_train():
    batch_size = 16
    max_num_epoch = 100
    min_num_epoch = 5
    h_channels = [2, 8]

    torch.multiprocessing.set_start_method('spawn', force=True)
    for h_channel in h_channels:
        hps = {'lr': 0.05}
        model_params = {
            "n_conv"    : 2,
            "h_channels": h_channel
        }
        experiment_name = f"q1-1_nlayer={model_params['n_conv']}_hc={model_params['h_channels']}_lr={hps['lr']}"
        train(
                batch_size,
                min_num_epoch,
                max_num_epoch,
                hps['lr'],
                experiment_name=experiment_name,
                device=None,
                visualize=True,
                load_checkpoint=False,
                model_params=model_params
        )


if __name__ == "__main__":
    grid_search_train()
