# MIT License
#
# Copyright (c) 2018 Bjarte Mehus Sunde
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
"""
Taken and adapted from Bjarten's implementation of early stopping manager.
Ref: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""

	def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement.
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
			path (str): Path for the checkpoint to be saved to.
							Default: 'checkpoint.pt'
			trace_func (function): trace print function.
							Default: print
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.path = path
		self.trace_func = trace_func

	def __call__(self, val_loss, model, training_params):
		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, training_params)
		elif score < self.best_score + self.delta:
			self.counter += 1
			self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, training_params)
			self.counter = 0

	def save_checkpoint(self, val_loss, model, training_params: Dict[str, Any]):
		"""Saves model when validation loss decrease."""
		if self.verbose:
			self.trace_func(
					f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.path)
		with open(Path(self.path).parent / "training_args.json", "w") as fd_out:
			json.dump(training_params, fd_out, default=str, indent=2)
		self.val_loss_min = val_loss
