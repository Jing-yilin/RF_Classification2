"""An implementation of a typical multi-class classification training loop.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import os
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from rfml.data import build_dataset

from rfml.nn.eval import (
    compute_accuracy,
    compute_accuracy_on_cross_sections,
    compute_confusion,
)

# Internal Includes
from .base import TrainingStrategy
from rfml.data import Dataset, Encoder
from rfml.nn.model import Model


import logging
logger = logging.getLogger(__name__)


class StandardTrainingStrategy(TrainingStrategy):
    """A typical strategy that would be used to train a multi-class classifier.

    Args:
        lr (float, optional): Learning rate to be used by the optimizer.
                              Defaults to 10e-4.
        max_epochs (int, optional): Maximum number of epochs to train before
                                    stopping training to preserve computing
                                    resources (even if the network is still
                                    improving). Defaults to 50.
        patience (int, optional): Maximum number of epochs to continue to train
                                  for even if the network is not still
                                  improving before deciding that overfitting is
                                  occurring and stopping. Defaults to 5.
        batch_size (int, optional): Number of examples to give to the model at
                                    one time.  If this value is set too high,
                                    then an out of memory error could occur.  If
                                    the value is set too low then training will
                                    take a longer time (and be more variable).
                                    Defaults to 512.
        gpu (bool, optional): Flag describing whether the GPU is used or the
                              training is performed wholly on the CPU.
                              Defaults to True.
    """

    def __init__(
        self,
        lr: float = 10e-4,
        max_epochs: int = 50,
        patience: int = 5,
        batch_size: int = 512,
        device: str = "cuda:1",
    ):
        super().__init__()
        if lr <= 0.0 or lr >= 1.0:
            raise ValueError(
                "A sane human would choose a learning rate between 0-1, but, you chose "
                "{}".format(lr)
            )
        if max_epochs < 1:
            raise ValueError(
                "You must train for at least 1 epoch, you set the max epochs as "
                "{}".format(max_epochs)
            )
        if patience < 1:
            raise ValueError(
                "Patience must be a positive integer, not {}".format(patience)
            )
        if batch_size < 1:
            raise ValueError(
                "The batch size must be a positive integer, not {}".format(
                    batch_size)
            )

        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device

    def __call__(
        self, model: Model, training: Dataset, validation: Dataset, le: Encoder
    ):
        """Train the model on the provided training data.

        Args:
            model (Model): Model to fit to the training data.
            training (Dataset): Data used in the training loop.
            validation (Dataset): Data only used for early stopping validation.
            le (Encoder): Mapping from human readable labels to model readable.
        """
        # The PyTorch implementation of CrossEntropyLoss combines the softmax
        # and natural log likelihood loss into one (so none of the models
        # should have softmax included in them)
        criterion = CrossEntropyLoss()

        # if self.gpu:
        #     model.cuda()
        #     criterion.cuda()

        # device
        model.to(self.device)
        criterion.to(self.device)

        optimizer = Adam(model.parameters(), lr=self.lr)

        train_data = DataLoader(
            training.as_torch(le=le), shuffle=True, batch_size=self.batch_size
        )
        val_data = DataLoader(
            validation.as_torch(le=le), shuffle=True, batch_size=self.batch_size
        )

        # Fit the data for the maximum number of epochs, bailing out early if
        # the early stopping condition is reached.  Set the initial "best" very
        # high so the first epoch is always an improvement
        best_val_loss = 10e10
        epochs_since_best = 0
        best_epoch = 0

        train_acc_list = []
        val_acc_list = []

        for epoch in range(0, self.max_epochs):
            train_loss = self._train_one_epoch(
                model=model, data=train_data, loss_fn=criterion, optimizer=optimizer
            )
            self._dispatch_epoch_completed(mean_loss=train_loss, epoch=epoch)

            val_loss = self._validate_once(
                model=model, data=val_data, loss_fn=criterion
            )
            self._dispatch_validation_completed(
                mean_loss=val_loss, epoch=epoch)

            train, val, test, le = build_dataset(
                dataset_name="RML2016.10a", path="./rfml/data/RML2016.10a_dict.pkl"
            )

            # do one time prediction and save accuracy
            # acc = compute_accuracy(model=model, data=test, le=le,
            #                        store_data=True, store_csv_file=store_train_csv_file)

            train_acc = compute_accuracy(
                model=model,
                data = train,
                le=le
            )

            val_acc = compute_accuracy(
                model=model,
                data = val,
                le=le
            )

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            logger.info(f"train acc: {train_acc}, val acc: {val_acc}")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_best = 0
                best_epoch = epoch
                model.save()
            else:
                epochs_since_best += 1

            if epochs_since_best >= self.patience:
                break

        # save the accuracy
        import pandas as pd
        data_log_dir = "./RML2016_data_log/"
        if not os.path.exists(data_log_dir):
            os.makedirs(data_log_dir)

        all_data_df = pd.DataFrame(
            {
                "train_acc": train_acc_list,
                "val_acc": val_acc_list,
            }   
        )
        all_data_df.to_csv(f"{data_log_dir}/all_data_acc.csv", index=True)

        # Reload the "best" weights
        model.load()
        self._dispatch_training_completed(
            best_loss=best_val_loss, best_epoch=best_epoch, total_epochs=epoch
        )

    def _train_one_epoch(
        self, model: Model, data: DataLoader, loss_fn: CrossEntropyLoss, optimizer: Adam
    ) -> float:
        total_loss = 0.0
        # Switch the model mode so it remembers gradients, induces dropout, etc.
        model.train()

        for i, batch in enumerate(data):
            x, y = batch

            # Push data to GPU
            # if self.gpu:
            #     x = Variable(x.cuda())
            #     y = Variable(y.cuda())
            # else:
            #     x = Variable(x)
            #     y = Variable(y)

            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass of prediction
            outputs = model(x)

            # Zero out the parameter gradients, because they are cumulative,
            # compute loss, compute gradients (backward), update weights
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / (i + 1.0)
        return mean_loss

    def _validate_once(
        self, model: Model, data: DataLoader, loss_fn: CrossEntropyLoss
    ) -> float:
        total_loss = 0.0
        # Switch the model back to test mode (so that batch norm/dropout doesn't
        # take effect)
        model.eval()
        for i, batch in enumerate(data):
            x, y = batch

            # if self.gpu:
            #     x = x.cuda()
            #     y = y.cuda()

            x = x.to(self.device)
            y = y.to(self.device)

            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()

        mean_loss = total_loss / (i + 1.0)
        return mean_loss
