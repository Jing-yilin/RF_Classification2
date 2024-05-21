import os
import torch
import csv
import numpy as np
from rfml.nn.model.resnet_CT import resnet18, resnetx
from rfml.nn.model.transformer import Transformer
import dataloader_rf1024 as dataloader
import torch.optim as optim
import torch.nn as nn
import argparse


# Logging
import logging
from rich.logging import RichHandler
from datetime import datetime

log_dir = "logs/RF1024_Train"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = f"{log_dir}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

file_handler = logging.FileHandler(
    filename=log_file, mode="w"
)

file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(), file_handler],
)
logger = logging.getLogger(__name__)


# arguments
parser = argparse.ArgumentParser(description="RF1024 Training")
parser.add_argument(
    "--model", default="resnet18", type=str, help="resnet18 / conv5 / CNN / CLDNN"
)
parser.add_argument("--lr", default=0.001, type=int,
                    help="Learning rate for training")
parser.add_argument("--momentum", default=0.9, type=int,
                    help="Momentum for training")
parser.add_argument("--epoch", default=20, type=int,
                    help="Number of training epochs")
parser.add_argument("--model_save", default=True,
                    type=bool, help="resnet18 or conv5")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--shuffle", default=True, type=bool, help="Shuffle data")
parser.add_argument("--drop_last", default=True, type=bool, help="Drop last")
parser.add_argument("--num_workers", default=0,
                    type=int, help="Number of workers")
parser.set_defaults(augment=True)

global args
args = parser.parse_args()

# log args
logger.info(f"🚀Model: {args.model}")
logger.info(f"🚀Learning Rate: {args.lr}")
logger.info(f"🚀Momentum: {args.momentum}")
logger.info(f"🚀Epoch: {args.epoch}")
logger.info(f"🚀Model Save: {args.model_save}")
logger.info(f"🚀Batch Size: {args.batch_size}")
logger.info(f"🚀Shuffle: {args.shuffle}")
logger.info(f"🚀Drop Last: {args.drop_last}")
logger.info(f"🚀Number of Workers: {args.num_workers}")


if torch.backends.mps.is_available():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "mps")
else:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

logger.info(f"🚀Device: {device}")


def write_csv(file, newrow):
    with open(file, mode="a") as f:
        f_writer = csv.writer(
            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        f_writer.writerow(newrow)


valid_loss_min = np.Inf

data_log_dir = "RF1024_data_log"
if not os.path.exists(data_log_dir):
    os.makedirs(data_log_dir)

if __name__ == "__main__":
    train_dataset = dataloader.train_data(
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
    )
    val_dataset = dataloader.val_data(
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
    )
    if args.model == "resnet18":
        model = resnet18()
    elif args.model == "conv5":
        model = resnetx()
    elif args.model == "transformer":
        model = Transformer()
    else:
        raise NotImplementedError(
            f"Model: {args.model} not implemented. Choose from [resnet18, conv5]"
        )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    model.to(device)
    total_step = len(train_dataset)
    logger.info(f"Total Step: {total_step}")
    val_loss_list = []
    val_acc_list = []
    train_loss_list = []
    train_acc_list = []
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        logger.info(
            f"===========================Epoch: {epoch}==========================="
        )
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == labels).item()
            total += labels.size(0)
        train_acc_list.append(100 * correct / total)
        train_loss_list.append(running_loss / total_step)
        logger.info(
            f"train loss: {np.mean(train_loss_list):.4f}, train acc: {(100 * correct / total):.4f}"
        )
        write_csv(
            f"{data_log_dir}/train.csv", [epoch,
                                          (100 * correct / total), np.mean(train_loss_list)]
        )
        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in val_dataset:
                data_t, target_t = data_t.to(
                    device), target_t.to(device)  # on GPU
                target_t = target_t.long()
                outputs_t = model(data_t.float())
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc_list.append(100 * correct_t / total_t)
            val_loss_list.append(batch_loss / len(val_dataset))
            network_learned = batch_loss < valid_loss_min
            logger.info(
                f"validation loss: {np.mean(val_loss_list):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n"
            )
            write_csv(
                f"{data_log_dir}/valid.csv",
                [epoch, (100 * correct_t / total_t), np.mean(val_loss_list)],
            )
            # Saving the best weight
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), "best_model.pt")
                logger.info(
                    "Detected network improvement, saving current model")
        model.train()

    # merge all data into df
    import pandas as pd

    all_data_df = pd.DataFrame(
        {
            "train_loss": train_loss_list,
            "train_acc": train_acc_list,
            "val_loss": val_loss_list,
            "val_acc": val_acc_list,
        }
    )
    all_data_df.to_csv(f"{data_log_dir}/all_data.csv", index=True)

