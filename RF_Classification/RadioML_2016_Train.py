import argparse
import torch
from rfml.data import build_dataset
from rfml.nn.eval import (
    compute_accuracy,
    compute_accuracy_on_cross_sections,
    compute_confusion,
)
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener

import os

# Logging
import logging
from rich.logging import RichHandler
from datetime import datetime

log_dir = "logs/RadioML_2016_Train"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = f"{log_dir}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

file_handler = logging.FileHandler(filename=log_file, mode="w")

file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(), file_handler],
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="RML2016 Training")
parser.add_argument(
    "--model",
    default="transformer",
    type=str,
    help="transformer /resnet18 / conv5 / CNN / CLDNN",
)
parser.add_argument("--model_save", default=True, type=bool, help="resnet18 or conv5")
parser.set_defaults(augment=True)

global args
args = parser.parse_args()


# log args
logger.info(f"🚀Model: {args.model}")
logger.info(f"🚀Model Save: {args.model_save}")

if torch.backends.mps.is_available():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "mps")
else:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

logger.info(f"🚀Device: {device}")


data_log_dir = f"./RML2016_data_log/{args.model}"
if not os.path.exists(data_log_dir):
    os.makedirs(data_log_dir)


train, val, test, le = build_dataset(
    dataset_name="RML2016.10a", path="./rfml/data/RML2016.10a_dict.pkl"
)
logger.info(f"classes: {len(le)}")

model = build_model(
    model_name=args.model,
    input_samples=128,
    n_classes=11,
    num_heads=4,
    num_layers=3,
    dropout=0.1,
)
logger.info(model)
trainer = build_trainer(
    strategy="standard", max_epochs=30, device=device, patience=10
)  # Note: Disable the GPU here if you do not have one

trainer.register_listener(PrintingTrainingListener())
trainer(model=model, training=train, validation=val, le=le)

store_train_csv_file = os.path.join(data_log_dir, f"train.csv")
acc = compute_accuracy(
    model=model, data=test, le=le, store_data=False, store_csv_file=store_train_csv_file
)

acc_vs_snr, snr = compute_accuracy_on_cross_sections(
    model=model, data=test, le=le, column="SNR"
)
#
cmn = compute_confusion(model=model, data=test, le=le)

# Calls to a plotting function could be inserted here
# For simplicity, this script only prints the contents as an example
logger.info("===============================")
logger.info("Overall Testing Accuracy: {:.4f}".format(acc))
logger.info("SNR (dB)\tAccuracy (%)")
logger.info("===============================")
for acc, snr in zip(acc_vs_snr, snr):
    logger.info("{snr:d}\t{acc:0.1f}".format(snr=snr, acc=acc * 100))
logger.info("===============================")
logger.info("Confusion Matrix:")
logger.info(cmn)
if args.model_save == True:
    model.save(f"RM2016_best_{args.model}.pt")
