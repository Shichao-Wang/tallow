import logging
import os

import torch
import torchmetrics as tm
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from tallow import evaluators, trainers
from tallow.data import datasets

logger = logging.getLogger(__name__)


class MNISTDataset(datasets.MapDataset):
    def __init__(self, data_folder: str = "/tmp", train: bool = True) -> None:
        self._mnist = MNIST(
            data_folder, train=train, transform=ToTensor(), download=True
        )
        super().__init__(len(self._mnist), shuffle=train)

    def __getitem__(self, index: int):
        img, label = self._mnist[index]
        return {"img": img, "label": label}


def load_datasets(data_folder: str, batch_size: int):
    return {
        "train": MNISTDataset(data_folder, train=True).batch(batch_size),
        "val": MNISTDataset(data_folder, train=False).batch(batch_size),
    }


class LinearModel(nn.Module):
    def __init__(self, input_size: int, num_labels: int) -> None:
        super().__init__()
        self._linear = nn.Linear(input_size, num_labels)

    def forward(self, img: torch.Tensor):
        x = img.view(img.size(0), -1)
        x = self._linear(x)
        return {"logit": x}


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, logit: torch.Tensor, label: torch.Tensor):
        return super().forward(logit, label)


class Accuracy(tm.Accuracy):
    def update(self, logit: torch.Tensor, label: torch.Tensor) -> None:
        return super().update(logit, label)


def main():
    save_folder_path = "saved_models"
    model = LinearModel(28 * 28, 10)

    def _init_optim(m: nn.Module):
        return optim.Adam(m.parameters())

    val_metric = Accuracy()
    os.makedirs(save_folder_path, exist_ok=True)
    earlystop = trainers.hooks.EarlyStopHook(save_folder_path, 2)
    ckpt = trainers.hooks.CheckpointHook(save_folder_path, reload_on_begin=True)
    trainer = trainers.SupervisedTrainer(
        model,
        criterion=CrossEntropyLoss(),
        init_optim=_init_optim,
        val_metric=val_metric,
        hooks=[ckpt, earlystop],
    )

    datasets = load_datasets("/tmp", batch_size=128)
    trainer.execute(datasets["train"], datasets["val"])

    print(earlystop.best_model_path)
    model.load_state_dict(torch.load(earlystop.best_model_path)["model"])
    full_metric = Accuracy()
    evaluator = evaluators.trainer_evaluator(trainer, full_metric)
    results = evaluator.execute(datasets)
    print(results)


if __name__ == "__main__":
    main()
