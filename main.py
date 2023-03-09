from ast import literal_eval as make_tuple
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from streaming import StreamingDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

BATCH_SIZE = 4


class Dataset(StreamingDataset):
    def _process(self, obj):
        image = obj["image"]
        dtype = obj["dtype"]
        shape = make_tuple(obj["shape"])
        object = np.frombuffer(image, dtype=dtype)

        image = object.reshape(shape)
        return image.astype(np.float32)

    def __init__(self, local, remote, transform=None, **kwargs):
        super().__init__(local, remote, **kwargs)

        self.transform = transform

    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)

        label = obj["class"]
        filename = obj["filename"]
        data = self._process(obj)

        if self.transform:
            data = self.transform(data)

        return data, label, filename


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        aux_params = dict(
            pooling="avg",
            activation="softmax",
            classes=62,
        )
        self.model = smp.Unet(
            encoder_name="resnet50",
            in_channels=8,
            classes=62,
            aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y, _ = batch
        _, y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    # Init a random seed to fix a bug
    np.random.seed(12341)

    # Init our model
    mnist_model = Model()

    # Init our dataset
    dataset = Dataset(
        local="/tmp/testing/cache",
        remote="s3://nharada-public-data/datasets/fmow_test_set",
        split="train",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.RandomCrop((256, 256)),
            ]
        ),
    )

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Initialize a trainer
    trainer = Trainer(
        max_epochs=200,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        strategy="ddp",
        accelerator="cpu",
        devices=4,
    )

    # Train the model
    trainer.fit(mnist_model, train_loader)


if __name__ == "__main__":
    main()
