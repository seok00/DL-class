import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
# from pytorch_lightning.metrics import functional as FM
from torchmetrics import functional as FM


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.layer_2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.layer_3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)
        self.save_hyperparameters()
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.layer_1(x)))
        x = self.maxpool(self.relu(self.layer_2(x)))
        x = self.maxpool(self.relu(self.layer_3(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == '__main__':
    import os
    import argparse
    from torchvision import transforms
    from torchvision.datasets.cifar import CIFAR10
    from torch.utils.data import DataLoader, random_split
    from pytorch_lightning.callbacks import ModelCheckpoint

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--n_gpus', default=0, type=int)
    parser.add_argument('--save_top_k', default=5, type=int)
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # dataloaders
    dataset = CIFAR10('../data/CIFAR_10', train=True, download=True, transform=transforms.ToTensor())
    train_dataset, val_dataset = random_split(dataset, [45000, 5000])
    test_dataset = CIFAR10('../data/CIFAR_10', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = Classifier()

    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(project='2017125074_허진석_pytorch lightning Cifar10')

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('checkpoints', '{epoch:d}'),
        verbose=True,
        save_last=True,
        save_top_k=args.save_top_k,
        monitor='val_acc',
        mode='max'
    )

    # training
    trainer_args = {
        'callbacks': [checkpoint_callback],
        'max_epochs': args.n_epochs,
        'gpus': args.n_gpus
    }
    if args.checkpoint:
        trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args.checkpoint)

    trainer = pl.Trainer(**trainer_args, logger = wandb_logger)
    trainer.fit(model, train_loader, val_loader)

    # testing
    trainer.test(dataloaders=test_loader)