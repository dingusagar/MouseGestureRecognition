import os
import random
from shutil import copyfile

from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from sklearn.model_selection import train_test_split
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

dataset_dir = Path('image_dataset')
classes = os.listdir(dataset_dir / 'train')
BATCH_SIZE = 16

model_name = 'resnet18'
# train_dir = dataset_dir / 'train'
# test_dir = dataset_dir / 'test'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# Load the datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x]) for x in ['train', 'test']}

# Create the data loaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'test']}

class FineTuningModel(LightningModule):
    def __init__(self, model_name, num_classes, data_dir, batch_size, lr):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'test')
        self.batch_size = batch_size
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss



model = FineTuningModel(model_name='resnet18',
                        num_classes=len(classes),
                        data_dir=dataset_dir,
                        batch_size=BATCH_SIZE,
                        lr=0.0001)

callbacks = [
    EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, mode="min")
]
trainer = Trainer(callbacks=callbacks)
trainer.fit(model, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['test'])
