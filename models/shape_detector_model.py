import os

import timm
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms

from cfg import ModelConfig

dataset_dir = ModelConfig.data_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShapeDetectionModel(LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        print(f'Initialising model {model_cfg.model_name} with classification head for classes : {model_cfg.classes}')
        self.model = timm.create_model(model_cfg.model_name, pretrained=True, num_classes=len(model_cfg.classes))
        data_dir = model_cfg.data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'test')
        self.batch_size = model_cfg.batch_size
        self.lr = model_cfg.lr

        # Define the data transforms
        self.data_transforms = {
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

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = self.data_transforms['test'](X)
            if len(X.shape) == 3: # single image, add batch dimension
                X = X.unsqueeze(0)
            X = X.to(device)
            y_hat = self(X)
            return y_hat