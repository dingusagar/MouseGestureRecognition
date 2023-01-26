import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import datasets

from cfg import ModelConfig
from models.shape_detector_model import ShapeDetectionModel


def start_training():
    model = ShapeDetectionModel(ModelConfig)

    # Load the datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(ModelConfig.data_dir, x), model.data_transforms[x]) for x in
                      ['train', 'test']}

    # Create the data loaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=ModelConfig.batch_size, shuffle=True, num_workers=4) for
                   x
                   in ['train', 'test']}

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, mode="min"),
        ModelCheckpoint(monitor='val_loss', filename='best_checkpoint', dirpath='lightning_logs/latest/')

    ]
    trainer = Trainer(callbacks=callbacks)
    trainer.fit(model, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['test'])


if __name__ == "__main__":
    start_training()
