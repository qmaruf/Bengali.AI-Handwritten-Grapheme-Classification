import os
import ast
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDataset
from argparse import ArgumentParser
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore")

TRAINING_FOLDS_CSV=os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT=int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH=int(os.environ.get("IMG_WIDTH"))
EPOCHS=int(os.environ.get("EPOCHS"))
LEARNING_RATE=float(os.environ.get("LEARNING_RATE"))

TRAIN_BATCH_SIZE=int(os.environ.get("TRAIN_BATCH_SIZE"))
VALID_BATCH_SIZE=int(os.environ.get("VALID_BATCH_SIZE"))
MODEL_MEAN=ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD=ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS=ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATIONS_FOLDS=ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
TRAINING_FOLDS=ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
BASE_MODEL=os.environ.get("BASE_MODEL")


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()
        self.hparams = hparams
        self.model = MODEL_DISPATCHER[BASE_MODEL]()
        print (self.model)
        # exit()

    def forward(self, x):
        return self.model(x)        

    def criterion(self, preds, targets):
        y1pred, y2pred, y3pred = preds
        y1, y2, y3 = targets        
        # l1 = nn.CrossEntropyLoss()(y1pred, y1)
        # l2 = nn.CrossEntropyLoss()(y2pred, y2)
        # l3 = nn.CrossEntropyLoss()(y3pred, y3)
        # print (l3)

        l1 = LabelSmoothingLoss(168)(y1pred, y1)
        l2 = LabelSmoothingLoss(11)(y2pred, y2)
        l3 = LabelSmoothingLoss(7)(y3pred, y3)
        # print (l1.item(), l2.item(), l3.item())

        avgl = (l1 * 0.5 + l2 * 0.25 + l3 * 0.25)/3.0        
        return avgl
        
        
    
    def get_data(self, batch):
        image = batch['image']
        target1 = batch['grapheme_root']
        target2 = batch['vowel_diacritic']
        target3 = batch['consonant_diacritic']
        return image, [target1, target2, target3]
    
    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, targets = self.get_data(batch)        
        preds = self.forward(x)
        return {'loss': self.criterion(preds, targets)}

    def validation_step(self, batch, batch_idx):
        
        x, targets = self.get_data(batch)        
        preds = self.forward(x)
        return {'val_loss': self.criterion(preds, targets)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True, )
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        train_dataset = BengaliDataset(
            folds = TRAINING_FOLDS,
            img_height = IMG_HEIGHT,
            img_width = IMG_WIDTH,
            mean = MODEL_MEAN,
            std = MODEL_STD
        )
        
        train_dl = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = TRAIN_BATCH_SIZE,
            shuffle = True,
            num_workers = 2
        )
        
        return train_dl 
        

    @pl.data_loader
    def val_dataloader(self):
        val_dataset = BengaliDataset(
            folds = VALIDATIONS_FOLDS,
            img_height = IMG_HEIGHT,
            img_width = IMG_WIDTH,
            mean = MODEL_MEAN,
            std = MODEL_STD
        )
        
        val_dl = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size = VALID_BATCH_SIZE,
            shuffle = False,
            num_workers = 2
        )
        
        return val_dl 

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)

        # training specific (for this model)
        # parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser
    

early_stop_callback = EarlyStopping( monitor='val_loss', patience=5, verbose=True, mode='min')
checkpoint_callback = ModelCheckpoint(filepath='./checkpoint_fold_%d/'%VALIDATIONS_FOLDS, save_top_k=1, verbose=True, monitor='val_loss', mode='min', prefix='')



def main(hparams):
    mymodel = CoolSystem(hparams)
    print (hparams)
    # exit()
    trainer = Trainer(
        max_nb_epochs=EPOCHS,
        gpus=[0,1],
        early_stop_callback=early_stop_callback, 
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(mymodel)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = CoolSystem.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)