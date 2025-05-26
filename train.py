import argparse
import functools
import logging
import os
import shutil
from typing import Sequence

import gin
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import dataset as dataset_lib
import model as model_lib  # Used in Gin config.


def _multivariate_gnll_loss(
    means: torch.Tensor,
    targets: torch.Tensor,
    covariance: torch.Tensor,
    eps: float = 1e-6,
    device: str = 'cpu'
) -> torch.Tensor:
    """Computes the multivariate Gaussian negative log-likelihood loss."""
    variance_loss = torch.maximum(
        torch.logdet(covariance), torch.tensor(eps, device=device=device)
    )
    diff = (means - targets).unsqueeze(-1)
    mean_loss = torch.transpose(diff, 1, 2) @ torch.inverse(covariance) @ diff
    return torch.mean(mean_loss.squeeze() / 2 + variance_loss / 2)


@gin.configurable
class TrainingLoop:
    """The training loop which trains and evaluates a model."""

    def __init__(
        self,
        *,
        model: nn.Module,
        save_path: str,
        loss_type: str = 'mgnll',
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        weight_decay: float = 0.0,
        dataset_cls: torch.utils.dataset.Dataset = dataset_lib.NisqaFeatures,
        num_epochs: int = 500,
        learning_rate: float = 1e-4,
        batch_size_train: int = 64,
        ssl_layer: int = 11,
    ):
        """Initializes the instance.
        
        Args:
           model: The model to train.
            save_path: Path to the directory where to save the model and logs.
            loss_type: The type of loss to use, either 'mgnll' or 'mse'.
            optimizer: The optimizer to use for training.
            weight_decay: The weight decay for the optimizer.
            dataset_cls: The dataset class to use for loading the data.
            num_epochs: The number of epochs to train the model.
            learning_rate: The learning rate for the optimizer.
            batch_size_train: The batch size for training.
            ssl_layer: The layer of the SSL model to use for feature extraction.
        """
        # Setup logging and paths.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print('New directory added!')
        log_path = os.path.join(save_path, 'train.log')
        self._save_path = save_path
        logging.basicConfig(filename=log_path, level=logging.INFO)

        # Datasets.
        dataset_cls_partial = functools.partial(dataset_cls, layer=ssl_layer)
        def _get_dataloaders(dataset_cls, names):
            dataloaders = []
            for name in names:
                dataloaders.append(dataset_lib.get_dataloader(
                    dataset=dataset_cls(dataset_name=name),
                    batch_size=1
                ))
            return dataloaders
        train_dataset = dataset_cls_partial(dataset_name='train')
        self._label_type = train_dataset.label_type
        self._train_loader = dataset_lib.get_dataloader(
            dataset=train_dataset,
            batch_size=batch_size_train
        )
        self._valid_loaders = _get_dataloaders(
            dataset_cls_partial,
            ['NISQA_VAL_SIM', 'NISQA_VAL_LIVE']
        )
        self._test_loaders = _get_dataloaders(
            dataset_cls_partial,
            ['NISQA_TEST_LIVETALK','NISQA_TEST_FOR', 'NISQA_TEST_P501']
        )

        # Model and optimizers.
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'Device={self._device}')
        self._model = model(
            device=self._device,
            in_shape=train_dataset.features_shape
        ).to(self._device)
        self._best_pcc = -1
        # TODO: Explore some learning rate scheduler.
        self._optimizer = optimizer(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self._optimizer.zero_grad()
        self._loss_type = loss_type
        if loss_type == 'mgnll':
            self._loss_fn = functools.partial(
                _multivariate_gnll_loss,
                device=self._device
            )
        elif loss_type == 'mse':
            self._loss_fn = nn.MSELoss()
        else:
            raise ValueError(f'{loss_type=} is an invalid loss type.')
        self._all_loss = []
        self._epoch = 0
        self._num_epochs = num_epochs
    
    @property
    def save_path(self):
        """The path to the log directory."""
        return self._save_path
            
    def _train_once(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Performs forward and backward pass on batch.

        Args:
            batch: The batch consisting of the spectrograms and labels.
        """
        features, labels = batch
        features = features.to(self._device)
        labels = labels.to(self._device)

        # Forward
        if self._loss_type == 'mgnll':
            means, covariance = self._model(features)
            loss = self._loss_fn(means, labels, covariance)
        elif self._loss_type == 'mse':
            means = self._model(features)
            loss = self._loss_fn(means, labels)
                
        # Backwards
        loss.backward()
        self._all_loss.append(loss.item())
        del loss

        # Gradient clipping
        nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5)
        self._optimizer.step()
        self._optimizer.zero_grad()
    
    def train(self, valid_each_epoch: bool = True) -> None:
        """Trains the model on the train data `self._num_epochs` number of epochs.
        
        Args:
            valid_each_epoch: If to compute the validation performance.
        """
        self._model.train()
        while self._epoch <= self._num_epochs:
            self._all_loss = list()
            for batch in tqdm.tqdm(
                self._train_loader,
                ncols=0,
                desc="Train",
                unit=" step"
            ):
                self._train_once(batch)

            average_loss = torch.FloatTensor(self._all_loss).mean().item()
            logging.info(f'Average loss={average_loss}')

            if valid_each_epoch:
                self.valid()
            self._epoch += 1

    def _evaluate(
        self,
        dataloaders: Sequence[torch.utils.data.DataLoader],
        prefix: str,
    ) -> None:
        """Evaluates the model on the data based on quality prediction."""
        self._model.eval()
        for dataloader in dataloaders:
            label_names = ['mos', 'noi', 'col', 'dis', 'loud']
            predictions = {name: [] for name in label_names}
            labels = {name: [] for name in label_names}
            for i, batch in enumerate(tqdm.tqdm(
                dataloader,
                ncols=0,
                desc=prefix,
                unit=' step'
            )):
                feature, label = batch
                feature = feature.to(self._device)

                with torch.no_grad():
                    prediction = self._model(feature)
                    if self._loss_type == 'mgnll':
                        prediction, _ = prediction
                    prediction = prediction.cpu().detach().numpy()
                    for i, name in enumerate(label_names):
                        predictions[name].extend(prediction[:, i].tolist())
                        labels[name].extend(label[:, i].tolist())
           
            for name in label_names:
                pred_cur = np.array(predictions[name])
                target_cur = np.array(labels[name])
                utt_mse = np.mean((target_cur - pred_cur) ** 2)
                utt_pcc = np.corrcoef(target_cur, pred_cur)[0][1]
                utt_srcc = scipy.stats.spearmanr(target_cur, pred_cur)[0]
                if utt_pcc > self._best_pcc and name == 'mos' and prefix == 'Valid':
                    self._best_pcc = utt_pcc
                    self.save_model('model_best.pt')
                logging.info(
                    f"\n[{dataloader.dataset.dataset_name}][{name}][{self._epoch}][UTT][ MSE = {utt_mse:.4f} | LCC = {utt_pcc:.4f} | SRCC = {utt_srcc:.4f} ]"
                )
        self._model.train()
    
    def valid(self):
        """Evaluates the model on validation data."""
        self._evaluate(self._valid_loaders, 'Valid')
    
    def test(self, plot: bool = False) -> None:
        """Evaluates the model on test data."""
        self._model = torch.jit.load(
            os.path.join(self._save_path, 'model_best.pt')
        ).to(self._device)
        self._evaluate(self._valid_loaders, 'Test')
        predictions, labels = self._evaluate(self._test_loaders, 'Test')
        if plot:
            plt.scatter(labels['mos'], predictions['mos'])
            plt.xlim([0.9, 5.1])
            plt.ylim([0.9, 5.1])
            plt.xlabel(self._label_type)
            plt.ylabel('Predictions')
            plt.title('Test data predictions vs targets')
            plt.savefig(os.path.join(self._save_path, 'test_scatter.png'))
            plt.close()

    def save_model(self, model_name: str = 'model.pt') -> None:
        """Saves the model."""
        model_scripted = torch.jit.script(self._model)
        model_scripted.save(os.path.join(self._save_path, model_name))


def main():
    """Main."""
    parser = argparse.ArgumentParser(description='Gin and save path.')
    parser.add_argument(
        '--gin_path',
        type=str,
        help='Path to the gin-config.',
        default='configs/tot.gin'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Path to directory storing results.',
    )
    parser.add_argument(
        '--layer',
        type=int,
        help='Layer of SSL model, leave as default if trained on input specs.',
        default=11,
    )
    args = parser.parse_args()

    gin.external_configurable(
            torch.nn.modules.activation.ReLU,
            module='torch.nn.modules.activation'
            )
    gin.external_configurable(
            torch.nn.modules.activation.SiLU,
            module='torch.nn.modules.activation'
            )
    gin.parse_config_file(args.gin_path)
    train_loop = TrainingLoop(save_path=args.save_path)
    new_gin_path = os.path.join(train_loop.save_path, 'config.gin')
    shutil.copyfile(args.gin_path, new_gin_path)
    train_loop.train()
    train_loop.test(plot=True)


if __name__ == '__main__':
    main()
