"""Dataset loader for NISQA feature dataset."""

import os

import gin
import numpy as np
import pandas as pd
import torch
import torch.utils.data 
import torch.utils.data.dataset 
import tqdm

    
@gin.configurable
class NisqaFeatures(torch.utils.data.dataset.Dataset):
    """The NISQA dataset, preprocessed by SSL model."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/NISQA_Corpus',
        dataset_name: str = 'NISQA_VAL_SIM',
        ssl_model_name: str = 'w2v2_xlsr_2b',
        layer: int = 11,
        debug: bool = False,
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the NISQA dataset.
            dataset_name: Name of the dataset to load (e.g., 'train',
                'NISQA_VAL_SIM'). If 'train', it loads both the simulated and
                live training sets, and is considered to be the training set.
            sample_rate: Sample rate for audio processing.
            ssl_model_name: Name of the SSL model to use for feature extraction.
            layer: Layer of the SSL model to extract features from.
            debug: If True, limits the number of samples for debugging purposes.
        """
        self._data_path = data_path
        self._valid = dataset_name
        self._ssl_model_name = ssl_model_name
        self._layer = layer
        self._debug = debug

        self._label_names = []
        self._label_names = ['mos', 'noi', 'col', 'dis', 'loud']

        if dataset_name == 'train':
            self._df = pd.read_csv(os.path.join(
                data_path,
                'NISQA_TRAIN_SIM',
                'NISQA_TRAIN_SIM_file.csv'
            ))
            self._num_samples = len(self._df)
            labels_sim = self._load_labels()
            clips_sim = self._load_features()
            self._df = pd.read_csv(os.path.join(
                data_path,
                'NISQA_TRAIN_LIVE',
                'NISQA_TRAIN_LIVE_file.csv'
            ))
            self._num_samples = len(self._df)
            labels_live = self._load_labels()
            clips_live = self._load_features()
            self._labels = pd.concat(
                [labels_sim, labels_live]
            )[self._label_names]
            self._features = clips_sim + clips_live
        else:
            self._df = pd.read_csv(os.path.join(
                data_path,
                dataset_name,
                f'{dataset_name}_file.csv'
            ))
            self._num_samples = len(self._df)
            self._labels = self._load_labels()
            self._features = self._load_features()
        self._num_samples = len(self._features)

    @property
    def features_shape(self) -> np.ndarray:
        """Returns the shape of the features."""
        return self._features[0].shape
    
    @property
    def dataset_name(self):
        """Returns the name of the dataset."""
        return self._valid

    def _load_labels(self) -> pd.DataFrame:
        """Loads the labels."""
        return self._df[self._label_names]

    def _load_features(self) -> list[np.ndarray]:
        """Loads the features."""
        features = []
        for i, path in tqdm.tqdm(
            enumerate(self._df['filepath_deg']),
            total=self._num_samples,
            desc='Loading features...',
        ):
            feature_path = path.replace(
                'deg/',
                f'deg_feature_{self._ssl_model_name}_layer{self._layer}/'
            ).replace('.wav', '.npy')
            feature = np.load(os.path.join(self._data_path, feature_path))
            features.append(feature)
            if self._debug and i == 100:
                break
        return features
   
    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        """Returns a feature and labels thereof."""
        return self._features[idx], self._labels.iloc[idx].to_numpy()
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._features)
 
    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
        features, labels = zip(*batch)
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(np.array(labels))
        return features, labels


@gin.configurable
def get_dataloader(
    dataset: torch.utils.data.dataset.Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Returns a dataloader of the dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
