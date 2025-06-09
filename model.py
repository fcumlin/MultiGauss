"""Multivariate Gaussian SSL-based projection head."""

import functools
from typing import Union, Sequence

import gin
import torch
import torch.nn as nn


def _get_conv1d_layer(
    in_channels: int,
    out_channels: int,
    ln_shape: tuple[int],
    kernel_size: int,
    pool_size: int,
    use_pooling: bool,
    use_normalization: bool,
    dropout_rate: float,
    ) -> nn.Module:
    """Returns a 1D conv layer with optional normalization and pooling."""
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)]
    if use_normalization:
        layers.append(nn.LayerNorm(ln_shape))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    if use_pooling:
        layers.append(nn.MaxPool1d(kernel_size=pool_size))
    return nn.Sequential(*layers)


@gin.configurable
class ProjectionHead(nn.Module):

    def __init__(
            self,
            in_shape: tuple[int],
            conv_channels: Sequence[int] = (32, 32),
            dense_neurons: Sequence[int] = (64, 32, 20),
            use_poolings: Sequence[bool] = (True, True),
            use_normalizations: Sequence[bool] = (True, True),
            kernel_size: int = 5,
            pool_size: int = 5,
            dropout_rate: float = 0.3,
            linear_transform: bool = True,
            device: str = 'cpu',
            ):
        super(ProjectionHead, self).__init__()
        assert len(dense_neurons) == 3
        assert len(conv_channels) == len(use_poolings)
        self._linear_transform = linear_transform
        self._device = device

        conv_layer = functools.partial(
            _get_conv1d_layer,
            kernel_size=kernel_size,
            pool_size=pool_size,
            dropout_rate=dropout_rate,
        )
        ln_shape = (
            conv_channels[0],
            self._get_time_shape(in_shape[1], ksize=5),
        )
        layers = [conv_layer(
            in_shape[0],
            conv_channels[0],
            ln_shape=ln_shape,
            use_pooling=use_poolings[0],
            use_normalization=use_normalizations[0]
        )]
        for in_channels, out_channels, use_pooling, use_pooling_shape, use_normalization in zip(
            conv_channels[:-1],
            conv_channels[1:],
            use_poolings[1:],
            use_poolings[:-1],
            use_normalizations,
        ):
            ln_shape = (out_channels, self._get_time_shape(
                ln_shape[-1],
                ksize=kernel_size,
                pool_size=1 if not use_pooling_shape else pool_size
            ))
            layers.append(conv_layer(
                in_channels,
                out_channels,
                ln_shape=ln_shape,
                use_pooling=use_pooling,
                use_normalization=use_normalization,
            ))
        self._encoder = nn.Sequential(*layers)
        self._flatten = nn.Flatten()
        in_dense = self._get_in_dense(in_shape)
        hidden_dense1, hidden_dense2, out_dense = dense_neurons
        self._head = nn.Sequential(
            nn.Linear(in_dense, hidden_dense1),
            nn.ReLU(),
            nn.Linear(hidden_dense1, hidden_dense2),
            nn.ReLU(),
            nn.Linear(hidden_dense2, out_dense),
        )
        self._softplus = nn.Softplus()
        indices = [(i, j) for i in range(5) for j in range(i+1, 5)]
        if out_dense == 17:
            indices.remove((1, 2))
            indices.remove((1, 3))
            indices.remove((2, 3))
        elif out_dense == 10:
            indices = [(-1, -1)]
        self._indices = indices

    def _get_time_shape(
        self,
        time_dim: int,
        ksize: int,
        dilation: int = 1,
        stride: int = 1,
        padding: int = 0,
        pool_size: int = 1,
    ) -> int:
        return int(((time_dim // pool_size + 2 * padding - dilation * (ksize-1) - 1) / stride + 1))

    def _get_in_dense(self, in_shape: tuple[int]) -> int:
        x = torch.zeros((1,) + in_shape)
        x = self._encoder(x)
        return self._flatten(x).shape[-1]

    def forward(
        self,
        x: torch.Tensor
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = self._encoder(x)
        x = self._flatten(x)
        predictions = self._head(x)
        if predictions.shape[-1] == 5:
            return 2 * predictions + 3
        mean_predictions = predictions[:, 0:5]
        var_predictions = self._softplus(predictions[:, 5:10])

        # Cholesky decomposition trick for proper covariance prediction.
        covariance_predictions = predictions[:, 10:]
        corr_matrix = torch.zeros(
            (predictions.shape[0], 5, 5), device=self._device
        )
        for i in range(5):
            corr_matrix[:, i, i] = var_predictions[:, i]
        for num, (i, j) in enumerate(self._indices):
            if (i, j) == (-1, -1):
                break
            corr_matrix[:, i, j] = covariance_predictions[:, num]
        corr_matrix = torch.matmul(
            corr_matrix, torch.transpose(corr_matrix, 1, 2)
        )
        
        # Apply linear transformation if specified, for unbiased estimator.
        if self._linear_transform:
            constant_values = torch.full(
                (mean_predictions.shape[0], 5), 2.0, device=self._device
            )
            A = torch.diag_embed(constant_values)
            b = torch.full(
                (mean_predictions.shape[0], 5), 3.0, device=self._device
            )
            transformed_mean = torch.matmul(
                A, mean_predictions.unsqueeze(2)
            ).squeeze(2) + b
            transformed_corr = torch.matmul(
                torch.matmul(A, corr_matrix), torch.transpose(A, 1, 2)
            )
            return transformed_mean, transformed_corr
        return mean_predictions, corr_matrix

