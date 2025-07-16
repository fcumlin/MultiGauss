"""Multivariate Gaussian SSL-based projection head."""

from typing import Sequence

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
        apply_linear_transform: bool = True,
    ):
        """Neural-based projection head for 'NISQA' speech quality prediction.

        Expects speech clip features from SSL models (e.g., wav2vec 2.0).
        Predicts a 5-dimensional distribution over speech quality.

        Args:
            in_shape: Input shape (D, T) where D=features, T=time dimension.
            conv_channels: Number of channels for each 1D convolution layer.
            dense_neurons: Nodes for each dense layer. Last value is output dim:
                - 20 for probabilistic model;
                - 10 for probabilistic model, with diagonal covariance; and
                - 5 for non-probabilistic model.
            use_poolings: Whether to use pooling for each conv layer.
            use_normalizations: Whether to use normalization for each conv
                layer.
            kernel_size: Convolution kernel size.
            pool_size: Pooling kernel size.
            dropout_rate: Dropout rate.
            apply_linear_transform: Whether to apply linear transform for
                unbiased estimation.
        """
        super().__init__()

        if len(conv_channels) != len(use_poolings):
            raise ValueError(
                f"{conv_channels=} and {use_poolings=} must have same length."
            )
        if len(conv_channels) != len(use_normalizations):
            raise ValueError(
                f"{conv_channels=} and {use_normalizations=} must have same length."
            )
        
        # Build encoder.
        self._encoder = self._build_encoder(
            in_shape,
            conv_channels,
            use_poolings,
            use_normalizations,
            kernel_size,
            pool_size,
            dropout_rate
        )
        
        # Build dense head.
        self._flatten = nn.Flatten()
        in_dense = self._calculate_dense_input_size(in_shape)
        self._head = self._build_dense_head(in_dense, dense_neurons)

        self._apply_linear_transform = apply_linear_transform
        self._softplus = nn.Softplus()
        self._covariance_indices = self._get_covariance_indices(
            dense_neurons[-1]
        )

    def _build_encoder(
        self,
        in_shape: tuple[int],
        conv_channels: Sequence[int],
        use_poolings: Sequence[bool],
        use_normalizations: Sequence[bool],
        kernel_size: int,
        pool_size: int,
        dropout_rate: float
    ) -> nn.Module:
        """Builds the encoder."""
        layers = []
        
        # First layer.
        current_ln_time_dim = self._calculate_conv_output_size(
            in_shape[1],
            kernel_size,
            pool_size=1,  # Normalization is applied before pooling.
        )
        layers.append(
            _get_conv1d_layer(
                in_shape[0],
                conv_channels[0],
                kernel_size=kernel_size,
                pool_size=pool_size,
                dropout_rate=dropout_rate,
                use_pooling=use_poolings[0],
                use_normalization=use_normalizations[0],
                ln_shape=(conv_channels[0], current_ln_time_dim),
            )
        )
        
        # Remaining layers.
        for i in range(1, len(conv_channels)):
            prev_ln_time_dim = current_ln_time_dim
            current_ln_time_dim = self._calculate_conv_output_size(
                prev_ln_time_dim,
                kernel_size,
                pool_size if use_poolings[i-1] else 1
            )
            
            layers.append(
                _get_conv1d_layer(
                    conv_channels[i-1],
                    conv_channels[i],
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    dropout_rate=dropout_rate,
                    use_pooling=use_poolings[i],
                    use_normalization=use_normalizations[i],
                    ln_shape=(conv_channels[i], current_ln_time_dim)
                )
            )
        
        return nn.Sequential(*layers)

    def _calculate_conv_output_size(
        self,
        input_size: int,
        kernel_size: int,
        pool_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1
    ) -> int:
        """Calculate output size after convolution and pooling."""
        conv_time_output = (
            input_size // pool_size + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        return conv_time_output

    def _calculate_dense_input_size(self, in_shape: tuple[int]) -> int:
        """Calculate input size for dense layers by doing a forward pass."""
        with torch.no_grad():
            x = torch.zeros((1,) + in_shape)
            x = self._encoder(x)
            return self._flatten(x).shape[-1]

    def _build_dense_head(
        self,
        in_features: int,
        dense_neurons: Sequence[int]
    ) -> nn.Module:
        """Builds the dense head."""
        layers = []
        layers.extend(
            [nn.Linear(in_features, dense_neurons[0]), nn.ReLU()]
        )
        for i in range(len(dense_neurons) - 1):
            layers.append(nn.Linear(dense_neurons[i], dense_neurons[i + 1]))
            # Add ReLU for all layers except the last (output) layer.
            if i < len(dense_neurons) - 2:
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)

    def _get_covariance_indices(self, output_dim: int) -> list[tuple[int, int]]:
        """Get indices for covariance matrix construction."""
        if output_dim == 5:
            return []  # Non-probabilistic case.
        elif output_dim == 10:
            return [(-1, -1)]  # Special case; diagonal covariance.
        elif output_dim == 20:
            # Full covariance matrix.
            return [(i, j) for i in range(5) for j in range(i+1, 5)]
        else:
            raise ValueError(f"Unsupported output dimension: {output_dim=}.")

    def _get_covariance_matrix(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix using Cholesky decomposition approach."""
        batch_size, *_ = predictions.shape
        cov_matrix = torch.zeros(
            (batch_size, 5, 5),
            device=predictions.device
        )
        var_predictions = self._softplus(predictions[:, 5:10])
        cov_predictions = predictions[:, 10:]
        for i in range(5):
            cov_matrix[:, i, i] = var_predictions[:, i]
        
        for idx, (i, j) in enumerate(self._covariance_indices):
            if (i, j) == (-1, -1):  # Special case marker for diagonal cov.
                break
            cov_matrix[:, i, j] = cov_predictions[:, idx]
        
        # Apply Cholesky decomposition: L @ L^T.
        return torch.matmul(cov_matrix, cov_matrix.transpose(1, 2))

    def _linear_transform(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply linear transformation: y = Ax + b for unbiased estimator."""
        batch_size = mean.shape[0]
        device = mean.device
        
        A = torch.diag_embed(torch.full((batch_size, 5), 2.0, device=device))
        b = torch.full((batch_size, 5), 3.0, device=device)
        
        # Transform mean: A @ mean + b.
        transformed_mean = torch.matmul(A, mean.unsqueeze(-1)).squeeze(-1) + b
        
        # Transform covariance: A @ cov @ A^T.
        transformed_cov = torch.matmul(
            torch.matmul(A, covariance), A.transpose(1, 2)
        )
        return transformed_mean, transformed_cov

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x = self._encoder(x)
        x = self._flatten(x)
        predictions = self._head(x)
        
        # Non-probabilistic case.
        if predictions.shape[-1] == 5:
            if self._apply_linear_transform:
                return 2 * predictions + 3
            return predictions
        
        # Probabilistic case.
        mean_predictions = predictions[:, :5]
        cov_predictions = self._get_covariance_matrix(predictions)
        if self._apply_linear_transform:
            mean_predictions, cov_predictions = self._linear_transform(
                mean_predictions,
                cov_predictions
            )
            
        return mean_predictions, cov_predictions
