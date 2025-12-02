"""Temporal Convolutional Network (TCN) implemented in PyTorch with
length-preserving padding (commonly called "same" padding).

This module implements a non-causal TCN that stacks dilated 1D
convolutional layers with exponentially increasing dilation. Each
convolution is followed by a ReLU activation and dropout. Per-layer
outputs are collected as skip connections, summed elementwise, and
passed through a final ReLU and a 1x1 projection.

The module expects inputs shaped ``(batch, seq_len, features)`` and
returns outputs shaped ``(batch, seq_len, outputs)``.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

__contributors__ = ["Griffin C. Sipes", "Melany D. Opolz"]
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (c) 2025 The Board of Trustees of the University of Illinois"


class TCN_Model(nn.Module):
    """Temporal Convolutional Network model.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    num_filters : int
        Number of filters used for each Conv1D layer.
    num_layers : int
        Number of dilated Conv1D layers.
    output_size : int
        Number of output features per timestep.
    kernel_size : int, optional
        Convolution kernel width (default is 3).
    dropout : float, optional
        Dropout probability applied after each convolution (default is 0.4).
    dilation_base : int, optional
        Base for exponential dilation (default is 2).

    Notes
    -----
    Each layer produces ``num_filters`` channels. Layer outputs are
    summed (skip connections) before the final 1x1 projection. Padding
    is applied so the sequence length is preserved (length-preserving
    padding, commonly called "same" padding).
    """

    def __init__(
        self,
        input_size: int,
        num_filters: int,
        num_layers: int,
        output_size: int,
        kernel_size: int = 3,
        dropout: float = 0.4,
        dilation_base: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be >= 1')

        self.input_size = input_size
        self.num_filters = int(num_filters)
        self.num_layers = int(num_layers)
        self.output_size = output_size
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)
        self.dilation_base = int(dilation_base)

        # Build per-layer convs. The first layer accepts `input_size`
        # channels, subsequent layers accept `num_filters` channels.
        # Use explicit constant padding (left/right).
        self.convs = nn.ModuleList()
        self.pad_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_ch = input_size
        for i in range(self.num_layers):
            dilation = self.dilation_base ** i
            full_pad = (self.kernel_size - 1) * dilation
            left_pad = full_pad // 2
            right_pad = full_pad - left_pad
            # ConstantPad1d expects (left, right)
            pad_layer = nn.ConstantPad1d((left_pad, right_pad), 0.0)
            conv = nn.Conv1d(in_ch, self.num_filters, kernel_size=self.kernel_size,
                             padding=0, dilation=dilation)
            self.pad_layers.append(pad_layer)
            self.convs.append(conv)
            self.dropouts.append(nn.Dropout(self.dropout_p))
            in_ch = self.num_filters

        # Final projection: 1x1 conv (time-distributed dense equivalent)
        self.head = nn.Conv1d(self.num_filters, self.output_size, kernel_size=1)
        self.relu = nn.ReLU()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Union[str, torch.device] = "cpu",
        strict: bool = True,
        **model_kwargs,
    ) -> nn.Module:
        """Construct a model and load weights from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to a PyTorch checkpoint. Accepts a serialized ``nn.Module``
            object or a state dict (mapping) saved with ``torch.save``.
        device : str or torch.device, optional
            Device to map the loaded tensors to (default: 'cpu').
        strict : bool, optional
            If ``True``, ``load_state_dict`` will be called with strict=True.
        **model_kwargs
            Keyword arguments forwarded to the model constructor if a
            template model needs to be created (e.g. ``input_size``,
            ``output_size``, ``num_filters``, ...). If the checkpoint
            contains a serialized module instance, constructor kwargs are
            ignored.

        Returns
        -------
        torch.nn.Module
            The loaded model on the requested device in eval mode.

        Raises
        ------
        ValueError
            If a state dict is found but no ``model_kwargs`` were provided
            to construct a template model.
        """
        map_location = device
        loaded = torch.load(checkpoint_path, map_location=map_location)

        # If the checkpoint is a Module instance, return it directly
        if isinstance(loaded, nn.Module):
            return loaded.to(device).eval()

        # If the checkpoint is a dict, try to find the state dict
        if isinstance(loaded, dict):
            # common container keys used by training scripts
            for key in ('state_dict', 'model_state_dict', 'weights'):
                if key in loaded and isinstance(loaded[key], dict):
                    state = loaded[key]
                    break
            else:
                # assume the dict is the state_dict itself
                state = loaded

            # If model kwargs provided, construct template and load
            if model_kwargs:
                model = cls(**model_kwargs)
                model.load_state_dict(state, strict=strict)
                return model.to(device).eval()

            raise ValueError(
                "Checkpoint appears to contain a state dict but no model "
                "constructor arguments were provided. Pass e.g. "
                "input_size=..., output_size..."
            )

        raise TypeError(f"Unsupported checkpoint format: {type(loaded)!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, seq_len, features)``.

        Returns
        -------
        torch.Tensor
            Output tensor with shape ``(batch, seq_len, output_size)``.
        """
        # Convert to (batch, channels, seq_len) for Conv1d
        out = x.permute(0, 2, 1)

        skips = []
        for idx, conv in enumerate(self.convs):
            # apply symmetric/asymmetric padding
            out_padded = self.pad_layers[idx](out)
            o = conv(out_padded)
            o = self.relu(o)
            o = self.dropouts[idx](o)
            skips.append(o)
            # next layer input is the layer output
            out = o

        # sum skip connections (elementwise)
        # shape: (num_layers, batch, channels, seq_len) -> (batch, channels, seq_len)
        s = torch.stack(skips, dim=0).sum(dim=0)
        s = self.relu(s)

        out = self.head(s)

        # return to (batch, seq_len, output_size)
        return out.permute(0, 2, 1)



__all__ = ["TCN_Model"]
