from architectures.components import CNNBlock, TransposeCNNBlock

import torch
import torch.nn as nn
from typing import Tuple


class RasterEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super(RasterEncoder, self).__init__()
        self._input_shape = input_shape

        self._convs = nn.ModuleList([
            CNNBlock(in_channels=input_shape[0], out_channels=32, kernel_size=7),
            CNNBlock(in_channels=32, out_channels=64, kernel_size=7),
            CNNBlock(in_channels=64, out_channels=128, kernel_size=5),
            CNNBlock(in_channels=128, out_channels=256, kernel_size=5),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3)
        ])
        self._maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert tuple(x.shape[1:]) == self._input_shape, f'Wrong input shape: Expected {self._input_shape} but got {tuple(x.shape[1:])}'

        for conv in self._convs[:-1]:
            x = self._maxpool(conv(x))
        x = self._convs[-1](x)

        return x


class HeatmapOutputDecoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super(HeatmapOutputDecoder, self).__init__()
        self._input_shape = input_shape

        self._transpose_convs = nn.ModuleList([
            TransposeCNNBlock(in_channels=input_shape[0], out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=0),
            TransposeCNNBlock(in_channels=256, out_channels=128, kernel_size=3, stride=2, output_padding=0),
            TransposeCNNBlock(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=0),
            TransposeCNNBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, output_padding=1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert tuple(x.shape[1:]) == self._input_shape, f'Wrong input shape: Expected {self._input_shape} but got {tuple(x.shape[1:])}'

        for trans_conv in self._transpose_convs:
            x = trans_conv(x)

        return x


class TrajectoryEncoder(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        super(TrajectoryEncoder, self).__init__()
        self._lstm1 = nn.LSTM(input_size=n_features, hidden_size=32)
        self._lstm2 = nn.LSTM(input_size=32, hidden_size=64)
        self._linear = nn.Linear(in_features=trajectory_length*64, out_features=128)
        self._lrelu = nn.LeakyReLU(0.3)
        self._batch_norm = nn.BatchNorm1d(128)
        self._flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._lstm1(x)
        x = self._lrelu(x)
        x, _ = self._lstm2(x)
        x = self._lrelu(x)
        x = self._flatten(x)

        x = self._batch_norm(self._linear(x))

        return x


class HeatmapModel(nn.Module):
    def __init__(self, encoder_input_shape: Tuple[int, int, int], decoder_input_shape: Tuple[int, int, int], traj_features: int, traj_length: int):
        super(HeatmapModel, self).__init__()
        self._encoder = RasterEncoder(encoder_input_shape)
        self._decoder = HeatmapOutputDecoder((decoder_input_shape[0] + 128, decoder_input_shape[1], decoder_input_shape[2]))
        self._trajectory_encoder = TrajectoryEncoder(traj_features, traj_length)

        self._conv1 = CNNBlock(in_channels=encoder_input_shape[0]+32, out_channels=16, kernel_size=7, padding='same')
        self._conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=7, padding='same')
        self._sigmoid = nn.Sigmoid()

    def forward(self, raster: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        raster_features = self._encoder(raster)
        traj_features = self._trajectory_encoder(trajectory)
        expanded_traj_features = traj_features.view(*traj_features.shape[-2:], 1, 1) \
            .expand(*traj_features.shape[-2:], *raster_features.shape[-2:])

        features = torch.concat([raster_features, expanded_traj_features], dim=1)

        decoder_output = self._decoder(features)
        merged_output = torch.concat([decoder_output, raster], dim=1)
        final_output = self._sigmoid(self._conv2(self._conv1(merged_output)))
        return final_output


def test():
    # test encoder
    encoder = RasterEncoder(input_shape=(48, 224, 224))
    inputs = torch.rand(4, 48, 224, 224)
    features = encoder(inputs)

    expected_shape = (512, 14, 14)
    assert tuple(features.shape[1:]) == expected_shape, f'{expected_shape} != {features.shape[1:]}'

    # test decoder
    decoder = HeatmapOutputDecoder(input_shape=(512, 14, 14))
    heatmap = decoder(features)

    expected_shape = (32, 224, 224)
    assert tuple(heatmap.shape[1:]) == expected_shape, f'{expected_shape} != {heatmap.shape[1:]}'

    # test trajectory encoder
    traj_encoder = TrajectoryEncoder(3, 20)
    trajectory = torch.rand(4, 20, 3)
    traj_features = traj_encoder(trajectory)

    expected_shape = (128, )
    assert tuple(traj_features.shape[1:]) == expected_shape, f'{expected_shape} != {heatmap.shape[1:]}'

    # test heatmap model
    heatmap_model = HeatmapModel(encoder_input_shape=(48, 224, 224), decoder_input_shape=(512, 14, 14), traj_features=3, traj_length=20)
    heatmap = heatmap_model(inputs, trajectory)

    expected_shape = (1, 224, 224)
    assert tuple(heatmap.shape[1:]) == expected_shape, f'{expected_shape} != {heatmap.shape[1:]}'


if __name__ == '__main__':
    test()
