from architectures.components import CNNBlock, TransposeCNNBlock, MultiHeadAttention

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


class TrajectoryObjectEncoder(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        super(TrajectoryObjectEncoder, self).__init__()
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


class TrajectoryAttentionEncoder(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        super(TrajectoryAttentionEncoder, self).__init__()
        self._agent_encoder = TrajectoryObjectEncoder(n_features=n_features, trajectory_length=trajectory_length)
        self._object_encoder = TrajectoryObjectEncoder(n_features=n_features, trajectory_length=trajectory_length)
        self._attention = MultiHeadAttention(in_features=128, head_num=8, activation=nn.ReLU())
        self._linear = nn.Linear(in_features=256, out_features=128)
        self._lrelu = nn.LeakyReLU(0.3)


    def forward(self, agent_hist: torch.Tensor, objects_hist: torch.Tensor) -> torch.Tensor:
        # trajectory encoding
        batch_size, n_objects = objects_hist.shape[:2]
        agent_features = self._agent_encoder(agent_hist)
        agent_features_expanded = agent_features.view(agent_features.shape[0], 1, *agent_features.shape[1:])

        objects_hist = objects_hist.view(-1, *objects_hist.shape[-2:])
        objects_features = self._object_encoder(objects_hist)
        objects_features = objects_features.view(batch_size, n_objects, objects_features.shape[-1])
        all_features = torch.cat([agent_features_expanded, objects_features], dim=1)

        # attention
        att_out = self._attention(all_features)
        agent_features = torch.cat([agent_features, att_out[:, 0, :]], dim=-1)

        return self._lrelu(self._linear(agent_features))


class HeatmapModel(nn.Module):
    def __init__(self, encoder_input_shape: Tuple[int, int, int], decoder_input_shape: Tuple[int, int, int], traj_features: int, traj_length: int):
        super(HeatmapModel, self).__init__()
        self._encoder = RasterEncoder(encoder_input_shape)
        self._decoder = HeatmapOutputDecoder((decoder_input_shape[0] + 128, decoder_input_shape[1], decoder_input_shape[2]))
        self._trajectory_encoder = TrajectoryAttentionEncoder(traj_features, traj_length)

        self._conv1 = CNNBlock(in_channels=encoder_input_shape[0]+32, out_channels=16, kernel_size=7, padding='same')
        self._conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=7, padding='same')
        self._sigmoid = nn.Sigmoid()

    def forward(self, raster: torch.Tensor, agent_hist: torch.Tensor, objects_hist: torch.Tensor) -> torch.Tensor:
        raster_features = self._encoder(raster)
        traj_features = self._trajectory_encoder(agent_hist, objects_hist)
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
    inputs = torch.randn(4, 48, 224, 224)
    features = encoder(inputs)

    expected_shape = (512, 14, 14)
    assert tuple(features.shape[1:]) == expected_shape, f'{expected_shape} != {features.shape[1:]}'

    # test decoder
    decoder = HeatmapOutputDecoder(input_shape=(512, 14, 14))
    heatmap = decoder(features)

    expected_shape = (32, 224, 224)
    assert tuple(heatmap.shape[1:]) == expected_shape, f'{expected_shape} != {heatmap.shape[1:]}'

    # test object trajectory encoder
    traj_encoder = TrajectoryObjectEncoder(3, 20)
    trajectory = torch.randn(4, 20, 3)
    traj_features = traj_encoder(trajectory)

    expected_shape = (128, )
    assert tuple(traj_features.shape[1:]) == expected_shape, f'{expected_shape} != {traj_features.shape[1:]}'

    # test full trajectory encoder
    full_traj_encoder = TrajectoryAttentionEncoder(3, 20)
    agent_hist = torch.randn(4, 20, 3)
    objects_hist = torch.randn(4, 20, 20, 3)
    full_traj_features = full_traj_encoder(agent_hist, objects_hist)

    expected_shape = (128, )
    assert tuple(full_traj_features.shape[1:]) == expected_shape, f'{expected_shape} != {full_traj_features.shape}'

    # test heatmap model
    heatmap_model = HeatmapModel(encoder_input_shape=(48, 224, 224), decoder_input_shape=(512, 14, 14), traj_features=3, traj_length=20)
    heatmap = heatmap_model(inputs, trajectory)

    expected_shape = (1, 224, 224)
    assert tuple(heatmap.shape[1:]) == expected_shape, f'{expected_shape} != {heatmap.shape[1:]}'


if __name__ == '__main__':
    test()
