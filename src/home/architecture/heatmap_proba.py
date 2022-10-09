"""
Heatmap estimation block
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from home.architecture.loss import PixelFocalLoss
from home.architecture.sampler import TorchModalitySampler  # Used for FDE evaluation
from library.ml.building_blocks import CNNBlock, TransposeCNNBlock, MultiHeadAttention


class RasterEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        """
        Encodes rasterized HD map features

        Args:
            input_shape: Raster input shape
        """
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
        """
        Args:
            x: Rasterized features

        Returns: extracted features
        """
        assert tuple(x.shape[1:]) == self._input_shape, f'Wrong input shape: Expected {self._input_shape} but got {tuple(x.shape[1:])}'

        for conv in self._convs[:-1]:
            x = self._maxpool(conv(x))
        x = self._convs[-1](x)

        return x


class HeatmapOutputDecoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        """
        Estimates probabilities in form of a heatmap for agent end points

        Args:
            input_shape: Input shape
        """
        super(HeatmapOutputDecoder, self).__init__()
        self._input_shape = input_shape

        self._transpose_convs = nn.ModuleList([
            TransposeCNNBlock(in_channels=input_shape[0], out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=0),
            TransposeCNNBlock(in_channels=256, out_channels=128, kernel_size=3, stride=2, output_padding=0),
            TransposeCNNBlock(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=0),
            TransposeCNNBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, output_padding=1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features

        Returns: Heatmap
        """
        assert tuple(x.shape[1:]) == self._input_shape, f'Wrong input shape: Expected {self._input_shape} but got {tuple(x.shape[1:])}'

        for trans_conv in self._transpose_convs:
            x = trans_conv(x)

        return x


class TrajectoryObjectEncoder(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        """
        Encodes object trajectory

        Args:
            n_features: Trajectory node features dimension
            trajectory_length: Trajectory length
        """
        super(TrajectoryObjectEncoder, self).__init__()
        self._lstm1 = nn.LSTM(input_size=n_features, hidden_size=32, batch_first=True)
        self._dropout = nn.Dropout(0.5)
        self._linear = nn.Linear(in_features=trajectory_length*32, out_features=64)
        self._lrelu = nn.LeakyReLU(0.1)
        self._batch_norm = nn.BatchNorm1d(64)
        self._flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Agent history

        Returns: Agent history features
        """
        x, _ = self._lstm1(x)
        x = self._dropout(x)
        x = self._lrelu(x)
        x = self._flatten(x)
        x = self._batch_norm(self._linear(x))

        return x


class TrajectoryAttentionEncoder(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        """
        Encodes agent and neighbours trajectories with attention applied between them to filter
        important features of neighbours for agent

        Args:
            n_features: Trajectory node features dimension
            trajectory_length: Trajectory length
        """
        super(TrajectoryAttentionEncoder, self).__init__()
        self._agent_encoder = TrajectoryObjectEncoder(n_features=n_features, trajectory_length=trajectory_length)
        self._object_encoder = TrajectoryObjectEncoder(n_features=n_features, trajectory_length=trajectory_length)
        self._attention = MultiHeadAttention(in_features=64, head_num=4, activation=nn.ReLU())
        self._layernorm = nn.LayerNorm(128)
        self._linear = nn.Linear(in_features=128, out_features=128)
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
        agent_features = self._layernorm(torch.cat([agent_features, att_out[:, 0, :]], dim=-1))

        return self._lrelu(self._linear(agent_features))


class HeatmapModel(nn.Module):
    def __init__(self, encoder_input_shape: Tuple[int, int, int], decoder_input_shape: Tuple[int, int, int], traj_features: int, traj_length: int):
        """
        Estimates probabilities in form of a heatmap for agent end points
        given rasterized HD map, agent trajectory and neighbours trajectories

        Args:
            encoder_input_shape: Encoder Input Shape
            decoder_input_shape: Decoder Input Shape
            traj_features: Trajectory node features dimension
            traj_length: Trajectory length
        """
        super(HeatmapModel, self).__init__()
        self._encoder = RasterEncoder(encoder_input_shape)
        self._decoder = HeatmapOutputDecoder((decoder_input_shape[0] + 128, decoder_input_shape[1], decoder_input_shape[2]))
        self._trajectory_encoder = TrajectoryAttentionEncoder(traj_features, traj_length)

        self._conv1 = CNNBlock(in_channels=encoder_input_shape[0]+32, out_channels=16, kernel_size=7, padding='same')
        self._conv2 = CNNBlock(in_channels=16, out_channels=1, kernel_size=3, padding='same')
        self._sigmoid = nn.Sigmoid()

    def forward(self, raster: torch.Tensor, agent_hist: torch.Tensor, objects_hist: torch.Tensor) -> torch.Tensor:
        raster_features = self._encoder(raster)
        traj_features = self._trajectory_encoder(agent_hist, objects_hist)
        expanded_traj_features = traj_features.view(*traj_features.shape[-2:], 1, 1) \
            .expand(*traj_features.shape[-2:], *raster_features.shape[-2:])

        features = torch.concat([raster_features, expanded_traj_features], dim=1)

        decoder_output = self._decoder(features)
        merged_output = torch.concat([decoder_output, raster], dim=1)  # skip connection
        final_output = self._sigmoid(self._conv2(self._conv1(merged_output)))
        return final_output


class LightningHeatmapModel(LightningModule):
    def __init__(
        self,
        encoder_input_shape: Tuple[int, int, int],
        decoder_input_shape: Tuple[int, int, int],
        traj_features: int,
        traj_length: int,
        sampler_targets: int,
        sampler_radius: int,
        base_lr: Optional[float],
        sched_step: Optional[int],
        sched_gamma: Optional[float]
    ):
        """
        Pytorch Lightning wrapper for HeatmapModel for training purposes

        Args:
            encoder_input_shape: Encoder Input Shape
            decoder_input_shape: Decoder Input Shape
            traj_features: Trajectory node features dimension
            traj_length: Trajectory length
            sampler_targets: Number of targets to sample by sampler (used for validation)
            sampler_radius: Sampler target radius (used for validation)
            base_lr: Base model learning rate (can be none for inference)
            sched_step: Scheduler step (can be none for inference)
            sched_gamma: Scheduler lr decay (can be none for inference)
        """
        super(LightningHeatmapModel, self).__init__()
        self._heatmap_estimator = HeatmapModel(
            encoder_input_shape=encoder_input_shape,
            decoder_input_shape=decoder_input_shape,
            traj_features=traj_features,
            traj_length=traj_length
        )
        self._loss = PixelFocalLoss()
        self._sampler = TorchModalitySampler(
            n_targets=sampler_targets,
            radius=sampler_radius,
            swap_rc=True
        )
        self._log_history = {
            'train/loss': [],
            'val/loss': [],
            'e2e/min_fde': [],
        }

        # e2e parameters
        self._n_targets = sampler_targets

        # parameters
        self._base_lr = base_lr
        self._sched_step = sched_step
        self._sched_gamma = sched_gamma

    def forward(self, raster: torch.Tensor, agent_hist: torch.Tensor, objects_hist: torch.Tensor) -> torch.Tensor:
        return self._heatmap_estimator(raster, agent_hist, objects_hist)

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        data = batch
        raster, agent_hist, objects_hist, da_area, true_heatmap = data['raster'], data['agent_traj_hist'], \
            data['objects_traj_hist'], data['da_area'], data['heatmap']
        pred_heatmap = self(raster, agent_hist, objects_hist)
        loss = self._loss(pred_heatmap, true_heatmap, da_area)
        self._log_history['train/loss'].append(loss)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> torch.Tensor:
        data = batch
        raster, agent_hist, objects_hist, da_area, true_heatmap, agent_traj_gt_end_point = data['raster'], data['agent_traj_hist'], \
            data['objects_traj_hist'], data['da_area'], data['heatmap'], data['agent_traj_gt_end_point']
        pred_heatmap = self(raster, agent_hist, objects_hist)
        loss = self._loss(pred_heatmap, true_heatmap, da_area)
        self._log_history['val/loss'].append(loss)

        # semi e2e loss
        raw_targets, _ = self._sampler(pred_heatmap)  # Note: Confidences are not currently used
        targets = raw_targets - pred_heatmap.shape[-1] // 2
        ground_truth_expanded = agent_traj_gt_end_point.unsqueeze(1).repeat(1, self._n_targets, 1)
        batch_fde, _ = torch.min(torch.sqrt(
            (targets[..., 0] - ground_truth_expanded[..., 0]) ** 2 + (targets[..., 1] - ground_truth_expanded[..., 1]) ** 2), dim=-1)
        fde = torch.mean(batch_fde)

        self._log_history['e2e/min_fde'].append(fde)

        loss = self._loss(pred_heatmap, true_heatmap, da_area)
        return loss

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        for log_name, sample in self._log_history.items():
            if len(sample) == 0:
                continue

            self.log(log_name, sum(sample) / len(sample), prog_bar=True)
            self._log_history[log_name] = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self._heatmap_estimator.parameters(), lr=self._base_lr)

        sched = {
            'scheduler': torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self._sched_step,
                gamma=self._sched_gamma
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [sched]


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

    expected_shape = (64, )
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
    heatmap = heatmap_model(inputs, trajectory, objects_hist)

    expected_shape = (1, 224, 224)
    assert tuple(heatmap.shape[1:]) == expected_shape, f'{expected_shape} != {heatmap.shape[1:]}'


if __name__ == '__main__':
    test()
