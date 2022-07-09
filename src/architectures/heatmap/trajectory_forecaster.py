from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F


from configparser.raster import RasterizationTrainTrajectoryForecasterParametersConfig


class TrajectoryForecaster(nn.Module):
    def __init__(self, in_features: int, trajectory_future_length: int):
        super(TrajectoryForecaster, self).__init__()

        # RNN
        self._rnn = nn.LSTM(input_size=in_features, hidden_size=128, proj_size=64)
        self._hidden = None
        self._layernom = nn.LayerNorm(64)
        self._relu = nn.ReLU()

        self._decoder = nn.Sequential(
            nn.Linear(66, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, trajectory_future_length*2),
        )

    def forward(self, traj_hist: torch.Tensor, end_point: torch.Tensor) -> torch.Tensor:
        encoded, self._hidden = self._rnn(traj_hist, self._hidden) if self._hidden is not None else self._rnn(traj_hist)
        self._hidden[0].detach_()
        self._hidden[1].detach_()

        encoded = encoded[:, -1, :]  # Obtaining last output (format: TxBxF)
        encoded_expanded = encoded.unsqueeze(1).repeat(1, end_point.shape[1], 1)
        encoded_with_end_point = torch.concat([encoded_expanded, end_point], dim=-1)
        output = self._decoder(encoded_with_end_point)
        output = output.view(*output.shape[:2], -1, 2)
        return output


class LightningTrajectoryForecaster(LightningModule):
    def __init__(self, train_config: RasterizationTrainTrajectoryForecasterParametersConfig, in_features: int, trajectory_future_length: int):
        super(LightningTrajectoryForecaster, self).__init__()
        self._model = TrajectoryForecaster(in_features, trajectory_future_length)
        self._train_config = train_config

        self._log_history = {
            'training_loss': [],
            'val_loss': []
        }

    def forward(self, traj_hist: torch.Tensor, end_point: torch.Tensor) -> torch.Tensor:
        return self._model(traj_hist, end_point)

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        traj, gt_traj, end_point = batch
        outputs = self._model(traj, end_point)
        loss = F.mse_loss(outputs, gt_traj)
        self._log_history['training_loss'].append(loss)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> torch.Tensor:
        traj, gt_traj, end_point = batch
        outputs = self._model(traj, end_point)
        loss = F.mse_loss(outputs, gt_traj)
        self._log_history['val_loss'].append(loss)
        return loss

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        for log_name, sample in self._log_history.items():
            if len(sample) == 0:
                continue

            self.log(log_name, sum(sample) / len(sample), prog_bar=True)
            self._log_history[log_name] = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._train_config.lr)

        sched = {
            'scheduler': torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self._train_config.sched_step,
                gamma=self._train_config.sched_gamma
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [sched]


def test():
    model = TrajectoryForecaster(in_features=3, trajectory_future_length=20)
    inputs, end_point = torch.randn(8, 30, 3), torch.randn(8, 6, 2)
    outputs = model(inputs, end_point)
    print(outputs.shape)
    outputs.sum().backward()


if __name__ == '__main__':
    test()
