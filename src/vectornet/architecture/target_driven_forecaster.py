"""
VectorNet TNT end-to-end model (training and inference)
"""
from typing import Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningModule

from configparser.graph import GraphTrainConfigParameters
from library.ml.metrics import metrics
from vectornet.architecture.loss import LiteTNTLoss
from vectornet.architecture.target_generator import TargetGenerator
from vectornet.architecture.trajectory_forecaster import TrajectoryForecaster, TrajectoryScorer


class TargetDrivenForecaster(LightningModule):
    def __init__(
        self,
        cluster_size: int,
        trajectory_length: int,
        polyline_features: int,
        n_targets: int,
        n_trajectories: int,
        use_traj_scoring: bool = True,
        traj_scale: float = 1.0,

        train_config: Optional[GraphTrainConfigParameters] = None
    ):
        """
        TNT-Vectornet

        Args:
            cluster_size: Polyline length
            trajectory_length: Output trajectory length
            polyline_features: Number of feagures for polyline node
            n_targets: Number of targets to filter using target generator
            n_trajectories: Number of trajectories to filter after target sampling and trajectory estimation
            use_traj_scoring: Use trajectory scoring (is fale then n_targets == n_trajectories)
            traj_scale: Normalization parameter for trajectory scaling
            train_config: Train configuration (only for training)
        """
        super(TargetDrivenForecaster, self).__init__()

        # model and loss
        if not use_traj_scoring:
            assert n_targets == n_trajectories, \
                'Number of targets and trajectories must match if not using trajectory scoring! '
        assert n_targets >= n_trajectories, 'Number of end point targets must be greater or equal than number of trajectories!'

        self._use_traj_scoring = use_traj_scoring
        self._n_targets = n_targets
        self._n_trajectories = n_trajectories
        self._target_generator = TargetGenerator(cluster_size=cluster_size, polyline_features=polyline_features)
        self._trajectory_forecaster = TrajectoryForecaster(n_features=256, trajectory_length=trajectory_length)
        self._trajectory_scorer = TrajectoryScorer(n_features=256, trajectory_length=trajectory_length)
        self._loss = LiteTNTLoss(traj_scoring=use_traj_scoring)

        # training
        self._train_config = train_config

        # evaluation
        self._traj_scale = traj_scale

        # logging
        self._log_history = {
            'train/loss': [],
            'train/tg_confidence_loss': [],
            'train/tg_huber_loss': [],
            'train/tf_huber_loss': [],
            'train/tf_confidence_loss': [],
            'val/loss': [],
            'val/tg_confidence_loss': [],
            'val/tg_huber_loss': [],
            'val/tf_huber_loss': [],
            'val/tf_confidence_loss': [],
            'e2e/min_ade': [],
            'e2e/min_fde': []
        }

    def _filter_targets(
        self,
        batch_size: int,
        target_confidences: torch.Tensor,
        anchors: torch.Tensor,
        offsets: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Filters `self.n_targets` with highest confidence

        Args:
            batch_size: batch_size
            target_confidences: Estimated confidence for each anchor
            anchors: Sampler target end point without corrections
            offsets: Anchor corrections
            targets: anchors + offsets

        Returns: Filtered `self.n_targets` targets by confidence
        """
        batch_filtered_anchors, batch_filtered_offsets, batch_filtered_targets, batch_filtered_target_confidences = [], [], [], []
        for batch_index in range(batch_size):
            # for each instance in batch: choose top N targets
            instance_filter_indexes = torch.argsort(target_confidences[batch_index], descending=True)[:self._n_targets]
            instance_filtered_anchors = anchors[batch_index, instance_filter_indexes]
            instance_filtered_offsets = offsets[batch_index, instance_filter_indexes]
            instance_filtered_targets = targets[batch_index, instance_filter_indexes]
            instance_filtered_target_confidences = target_confidences[batch_index, instance_filter_indexes]

            batch_filtered_anchors.append(instance_filtered_anchors)
            batch_filtered_offsets.append(instance_filtered_offsets)
            batch_filtered_targets.append(instance_filtered_targets)
            batch_filtered_target_confidences.append(instance_filtered_target_confidences)

        # form batch from filtered targets
        filtered_anchors = torch.stack(batch_filtered_anchors)
        filtered_offsets = torch.stack(batch_filtered_offsets)
        filtered_targets = torch.stack(batch_filtered_targets)
        filtered_confidences = torch.stack(batch_filtered_target_confidences)

        return filtered_anchors, filtered_offsets, filtered_targets, filtered_confidences

    def forward(self, polylines: torch.Tensor, anchors: torch.Tensor) -> Dict[str, torch.Tensor]:
        features, offsets, target_confidences = self._target_generator(polylines, anchors)
        targets = anchors + offsets

        # Filter by target scores
        n_batches = features.shape[0]

        filtered_anchors, filtered_offsets, filtered_targets, filtered_confidences = self._filter_targets(
            batch_size=n_batches,
            anchors=anchors,
            target_confidences=target_confidences,
            targets=targets,
            offsets=offsets
        )

        trajectories = self._trajectory_forecaster(features, filtered_targets)
        trajectory_scores = self._trajectory_scorer(features, trajectories)

        if self._use_traj_scoring:
            # Filter by trajectory scores
            batch_filtered_trajectories, batch_filtered_traj_scores = [], []
            for batch_index in range(n_batches):
                # for each instance in batch: choose top N trajectories
                instance_filter_indexes = torch.argsort(trajectory_scores[batch_index], descending=True)[:self._n_trajectories]
                instance_filtered_trajectories = trajectories[batch_index, instance_filter_indexes]
                instance_filtered_scores = trajectory_scores[batch_index, instance_filter_indexes]

                batch_filtered_trajectories.append(instance_filtered_trajectories)
                batch_filtered_traj_scores.append(instance_filtered_scores)

            filtered_trajectories = torch.stack(batch_filtered_trajectories)
            filtered_traj_scores = torch.stack(batch_filtered_traj_scores)
        else:
            filtered_trajectories = trajectories
            filtered_traj_scores = None

        return {
            'all_anchors': anchors,
            'all_offsets': offsets,
            'all_target_confidences': target_confidences,
            'anchors': filtered_anchors,
            'offsets': filtered_offsets,
            'targets': filtered_targets,
            'confidences': filtered_confidences,
            'all_forecasts': trajectories,
            'all_forecast_scores': trajectory_scores,
            'forecasts': filtered_trajectories,
            'traj_scores': filtered_traj_scores
        }

    def forward_backward_step(self, polylines: torch.Tensor, anchors: torch.Tensor, ground_truth: torch.Tensor, gt_traj: torch.Tensor) \
            -> Tuple[torch.Tensor, ...]:
        # Generate targets from anchors
        features, offsets, confidences = self._target_generator(polylines, anchors)

        # Forecast trajectories for ground truth targets
        ground_truth_expanded = ground_truth.unsqueeze(1).repeat(1, self._n_targets, 1)
        forecasted_trajectories_gt_end_point = self._trajectory_forecaster(features, ground_truth_expanded)

        # Forecast trajectories for predicted targets
        _, _, filtered_targets, _ = self._filter_targets(
            batch_size=features.shape[0],
            anchors=anchors,
            target_confidences=confidences,
            targets=anchors + offsets,
            offsets=offsets
        )
        forecasted_trajectories = self._trajectory_forecaster(features, filtered_targets)
        traj_conf = self._trajectory_scorer(features, forecasted_trajectories)

        return self._loss(
            anchors=anchors,
            offsets=offsets,
            confidences=confidences,
            ground_truth=ground_truth,
            forecasted_trajectories_gt_end_point=forecasted_trajectories_gt_end_point,
            forecasted_trajectories=forecasted_trajectories,
            traj_conf=traj_conf,
            gt_traj=gt_traj
        )

    def training_step(self, batch, *args, **kwargs) -> dict:
        polylines, anchors, ground_truth, gt_traj = batch

        loss, tg_ce_loss, tg_huber_loss, tf_huber_loss, tf_bce_loss = self.forward_backward_step(polylines, anchors, ground_truth, gt_traj)

        self._log_history['train/loss'].append(loss)
        self._log_history['train/tg_confidence_loss'].append(tg_ce_loss)
        self._log_history['train/tg_huber_loss'].append(tg_huber_loss)
        self._log_history['train/tf_huber_loss'].append(tf_huber_loss)
        self._log_history['train/tf_confidence_loss'].append(tf_bce_loss)

        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch, *args, **kwargs) -> dict:
        polylines, anchors, ground_truth, gt_traj = batch
        self.eval()
        outputs = self(polylines, anchors)
        self.train()

        # loss info
        loss, tg_ce_loss, tg_huber_loss, tf_huber_loss, tf_bce_loss = self.forward_backward_step(polylines, anchors, ground_truth, gt_traj)

        self._log_history['val/loss'].append(loss)
        self._log_history['val/tg_confidence_loss'].append(tg_ce_loss)
        self._log_history['val/tg_huber_loss'].append(tg_huber_loss)
        self._log_history['val/tf_huber_loss'].append(tf_huber_loss)
        self._log_history['val/tf_confidence_loss'].append(tf_bce_loss)

        # e2e metrics
        normalized_forecasts = outputs['forecasts'].cumsum(axis=2) * self._traj_scale
        gt_traj_normalized = gt_traj.cumsum(axis=1) * self._traj_scale
        expanded_gt_traj_normalized = gt_traj_normalized.unsqueeze(1).repeat(1, self._n_trajectories, 1, 1)
        min_ade, _ = metrics.minADE(normalized_forecasts, expanded_gt_traj_normalized)
        min_fde, _ = metrics.minFDE(normalized_forecasts, expanded_gt_traj_normalized)

        self._log_history['e2e/min_ade'].append(min_ade)
        self._log_history['e2e/min_fde'].append(min_fde)

        return {'val_loss': loss}

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        for log_name, sample in self._log_history.items():
            if len(sample) == 0:
                continue

            self.log(log_name, sum(sample) / len(sample), prog_bar=True)
            self._log_history[log_name] = []

    def configure_optimizers(self):
        assert self._train_config is not None, 'Error: Training config not set'
        tg_opt = torch.optim.Adam(self._target_generator.parameters(), lr=self._train_config.tg_lr)
        tf_opt = torch.optim.Adam(self._trajectory_forecaster.parameters(), lr=self._train_config.tf_lr)
        tfs_opt = torch.optim.Adam(self._trajectory_scorer.parameters(), lr=self._train_config.tfs_lr)

        tg_sched = {
            'scheduler': torch.optim.lr_scheduler.StepLR(
                optimizer=tg_opt,
                step_size=self._train_config.tg_sched_step,
                gamma=self._train_config.tg_sched_gamma
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        tf_sched = {
            'scheduler': torch.optim.lr_scheduler.StepLR(
                optimizer=tf_opt,
                step_size=self._train_config.tf_sched_step,
                gamma=self._train_config.tf_sched_gamma
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        tfs_sched = {
            'scheduler': torch.optim.lr_scheduler.StepLR(
                optimizer=tfs_opt,
                step_size=self._train_config.tfs_sched_step,
                gamma=self._train_config.tfs_sched_gamma
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [tg_opt, tf_opt, tfs_opt], [tg_sched, tf_sched, tfs_sched]


def test():
    polylines = torch.randn(4, 200, 20, 14)
    anchors = torch.randn(4, 75, 2)
    tdf = TargetDrivenForecaster(cluster_size=20, polyline_features=14, trajectory_length=20, n_targets=15, n_trajectories=10)
    outputs = tdf(polylines, anchors)
    print(outputs['forecasts'].shape, outputs['confidences'].shape, outputs['targets'].shape, outputs['traj_scores'].shape)


if __name__ == '__main__':
    test()
