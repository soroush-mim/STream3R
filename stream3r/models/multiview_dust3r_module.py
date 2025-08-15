# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import re
import roma
import torch
from torch.distributed import all_gather_object, barrier
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import MaxMetric, MeanMetric, MinMetric, SumMetric, Metric
from torchmetrics.aggregation import BaseAggregator
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from concurrent.futures import ThreadPoolExecutor

from stream3r.dust3r.post_process import estimate_focal_knowing_depth_and_confidence_mask
from stream3r.dust3r.model import FlashDUSt3R
from stream3r.models.stream3r.stream3r import STream3R
from stream3r.dust3r.cloud_opt.init_im_poses import fast_pnp
from stream3r.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class AccumulatedSum(BaseAggregator):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fn="sum",
            default_value=torch.tensor(0.0, dtype=torch.long),
            nan_strategy='warn',
            state_name="sum_value",
            **kwargs,
        )

    def update(self, value: int) -> None:
        self.sum_value += value

    def compute(self) -> torch.LongTensor:
        return self.sum_value

def gather_deduplicated_scene_metrics(reconstruction_metrics_per_epoch):
    """Gathers and deduplicates scene-specific metrics across all ranks by dataset."""
    gathered_metrics = [None] * torch.distributed.get_world_size()
    all_gather_object(gathered_metrics, reconstruction_metrics_per_epoch)

    # Flatten and deduplicate metrics across all ranks
    all_metrics = {}
    for rank_metrics in gathered_metrics:
        for dataset_name, scenes in rank_metrics.items():
            if dataset_name not in all_metrics:
                all_metrics[dataset_name] = {}
            all_metrics[dataset_name].update(scenes)  # Keeps the first occurrence of each scene

    return all_metrics

class MultiViewDUSt3RLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        train_criterion: torch.nn.Module,
        validation_criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        pretrained: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        eval_use_pts3d_from_local_head: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net', 'train_criterion', 'validation_criterion'])

        self.net = net
        self.train_criterion = train_criterion
        self.validation_criterion = validation_criterion
        self.pretrained = pretrained
        self.resume_from_checkpoint = resume_from_checkpoint
        self.eval_use_pts3d_from_local_head = eval_use_pts3d_from_local_head

        # use register_buffer to save these with checkpoints
        # so that when we resume training, these bookkeeping variables are preserved
        self.register_buffer("epoch_fraction", torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer("train_total_samples", torch.tensor(0, dtype=torch.long, device=self.device))
        self.register_buffer("train_total_images", torch.tensor(0, dtype=torch.long, device=self.device))

        self.train_total_samples_per_step = AccumulatedSum()  # these need to be reduced across GPUs, so use Metric
        self.train_total_images_per_step = AccumulatedSum()  # these need to be reduced across GPUs, so use Metric

        self.val_loss = MeanMetric()

        # Initialize metrics
        self.RRA_thresholds = [5, 15, 30]
        self.RTA_thresholds = [5, 15, 30]
        # Initialize RRA and RTA metrics as attributes
        for tau in self.RRA_thresholds:
            setattr(self, f'val_RRA_{tau}', MeanMetric())
        for tau in self.RTA_thresholds:
            setattr(self, f'val_RTA_{tau}', MeanMetric())

        self.val_mAA = MeanMetric()

        # Reconstruction evaluation metrics
        self.dataset_names_with_samples_of_uneven_num_of_views = ['dtu', '7scenes', 'nrgbd']
        self.reconstruction_metrics_per_epoch = {}  # Accumulate all reconstruction metrics by dataset and scene for the epoch
        # New dictionary to store detailed losses for datasets with uneven number of views
        self.uneven_view_detailed_losses = {}

    @classmethod
    def load_for_inference(cls, net: STream3R):
        lit_module = cls(net=net, train_criterion=None, validation_criterion=None, optimizer=None, scheduler=None, compile=False)
        lit_module.eval()
        return lit_module

    def forward(self, views: List[Dict[str, torch.Tensor]]) -> Any:
        return self.net(views)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Legacy: if the checkpoint does not contain the epoch_fraction, train_total_samples, and train_total_images
        # we manually add them to the checkpoint
        # if self.trainer.strategy.strategy_name != "deepseed":
        #     if checkpoint["state_dict"].get("epoch_fraction") is None:
        #         checkpoint["state_dict"]["epoch_fraction"] = self.epoch_fraction
        #     if checkpoint["state_dict"].get("train_total_samples") is None:
        #         checkpoint["state_dict"]["train_total_samples"] = self.train_total_samples
        #     if checkpoint["state_dict"].get("train_total_images") is None:
        #         checkpoint["state_dict"]["train_total_images"] = self.train_total_images
        pass

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        # the wandb logger lives in self.loggers
        # find the wandb logger and watch the model and gradients
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                # log gradients, parameter histogram and model topology
                self.wandb_logger.watch(self.net, log="all", log_freq=500, log_graph=False)

    def on_train_epoch_start(self) -> None:
        # save initial checkpoint to check pretrained model
        # if self.trainer.global_step == 0:
        #     checkpoint_path = os.path.join(self.trainer.checkpoint_callback.dirpath, "step_0.ckpt")
        #     self.trainer.save_checkpoint(checkpoint_path)

        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.train_dataloader, "dataset") and hasattr(self.trainer.train_dataloader.dataset, "set_epoch"):
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.train_dataloader, "sampler") and hasattr(self.trainer.train_dataloader.sampler, "set_epoch"):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        for loader in self.trainer.val_dataloaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(0)

    def model_step(
        self, batch: List[Dict[str, torch.Tensor]], criterion: torch.nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        device = self.device

        # Move data to device
        for view in batch:
            for name in "img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres".split():
                if name in view:
                    view[name] = view[name].to(device, non_blocking=True)

        views = batch

        preds = self.forward(views)

        # Compute the loss in higher precision
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            loss, loss_details = criterion(views, preds) if criterion is not None else None

        return views, preds, loss, loss_details

    def training_step(
        self, batch: List[Dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        views, preds, loss, loss_details = self.model_step(batch, self.train_criterion)

        if not isinstance(loss, (torch.Tensor, dict, type(None))):  # this will cause a lightning.fabric.utilities.exceptions.MisconfigurationException
            # log loss and the batch information to help debugging
            # use print instead of log because the logger only logs on rank 0, but this could happen on any rank
            print(f"Loss is not a tensor or dict but {type(loss)}, value: {loss}")
            print(f"Loss details: {loss_details}")
            print(f"Batch: {batch}")
            print(f"Batch index: {batch_idx}")
            print(f"Views: {views}")
            print(f"Preds: {preds}")
            loss = None  # set loss to None will still break the training loop in DDP, this is intended - we should fix the data to avoid nan loss in the first place
            return loss

        self.epoch_fraction = torch.tensor(self.trainer.current_epoch + batch_idx / self.trainer.num_training_batches, device=self.device)

        self.log("trainer/epoch", self.epoch_fraction, on_step=True, on_epoch=False, prog_bar=True)
        self.log("trainer/lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # log the details of the loss
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(f"train_detail_{key}", value, on_step=True, on_epoch=False, prog_bar=False)
                match = re.search(r'/(\d{1,2})$', key)
                if match:
                    stripped_key = key[:match.start()]
                    self.log(f"train/{stripped_key}", value, on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of samples seen so far
        batch_size = views[0]["img"].shape[0]
        self.train_total_samples_per_step(batch_size)  # aggregate across all GPUs
        self.train_total_samples += self.train_total_samples_per_step.compute()  # accumulate across all steps
        self.train_total_samples_per_step.reset()
        self.log("trainer/total_samples", self.train_total_samples, on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of images seen so far
        num_views = len(views)
        n_image_cur_step = batch_size * num_views
        self.train_total_images_per_step(n_image_cur_step)  # aggregate across all GPUs
        self.train_total_images += self.train_total_images_per_step.compute()  # accumulate across all steps
        self.train_total_images_per_step.reset()
        self.log("trainer/total_images", self.train_total_images, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(
        self, batch: List[Dict[str, torch.Tensor]], batch_idx: int, dataloader_idx: int = 0,
    ) -> torch.Tensor:
        views, preds, loss, loss_details = self.model_step(batch, self.validation_criterion)

        # Extract the dataset name and batch size
        dataset_name = views[0]['dataset'][0]  # all views should have the same dataset name because we use "sequential" mode of CombinedLoader
        batch_size = views[0]["img"].shape[0]

        # Log the overall validation loss
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=True, batch_size=batch_size)
        self.val_loss(loss)
        # self.log(f"val/loss_{dataset_name}", loss, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)

        # Log the details of the loss with dataset name and view number in the key
        if loss_details is not None:
            if dataset_name not in self.dataset_names_with_samples_of_uneven_num_of_views:
                for key, value in loss_details.items():
                    self.log(
                        f"val_detail_{dataset_name}_{key}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        reduce_fx="mean",
                        sync_dist=True,
                        add_dataloader_idx=False,
                        batch_size=batch_size,
                    )
                    match = re.search(r'/(\d{1,2})$', key)
                    if match:
                        stripped_key = key[:match.start()]
                        self.log(f"val/{dataset_name}_{stripped_key}", value, on_step=False, on_epoch=True, prog_bar=False, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
            else:
                # if the dataset name is in self.dataset_names_with_samples_of_uneven_num_of_views, do not use self.val_loss but log it to the holder so that we can use a custom aggregation logic to reduce the loss
                # this is because the there are different number of views per sample in these datasets, but self.log assumes that all samples have the same number of views,
                # otherwise self.log will enter a deadlock because it will wait for the same number of samples from all ranks

                # Store in dictionary instead of logging directly
                for key, value in loss_details.items():
                    if dataset_name not in self.uneven_view_detailed_losses:
                        self.uneven_view_detailed_losses[dataset_name] = {}
                    new_key = f"val_detail_{dataset_name}_{key}"
                    if new_key not in self.uneven_view_detailed_losses[dataset_name]:
                        self.uneven_view_detailed_losses[dataset_name][new_key] = []
                    self.uneven_view_detailed_losses[dataset_name][new_key].append(value)

        loss_value = loss.detach().cpu().item()
        del loss, loss_details
        torch.cuda.empty_cache()

        # Evaluate point clouds only for the reconstruction datasets (DTU, 7-Scenes, and NRGBD)
        # eval only every 5 epochs because it's slow
        # if dataset_name in ['dtu', '7scenes', 'nrgbd', "Co3d_v2", "PointOdyssey"] and (self.current_epoch % 5 == 4 or self.current_epoch == 0):
        # # if dataset_name in ['dtu', '7scenes', 'nrgbd']:
        #     self.evaluate_reconstruction(views, preds, dataset_name=dataset_name,
        #                                  use_pts3d_from_local_head=self.eval_use_pts3d_from_local_head,
        #                                  min_conf_thr_percentile_for_local_alignment_and_icp=85,
        #                                  min_conf_thr_percentile_for_metric_cacluation=0)  # use only the very confident points for alignment and use all of the points for metric calculation

        # Evaluate metrics for camera poses
        # Yihang: this must behind the reconstruction evaluation since it has inplace modification of preds
        # if dataset_name == "Co3d_v2":
        #     self.evaluate_camera_poses(views, preds, niter_PnP=100, focal_length_estimation_method='first_view_from_global_head')

        del views, preds
        torch.cuda.empty_cache()

        return loss_value

    def on_validation_epoch_end(self) -> None:
        self.log("val/loss", self.val_loss, prog_bar=True)

        # if we dont do these, wandb for some reason cannot display the validation loss with them as the x-axis
        self.log("trainer/epoch", self.epoch_fraction, sync_dist=True)
        self.log("trainer/total_samples", self.train_total_samples.cpu().item(), sync_dist=True)
        self.log("trainer/total_images", self.train_total_images.cpu().item(), sync_dist=True)

        # self.aggregate_and_log_reconstruction_detail_losses()

        # Log the 3D reconstruction metrics
        self.aggregate_and_log_reconstruction_metrics()

    # def test_step(
    #     self, batch: List[Dict[str, torch.Tensor]], batch_idx: int
    # ) -> None:
    #     pass

    def aggregate_and_log_reconstruction_detail_losses(self):
        # log the detailes loss for uneven view datasets
        # Gather and aggregate detailed losses for uneven-view datasets across all ranks
        if torch.distributed.is_initialized():
            gathered_detailed_losses = [None] * torch.distributed.get_world_size() if self.global_rank == 0 else None
            # all_gather_object(gathered_detailed_losses, self.uneven_view_detailed_losses)
            # gather detailed losses from all ranks to rank 0
            torch.distributed.gather_object(self.uneven_view_detailed_losses, gathered_detailed_losses, dst=0)

            # log the detailed losses in rank 0
            if self.global_rank == 0:
                # Aggregate gathered data
                aggregated_losses = {}
                for rank_losses in gathered_detailed_losses:
                    for dataset_name, loss_dict in rank_losses.items():
                        if dataset_name not in aggregated_losses:
                            aggregated_losses[dataset_name] = {}
                        for key, values in loss_dict.items():
                            if key not in aggregated_losses[dataset_name]:
                                aggregated_losses[dataset_name][key] = []
                            aggregated_losses[dataset_name][key].extend(values)

                # Compute and log the mean of each loss
                for dataset_name, loss_dict in aggregated_losses.items():
                    for key, values in loss_dict.items():
                        mean_value = np.mean(values)
                        self.log(key, mean_value, rank_zero_only=True)

            # Clear the dictionary after logging
            self.uneven_view_detailed_losses.clear()

            # # Aggregate gathered data
            # aggregated_losses = {}
            # for rank_losses in gathered_detailed_losses:
            #     for dataset_name, loss_dict in rank_losses.items():
            #         if dataset_name not in aggregated_losses:
            #             aggregated_losses[dataset_name] = {}
            #         for key, values in loss_dict.items():
            #             if key not in aggregated_losses[dataset_name]:
            #                 aggregated_losses[dataset_name][key] = []
            #             aggregated_losses[dataset_name][key].extend(values)

            # # Compute and log the mean of each loss
            # for dataset_name, loss_dict in aggregated_losses.items():
            #     for key, values in loss_dict.items():
            #         mean_value = np.mean(values)
            #         self.log(key, mean_value, sync_dist=True)

            # # Clear the dictionary after logging
            # self.uneven_view_detailed_losses.clear()

    def aggregate_and_log_reconstruction_metrics(self):
        # Gather and deduplicate metrics by dataset across all ranks after all batches
        if torch.distributed.is_initialized():
            self.reconstruction_metrics_per_epoch = gather_deduplicated_scene_metrics(self.reconstruction_metrics_per_epoch)

        # Log each dataset's scene-specific metric after deduplication
        for dataset_name, scenes in self.reconstruction_metrics_per_epoch.items():
            for scene_name, metrics in scenes.items():
                self.log(f"val_recon_{dataset_name}_detail/{scene_name}/accuracy", metrics["accuracy"], sync_dist=True)
                self.log(f"val_recon_{dataset_name}_detail/{scene_name}/accuracy_median", metrics["accuracy_median"], sync_dist=True)
                self.log(f"val_recon_{dataset_name}_detail/{scene_name}/completion", metrics["completion"], sync_dist=True)
                self.log(f"val_recon_{dataset_name}_detail/{scene_name}/completion_median", metrics["completion_median"], sync_dist=True)
                self.log(f"val_recon_{dataset_name}_detail/{scene_name}/nc1", metrics["nc1"], sync_dist=True)
                self.log(f"val_recon_{dataset_name}_detail/{scene_name}/nc1_median", metrics["nc1_median"], sync_dist=True)
                self.log(f"val_recon_{dataset_name}_detail/{scene_name}/nc2", metrics["nc2"], sync_dist=True)
                self.log(f"val_recon_{dataset_name}_detail/{scene_name}/nc2_median", metrics["nc2_median"], sync_dist=True)

        # Aggregate global metrics per dataset using deduplicated data
        for dataset_name, scenes in self.reconstruction_metrics_per_epoch.items():
            acc_list = [metrics["accuracy"] for metrics in scenes.values()]
            acc_med_list = [metrics["accuracy_median"] for metrics in scenes.values()]
            comp_list = [metrics["completion"] for metrics in scenes.values()]
            comp_med_list = [metrics["completion_median"] for metrics in scenes.values()]
            nc1_list = [metrics["nc1"] for metrics in scenes.values()]
            nc1_med_list = [metrics["nc1_median"] for metrics in scenes.values()]
            nc2_list = [metrics["nc2"] for metrics in scenes.values()]
            nc2_med_list = [metrics["nc2_median"] for metrics in scenes.values()]

            # Log global aggregated metrics per dataset
            mean_accuracy = np.mean(acc_list)
            median_accuracy = np.mean(acc_med_list)
            mean_completion = np.mean(comp_list)
            median_completion = np.mean(comp_med_list)
            mean_nc1 = np.mean(nc1_list)
            median_nc1 = np.mean(nc1_med_list)
            mean_nc2 = np.mean(nc2_list)
            median_nc2 = np.mean(nc2_med_list)

            self.log(f"val_recon_{dataset_name}/accuracy", mean_accuracy, sync_dist=True)
            self.log(f"val_recon_{dataset_name}/accuracy_median", median_accuracy, sync_dist=True)
            self.log(f"val_recon_{dataset_name}/completion", mean_completion, sync_dist=True)
            self.log(f"val_recon_{dataset_name}/completion_median", median_completion, sync_dist=True)
            self.log(f"val_recon_{dataset_name}/nc1", mean_nc1, sync_dist=True)
            self.log(f"val_recon_{dataset_name}/nc1_median", median_nc1, sync_dist=True)
            self.log(f"val_recon_{dataset_name}/nc2", mean_nc2, sync_dist=True)
            self.log(f"val_recon_{dataset_name}/nc2_median", median_nc2, sync_dist=True)

        # Clear all dataset metrics after logging
        self.reconstruction_metrics_per_epoch.clear()

    # Function to estimate camera poses using fast_pnp
    @staticmethod
    def estimate_camera_poses(preds, views=None, niter_PnP=10, focal_length_estimation_method='individual'):
        """Estimate camera poses and focal lengths using fast_pnp in parallel."""

        batch_size = len(preds[0]["pts3d_in_other_view"])  # Get the batch size

        # Prepare data_for_processing
        data_for_processing = []

        for i in range(batch_size):
            # Collect preds for each sample in the batch
            sample_preds = [{key: value[i].cpu() for key, value in view.items()} for view in preds]

            data_for_processing.append(sample_preds)

        # Estimate the focal length
        def estimate_focal_for_sample(sample_preds):
            if focal_length_estimation_method == 'first_view_from_global_head':
                # Use global head outputs for focal length estimation
                pts3d_i = sample_preds[0]["pts3d_in_other_view"].unsqueeze(0)  # Shape: (1, H, W, 3)
                conf_i = sample_preds[0]["conf"].unsqueeze(0)                  # Shape: (1, H, W)
            elif focal_length_estimation_method == 'first_view_from_local_head':
                # Use local head outputs for focal length estimation
                pts3d_i = sample_preds[0]["pts3d_local_aligned_to_global"].unsqueeze(0)  # Shape: (1, H, W, 3)
                conf_i = sample_preds[0]["conf_local"].unsqueeze(0)                       # Shape: (1, H, W)
            elif focal_length_estimation_method == 'individual':
                # Focal length will be estimated individually per view
                return sample_preds
            else:
                raise ValueError(f"Unknown focal_length_estimation_method: {focal_length_estimation_method}")

            # Estimate focal length using the provided function and confidence mask
            estimated_focal = estimate_focal(pts3d_i, conf_i, min_conf_thr_percentile=10)

            # Store the estimated focal length in sample_preds
            for view_pred in sample_preds:
                view_pred["focal_length"] = estimated_focal
                # view_pred["focal_length"] = 256.64

            return sample_preds

        with ThreadPoolExecutor() as executor:
            data_for_processing = list(executor.map(estimate_focal_for_sample, data_for_processing))

        # Estimate the camera poses
        # Use ProcessPoolExecutor to parallelize processing across samples in the batch
        poses_c2w_all = []
        estimated_focals_all = []

        # Use partial to fix arguments
        from functools import partial

        estimate_cam_pose_one_sample_partial = partial(estimate_cam_pose_one_sample, niter_PnP=niter_PnP, min_conf_thr_percentile=85)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(estimate_cam_pose_one_sample_partial, data_for_processing))

        # Collect results from all processed samples
        for poses_c2w_sample, estimated_focals_sample in results:
            poses_c2w_all.append(poses_c2w_sample)
            estimated_focals_all.append(estimated_focals_sample)

        return poses_c2w_all, estimated_focals_all

    @staticmethod
    def correct_preds_orientation(preds, views):
        # *In-place* correction of the orientation of the predicted points and confidence maps

        # correct the shape of the predicted points and confidence maps if the view is portrait
        # this is because the data loader transposed the input images and valid_masks to landscape
        # see datasets/base/base_stereo_view_dataset.py
        if views is not None:
            for pred, view in zip(preds, views):
                # debug: use GT point map to estimate poses
                # pred["pts3d_in_other_view"] = view["pts3d"]  # shape (B, H, W, 3)
                # pred["conf"] = view['valid_mask'].float() if "valid_mask" in view else torch.ones_like(pred["conf"])  # shape (B, H, W)
                # pred["focal_length"] = view["camera_intrinsics"][:, 0, :2].sum(1)
                # end debug

                # check if the view is protrait or landscape (true_shape: (H, W))
                conf_list = []
                pts3d_list = []

                for i in range(view["true_shape"].shape[0]):
                    H, W = view["true_shape"][i]
                    if H > W:  # portrait
                        # Transpose the tensors
                        transposed_conf = pred["conf"][i].transpose(0, 1)
                        transposed_pts3d = pred["pts3d_in_other_view"][i].transpose(0, 1)

                        # Append the transposed tensors to the lists
                        conf_list.append(transposed_conf)
                        pts3d_list.append(transposed_pts3d)
                    else:
                        # Append the original tensors to the lists
                        conf_list.append(pred["conf"][i])
                        pts3d_list.append(pred["pts3d_in_other_view"][i])

                pred["conf"] = conf_list
                pred["pts3d_in_other_view"] = pts3d_list

                if "pts3d_local" in pred:
                    conf_local_list = []
                    pts3d_local_list = []
                    if "pts3d_local_aligned_to_global" in pred:
                        pts3d_local_aligned_to_global_list = []

                    for i in range(view["true_shape"].shape[0]):
                        H, W = view["true_shape"][i]
                        if H > W:
                            # Transpose the tensors
                            transposed_conf_local = pred["conf_local"][i].transpose(0, 1)
                            transposed_pts3d_local = pred["pts3d_local"][i].transpose(0, 1)
                            if "pts3d_local_aligned_to_global" in pred:
                                transposed_pts3d_local_aligned_to_global = pred["pts3d_local_aligned_to_global"][i].transpose(0, 1)

                            # Append the transposed tensors to the lists
                            conf_local_list.append(transposed_conf_local)
                            pts3d_local_list.append(transposed_pts3d_local)
                            if "pts3d_local_aligned_to_global" in pred:
                                pts3d_local_aligned_to_global_list.append(transposed_pts3d_local_aligned_to_global)
                        else:
                            # Append the original tensors to the lists
                            conf_local_list.append(pred["conf_local"][i])
                            pts3d_local_list.append(pred["pts3d_local"][i])
                            if "pts3d_local_aligned_to_global" in pred:
                                pts3d_local_aligned_to_global_list.append(pred["pts3d_local_aligned_to_global"][i])

                    pred["conf_local"] = conf_local_list
                    pred["pts3d_local"] = pts3d_local_list
                    if "pts3d_local_aligned_to_global" in pred:
                        pred["pts3d_local_aligned_to_global"] = pts3d_local_aligned_to_global_list

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler_config = self.hparams.scheduler

            # HACK: if the class is pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR,
            # both warmup_epochs and max_epochs should be scaled.
            # more specifically, max_epochs should be scaled to total number of steps that we will have during training,
            # and warmup_epochs should be scaled up proportionally.
            if scheduler_config.func is LinearWarmupCosineAnnealingLR:
                # Extract the keyword arguments from the partial object
                scheduler_kwargs = {k: v for k, v in scheduler_config.keywords.items()}
                original_warmup_epochs = scheduler_kwargs['warmup_epochs']
                original_max_epochs = scheduler_kwargs['max_epochs']

                total_steps = self.trainer.estimated_stepping_batches  # total number of total steps in all training epochs

                # Scale warmup_epochs and max_epochs
                scaled_warmup_epochs = int(original_warmup_epochs * total_steps / original_max_epochs)
                scaled_max_epochs = total_steps

                # Update the kwargs with scaled values
                scheduler_kwargs.update({
                    'warmup_epochs': scaled_warmup_epochs,
                    'max_epochs': scaled_max_epochs
                })

                # Re-initialize the scheduler with updated parameters
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    **scheduler_kwargs
                )
            else:
                scheduler = scheduler_config(optimizer=optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'name': 'train/lr',  # put lr inside train group in loggers
                    'scheduler': scheduler,
                    'interval': 'step' if scheduler_config.func is LinearWarmupCosineAnnealingLR else 'epoch',
                    'frequency': 1,
                }
            }

        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        # Load pretrained weights if available and not resuming
        # note that if resume_from_checkpoint is set, the Trainer is responsible for actually loading the checkpoint
        # so we are only using resume_from_checkpoint as a check of whether we should load the pretrained weights
        if self.pretrained and not self.resume_from_checkpoint:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> None:
        log.info(f"Loading pretrained: {self.pretrained}")
        if isinstance(self.net, FlashDUSt3R):  # if the model is FlashDUSt3R, use the weights of the first head only
            ckpt = torch.load(self.pretrained)
            ckpt = self._update_ckpt_keys(ckpt, new_head_name='downstream_head', head_to_keep='downstream_head1', head_to_discard='downstream_head2')
            self.net.load_state_dict(ckpt["model"], strict=False)
            del ckpt  # in case it occupies memory
        elif isinstance(self.net, STream3R):
            if self.pretrained.endswith("DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"):
                # if the model is STream3R and the pretrained model is DUSt3R, load a subset of the weights into the net
                self.net.load_from_dust3r_checkpoint(self.pretrained)
            else:
                # if the checkpoint is also STream3R, load all weights
                log.info(f"Loading pretrained weights from {self.pretrained}")
                checkpoint = torch.load(self.pretrained)
                filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.startswith('net.')}
                # Remove the 'net.' prefix from the keys
                filtered_state_dict = {k[len('net.'):]: v for k, v in filtered_state_dict.items()}
                # Load the filtered state_dict into the model
                self.net.load_state_dict(filtered_state_dict, strict=True)

    @staticmethod
    def _update_ckpt_keys(ckpt, new_head_name='downstream_head', head_to_keep='downstream_head1', head_to_discard='downstream_head2'):
        """Helper function to use the weights of a model with multiple heads in a model with a single head.
        specifically, keep only the weights of the first head and delete the weights of the second head.
        """
        new_ckpt = {'model': {}}

        for key, value in ckpt['model'].items():
            if key.startswith(head_to_keep):
                new_key = key.replace(head_to_keep, new_head_name)
                new_ckpt['model'][new_key] = value
            elif key.startswith(head_to_discard):
                continue
            else:
                new_ckpt['model'][key] = value

        return new_ckpt


def estimate_cam_pose_one_sample(sample_preds, device='cpu', niter_PnP=10, min_conf_thr_percentile=0):
    poses_c2w = []
    estimated_focals = []

    # Define the function to process each view
    def process_view(view_idx):
        pts3d = sample_preds[view_idx]["pts3d_in_other_view"].cpu().numpy().squeeze()  # (H, W, 3)
        valid_mask = sample_preds[view_idx]["conf"].cpu().numpy().squeeze() > 1.0  # Confidence mask
        # use the confidence map to filter out low-confidence points
        # conf_threshold_value = torch.quantile(sample_preds[view_idx]["conf"].view(-1), min_conf_thr_percentile / 100.0)
        # valid_mask = sample_preds[view_idx]["conf"].cpu().numpy().squeeze() >= float(conf_threshold_value ) # Confidence mask
        focal_length = float(sample_preds[view_idx]["focal_length"]) if "focal_length" in sample_preds[view_idx] else None

        # Call fast_pnp with unflattened pts3d and mask
        focal_length, pose_c2w = fast_pnp(
            torch.tensor(pts3d),
            focal_length,  # Guess focal length
            torch.tensor(valid_mask, dtype=torch.bool),
            "cpu",
            pp=None,  # Use default principal point (center of image)
            niter_PnP=niter_PnP
        )

        if pose_c2w is None or focal_length is None:
            log.warning(f"Failed to estimate pose for view {view_idx}")
            return np.eye(4), focal_length  # Return identity pose in case of failure

        # Return the results for this view
        return pose_c2w.cpu().numpy(), focal_length

    # Use ThreadPoolExecutor to process views in parallel
    with ThreadPoolExecutor() as executor:
        # Map the process_view function to each view index
        results = list(executor.map(process_view, range(len(sample_preds))))

    # Collect the results
    for pose_c2w_result, focal_length_result in results:
        poses_c2w.append(pose_c2w_result)
        estimated_focals.append(focal_length_result)

    return poses_c2w, estimated_focals


def estimate_focal(pts3d_i, conf_i, pp=None, min_conf_thr_percentile=10):
    B, H, W, THREE = pts3d_i.shape
    assert B == 1  # Since we're processing one sample at a time

    if pp is None:
        pp = torch.tensor((W / 2, H / 2), device=pts3d_i.device).view(1, 2)  # Shape: (1, 2)

    # Flatten the confidence map using reshape instead of view
    conf_flat = conf_i.reshape(-1)

    # Compute the confidence threshold based on the percentile
    percentile = min_conf_thr_percentile / 100.0  # Convert to a fraction
    conf_threshold = torch.quantile(conf_flat, percentile)

    # Create the confidence mask based on the computed threshold
    conf_mask = conf_i >= conf_threshold
    conf_mask = conf_mask.view(B, H, W)  # Ensure shape is (B, H, W)

    # Check if there are enough valid points
    if conf_mask.sum() < 10:  # Adjust the minimum number as needed
        print("Not enough high-confidence points for focal estimation.")
        # Optionally, adjust the percentile or set conf_mask to all True
        # For example:
        # conf_mask = torch.ones_like(conf_mask, dtype=torch.bool)

    focal = estimate_focal_knowing_depth_and_confidence_mask(
        pts3d_i, pp.unsqueeze(0), conf_mask, focal_mode="weiszfeld"
    ).ravel()
    return float(focal)
