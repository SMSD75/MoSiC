import argparse
import copy
from datetime import date
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import deepspeed  # <-- NEW: Import DeepSpeed

from utils.clustering import PerDatasetClustering
from dataset.data_loader import CocoDataModule, PascalVOCDataModule, SamplingMode, VideoDataModule
from evaluation.eval_metrics import PredsmIoU
from evaluation.evaluation import HbirdEvaluation, get_hb_dataset
from evaluation.evaluator import LinearFinetuneModule
from models.models import RAFTFF, CoTrackerFF, FeatureExtractor, FeatureForwarder
from utils.my_utils import denormalize_video_cotracker_compatible, find_optimal_assignment, denormalize_video, overlay_video_cmap, process_and_save_gif
from matplotlib.colors import ListedColormap
from models.optimizer import MoSiCOptimizer
import torchvision.transforms as trn

from dataset.image_transformations import Compose, Resize
import dataset.video_transformations as video_transformations
import numpy as np
import random
from experiments.exp_mosic import MoSiC
from cotracker.utils.visualizer import Visualizer
from utils.my_utils import all_gather_concat
import torch.distributed as dist
from models.models import FixedMaskPatchDropout

import wandb
import timm
project_name = "MoSiC"
cmap = ListedColormap([
    '#FF0000', '#0000FF', '#008000', '#A52A2A', '#FFFF00', '#FFA500', '#800080',
    '#FFFFFF', '#000000', '#800000', '#808000', '#008080', '#000080', '#808080', '#C0C0C0'
])

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


class MoSiCCoTracker(MoSiC):
    def __init__(
        self, input_size, vit_model, num_prototypes=200, topk=5,
        context_frames=6, context_window=6, grid_size=32, logger=None,
        model_type='dino', latent_tracking=False, feature_upsampling='bilinear',
        sk_epsilon=0.05, sk_k=10, use_symmetric_loss=False, use_soft_assignment=False,
        use_EMA_teacher=False, teacher_momentum=0.9, mask_ratio=0, teacher_eval=False,
        use_lora=False, lora_r=8, lora_alpha=32, lora_dropout=0.1, teacher_feature_upsampling='bilinear'
    ):
        super(MoSiCCoTracker, self).__init__(
            input_size, vit_model, num_prototypes, topk, context_frames,
            context_window, logger, model_type, mask_ratio, use_lora, lora_r, lora_alpha, lora_dropout
        )
        self.grid_size = grid_size
        self.FF = CoTrackerFF(
            self.eval_spatial_resolution, context_frames, context_window,
            topk=topk, grid_size=grid_size, feature_head=None
        )
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.latent_tracking = latent_tracking
        self.feature_upsampling = feature_upsampling
        self.teacher_feature_upsampling = teacher_feature_upsampling
        self.sk_epsilon = sk_epsilon
        self.sk_k = sk_k
        self.use_symmetric_loss = use_symmetric_loss
        self.use_soft_assignment = use_soft_assignment
        self.teacher_eval = teacher_eval
        self.use_EMA_teacher = use_EMA_teacher
        self.mask_ratio = mask_ratio
        self.teacher_momentum = None
        
        if self.use_EMA_teacher:
            self.teacher = copy.deepcopy(self.feature_extractor)
            if 'clip' in model_type:
                self.teacher.model.patch_drop = None
            else:
                self.teacher.model.patch_drop = torch.nn.Identity()
            
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher_momentum = teacher_momentum
            # Add teacher's MLP head and prototypes
            self.teacher_mlp_head = copy.deepcopy(self.mlp_head)
            for param in self.teacher_mlp_head.parameters():
                param.requires_grad = False
            self.teacher_prototypes = nn.Parameter(copy.deepcopy(self.prototypes.data), requires_grad=False)
            self.teacher_upsampler = copy.deepcopy(self.up_sampler)
            for param in self.teacher_upsampler.parameters():
                param.requires_grad = False

        # if self.mask_ratio > 0:
        #     self.feature_extractor.model.patch_drop = FixedMaskPatchDropout(num_prefix_tokens=self.feature_extractor.model.num_prefix_tokens)
        

    def train(self, mode=True):
        """Override the train method to exclude FF from training mode."""
        super().train(mode)

        if self.mask_ratio > 0:
            self.feature_extractor.model.patch_drop.training = True
        if mode:
            self.FF.eval()  # freeze CoTrackerFF
        return self


    def eval(self):
        """Override the eval method to exclude FF from training mode."""
        super().eval()
        if self.mask_ratio > 0:
            self.feature_extractor.model.patch_drop.training = False
        self.FF.eval()  # Set FF to evaluation mode
        return self

    def train_step(self, datum, pred_tracks, pred_visibility):
        """
        One training step for MoSiCCoTracker.
        This is what we will call inside our DeepSpeed engine for the forward pass.
        """
        self.normalize_prototypes()
        if self.use_EMA_teacher:
            with torch.no_grad():
                w = self.teacher_prototypes.data.clone()
                w = F.normalize(w, dim=1, p=2)
                self.teacher_prototypes.copy_(w)    
        bs, nf, c, h, w = datum.shape
        denormalized_video = denormalize_video_cotracker_compatible(datum)
        if datum.dtype == torch.float16:
            denormalized_video = denormalized_video.half()

        if self.mask_ratio > 0:
            B = bs
            L = self.feature_extractor.eval_spatial_resolution**2
            num_keep = max(0, int(L * (self.mask_ratio)))
            keep_indices = torch.argsort(torch.randn(B, L, device=datum.device), dim=-1)[:, num_keep: ]
            keep_indices = keep_indices.unsqueeze(1).repeat(1, nf, 1)
            keep_indices.requires_grad = False
            keep_indices = keep_indices.flatten(0, 1)
        else:
            keep_indices = None

        dataset_teacher_features = None
        if self.use_EMA_teacher:
            dataset_teacher_features, _ = self.teacher.forward_features(datum.flatten(0, 1), mask=None)
            dataset_features, _ = self.feature_extractor.forward_features(datum.flatten(0, 1), mask=keep_indices)
        else:
            dataset_features, _ = self.feature_extractor.forward_features(datum.flatten(0, 1), mask=keep_indices)
        if self.mask_ratio > 0:
            filled_features = torch.zeros((bs*nf, self.feature_extractor.eval_spatial_resolution ** 2, self.feature_extractor.d_model), device=datum.device)
            # if dataset_teacher_features is not None:
            #     dataset_teacher_features = filled_features.scatter(
            #         dim=1, 
            #         index=keep_indices.unsqueeze(-1).expand(-1, -1, self.feature_extractor.d_model),
            #         src=dataset_teacher_features
            #     )
            # else:
            dataset_features = filled_features.scatter(
                dim=1, 
                index=keep_indices.unsqueeze(-1).expand(-1, -1, self.feature_extractor.d_model),
                src=dataset_features
            )
        pred_tracks, pred_visibility = self.FF.forward(denormalized_video)
        # gif_files = process_and_save_gif(
        #                     datum,
        #                     grid_size=self.grid_size,
        #                     draw_trails=False,
        #                     trail_length=10,
        #                     gif_path_template="tracked_{batch}.gif",
        #                     fps=15
        #                 )

        pred_tracks.requires_grad = False
        pred_visibility.requires_grad = False
        _, npix, dim = dataset_features.shape

        if self.latent_tracking:
            # New sampling approach
            reshaped_features = dataset_features.reshape(bs*nf, self.spatial_resolution, self.spatial_resolution, dim)
            reshaped_features = reshaped_features.permute(0, 3, 1, 2).contiguous()
            reshaped_features = reshaped_features.reshape(bs, nf, dim, self.spatial_resolution, self.spatial_resolution)

            if dataset_teacher_features is not None:
                reshaped_teacher_features = dataset_teacher_features.reshape(bs*nf, self.spatial_resolution, self.spatial_resolution, dim)
                reshaped_teacher_features = reshaped_teacher_features.permute(0, 3, 1, 2).contiguous()
                reshaped_teacher_features = reshaped_teacher_features.reshape(bs, nf, dim, self.spatial_resolution, self.spatial_resolution)

            pred_tracks = torch.clamp(pred_tracks, min=0, max=max(h-1, w-1))
            # Normalize pred_tracks to [-1, 1]
            pred_tracks_normalized = pred_tracks.float()
            pred_tracks_normalized[..., 0] = 2.0 * pred_tracks_normalized[..., 0] / (h - 1) - 1.0
            pred_tracks_normalized[..., 1] = 2.0 * pred_tracks_normalized[..., 1] / (w - 1) - 1.0

            if datum.dtype == torch.float16:
                pred_tracks_normalized = pred_tracks_normalized.half()

            grid = pred_tracks_normalized.reshape(bs, nf, -1, 1, 2)  # [bs, nf, num_points, 1, 2]

            # Sample features via grid_sample
            selected_features = []
            if dataset_teacher_features is not None:
                selected_teacher_features = []
            for t in range(nf):
                features_t = reshaped_features[:, t]  # [bs, dim, sr, sr]
                grid_t = grid[:, t]                  # [bs, num_points, 1, 2]
                sampled = F.grid_sample(
                    features_t, grid_t, mode='bilinear', align_corners=True
                )  # [bs, dim, num_points, 1]
                selected_features.append(sampled.squeeze(-1).permute(0, 2, 1))

                if dataset_teacher_features is not None:
                    teacher_features_t = reshaped_teacher_features[:, t]
                    sampled_teacher = F.grid_sample(
                        teacher_features_t, grid_t, mode='bilinear', align_corners=True
                    )
                    selected_teacher_features.append(sampled_teacher.squeeze(-1).permute(0, 2, 1))

            selected_features = torch.stack(selected_features, dim=1)  # [bs, nf, num_points, dim]
            if dataset_teacher_features is not None:
                selected_teacher_features = torch.stack(selected_teacher_features, dim=1)
        else:
            # Original sampling approach
            resized_reshaped_features = self._resize_features_orig_res(bs, nf, h, w, dataset_features, upsampling_mode=self.feature_upsampling)
            if dataset_teacher_features is not None:
                resized_reshaped_teacher_features = self._resize_features_orig_res(bs, nf, h, w, dataset_teacher_features, upsampling_mode=self.teacher_feature_upsampling)

            batch_idx = torch.arange(bs).view(bs, 1, 1).expand(-1, nf, pred_tracks.shape[2])
            time_idx = torch.arange(nf).view(1, nf, 1).expand(bs, -1, pred_tracks.shape[2])
            pred_tracks = torch.clamp(pred_tracks, min=0, max=max(h-1, w-1)).round().long()

            # pred_tracks = torch.randint(0, h, pred_tracks.shape)

            selected_features = resized_reshaped_features[
                batch_idx,
                time_idx,
                pred_tracks[..., 0].long(),  # height
                pred_tracks[..., 1].long(),  # width
            ]
            if dataset_teacher_features is not None:
                selected_teacher_features = resized_reshaped_teacher_features[
                    batch_idx,
                    time_idx,
                    pred_tracks[..., 0].long(),  # height
                    pred_tracks[..., 1].long(),  # width
                ]
                
        bs, nf, T, dim = selected_features.shape
        if self.mask_ratio > 0:
            # if dataset_teacher_features is not None:
            #     not_zero_rows = selected_teacher_features.sum(dim=-1) != 0
            # else:
            not_zero_rows = selected_features.sum(dim=-1) != 0
            not_zero_rows.requires_grad = False
        else:
            not_zero_rows = None


        projected_dataset_features = self.mlp_head(selected_features)
        if dataset_teacher_features is not None:
            projected_dataset_teacher_features = self.teacher_mlp_head(selected_teacher_features)
            projected_dim = projected_dataset_teacher_features.shape[-1]
            projected_dataset_teacher_features = projected_dataset_teacher_features.reshape(-1, projected_dim)
        projected_dim = projected_dataset_features.shape[-1]
        projected_dataset_features = projected_dataset_features.reshape(-1, projected_dim)
        normalized_projected_features = F.normalize(projected_dataset_features, dim=-1, p=2)
        if dataset_teacher_features is not None:
            normalized_projected_teacher_features = F.normalize(projected_dataset_teacher_features, dim=-1, p=2)

        dataset_scores = torch.einsum('bd,nd->bn', normalized_projected_features, self.prototypes)
        if self.use_EMA_teacher:
            student_sk_epsilon = 2 * self.sk_epsilon
        else:
            student_sk_epsilon = self.sk_epsilon
        if dataset_teacher_features is not None:
            dataset_teacher_scores = torch.einsum('bd,nd->bn', normalized_projected_teacher_features, self.teacher_prototypes)
            dataset_q = find_optimal_assignment(dataset_teacher_scores, self.sk_epsilon, self.sk_k, world_size=dist.get_world_size())
        else:
            dataset_q = find_optimal_assignment(dataset_scores, student_sk_epsilon, self.sk_k, world_size=dist.get_world_size())
        dataset_q = dataset_q.reshape(bs, nf, T, self.num_prototypes)
        dataset_scores = dataset_scores.reshape(bs, nf, T, self.num_prototypes)

        if self.use_soft_assignment:
            clustering_loss = self._soft_assignment(pred_visibility, dataset_scores, dataset_q, not_zero_rows)
        else:
            clustering_loss = self._hard_assignment(pred_visibility, dataset_scores, dataset_q, not_zero_rows)

        return clustering_loss

    def _resize_features_orig_res(self, bs, nf, h, w, dataset_features, upsampling_mode='bilinear'):
        dim = dataset_features.size(-1)
        reshaped_features = dataset_features.reshape(bs*nf, self.spatial_resolution, self.spatial_resolution, dim)
        reshaped_features = reshaped_features.permute(0, 3, 1, 2).contiguous()
        resized_features = []
        for feature in reshaped_features:
            if upsampling_mode == 'bilinear':
                feature = F.interpolate(feature.unsqueeze(0), size=(h, w), mode="bilinear")
            elif upsampling_mode == 'nearest':
                feature = F.interpolate(feature.unsqueeze(0), size=(h, w), mode="nearest")
            elif upsampling_mode == "student_learnable":
                feature = self.up_sampler(feature.unsqueeze(0))
            elif upsampling_mode == "teacher_learnable":
                feature = self.teacher_upsampler(feature.unsqueeze(0))
            feature = feature.permute(0, 2, 3, 1).squeeze(0)
            resized_features.append(feature)
        resized_reshaped_features = torch.stack(resized_features, dim=0)
        resized_reshaped_features = resized_reshaped_features.reshape(bs, nf, h, w, dim)
        return resized_reshaped_features


    def _hard_assignment(self, pred_visibility, dataset_scores, dataset_q, mask=None):
        bs, nf, T, _ = dataset_scores.shape
        f1_scores = dataset_scores[:, 0]
        f1_scores = f1_scores.unsqueeze(1).repeat(1, nf, 1, 1)
        dataset_q = dataset_q.argmax(dim=-1)
        f1_scores = f1_scores.permute(0, 3, 1, 2).contiguous()
        visibility_weights = pred_visibility.squeeze(-1).float().to(dataset_scores.device)  # B T N
        if mask is not None:
            visibility_weights = visibility_weights * mask
        clustering_loss = (self.criterion(f1_scores / 0.1, dataset_q.long()) * visibility_weights).mean()
        
        if self.use_symmetric_loss:
            f1_q = dataset_q[:, 0]
            f1_q = f1_q.unsqueeze(1).repeat(1, nf, 1)
            dataset_scores = dataset_scores.permute(0, 3, 1, 2).contiguous()
            visibility_weights = pred_visibility.squeeze(-1).float().to(dataset_scores.device)  # B T N
            if mask is not None:
                visibility_weights = visibility_weights * mask
            clustering_loss +=(self.criterion(dataset_scores / 0.1, f1_q.long()) * visibility_weights).mean()

        return clustering_loss

    def _soft_assignment(self, pred_visibility, dataset_scores, dataset_q, mask=None):
        assert torch.all(torch.isclose(dataset_q.sum(dim=-1), torch.ones_like(dataset_q.sum(dim=-1)), rtol=1e-5))

        bs, nf, T, _ = dataset_scores.shape
        f1_scores = dataset_scores[:, 0]
        f1_scores = f1_scores.unsqueeze(1).repeat(1, nf, 1, 1)
        # Remove argmax to keep soft assignments
        visibility_weights = pred_visibility.squeeze(-1).float().to(dataset_scores.device)  # B T N
        # Use KL divergence or cross-entropy for soft assignments
        normalized_f1_scores = F.log_softmax(f1_scores / 0.1, dim=-1)
        if mask is not None:
            visibility_weights = visibility_weights * mask
        clustering_loss = -1 * (torch.sum(dataset_q * normalized_f1_scores, dim=-1) * visibility_weights).mean()
        
        if self.use_symmetric_loss:
            f1_q = dataset_q[:, 0]
            f1_q = f1_q.unsqueeze(1).repeat(1, nf, 1, 1)
            visibility_weights = pred_visibility.squeeze(-1).float().to(dataset_scores.device)  # B T N
            normalized_dataset_scores = F.log_softmax(dataset_scores / 0.1, dim=-1)
            if mask is not None:
                visibility_weights = visibility_weights * mask
            clustering_loss += -1 *(torch.sum(normalized_dataset_scores * f1_q, dim=-1) * visibility_weights).mean()
        return clustering_loss
    
    def log_tracked_frames(self, datum):
        """
        For logging visualization in wandb or other frameworks.
        """
        denormalized_video = denormalize_video_cotracker_compatible(datum)
        pred_tracks, pred_visibility = self.FF.forward(denormalized_video)
        vis = Visualizer(save_dir=f"./co_tracker_saved_videos/", pad_value=120, linewidth=3)
        overlayed_video = vis.visualize(denormalized_video, pred_tracks, pred_visibility, save_video=False)
        return overlayed_video

    def update_teacher(self):
        """
        Update teacher model using exponential moving average of student parameters
        """
        if self.teacher_momentum is None:
            return
        with torch.no_grad():
            # Update feature extractor
            for param_t, param_s in zip(self.teacher.parameters(), self.feature_extractor.parameters()):
                param_t.data = (self.teacher_momentum) * param_t.data + (1 - self.teacher_momentum) * param_s.data
            
            # Update MLP head
            for param_t, param_s in zip(self.teacher_mlp_head.parameters(), self.mlp_head.parameters()):
                param_t.data = (self.teacher_momentum) * param_t.data + (1 - self.teacher_momentum) * param_s.data
            
            # Update prototypes
            self.teacher_prototypes.data = (self.teacher_momentum) * self.teacher_prototypes.data + (1 - self.teacher_momentum) * self.prototypes.data
            
            # Update up_sampler
            for param_t, param_s in zip(self.teacher_upsampler.parameters(), self.up_sampler.parameters()):
                param_t.data = (self.teacher_momentum) * param_t.data + (1 - self.teacher_momentum) * param_s.data
            
        self.teacher.eval()


class MoSiCTrainerDS:
    """
    A trainer class adapted for DeepSpeed. This replaces the manual optimizer usage
    with DeepSpeed engine calls (forward, backward, step).
    """
    def __init__(
        self, 
        training_dataloader, 
        test_dataloader, 
        hbird_datasets,
        num_epochs, 
        logger,
        ds_engine,
        save_dir="/ssdstore/ssalehi/MoSiC_models"
    ):
        """
        Args:
          ds_engine: The DeepSpeed engine that wraps our model & optimizer.
        """
        # We'll store some references for convenience.
        self.training_dataloader = training_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.logger = logger
        self.hbird_datasets = hbird_datasets
        self.max_val_score = -1
        
        # The underlying original model is accessible via ds_engine.module
        # but for code clarity, let's keep this pointer:
        self.ds_engine = ds_engine
        self.save_dir = save_dir

    def train(self):
        """
        The main training loop with DeepSpeed: 
        We iterate over epochs and steps using `self.ds_engine`.
        """
        for epoch in range(self.num_epochs):
            print(f"Epoch: {epoch}")
            # (Optional) you can call validate here at intervals
            self.train_one_epoch(epoch)


    def _print_gradient_info(self):
        for n, p in self.ds_engine.module.named_parameters():
            if p.requires_grad:
                print(f"\n{n}:")
                print(f"Grad exists: {p.grad is not None}")
                if p.grad is not None:
                    print(f"Grad norm: {p.grad.norm().item()}")

    def train_one_epoch(self, epoch):
        """
        One epoch of training. 
        We pull batches from ds_engine (which knows how to handle them) 
        and do forward/backward/step with DeepSpeed.
        """
        local_rank = self.ds_engine.local_rank  # Get the local rank from ds_engine
        epoch_loss = 0
        is_main_process = dist.get_rank() == 0

        for i, batch in enumerate(self.training_dataloader):
            if i % 60 == 0:
                self.validate_hb()
                # num_itr = len(self.training_dataloader)
                # self.validate(epoch * num_itr + i)
            datum, annotations, pred_tracks, pred_visibility = batch
            # Forward pass via ds_engine.module (the underlying model)
            annotations = annotations.squeeze(1).to(local_rank)
            datum = datum.squeeze(1).to(local_rank)
            pred_tracks = pred_tracks.to(local_rank)
            pred_visibility = pred_visibility.to(local_rank)
            # Check if half precision (fp16) is enabled in DeepSpeed config
            if self.ds_engine.fp16_enabled():
                datum = datum.half()
            
            self.ds_engine.train()
            clustering_loss = self.ds_engine.module.train_step(datum, pred_tracks, pred_visibility)

            # DeepSpeed backward & step
            self.ds_engine.backward(clustering_loss)
            # self._print_gradient_info()
            self.ds_engine.step()
            self.ds_engine.module.update_teacher()

            epoch_loss += clustering_loss.item()
            print(f"Iteration: {i}, Loss: {clustering_loss.item()}")
            if is_main_process:
                self.logger.log({"clustering_loss": clustering_loss.item()})
                lr = self.ds_engine.get_lr()[0] 
                self.logger.log({"lr": lr})
            
        if is_main_process:
            self.log_tracked_frames(datum)
        epoch_loss /= (i + 1)
        print(f"Epoch {epoch} Loss: {epoch_loss}")

        # (Optional) Log or do something after each epoch
        # Example: self.validate(epoch)

    def validate(self, epoch, val_spatial_resolution=56):
        """
        Distributed validation that gathers all eval_targets and cluster_maps
        on every rank, then only rank 0 computes the final metric.
        """
        # Access the underlying model
        model = self.ds_engine.module
        model.eval()
        
        # Current local rank / global rank
        local_rank = self.ds_engine.local_rank
        global_rank = self.ds_engine.global_rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # We'll store the local data here
        local_eval_features = []
        local_eval_targets = []

        model = self.ds_engine.module
        feature_spatial_resolution = model.feature_extractor.eval_spatial_resolution
        spatial_feature_dim = 50  # or model.get_dino_feature_spatial_dim()
        clustering_method = PerDatasetClustering(spatial_feature_dim, 21)
        overclustering_100_method = PerDatasetClustering(spatial_feature_dim, 100)
        overclustering_300_method = PerDatasetClustering(spatial_feature_dim, 300)
        overclustering_500_method = PerDatasetClustering(spatial_feature_dim, 500)
        
        with torch.no_grad():
            visualization_x = None
            for i, (x, y) in enumerate(self.test_dataloader):
                # Move input to local rank device
                img = x.to(local_rank)
                target = (y * 255).long().to(local_rank)

                if visualization_x is None and i == 0:
                    visualization_x = x  # keep a copy for logging on rank 0

                # 1) Get features
                spatial_features = model.validate_step(img)  # shape (B, np, dim)

                # 2) Resize your target if needed
                print(target.shape)
                resized_target = F.interpolate(
                    target.float(), 
                    size=(val_spatial_resolution, val_spatial_resolution),
                    mode="nearest"
                ).long()

                # 3) Collect them in local lists
                local_eval_features.append(spatial_features)
                local_eval_targets.append(resized_target)

        # Concatenate along batch dimension on each GPU
        if local_eval_features:
            local_eval_features = torch.cat(local_eval_features, dim=0)  # shape [local_B, np, dim]
            local_eval_targets = torch.cat(local_eval_targets, dim=0)    # shape [local_B, H, W]
        else:
            # In case a rank has 0 samples (e.g., data not perfectly divisible):
            local_eval_features = torch.zeros((0, 50, 256), device=local_rank)  # or appropriate shape
            local_eval_targets = torch.zeros((0, val_spatial_resolution, val_spatial_resolution), device=local_rank)

        # 4) Gather all rank's features/targets to every GPU
        #    so that each GPU has the entire dataset.
        gathered_eval_features = all_gather_concat(local_eval_features)  # shape [total_B, np, dim]
        gathered_eval_targets = all_gather_concat(local_eval_targets)    # shape [total_B, H, W]

        # 5) On each GPU, produce cluster_maps
        B, npix, dim = gathered_eval_features.shape
        gathered_eval_features = gathered_eval_features.reshape(
            B,
            feature_spatial_resolution,
            feature_spatial_resolution,
            dim
        )
        gathered_eval_features = gathered_eval_features.permute(0, 3, 1, 2).contiguous()
        gathered_eval_features = F.interpolate(
            gathered_eval_features, 
            size=(val_spatial_resolution, val_spatial_resolution),
            mode="bilinear"
        )
        # shape now [B, dim, val_spatial_resolution, val_spatial_resolution]
        gathered_eval_features = gathered_eval_features.reshape(B, dim, -1).permute(0, 2, 1)
        # shape [B, HW, dim]
        gathered_eval_features = gathered_eval_features.detach().cpu().unsqueeze(1)
        # shape [B, 1, HW, dim]

        cluster_maps = clustering_method.cluster(gathered_eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]

        overclustering_300_maps = overclustering_300_method.cluster(gathered_eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]
        
        overclustering_100_maps = overclustering_100_method.cluster(gathered_eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]

        overclustering_500_maps = overclustering_500_method.cluster(gathered_eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]

        cluster_maps = cluster_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
        # shape [B, 1, val_spatial_resolution, val_spatial_resolution]

        overclustering_300_maps = overclustering_300_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
        # shape [B, 1, val_spatial_resolution, val_spatial_resolution]

        overclustering_100_maps = overclustering_100_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
        # shape [B, 1, val_spatial_resolution, val_spatial_resolution]

        overclustering_500_maps = overclustering_500_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
        # shape [B, 1, val_spatial_resolution, val_spatial_resolution]

        # 6) Now compute the metric only on rank 0 to avoid duplication
        if global_rank == 0:
            if visualization_x is not None:
                denormalized_x = denormalize_video(visualization_x.cpu())
                self.log_cluster_maps(cluster_maps[:10], denormalized_x[:10])
            # valid_idx for ignoring 255
            # shape check: gathered_eval_targets is [B, H, W]
            # cluster_maps is [B, 1, H, W] => we can squeeze that to match
            
            valid_idx = gathered_eval_targets != 255
            valid_target = gathered_eval_targets[valid_idx].cpu()       # [some_size]
            valid_pred = cluster_maps[valid_idx.cpu()].cpu()         # [some_size]

            valid_oc_pred = overclustering_300_maps[valid_idx.cpu()].cpu()
            valid_oc_pred_100 = overclustering_100_maps[valid_idx.cpu()].cpu()
            valid_oc_pred_500 = overclustering_500_maps[valid_idx.cpu()].cpu()

            metric = PredsmIoU(21, 21)
            overclustering_300_metric = PredsmIoU(300, 21)
            overclustering_100_metric = PredsmIoU(100, 21)
            overclustering_500_metric = PredsmIoU(500, 21)
            metric.update(valid_target, valid_pred)
            overclustering_300_metric.update(valid_target, valid_oc_pred)
            overclustering_100_metric.update(valid_target, valid_oc_pred_100)
            overclustering_500_metric.update(valid_target, valid_oc_pred_500)
            jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
            oc_300_jac, oc_300_tp, oc_300_fp, oc_300_fn, oc_300_reordered_preds, oc_300_matched_bg_clusters = overclustering_300_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)
            oc_100_jac, oc_100_tp, oc_100_fp, oc_100_fn, oc_100_reordered_preds, oc_100_matched_bg_clusters = overclustering_100_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)
            oc_500_jac, oc_500_tp, oc_500_fp, oc_500_fn, oc_500_reordered_preds, oc_500_matched_bg_clusters = overclustering_500_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)

            print(f"[Rank 0] Validation finished, global mIoU: {jac}")
            print(f"[Rank 0] Validation finished, global overclustering mIoU: {oc_300_jac}")
            self.logger.log({"val_k=gt_miou": jac})
            self.logger.log({"val_k=300_miou": oc_300_jac})
            self.logger.log({"val_k=100_miou": oc_100_jac})
            self.logger.log({"val_k=500_miou": oc_500_jac})

        # If you want to ensure rank 0 finishes logging before others proceed, barrier:
        dist.barrier()
    
    @torch.no_grad()
    def validate_hb(self):
        device = self.ds_engine.device
        model = self.ds_engine.module
        model.eval()
        global_rank = self.ds_engine.global_rank
        eval_spatial_resolution = model.feature_extractor.eval_spatial_resolution
        embeddings_size = model.feature_extractor.d_model

        val_score = 0

        if model.teacher_eval:
            model = model.teacher.model
        else:
            model = model.feature_extractor.model
        
        for hbird_dataset in self.hbird_datasets:
            dataset_size = hbird_dataset.get_train_dataset_size()
            num_classes = hbird_dataset.get_num_classes()
            train_loader = hbird_dataset.train_dataloader()
            val_loader = hbird_dataset.val_dataloader()

            if hbird_dataset.get_name() == "VOCDataModule_HB":
                memory_size = (1024) * 10**2
                ignore_index = 255
            elif hbird_dataset.get_name() == "Ade20kDataModule":
                memory_size = (1024) * 10**2
                ignore_index = 0
                continue
            else:
                raise ValueError("Unknown dataset name")

            evaluator = HbirdEvaluation(model, train_loader, n_neighbours=30, 
                                augmentation_epoch=1, num_classes=num_classes, 
                                device=device, eval_spatial_resolution=eval_spatial_resolution, d_model=embeddings_size, nn_params=None, memory_size=memory_size, 
                                dataset_size=dataset_size)
                
            hbird_miou = evaluator.evaluate(val_loader, eval_spatial_resolution, return_knn_details=False, ignore_index=ignore_index)
            if global_rank == 0:
                val_score = hbird_miou
                self.logger.log({f"val_hb_miou_{hbird_dataset.get_name()}": hbird_miou})
                # print(f"val_hb_miou_{hbird_dataset.get_name()}: {hbird_miou}")
            # dist.barrier()

        # if global_rank == 0:
        #     if val_score > self.max_val_score:
        #         self.save_model(f"{self.save_dir}", self.num_epochs, val_score)
        #         self.max_val_score = val_score
        
        dist.barrier()
    

    def save_model(self, dir_name, epoch, val_score):
        ## just save feature_extractor.model
        model = self.ds_engine.module
        model.eval()
        model_type = model.feature_extractor.model_type
        model = model.feature_extractor.model
        if os.path.exists(dir_name):
            torch.save(model.state_dict(), f"{dir_name}/MoSiC_{model_type}_epoch_{epoch}_val_score_{val_score}.pth")
        else:   
            os.makedirs(dir_name)
            torch.save(model.state_dict(), f"{dir_name}/MoSiC_{model_type}_epoch_{epoch}_val_score_{val_score}.pth")
    
    
    

    def _log_video_to_wandb(self, video_tensor, fps=4, name="video"):
        """
        Log a video tensor to wandb.
        
        Args:
            video_tensor (torch.Tensor): Tensor of shape (T, 3, H, W) in uint8 format
            fps (int, optional): Frames per second. Defaults to 4.
            name (str, optional): Name of the video in wandb. Defaults to "video".
        """
        video = video_tensor.cpu().numpy()
        # video = np.transpose(video, (0, 2, 3, 1))
        self.logger.log({name: wandb.Video(video, fps=fps, format="mp4")})

    def log_cluster_maps(self, cluster_maps, denormalized_images):
        resized_denormalized_images= F.interpolate(denormalized_images, size=(cluster_maps.size(-2), cluster_maps.size(-1)), mode="bilinear")
        _, overlayed_video = overlay_video_cmap(cluster_maps.squeeze(1), resized_denormalized_images)
        self._log_images_to_wandb(overlayed_video, name="clustered_images")


    def _log_images_to_wandb(self, images, name="images"):
        self.logger.log({name: wandb.Image(images)})


    def log_tracked_frames(self, datum):
        """
        Process video frames and log the overlayed tracking visualization to wandb.
        
        Args:
            datum (torch.Tensor): Input video tensor
        """
        for d in datum[:10]:
            overlayed_video = self.ds_engine.module.log_tracked_frames(d.unsqueeze(0))  # B T 3 H W
            self._log_video_to_wandb(overlayed_video[0].to(torch.uint8), fps=4, name="overlayed_video")



def verify_arguments(args):

    if args.dataset == "epic_kitchens" and (args.frame_sampling_mode != "regular" or args.regular_step < 5):
        raise ValueError("Epic kitchens dataset is required for regular frame sampling mode and high regular step")
    
    if (args.model_choice != 'MoSiCCoTracker') and (args.grid_size != 0 or args.latent_tracking):
        raise ValueError("Grid size and latent tracking are only supported for MoSiCCoTracker")

    if args.frame_sampling_mode != 'regular' and args.regular_step > 0:
        raise ValueError("Regular step is only supported for regular frame sampling mode")
    
    if args.feature_upsampling != 'off' and args.latent_tracking:
        raise ValueError("Feature upsampling is not supported for latent tracking")
    
def run(args):
    """
    Main function adapted to launch DeepSpeed and do multi-GPU training.
    """
    verify_arguments(args)
    # ------------------------------
    # 1) SETUP WANDB and ARGS
    # ------------------------------
    device = f"cuda:{args.local_rank}" if args.local_rank >= 0 else "cuda:0"
    config = vars(args)
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    experiment_name = (
        f"model_choice:{args.model_choice}_grid:{args.grid_size}_latent:{args.latent_tracking}_"
        f"frame_sampling:{args.frame_sampling_mode}_model_type:{args.model_type}_batch_size:{args.batch_size}_"
        f"num_epochs:{args.num_epochs}_explaination:{args.explaination}_regular_step:{args.regular_step}_"
        f"num_clip_frames:{args.num_clip_frames}_num_clips:{args.num_clips}_num_gpus:{dist.get_world_size()}_num_epochs:{args.num_epochs}_feature_upsampling:{args.feature_upsampling}_num_prototypes:{args.num_prototypes}_sk_epsilon:{args.sk_epsilon}_sk_k:{args.sk_k}_use_symmetric_loss:{args.use_symmetric_loss}"
        f"_use_soft_assignment:{args.use_soft_assignment}_use_EMA_teacher:{args.use_EMA_teacher}_mask_ratio:{args.mask_ratio}_dataset:{args.dataset}_teacher_momentum:{args.teacher_momentum}_crop_scale:{args.crop_scale}_teacher_eval:{args.teacher_eval}"
        f"_use_lora:{args.use_lora}_lora_r:{args.lora_r}_lora_alpha:{args.lora_alpha}_lora_dropout:{args.lora_dropout}_mixed_datasets:{args.mixed_datasets}_sampling_ratios:{args.sampling_ratios}_teacher_feature_upsampling:{args.teacher_feature_upsampling}"
    )   
    if dist.get_rank() == 0:
        logger = wandb.init(
            project=project_name,
            group=d1,
            mode=args.wandb_mode,
            job_type=args.job_type,
            config=config,
            name=experiment_name
        )
    else:
        logger = None

    # ------------------------------
    # 2) PREPARE DATA
    # ------------------------------
    rand_color_jitter = video_transformations.RandomApply(
        [video_transformations.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )],
        p=0.8
    )
    data_transform_list = [
        rand_color_jitter,
        video_transformations.RandomGrayscale(),
        video_transformations.RandomGaussianBlur()
    ]
    data_transform = video_transformations.Compose(data_transform_list)
    if args.model_choice == 'MoSiCCoTracker':
        if args.crop_scale > 0:
            video_transform_list = [
                video_transformations.Resize((224, 224)),
                video_transformations.RandomHorizontalFlip(),
                video_transformations.RandomResizedCrop((224, 224), scale=(args.crop_scale, 1.0)),
                video_transformations.ClipToTensor(
                mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
                )
            ]
        else:
            video_transform_list = [
                video_transformations.Resize((224, 224)),
                video_transformations.RandomHorizontalFlip(),
                video_transformations.ClipToTensor(
                mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
                )
            ]
    else:
        video_transform_list = [
            video_transformations.Resize(224),
            video_transformations.RandomResizedCrop((224, 224), scale=(args.crop_scale, 1.0)),
            video_transformations.RandomHorizontalFlip(),
            video_transformations.ClipToTensor(
                mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
            )
        ]
    video_transform = video_transformations.Compose(video_transform_list)
    if args.model_choice == 'MoSiCCoTracker':
        transformations_dict = {
            "data_transforms": None,
            "target_transforms": None,
            "shared_transforms": video_transform
        }
    else:
        transformations_dict = {
            "data_transforms": data_transform,
            "target_transforms": None,
            "shared_transforms": video_transform
        }

    prefix = os.environ.get("DATA_PREFIX")
    path_dicts = {}
    if "ytvos" in args.dataset or ("ytvos" in args.mixed_datasets and args.dataset == "mixed"):
        data_path = os.path.join(prefix, "dataset/all_frames/train_all_frames/JPEGImages/")
        path_dicts['ytvos'] = {"class_directory": data_path, "annotation_directory": "", "meta_file_path": ""} 
        # data_path = os.path.join(prefix, "dataset/train1/")
    if "lvos" in args.dataset or ("lvos" in args.mixed_datasets and args.dataset == "mixed"):
        data_path = os.path.join(prefix, "LVOS/train/JPEGImages/")
        path_dicts['lvos'] = {"class_directory": data_path, "annotation_directory": "", "meta_file_path": ""}
    if "vspw" in args.dataset or ("vspw" in args.mixed_datasets and args.dataset == "mixed"):
        data_path = os.path.join(prefix, "VSPW/data/")
        path_dicts['vspw'] = {"class_directory": data_path, "annotation_directory": "", "meta_file_path": ""}
    if "epic_kitchens" in args.dataset or ("epic_kitchens" in args.mixed_datasets and args.dataset == "mixed"):
        data_path = os.path.join(prefix, "EpicKitchens/")
        annotation_path = os.path.join(prefix, "Annotations/")
        trajectory_path = os.path.join(prefix, "train1/trajectories/")
    if "mose" in args.dataset or ("mose" in args.mixed_datasets and args.dataset == "mixed"):
        data_path = os.path.join(prefix, "MOSE/train/JPEGImages/")
        path_dicts['mose'] = {"class_directory": data_path, "annotation_directory": "", "meta_file_path": ""}
        annotation_path = os.path.join(prefix, "mose/")
        trajectory_path = os.path.join(prefix, "mose/")
    if "kinetics" in args.dataset or ("kinetics" in args.mixed_datasets and args.dataset == "mixed"):
        data_path = os.path.join(prefix, "kinetics_images/")
        path_dicts['kinetics'] = {"class_directory": data_path, "annotation_directory": "", "meta_file_path": ""}
        annotation_path = os.path.join(prefix, "kinetics_images/")
        trajectory_path = os.path.join(prefix, "kinetics_trajectories/")
    if "co3d" in args.dataset or ("co3d" in args.mixed_datasets and args.dataset == "mixed"):
        # data_path = os.path.join(prefix, "co3d_images/")
        data_path = os.path.join("/nvmestore/ssalehi/", "co3d_zips/")
        path_dicts['co3d'] = {"class_directory": data_path, "mapping_path": "/nvmestore/ssalehi/all_frames_complete.json", "zip_mapping_path": "/nvmestore/ssalehi/zip_mapping.json"}
        annotation_path = None
        trajectory_path = None
    
    # if args.dataset == "co3d":
    #     path_dict = {"mapping_path": os.path.join(data_path, "/hddstore/ssalehi/all_frames_complete.json"), 
    #                  "zip_mapping_path": os.path.join(data_path, "/hddstore/ssalehi/zip_mapping.json")}
    # else:
    #     path_dict = {
    #         "class_directory": data_path,
    #         "annotation_directory": "",
    #         "meta_file_path": ""
    #     }

    if args.frame_sampling_mode == 'uniform':
        sampling_mode = SamplingMode.UNIFORM
    elif args.frame_sampling_mode == 'dense':
        sampling_mode = SamplingMode.DENSE
    elif args.frame_sampling_mode == 'full':
        sampling_mode = SamplingMode.FULL
    else:
        sampling_mode = SamplingMode.Regular
    
    if args.dataset == "mixed":
                video_data_module = VideoDataModule(
                "mixed",
                path_dicts,
                args.num_clips,
                args.num_clip_frames,
                sampling_mode,
                args.regular_step,
                args.batch_size,
                args.num_workers,
                mixed_datasets=args.mixed_datasets,
                sampling_ratios=args.sampling_ratios
                )
    else:
        video_data_module = VideoDataModule( #"ytvos_trj", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers, grid_size=args.grid_size
            "co3d" if args.dataset == "co3d" else "ytvos_trj", path_dicts[args.dataset],
            num_clips=args.num_clips,
            num_clip_frames=args.num_clip_frames,
            sampling_mode=sampling_mode,
            regular_step=args.regular_step,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            grid_size=args.grid_size
        )
    # video_data_module = VideoDataModule(
    #     "ytvos_trj", path_dict,
    #     num_clips=args.num_clips,
    #     num_clip_frames=args.num_clip_frames,
    #     sampling_mode=sampling_mode,
    #     regular_step=args.regular_step,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     grid_size=args.grid_size
    # )
    video_data_module.setup(transformations_dict)
    video_data_module.make_data_loader()  # Creates .data_loader

    # Prepare model
    if args.model_type == 'dino-s':
        # vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        vit_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    elif args.model_type == "clip-s":
        vit_model = timm.create_model('vit_small_patch14_224.clip', pretrained=True)
    elif args.model_type == "clip-b":
        vit_model = timm.create_model('eva02_base_patch16_clip_224', pretrained=True)
    elif args.model_type == 'dinov2-b':
        vit_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model_type == 'dinov2-s':
        vit_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model_type == 'dinov2-l':
        vit_model = timm.create_model('vit_large_patch14_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model_type == 'dinov2r-s':
        vit_model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model_type == 'dinov2r-b':
        vit_model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model_type == 'dinov2r-l':
        vit_model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model_type == 'leopart-s':
        from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch8_224
        prefix = os.environ.get("DATA_PREFIX")
        path_to_checkpoint = os.path.join(prefix, "leopart_knn/leopart_vits16.ckpt")
        vit_model = vit_small_patch16_224()  # or vit_base_patch8_224() if you want to use our larger model
        state_dict = torch.load(path_to_checkpoint)
        vit_model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()}, strict=False)
    elif args.model_type == 'TimeT-s':
        from timm.models.vision_transformer import vit_small_patch16_224
        path_to_checkpoint = os.path.join(prefix, "paneco_models/TimeT.pth")
        vit_model = vit_small_patch16_224()
        state_dict = torch.load(path_to_checkpoint)
        vit_model.load_state_dict({".".join(k.split(".")[2:]): v for k, v in state_dict.items()}, strict=False)
    elif args.model_type == 'NeCo-s':
        path_to_checkpoint = os.path.join(prefix, "paneco_models/neco_on_dinov2r_vit14_model.ckpt")
        vit_model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        state_dict = torch.load(path_to_checkpoint)
        vit_model.load_state_dict(state_dict, strict=False)

    if args.model_choice == 'MoSiCCoTracker':
        patch_prediction_model = MoSiCCoTracker(
            224, vit_model,
            logger=logger,
            model_type=args.model_type,
            grid_size=args.grid_size,
            latent_tracking=args.latent_tracking,
            feature_upsampling=args.feature_upsampling,
            num_prototypes=args.num_prototypes,
            use_symmetric_loss=args.use_symmetric_loss,
            use_soft_assignment=args.use_soft_assignment,
            use_EMA_teacher=args.use_EMA_teacher,
            teacher_momentum=args.teacher_momentum,
            mask_ratio=args.mask_ratio,
            teacher_eval=args.teacher_eval,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            teacher_feature_upsampling=args.teacher_feature_upsampling
        )
    else:
        patch_prediction_model = MoSiC(
            224, vit_model,
            logger=logger,
            model_type=args.model_type,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )



    # ------------------------------
    # 4) CALCULATE TOTAL STEPS
    # ------------------------------
    # The DataLoader is in video_data_module.data_loader
    steps_per_epoch = len(video_data_module.data_loader)
    # If we do gradient_accumulation_steps > 1, we must account for that:
    gradient_accumulation_steps = 1  # or read from config
    total_training_steps = (steps_per_epoch // gradient_accumulation_steps) * args.num_epochs
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_training_steps}")


    # ------------------------------
    # 5) DEEPSPEED CONFIG (With Cosine LR & WD Scheduling)
    # ------------------------------
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
                "weight_decay": 1e-2  # Fixed weight decay value
            }
        },
        
        "fp16": {
            "enabled": False
        },
        
        "scheduler": {
            "type": "CosineAnnealingLR",
            "params": {
                "T_max": total_training_steps,
                "eta_min": 0,
            }
        },
}


    # ------------------------------
    # 4) Initialize DeepSpeed
    # ------------------------------
    # Note: We pass our dataset to `training_data` so that the engine knows how to distribute it.
    # If you'd rather do it manually, you can pass model_engine.train_batch(...) yourself.
    model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
        args=args,
        model=patch_prediction_model,
        model_parameters=patch_prediction_model.get_optimization_params(),
        training_data=video_data_module.dataset,  # a PyTorch DataLoader
        config=ds_config
    )

    # Prepare val/test data
    image_val_transform = trn.Compose([
        trn.Resize((args.input_size, args.input_size)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    shared_val_transform = Compose([
        Resize(size=(args.input_size, args.input_size)),
    ])
    val_transforms = {
        "img": image_val_transform,
        "target": None,
        "shared": shared_val_transform
    }
    dataset = PascalVOCDataModule(
        batch_size=args.batch_size,
        train_transform=val_transforms,
        val_transform=val_transforms,
        test_transform=val_transforms,
        num_workers=args.num_workers,
        dir= os.environ.get("DATA_PREFIX") + "/VOCSegmentation"
    )

    # coco_data_dir = os.environ.get("DATA_PREFIX")
    # ignore_index = 255
    # file_list = os.listdir(os.path.join(coco_data_dir, "coco2017", "train2017"))
    # file_list_val = os.listdir(os.path.join(coco_data_dir, "coco2017", "val2017"))
    # random.shuffle(file_list_val)
    # # sample 10% of train images
    # random.shuffle(file_list)
    # file_list = file_list[:int(len(file_list)*0.1)]
    # print(f"sampled {len(file_list)} COCO images for training")
    # dataset = CocoDataModule(batch_size=args.batch_size,
    #                                 num_workers=args.num_workers,
    #                                 file_list=file_list,
    #                                 data_dir=os.environ.get("DATA_PREFIX"),
    #                                 file_list_val=file_list_val,
    #                                 mask_type="thing",
    #                                 train_transforms=image_val_transform,
    #                                 val_transforms=image_val_transform,
    #                                 val_target_transforms=shared_val_transform)

    dataset.setup()
    test_dataloader = dataset.get_test_dataloader()

    ignore_index, pascal_hb_dataset = get_hb_dataset('voc', os.path.join(os.environ.get("DATA_PREFIX"), "VOCSegmentation_tiny"), 32, 224, 2)
    ignore_index, ade20k_hb_dataset = get_hb_dataset('ade20k', os.path.join(os.environ.get("DATA_PREFIX"), "ADE20k/ADEChallengeData2016"), 32, 224, 2)
    hb_datasets = [ade20k_hb_dataset, pascal_hb_dataset]

    # Create our new trainer (DeepSpeed version)
    trainer = MoSiCTrainerDS(
        training_dataloader=training_dataloader,
        test_dataloader=test_dataloader,
        hbird_datasets=hb_datasets  ,
        num_epochs=args.num_epochs,
        logger=logger,
        ds_engine=model_engine,
        save_dir=args.save_dir
    )

    # ------------------------------
    # 5) TRAIN
    # ------------------------------
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DeepSpeed requires local_rank for multi-GPU usage
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DeepSpeed')

    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument(
        '--frame_sampling_mode', type=str,
        choices=['uniform', 'dense', 'full', 'regular'], default='dense',
        help='Frame sampling mode: uniform, dense, full, or regular'
    )
    parser.add_argument(
        '--model_type', type=str,
        choices=['dino-s', 'leopart-s', 'dinov2r-s', 'dinov2r-b', 'dinov2r-l', 'dinov2-b', 'dinov2-s', 'dinov2-l', 'TimeT-s', 'NeCo-s', 'clip-s', 'clip-b'], default='dino-s',
        help='Select model type: dino or dinov2'
    )
    parser.add_argument(
        '--regular_step', type=int, default=1,
        help="Regular step for the video dataset"
    )
    parser.add_argument(
        '--job_type', type=str, default="debug_clustering_ytvos",
        help="Job type for wandb"
    )
    parser.add_argument("--explaination", type=str, default="full_frames")
    parser.add_argument('--grid_size', type=int, default=16, help='Grid size for the model')
    parser.add_argument('--latent_tracking', type=bool, default=False)
    parser.add_argument("--num_clip_frames", type=int, default=4)
    parser.add_argument("--num_clips", type=int, default=1)
    parser.add_argument("--num_prototypes", type=int, default=200)
    parser.add_argument("--sk_epsilon", type=float, default=0.06)
    parser.add_argument("--sk_k", type=int, default=3)
    parser.add_argument("--dataset", type=str, choices=['ytvos', 'epic_kitchens', 'mose', 'kinetics', 'co3d','vspw', 'lvos', "mixed"], default="ytvos")
    parser.add_argument(
        '--model_choice', type=str,
        choices=['MoSiCCoTracker', 'MoSiC'],
        default='MoSiCCoTracker',
        help='Select model: MoSiCCoTracker or MoSiC'
    )
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--feature_upsampling", type=str, choices=['bilinear', 'nearest', 'off', 'student_learnable'], default='bilinear')
    parser.add_argument("--teacher_feature_upsampling", type=str, choices=['bilinear', 'nearest', 'off', 'teacher_learnable'], default='bilinear')
    parser.add_argument("--use_symmetric_loss", type=bool, help="Use symmetric loss", default=False)
    parser.add_argument("--use_soft_assignment", type=bool, help="Use soft assignment", default=False)
    parser.add_argument("--use_EMA_teacher", type=bool, help="Use EMA teacher", default=False)
    parser.add_argument("--teacher_momentum", type=float, help="Teacher momentum", default=0.99)
    parser.add_argument("--mask_ratio", type=float, help="Mask ratio", default=0)
    parser.add_argument("--crop_scale", type=float, help="Crop scale", default=0.0)
    parser.add_argument("--teacher_eval", type=bool, help="Teacher eval", default=False)
    parser.add_argument("--sampling_ratios", type=float, nargs='+', default=[1.0, 1.0], help='Sampling ratios for mixed datasets')
    parser.add_argument("--mixed_datasets", type=str, nargs='+', default=["ytvos", "mose"], help='Mixed datasets')
    parser.add_argument('--use_lora', type=bool, default=False, help='Use LoRA fine-tuning instead of unfreezing layers')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank parameter')
    parser.add_argument('--save_dir', type=str, default="/ssdstore/ssalehi/MoSiC_models", help='Save directory')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout parameter')
    parser.add_argument("--hb_dataset_name", type=str, default="voc")
    args = parser.parse_args()

    # This call sets up distributed training for DeepSpeed
    deepspeed.init_distributed()

    run(args)
