import torch
from utils.clustering import PerDatasetClustering
from models.models import FeatureExtractor
import torchvision.transforms as trn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
import scann
import argparse
import random
import time
import os
import gc
from evaluation.eval_metrics import PredsmIoU
import numpy as np
from dataset.image_transformations import CombTransforms
from torchvision import transforms
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch8_224
import timm
from models.models import FeatureExtractorBeta as FeatureExtractor
from models.models import FeatureExtractorSimple
import tqdm
from dataset.image_transformations import get_hbird_train_transforms, get_hbird_val_transforms
from dataset.data_loader import PascalVOCDataModule, Ade20kDataModule, VOCDataModule_HB
from utils.my_utils import all_gather_concat, gather_in_chunks, get_cluster_map, merge_models, overlay_overclustering_maps
import torch.distributed as dist
import deepspeed  # <-- Added DeepSpeed import
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dataset.image_transformations import Compose, Resize


class HbirdEvaluation():
    def __init__(self, model, train_loader, n_neighbours, augmentation_epoch, num_classes, device, eval_spatial_resolution=None, d_model=None, nn_params=None, memory_size=None, dataset_size=None, f_mem_p=None, l_mem_p=None):
        if nn_params is None:
            nn_params = {}
        self.model = model
        self.device = device
        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.n_neighbours = n_neighbours
        self.model.eval()
        self.model = self.model.to(self.device)
        self.num_classes = num_classes
        self.num_sampled_features = None
        self.f_mem_p = f_mem_p
        self.l_mem_p = l_mem_p
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
        self.feature_memory = list()
        self.label_memory = list()
        world_size = dist.get_world_size()
        if self.memory_size is not None:
            self.num_sampled_features = self.memory_size // (dataset_size * self.augmentation_epoch)
            ## create memory of specific size
            self.feature_memory = torch.zeros(((self.memory_size // world_size) + 1, self.d_model))
            self.label_memory = torch.zeros(((self.memory_size // world_size) + 1, self.num_classes))
        self.create_memory(train_loader, num_classes=self.num_classes, eval_spatial_resolution=eval_spatial_resolution)
        self.save_memory()
        self.feature_memory = self.feature_memory.to(self.device)
        self.label_memory = self.label_memory.to(self.device)
        self.create_NN(self.n_neighbours, **nn_params)
    
    def create_NN(self, n_neighbours=30, distance_measure="dot_product", num_leaves=512, num_leaves_to_search=32, anisotropic_quantization_threshold=0.2, num_reordering_candidates=120, dimensions_per_block=4):
        self.NN_algorithm = scann.scann_ops_pybind.builder(self.feature_memory.detach().cpu().numpy(), n_neighbours, distance_measure).tree(
    num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=self.feature_memory.size(0)).score_ah(
    2, anisotropic_quantization_threshold=anisotropic_quantization_threshold, dimensions_per_block=dimensions_per_block).reorder(num_reordering_candidates).build()

    @torch.no_grad()
    def create_memory(self, train_loader, num_classes, eval_spatial_resolution):
        num_itr = len(train_loader)
        idx = 0
        features_memory = []
        label_memory = []
        with torch.no_grad():
            for j in tqdm.tqdm(range(self.augmentation_epoch), desc='Augmentation loop'):
                for i, (x, y) in enumerate(tqdm.tqdm(train_loader, desc='Memory Creation loop')):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y = (y * 255).long()
                    y[y == 255] = 0
                    features = self.model.forward_features(x)[:, self.model.num_prefix_tokens:] # features of shape (BS, PS, D)
                    # features = self.model.forward_features(x)["x_norm_patchtokens"]
                    input_size = x.shape[-1]
                    patch_size = input_size // eval_spatial_resolution
                    patchified_gts = self.patchify_gt(y, patch_size) ## (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                    one_hot_patch_gt = F.one_hot(patchified_gts, num_classes=num_classes).float()
                    label = one_hot_patch_gt.mean(dim=3)
                    if self.memory_size is None:
                        # Memory Size is unbounded so we store all the features
                        normalized_features = features / torch.norm(features, dim=2, keepdim=True)

                        normalized_features = normalized_features.flatten(0, 1)
                        label = label.flatten(0, 2)
                        # self.feature_memory[idx:idx+normalized_features.size(0)] = normalized_features.detach().cpu()
                        # self.label_memory[idx:idx+label.size(0)] = label.detach().cpu()
                        # idx += normalized_features.size(0)
                        features_memory.append(normalized_features)
                        label_memory.append(label)
                    else:
                        # Memory Size is bounded so we need to select/sample some features only
                        sampled_features, sampled_indices = self.sample_features(features, patchified_gts)
                        normalized_sampled_features = sampled_features / torch.norm(sampled_features, dim=2, keepdim=True)
                        label = label.flatten(1, 2)
                        ## select the labels of the sampled features
                        sampled_indices = sampled_indices.to(self.device)
                        ## repeat the label for each sampled feature
                        label_hat = label.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1]))

                        # label_hat = label.gather(1, sampled_indices)
                        normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
                        label_hat = label_hat.flatten(0, 1)
                        self.feature_memory[idx:idx+normalized_sampled_features.size(0)] = normalized_sampled_features.detach().cpu()
                        self.label_memory[idx:idx+label_hat.size(0)] = label_hat.detach().cpu()
                        idx += normalized_sampled_features.size(0)

            if dist.is_initialized():
                if len(features_memory) > 0:
                    self.feature_memory = torch.cat(features_memory)
                    self.label_memory = torch.cat(label_memory)
                else:
                    self.feature_memory = self.feature_memory.to(self.device)
                    ## cast label memory to float16
                    # self.label_memory = self.label_memory.to(torch.float16)
                    self.label_memory = self.label_memory.to(self.device)
                # Gather memory in chunks and detach from GPU
                self.feature_memory = torch.cat(gather_in_chunks([self.feature_memory]))
                self.label_memory = torch.cat(gather_in_chunks([self.label_memory]))
            else:
                if len(features_memory) > 0:
                    self.feature_memory = torch.cat(features_memory)
                    self.label_memory = torch.cat(label_memory)
                else:
                    self.feature_memory = self.feature_memory
                    self.label_memory = self.label_memory

    def save_memory(self):
        if self.f_mem_p is not None:
            torch.save(self.feature_memory.cpu(), self.f_mem_p)
        if self.l_mem_p is not None:
            torch.save(self.label_memory.cpu(), self.l_mem_p)
    def load_memory(self):
        if self.f_mem_p is not None and self.l_mem_p is not None and os.path.isfile(self.f_mem_p) and os.path.isfile(self.l_mem_p):
            self.feature_memory = torch.load(self.f_mem_p).to(self.device)
            self.label_memory = torch.load(self.l_mem_p).to(self.device)
            return True
        return False
    def sample_features(self, features, pathified_gts):
        sampled_features = []
        sampled_indices = []
        for k, gt in enumerate(tqdm.tqdm(pathified_gts)):
            class_frequency = self.get_class_frequency(gt)
            patch_scores, nonzero_indices = self.get_patch_scores(gt, class_frequency)

            patch_scores = patch_scores.flatten()
            nonzero_indices = nonzero_indices.flatten()

            # assert zero_score_idx[0].size(0) != 0 ## for pascal every patch should belong to one class
            patch_scores[~nonzero_indices] = 1e6

            # sample uniform distribution with the same size as the
            # number of nonzero indices (we use the sum here as the
            # nonzero_indices matrix is a boolean mask)
            uniform_x = torch.rand(nonzero_indices.sum())
            patch_scores[nonzero_indices] *= uniform_x
            feature = features[k]

            ### select the least num_sampled_features score indices
            _, indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)

            sampled_indices.append(indices)
            samples = feature[indices]
            sampled_features.append(samples)

        sampled_features = torch.stack(sampled_features)
        sampled_indices = torch.stack(sampled_indices)

        return sampled_features, sampled_indices

    def get_class_frequency(self, gt):
        class_frequency = torch.zeros((self.num_classes), device=self.device)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                patch_classes = gt[i, j].unique()
                class_frequency[patch_classes] += 1

        return class_frequency

    def get_patch_scores(self, gt, class_frequency):
        patch_scores = torch.zeros((gt.shape[0], gt.shape[1]))
        nonzero_indices = torch.zeros((gt.shape[0], gt.shape[1]), dtype=torch.bool)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                patch_classes = gt[i, j].unique()
                patch_scores[i, j] = class_frequency[patch_classes].sum()
                nonzero_indices[i, j] = patch_classes.shape[0] > 0

        return patch_scores, nonzero_indices

    def patchify_gt(self, gt, patch_size):
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h//patch_size, patch_size, w//patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h//patch_size, w//patch_size, c*patch_size*patch_size)
        return gt

    def cross_attention(self, q, k, v, beta=0.02):
        """
        Args: 
            q (torch.Tensor): query tensor of shape (bs, num_patches, d_k)
            k (torch.Tensor): key tensor of shape (bs, num_patches,  NN, d_k)
            v (torch.Tensor): value tensor of shape (bs, num_patches, NN, label_dim)
        """
        d_k = q.size(-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q.unsqueeze(2) ## (bs, num_patches, 1, d_k)
        attn = torch.einsum("bnld,bnmd->bnlm", q, k) / beta
        attn = attn.squeeze(2)
        attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(-1)
        label_hat = torch.einsum("blms,blmk->blsk", attn, v)
        label_hat = label_hat.squeeze(-2)
        return label_hat
    
    def find_nearest_key_to_query(self, q):
        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs*num_patches, d_k)
        neighbors, distances = self.NN_algorithm.search_batched(reshaped_q)
        neighbors = neighbors.astype(np.int64)
        neighbors = torch.from_numpy(neighbors).to(self.device)
        neighbors = neighbors.flatten()
        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(bs, num_patches, self.n_neighbours, -1)
        key_labels = self.label_memory[neighbors]
        key_labels = key_labels.reshape(bs, num_patches, self.n_neighbours, -1)
        return key_features, key_labels

    def evaluate(self, val_loader, eval_spatial_resolution, return_knn_details=False, ignore_index=255):
        ## clear the gpu memory
        torch.cuda.empty_cache()
        metric = PredsmIoU(self.num_classes, self.num_classes)
        self.model = self.model.to(self.device)
        label_hats = []
        lables = []
        knns = []
        knns_labels = []
        knns_ca_labels = []
        gathered_labels = []
        gathered_label_hats = []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm.tqdm(val_loader, desc='Evaluation loop')):
                x = x.to(self.device)
                _, _, h, w = x.shape
                features = self.model.forward_features(x.to(self.device))[:, self.model.num_prefix_tokens:]
                # features = self.model.forward_features(x)["x_norm_patchtokens"]
                features = features.to(self.device)
                # y = y.to(self.device)
                y = (y * 255).to(torch.int32)
                ## copy the data of features to another variable
                q = features.clone()
                q = q.detach().cpu().numpy()
                key_features, key_labels = self.find_nearest_key_to_query(q)     
                key_features = key_features.to(self.device)
                key_labels = key_labels.to(self.device)
                label_hat =  self.cross_attention(features, key_features, key_labels)
                if return_knn_details:
                    knns.append(key_features.detach().cpu())
                    knns_labels.append(key_labels.detach().cpu())
                    knns_ca_labels.append(label_hat.detach().cpu())
                bs, _, label_dim = label_hat.shape
                label_hat = label_hat.reshape(bs, eval_spatial_resolution, eval_spatial_resolution, label_dim).permute(0, 3, 1, 2)
                resized_label_hats = F.interpolate(label_hat.float(), size=(h, w), mode="bilinear")
                cluster_map = resized_label_hats.argmax(dim=1).unsqueeze(1).to(torch.int32)
                label_hats.append(cluster_map.detach().cpu())
                lables.append(y.detach().cpu())

                ## only for visualization
                # overlaid_images = overlay_overclustering_maps(
                #     x.detach().cpu(),
                #     cluster_map.detach().cpu(),
                #     model="MoSiC_dinov2",
                #     save_dir="./clustering_results",
                #     prefix=f"HB_MoSiC_dinov2",
                #     nrow=11  # Number of images per row in the grid
                # )

            
        
        # Concatenate local results
        local_labels = torch.cat(lables)
        local_label_hats = torch.cat(label_hats)

        # Gather results from all GPUs if running distributed
        ## detach self.feature_memory from the GPU to save memory
        self.feature_memory = self.feature_memory.detach().cpu()
        self.label_memory = self.label_memory.detach().cpu()
        if dist.is_initialized():
            local_labels = local_labels.to(self.device)
            local_label_hats = local_label_hats.to(self.device)
            gathered_labels = torch.cat(gather_in_chunks([local_labels]))
            gathered_label_hats = torch.cat(gather_in_chunks([local_label_hats]))
        else:
            gathered_labels = local_labels
            gathered_label_hats = local_label_hats
        
        gathered_labels = gathered_labels.detach().cpu()
        gathered_label_hats = gathered_label_hats.detach().cpu()

        if dist.get_rank() == 0:
            valid_idx = gathered_labels != ignore_index
            valid_target = gathered_labels[valid_idx]
            valid_cluster_maps = gathered_label_hats[valid_idx]
            metric.update(valid_target, valid_cluster_maps)
            jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
            if return_knn_details:
                knns = torch.cat(knns)
                knns_labels = torch.cat(knns_labels)
                knns_ca_labels = torch.cat(knns_ca_labels)
                return jac, {"knns": knns, "knns_labels": knns_labels, "knns_ca_labels": knns_ca_labels}
            else:
                return jac


def hbird_evaluation(model, d_model, patch_size, dataset_name:str, data_dir:str, batch_size=64, input_size=224, 
                        augmentation_epoch=1, device='cpu', return_knn_details=False, n_neighbours=30, nn_params=None, 
                        ftr_extr_fn=None, memory_size=None, num_workers=8, ignore_index=255):
    eval_spatial_resolution = input_size // patch_size
    if ftr_extr_fn is None:
        feature_extractor = FeatureExtractor(model, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    else:
        feature_extractor = FeatureExtractorSimple(model, ftr_extr_fn=ftr_extr_fn, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
        
    ignore_index, dataset = get_hb_dataset(dataset_name, data_dir, batch_size, input_size, num_workers)
    dataset_size = dataset.get_train_dataset_size()
    num_classes = dataset.get_num_classes()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    
    evaluator = HbirdEvaluation(model, train_loader, n_neighbours=n_neighbours, 
                        augmentation_epoch=augmentation_epoch, num_classes=num_classes, 
                        device=device, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model, nn_params=nn_params, memory_size=memory_size, 
                        dataset_size=dataset_size)
    return evaluator.evaluate(val_loader, eval_spatial_resolution, return_knn_details=return_knn_details, ignore_index=ignore_index)

def get_hb_dataset(dataset_name, data_dir, batch_size, input_size, num_workers):
    train_transforms_dict = get_hbird_train_transforms(input_size)
    val_transforms_dict = get_hbird_val_transforms(input_size)

    train_transforms = CombTransforms(img_transform=train_transforms_dict['img'], tgt_transform=None, img_tgt_transform=train_transforms_dict['shared'])
    val_transforms = CombTransforms(img_transform=val_transforms_dict['img'], tgt_transform=None, img_tgt_transform=val_transforms_dict['shared'])
    
    dataset_size = 0
    num_classes = 0
    ignore_index = -1   
    if dataset_name == "voc":
        ignore_index = 255
        dataset = VOCDataModule_HB(batch_size=batch_size,
                                    num_workers=num_workers,
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=train_transforms,
                                    val_transforms=val_transforms,
                                    shuffle=False,
                                    return_masks=True)
        dataset.setup()
    elif dataset_name == "ade20k":
        ignore_index = 0
        dataset = Ade20kDataModule(data_dir,
                 train_transforms=train_transforms,
                 val_transforms=val_transforms,
                 shuffle=False,
                 num_workers=num_workers,
                 batch_size=batch_size)
        dataset.setup()
    else:
        raise ValueError("Unknown dataset name")
    return ignore_index,dataset




def overclustering(model, model_name, feature_spatial_resolution, test_dataloader, val_spatial_resolution, num_classes, ignore_index, device):

        model.eval()
        
        # We'll store the local data here
        eval_features = []
        eval_targets = []

        spatial_feature_dim = 50  # or model.get_dino_feature_spatial_dim()
        clustering_method = PerDatasetClustering(spatial_feature_dim, num_classes)
        overclustering_100_method = PerDatasetClustering(spatial_feature_dim, 50)
        overclustering_300_method = PerDatasetClustering(spatial_feature_dim, 100)
        overclustering_500_method = PerDatasetClustering(spatial_feature_dim, 200)

        visualization_x = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(test_dataloader):
                # Move input to local rank device
                visualization_x.append(x)
                img = x.to(device)
                target = (y * 255).long().to(device)
                # 1) Get features
                spatial_features = model.forward_features(img)[:, 1:]  # shape (B, np, dim)

                # 2) Resize your target if needed
                resized_target = F.interpolate(
                    target.float(), 
                    size=(val_spatial_resolution, val_spatial_resolution),
                    mode="nearest"
                ).long()

                # 3) Collect them in local lists
                eval_features.append(spatial_features)
                eval_targets.append(resized_target)


        # 5) On each GPU, produce cluster_maps
        visualization_x = torch.cat(visualization_x, dim=0)
        eval_features = torch.cat(eval_features, dim=0)
        B, npix, dim = eval_features.shape
        eval_features = eval_features.reshape(
            B,
            feature_spatial_resolution,
            feature_spatial_resolution,
            dim
        )
        B = 500
        eval_features = eval_features[0:B]
        visualization_x = visualization_x[0:B]
        eval_features = eval_features.permute(0, 3, 1, 2).contiguous()
        eval_features = F.interpolate(
            eval_features, 
            size=(val_spatial_resolution, val_spatial_resolution),
            mode="bilinear"
        )

        # shape now [B, dim, val_spatial_resolution, val_spatial_resolution]
        eval_features = eval_features.reshape(B, dim, -1).permute(0, 2, 1)
        # shape [B, HW, dim]
        eval_features = eval_features.detach().cpu().unsqueeze(1)
        # shape [B, 1, HW, dim]

        cluster_maps = clustering_method.cluster(eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]

        overclustering_300_maps = overclustering_300_method.cluster(eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]
        
        overclustering_100_maps = overclustering_100_method.cluster(eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]

        overclustering_500_maps = overclustering_500_method.cluster(eval_features)
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

        selection = [10, 20, 30, 40, 4, 14, 7, 18, 19, 80, 61]
        overlaid_images = overlay_overclustering_maps(
            visualization_x[selection],
            overclustering_100_maps[selection],
            model=model_name,
            save_dir="./clustering_results",
            prefix=f"overclustering_100_{model_name}",
            nrow=11  # Number of images per row in the grid
        )

        overlaid_images = overlay_overclustering_maps(
            visualization_x[selection],
            overclustering_300_maps[selection],
            model=model_name,
            save_dir="./clustering_results",
            prefix=f"overclustering_300_{model_name}",
            nrow=11  # Number of images per row in the grid
        )

        overlaid_images = overlay_overclustering_maps(
            visualization_x[selection],
            overclustering_500_maps[selection],
            model=model_name,
            save_dir="./clustering_results",
            prefix=f"overclustering_500_{model_name}",
            nrow=11  # Number of images per row in the grid
        )
            
        valid_idx = eval_targets != ignore_index
        valid_target = eval_targets[valid_idx].cpu()       # [some_size]
        valid_pred = cluster_maps[valid_idx.cpu()].cpu()         # [some_size]

        valid_oc_pred = overclustering_300_maps[valid_idx.cpu()].cpu()
        valid_oc_pred_100 = overclustering_100_maps[valid_idx.cpu()].cpu()
        valid_oc_pred_500 = overclustering_500_maps[valid_idx.cpu()].cpu()

        metric = PredsmIoU(num_classes, num_classes)
        overclustering_300_metric = PredsmIoU(300, num_classes)
        overclustering_100_metric = PredsmIoU(100, num_classes)
        overclustering_500_metric = PredsmIoU(500, num_classes)
        metric.update(valid_target, valid_pred)
        overclustering_300_metric.update(valid_target, valid_oc_pred)
        overclustering_100_metric.update(valid_target, valid_oc_pred_100)
        overclustering_500_metric.update(valid_target, valid_oc_pred_500)
        jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
        oc_300_jac, oc_300_tp, oc_300_fp, oc_300_fn, oc_300_reordered_preds, oc_300_matched_bg_clusters = overclustering_300_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)
        oc_100_jac, oc_100_tp, oc_100_fp, oc_100_fn, oc_100_reordered_preds, oc_100_matched_bg_clusters = overclustering_100_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)
        oc_500_jac, oc_500_tp, oc_500_fp, oc_500_fn, oc_500_reordered_preds, oc_500_matched_bg_clusters = overclustering_500_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)

        print(f"Validation finished, global mIoU: {jac}")
        print(f"Validation finished, global overclustering mIoU: {oc_300_jac}")
        print(f"Validation finished, global overclustering mIoU: {oc_100_jac}")
        print(f"Validation finished, global overclustering mIoU: {oc_500_jac}")



def image_knn(model, train_dataset, val_dataset, input_size, device, n_neighbours=30, nn_params=None):
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=8
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8
    )
    
    # Extract features and labels from training set
    model.eval()
    features_list = []
    labels_list = []
    
    print("Extracting training features...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            # Get CLS token or average pooling of patch tokens
            features = model.forward_features(images)
            if isinstance(features, dict):
                features = features['x_norm_clstoken']  # For DINO models
            else:
                features = features[:, 0]  # CLS token
            features = F.normalize(features, dim=1)
            features_list.append(features.cpu())
            labels_list.append(labels)
    
    # Concatenate all features and labels
    train_features = torch.cat(features_list, dim=0)
    train_labels = torch.cat(labels_list, dim=0)
    
    # Build ScaNN index
    print("Building ScaNN index...")
    if nn_params is None:
        nn_params = {
            "num_leaves": 512,
            "num_leaves_to_search": 32,
            "anisotropic_quantization_threshold": 0.2,
            "num_reordering_candidates": 120,
            "dimensions_per_block": 4
        }
        
    searcher = scann.scann_ops_pybind.builder(
        train_features.numpy(), 
        n_neighbours, 
        "dot_product"
    ).tree(
        num_leaves=nn_params["num_leaves"],
        num_leaves_to_search=nn_params["num_leaves_to_search"],
        training_sample_size=train_features.size(0)
    ).score_ah(
        2,
        anisotropic_quantization_threshold=nn_params["anisotropic_quantization_threshold"],
        dimensions_per_block=nn_params["dimensions_per_block"]
    ).reorder(
        nn_params["num_reordering_candidates"]
    ).build()
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader):
            images = images.to(device)
            features = model.forward_features(images)
            if isinstance(features, dict):
                features = features['x_norm_clstoken']
            else:
                features = features[:, 0]
            features = F.normalize(features, dim=1)
            
            # Find nearest neighbors
            neighbors, distances = searcher.search_batched(features.cpu().numpy())
            neighbors = torch.from_numpy(neighbors).to(device)
            
            # Get labels of neighbors
            neighbor_labels = train_labels[neighbors.long()]
            
            # Majority voting
            pred_labels = torch.mode(neighbor_labels, dim=1)[0]
            
            # Calculate accuracy
            correct += (pred_labels == labels.to(device)).sum().item()
            total += labels.size(0)
    
    accuracy = 100. * correct / total
    print(f"KNN Accuracy: {accuracy:.2f}%")
    return accuracy



def imagenet_100_main(model, dataset_name, data_dir, batch_size, input_size, device, n_neighbours=30, nn_params=None):


    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    import torchvision
    if dataset_name == "imagenet100":
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir+"/imagenet-100/train", transform=train_transforms)
        val_dataset = torchvision.datasets.ImageFolder(root=data_dir+"/imagenet-100/val", transform=val_transforms)
    elif dataset_name == "cifar-10":
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transforms, download=True)
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=val_transforms, download=True)
    elif dataset_name == "cifar-100":
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, transform=train_transforms, download=True)
        val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=val_transforms, download=True)
    elif dataset_name == "flowers102":
        train_dataset = torchvision.datasets.Flowers102(root=data_dir, split="train", transform=train_transforms, download=True)
        val_dataset = torchvision.datasets.Flowers102(root=data_dir, split="val", transform=val_transforms, download=True)
    elif dataset_name == "celeba":
        train_dataset = torchvision.datasets.CelebA(root=data_dir, split="train", transform=train_transforms, download=True)
        val_dataset = torchvision.datasets.CelebA(root=data_dir, split="test", transform=val_transforms, download=True)
    else:
        raise ValueError("Unknown dataset name")

    image_knn(model, train_dataset, val_dataset, input_size, device, n_neighbours, nn_params)


def main_knn(args):
    if args.model == "dino_vits16":
        # device = torch.device(args.device)
        # model = torch.hub.load("facebookresearch/dino:main", args.model).to(device)
        model = timm.create_model('vit_small_patch16_224.dino', img_size=args.input_size, pretrained=True)
    elif args.model == "timet":
        # device = torch.device(args.device)
        model = vit_small_patch16_224(img_size=224)  # First load with original size
        state_dict = torch.load("/ssdstore/ssalehi/paneco_models/TimeT.pth")
        msg = model.load_state_dict({".".join(k.split(".")[2:]): v for k, v in state_dict.items()}, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "dinov2":
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', img_size=224, pretrained=True)
        model.set_input_size(img_size=args.input_size)
    elif args.model == "MoSiC_dinov2":
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "merge":
        dinov2 = timm.create_model('vit_base_patch14_dinov2.lvd142m', img_size=224, pretrained=True)
        MoSiC_dinov2 = timm.create_model('vit_base_patch14_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2_epoch_8_val_score_0.6816405009172583.pth")
        msg = MoSiC_dinov2.load_state_dict(state_dict, strict=False)
        MoSiC_dinov2.set_input_size(img_size=args.input_size)
        model = merge_models(dinov2, MoSiC_dinov2, coeff=0)
        model.set_input_size(img_size=args.input_size)

    
    model = model.to(args.device)

    imagenet_100_main(model, "celeba", args.data_dir, args.batch_size, args.input_size, args.device, 30, None)


def main(args):
    print(f"the script arguments are {args}")

    if args.model == "dino_vits16":
        # device = torch.device(args.device)
        # model = torch.hub.load("facebookresearch/dino:main", args.model).to(device)
        model = timm.create_model('vit_small_patch16_224.dino', img_size=args.input_size, pretrained=True)
    elif args.model == "dinov2-s":
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model == "dinov2-b":
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model == "dinov2-l":
        model = timm.create_model('vit_large_patch14_dinov2.lvd142m', img_size=args.input_size, pretrained=True)
    elif args.model == "timet":
        # device = torch.device(args.device)
        model = vit_small_patch16_224(img_size=224)  # First load with original size
        state_dict = torch.load("/ssdstore/ssalehi/paneco_models/TimeT.pth")
        msg = model.load_state_dict({".".join(k.split(".")[2:]): v for k, v in state_dict.items()}, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "MoSiC_dino":
        model = timm.create_model('vit_small_patch16_224.dino', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dino.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "MoSiC_dinov2-s":
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2-s.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)

    elif args.model == "MoSiC_dinov2-b":
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2-b.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "MoSiC_dinov2-l":
        model = timm.create_model('vit_large_patch14_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2-l.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "MoSiC_dinov2r-s":
        model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2r-s.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "MoSiC_dinov2r-b":
        model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2r-b.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "MoSiC_clip-b":
        model = timm.create_model('eva02_base_patch16_clip_224', pretrained=True, dynamic_img_size=True)
        state_dict = torch.load("MoSiC_EVAclip-b.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        # model.set_input_size(img_size=args.input_size)
        print(msg)
    elif args.model == "clip-b":
        model = timm.create_model('eva02_base_patch16_clip_224', pretrained=True, dynamic_img_size=True)
        # state_dict = torch.load("MoSiC_EVA_clip-b.pth")
        # msg = model.load_state_dict(state_dict, strict=False)
        # model.set_input_size(img_size=args.input_size)
        # print(msg)


    ds_config = {
        "train_batch_size": args.batch_size,

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
        
    }

    model, optimizer, training_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        config=ds_config,
    )


    eval_spatial_resolution = args.input_size // args.patch_size
    train_transforms_dict = get_hbird_train_transforms(args.input_size)
    val_transforms_dict = get_hbird_val_transforms(args.input_size)

    train_transforms = CombTransforms(img_transform=train_transforms_dict['img'], tgt_transform=None, img_tgt_transform=train_transforms_dict['shared'])
    val_transforms = CombTransforms(img_transform=val_transforms_dict['img'], tgt_transform=None, img_tgt_transform=val_transforms_dict['shared'])
    
    dataset_size = 0
    num_classes = 0
    ignore_index = -1   
    if args.dataset == "voc":
        ignore_index = 255
        dataset = VOCDataModule_HB(batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_split=args.train_split,
                                    # train_split="trainaug",
                                    val_split="val",
                                    data_dir=args.data_dir,
                                    train_image_transform=train_transforms,
                                    val_transforms=val_transforms,
                                    shuffle=False,
                                    return_masks=True)
        dataset.setup()
    elif args.dataset == "ade20k":
        ignore_index = 0
        dataset = Ade20kDataModule(args.data_dir,
                 train_transforms=train_transforms,
                 val_transforms=val_transforms,
                 shuffle=False,
                 num_workers=args.num_workers,
                 batch_size=args.batch_size,
                 train_file_set=os.path.join(os.path.join(args.data_dir, f"sets/{args.train_split}.txt"),))
                # train_file_set=None)
        dataset.setup()
    else:
        raise ValueError("Unknown dataset name")

    dataset_size = dataset.get_train_dataset_size()
    num_classes = dataset.get_num_classes()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()



    model.eval()
    # model.to(device)

    evaluator = HbirdEvaluation(model, train_loader, n_neighbours=30, 
                        augmentation_epoch=2, num_classes=num_classes, 
                        device=model.device, eval_spatial_resolution=eval_spatial_resolution, d_model=args.embeddings_size, nn_params=None, memory_size=args.memory_size, 
                        dataset_size=dataset_size)
        
    hbird_miou = evaluator.evaluate(val_loader, eval_spatial_resolution, return_knn_details=False, ignore_index=ignore_index)



    # hbird_miou = hbird_evaluation(
    #     model.to(device),
    #     # Size of the embedding feature vectors of patches
    #     d_model=args.embeddings_size,
    #     patch_size=args.patch_size,
    #     batch_size = args.batch_size,
    #     input_size=args.input_size,
    #     # How many iterations of augmentations to use on top of the training dataset in order to generate the memory
    #     augmentation_epoch=2,
    #     device=device,
    #     # Whether to return additional NNs details
    #     return_knn_details=False,
    #     # The number of neighbors to fetch per image patch
    #     n_neighbours=30,
    #     # Other parameters to be used for the k-NN operator
    #     nn_params=None,
    #     # Function that extracts features from a vision encoder on images
    #     ftr_extr_fn=None,
    #     # The name of the dataset to use, currently only Pascal VOC is included.
    #     dataset_name="voc",
    #     # Path to the dataset to use for evaluation
    #     data_dir=args.data_dir,
    #     memory_size=args.memory_size
    # )

    print(f"Hummingbird Evaluation (mIoU): {hbird_miou}")


def main_overclustering(model_name="DINO"):
    import torchvision
    # Prepare val/test data
    input_size = 224
    batch_size = 32
    num_workers = 8
    device = "cuda:0"
    image_val_transform = trn.Compose([trn.Resize((input_size, input_size)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])
    val_transforms = {"img": image_val_transform, "target": None , "shared": shared_val_transform}
    dataset = PascalVOCDataModule(batch_size=batch_size, train_transform=val_transforms, val_transform=val_transforms, test_transform=val_transforms, num_workers=num_workers)
    dataset.setup()
    test_dataloader = dataset.get_test_dataloader()

    if model_name == "DINO":
        model = timm.create_model('vit_small_patch16_224.dino', img_size=224, pretrained=True)
        model.set_input_size(img_size=224)
        model.to(device)
        feature_spatial_resolution = 14
    elif model_name == "DINOv2":
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=224, pretrained=True)
        model.set_input_size(img_size=224)
        model.to(device)
        feature_spatial_resolution = 16
    elif model_name == "MoSiC(DINO)":
        model = timm.create_model('vit_small_patch16_224.dino', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dino.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=224)
        print(msg)
        model.to(device)
        feature_spatial_resolution = 14
    elif model_name == "MoSiC(DINOv2)":
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2-s.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        model.set_input_size(img_size=224)
        print(msg)
        model.to(device)
        feature_spatial_resolution = 16
    overclustering(model, model_name, feature_spatial_resolution, test_dataloader, 224, 21, 255, device)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    # Distributed / DeepSpeed argument (this is usually injected by deepspeed launcher)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank. Necessary for using the DeepSpeed launcher.')
    
    # Standard arguments
    parser.add_argument("--seed", default=42, type=int,
                        help="The seed for the random number generators")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224,
                        help="Size of the images fed to the model")
    parser.add_argument("--patch-size", type=int, default=14,
                        help="Size of the model patches")
    parser.add_argument("--memory-size", type=int,
                        help="The size of the memory bank. Unbounded if not specified", default=10240000)
    parser.add_argument("--model", type=str,
                        help="DINO model name", default="merge")
    parser.add_argument("--embeddings-size", type=int,
                        help="The size of the model embeddings", default=384)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run the model on")
    
    parser.add_argument("--train-split", type=str, default="trainaug_8_783",
                        help="The training split to use")
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="/nvmestore/ssalehi",
                        help="Path to the VOC dataset")
    parser.add_argument("--dataset", type=str, default="voc",
                        help="Dataset to use for evaluation")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers for the dataloader")

    args = parser.parse_args()


    seed_everything(args.seed)
    # main(args)
    get_cluster_map("/ssdstore/ssalehi/dataset/davis_2021/davis_data/JPEGImages/480p/car-turn/", "MoSiC(DINO)", "cuda:0", num_classes=6)
    # main_knn(args)
    # main_overclustering("MoSiC(DINOv2)")