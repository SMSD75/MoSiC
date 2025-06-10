import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from torch.nn import init
# from mmcv.cnn import ConvModule
from timm.models.vision_transformer import VisionTransformer
from timm.models._manipulate import checkpoint_seq
from timm.layers import PatchDropout   
from types import MethodType
from peft import LoraConfig, get_peft_model, TaskType
from cotracker.predictor import CoTrackerPredictor
import os
from utils.my_utils import track_grid_points_batch
## FCN Head for evaluation


class LearnableUpsampler(nn.Module):
    """
    A learnable upsampler that replaces bilinear interpolation.
    Uses transposed convolutions with learnable parameters.
    """
    def __init__(self, dim):
            super(LearnableUpsampler, self).__init__()
            self.up1 = nn.ConvTranspose2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=7,
                stride=7,
                padding=0
            )
            # 112 -> 224
            self.up2 = nn.ConvTranspose2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=2,
                stride=2,
                padding=0
            )

    def forward(self, x):
        # x: (batch_size, dim, 16, 16)
        x = self.up1(x)  # (batch_size, dim, 112, 112)
        x = self.up2(x)  # (batch_size, dim, 224, 224)
        return x


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.
    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(BaseDecodeHead, self).__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize weights of classification layer."""
        nn.init.normal(self.conv_seg, mean=0, std=0.01)

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = inputs
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output


class FixedMaskPatchDropout(PatchDropout):
    def __init__(self, num_prefix_tokens=0):
        super().__init__(num_prefix_tokens=num_prefix_tokens)
        self.keep_indices = None
    

    def set_mask(self, keep_indices):
        self.keep_indices = keep_indices

    def forward(self, x):
        if not self.training or self.keep_indices is None:
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        x = x.gather(1, self.keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        return x

class FeatureExtractor(torch.nn.Module):
    def __init__(self, vit_model, eval_spatial_resolution=14, d_model=768, model_type="dino", mask_ratio=0, use_lora=False, lora_config=None):
        super().__init__()
        self.model = vit_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
        self.model_type = model_type
        self.mask_ratio = mask_ratio
        self.use_lora = use_lora

        if self.mask_ratio > 0:
            self.model.patch_drop = FixedMaskPatchDropout(num_prefix_tokens=self.model.num_prefix_tokens)
            # self.model.patch_drop = FixedMaskPatchDropout(num_prefix_tokens=self.model.num_register_tokens)
        
        # Apply LoRA if specified
        if use_lora and lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def freeze_feature_extractor(self, unfreeze_layers=[]):
        if not self.use_lora:
            # Original freezing logic for non-LoRA approach
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                for unfreeze_layer in unfreeze_layers:
                    if unfreeze_layer in name:
                        param.requires_grad = True
                        break

    def get_intermediate_layer_feats(self, imgs, feat="k", layer_num=-1):
        bs, c, h, w = imgs.shape
        imgs = imgs.reshape(bs, c, h, w)
        ## hook to get the intermediate layers
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        self.model._modules["blocks"][layer_num]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        self.model(imgs)
        attentions = self.model.get_last_selfattention(imgs)
        # Scaling factor
        average_cls_attention = torch.mean(attentions[:, :, 0, 1:], dim=1)
        temp_mins, temp_maxs = average_cls_attention.min(dim=1)[0], average_cls_attention.max(dim=1)[0]
        normalized_cls_attention = (average_cls_attention - temp_mins[:, None]) / (temp_maxs[:, None] - temp_mins[:, None])
        # cls_attentions = process_attentions(attentions[:, :, 0, 1:], self.spatial_resolution)  
        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens
        qkv = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        if feat == "k":
            feats = k[:, 1:, :]
        elif feat == "q":
            feats = q[:, 1:, :]
        elif feat == "v":
            feats = v[:, 1:, :]
        return feats, normalized_cls_attention
    
    def forward_features(self, imgs, mask=None):
        if mask is not None:
            self.model.patch_drop.set_mask(mask)
        if self.model_type == "dino-s":
            # try:
            #     ## for the backbones that does not support the function
            #     features = self.model.get_intermediate_layers(imgs)[0]
            #     features = features[:, 1:]
            #     normalized_cls_attention = self.model.get_last_selfattention(imgs)
            # except:
            features = self.model.forward_features(imgs)[:, self.model.num_prefix_tokens:]
            normalized_cls_attention = None
        elif "dinov2r" in self.model_type:
            features = self.model.forward_features(imgs)[:, self.model.num_prefix_tokens:]
            normalized_cls_attention = None
        elif "dinov2" in self.model_type or self.model_type == "NeCo-s":
            features = self.model.forward_features(imgs)[:, self.model.num_prefix_tokens:]
            normalized_cls_attention = None
        elif self.model_type == "leopart-s" or self.model_type == "TimeT-s":
            features = self.model.forward_features(imgs)
            features = features[:, 1:]
            normalized_cls_attention = None
        elif "clip" in self.model_type:
            features = self.model.forward_features(imgs)
            features = features[:, self.model.num_prefix_tokens:]
            normalized_cls_attention = None
        return features, normalized_cls_attention




class FeatureExtractorSimple(torch.nn.Module):
    def __init__(self, vit_model, ftr_extr_fn, eval_spatial_resolution=14, d_model=768):
        super().__init__()
        self.model = vit_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
        self.ftr_extr_fn = ftr_extr_fn
    
    def forward_features(self, imgs):
        return self.ftr_extr_fn(self.model, imgs)



class FeatureExtractorBeta(torch.nn.Module):
    def __init__(self, vit_model, eval_spatial_resolution=14, d_model=768):
        super().__init__()
        self.model = vit_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
        self.define_forward_features()

    def define_forward_features(self):
        """
        This function defines the forward_features function for the model.
        """
        funcType = MethodType
        device = self.device

        def ff1(self, imgs, feat="k"):
            features, normalized_cls_attention = self.get_intermediate_layer_feats(imgs, feat=feat, layer_num=-1)
            return features, normalized_cls_attention

        def ff2(self, imgs, feat="k"):
            ## for the backbones that does not support the function
            features = self.model.get_intermediate_layers(imgs)[0]
            features = features[:, 1:]
            # Normalizing the attention map
            attentions = self.model.get_last_selfattention(imgs)
            average_cls_attention = torch.mean(attentions[:, :, 0, 1:], dim=1)
            temp_mins, temp_maxs = average_cls_attention.min(dim=1)[0], average_cls_attention.max(dim=1)[0]
            normalized_cls_attention = (average_cls_attention - temp_mins[:, None]) / (temp_maxs[:, None] - temp_mins[:, None])
            return features, normalized_cls_attention

        def ff3(self, imgs, feat="k"):
            # In case of dinov2 or simimilar models
            features_dict = self.model.forward_features(imgs)
            features = features_dict['x_norm_patchtokens']
            normalized_cls_attention = None
            return features, normalized_cls_attention

        def ff4(self, imgs, feat="k"):
            # Other models
            features = self.model.forward_features(imgs)
            features = features[:, 1:]
            normalized_cls_attention = None
            return features, normalized_cls_attention

        with torch.no_grad():
            (BS, C, H,W) = 32, 3, 224, 224
            ps_imgs = torch.rand((BS, C, H, W)).to(device)

            try:
                ff1(self, ps_imgs)
                # setattr(self, 'forward_features', ff1)
                self.forward_features = funcType(ff1, self)
                return
            except Exception as e:
                pass
            try:
                ff2(self, ps_imgs)
                # setattr(self, 'forward_features', ff2)
                self.forward_features = funcType(ff2, self)
                return
            except:
                pass
            try:
                self.model = self.model.cuda()
                fts, att = ff3(self, ps_imgs.cuda())
                self.forward_features = funcType(ff3, self)
                return
            except Exception as e:
                pass

            # setattr(self, 'forward_features', ff4)
            self.forward_features = funcType(ff4, self)
    
    def freeze_feature_extractor(self, unfreeze_layers=[]):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for unfreeze_layer in unfreeze_layers:
                if unfreeze_layer in name:
                    param.requires_grad = True
                    break

    def get_intermediate_layer_feats(self, imgs, feat="k", layer_num=-1):
        bs, c, h, w = imgs.shape
        imgs = imgs.reshape(bs, c, h, w)
        ## hook to get the intermediate layers
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        self.model._modules["blocks"][layer_num]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        # I think this is not necessary
        # self.model(imgs)
        attentions = self.model.get_last_selfattention(imgs)
        # Scaling factor
        average_cls_attention = torch.mean(attentions[:, :, 0, 1:], dim=1)
        temp_mins, temp_maxs = average_cls_attention.min(dim=1)[0], average_cls_attention.max(dim=1)[0]
        normalized_cls_attention = (average_cls_attention - temp_mins[:, None]) / (temp_maxs[:, None] - temp_mins[:, None])
        # cls_attentions = process_attentions(attentions[:, :, 0, 1:], self.spatial_resolution)  
        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens
        qkv = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        if feat == "k":
            feats = k[:, 1:, :]
        elif feat == "q":
            feats = q[:, 1:, :]
        elif feat == "v":
            feats = v[:, 1:, :]
        return feats, normalized_cls_attention
    
    @property
    def device(self):
        """
        Returns the device where the model is stored.
        """
        return next(self.model.parameters()).device



class FeatureForwarder():
    def __init__(self, spatial_resolution, context_frames, context_window, topk, feature_head=None):
        super().__init__()
        self.spatial_resolution = spatial_resolution
        self.feature_head = feature_head
        self.context_frames = context_frames
        self.context_window = context_window
        self.topk = topk  
        self.mask_neighborhood = self.restrict_neighborhood(self.spatial_resolution, self.spatial_resolution, self.context_window)

    
    def restrict_neighborhood(self, h, w, size_mask_neighborhood):
        # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
        mask = torch.zeros(h, w, h, w)
        for i in range(h):
            for j in range(w):
                for p in range(2 * size_mask_neighborhood + 1):
                    for q in range(2 * size_mask_neighborhood + 1):
                        if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                            continue
                        if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                            continue
                        mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

        mask = mask.reshape(h * w, h * w)
        return mask

    def norm_mask(self, mask):
        c, h, w = mask.size()
        for cnt in range(c):
            mask_cnt = mask[cnt,:,:]
            if(mask_cnt.max() > 0):
                mask_cnt = (mask_cnt - mask_cnt.min())
                mask_cnt = mask_cnt/mask_cnt.max()
                mask[cnt,:,:] = mask_cnt
        return mask
    

    def forward(self, feature_list, first_segmentation_map):
        with torch.no_grad():
            down_sampled_first_seg = nn.functional.interpolate(first_segmentation_map.type(torch.DoubleTensor).unsqueeze(0), size=(self.spatial_resolution, self.spatial_resolution), mode="nearest")
            first_seg = down_sampled_first_seg
            # The queue stores the n preceeding frames
            que = queue.Queue(self.context_frames)
            frame1_feat = feature_list[0]
            frame1_feat = frame1_feat.squeeze()
            frame1_feat = frame1_feat.T
            
            segmentation_list = []
            for cnt in tqdm(range(1, feature_list.size(0))):
                feature_tar = feature_list[cnt]

                # we use the first segmentation and the n previous ones
                used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
                used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

                frame_tar_avg, feat_tar = self.label_propagation(feature_tar, used_frame_feats, used_segs)

                # pop out oldest frame if neccessary
                if que.qsize() == self.context_frames:
                    que.get()
                # push current results into queue
                # seg = copy.deepcopy(frame_tar_avg.detach())
                seg = frame_tar_avg
                que.put([feat_tar, seg])
                # segmentation_list.append(self.norm_mask(frame_tar_avg.squeeze(0)))
                segmentation_list.append(frame_tar_avg.squeeze(0))
            return segmentation_list  



    def label_propagation(self, feature_tar, list_frame_feats, list_segs):
        """
        propagate segs of frames in list_frames to frame_tar
        """
        ## we only need to extract feature of the target frame

        h = w = self.spatial_resolution
        return_feat_tar = feature_tar.squeeze().T
        feat_tar = feature_tar
        ncontext = len(list_frame_feats)
        feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

        feat_tar = F.normalize(feat_tar, dim=1, p=2)
        feat_sources = F.normalize(feat_sources, dim=1, p=2)

        feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
        aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)
        aff = aff.to(feat_tar.device)
        if self.context_window > 0:
            if self.mask_neighborhood is None:
                self.mask_neighborhood = self.restrict_neighborhood(h, w, self.context_window)
                self.mask_neighborhood = self.mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
                self.mask_neighborhood = self.mask_neighborhood.to(feat_tar.device)
            self.mask_neighborhood = self.mask_neighborhood.to(aff.device)
            aff =  aff * self.mask_neighborhood

        aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
        tk_val, _ = torch.topk(aff, dim=0, k=self.topk)
        tk_val_min, _ = torch.min(tk_val, dim=0)
        aff[aff < tk_val_min] = 0

        aff = aff / torch.sum(aff, keepdim=True, axis=0)
        # aff = aff.softmax(dim=0)

        list_segs = [s.to(feat_tar.device) for s in list_segs]
        segs = torch.cat(list_segs)
        nmb_context, C, h, w = segs.shape
        segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
        seg_tar = torch.mm(segs.double(), aff.double())
        seg_tar = seg_tar.reshape(1, C, h, w)
        return seg_tar, return_feat_tar


class RAFTFF(torch.nn.Module):
    def __init__(self, spatial_resolution, context_frames, context_window, topk, grid_size = 10, feature_head=None):
        super().__init__()
        self.grid_size = grid_size
    
    def forward(self, video):
        ## video is a tensor of shape (bs, nf, c, h, w)
        pred_tracks, pred_visibility = track_grid_points_batch(video, self.grid_size) # B T N 2
        return pred_tracks, pred_visibility
    

class CoTrackerFF(torch.nn.Module):
    def __init__(self, spatial_resolution, context_frames, context_window, topk, grid_size = 10, feature_head=None):
        super().__init__()
        self.co_tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        # self.co_tracker = CoTrackerPredictor(
        #     checkpoint=os.path.join(
        #         'scaled_offline.pth'
        #     )
        # )
        self.grid_size = grid_size
    

    def forward(self, video):
        ## video is a tensor of shape (bs, nf, c, h, w)
        pred_tracks, pred_visibility = self.co_tracker(video, grid_size=self.grid_size) # B T N 2,  B T N 1
        return pred_tracks, pred_visibility

class CrossAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, kdim=input_dim, vdim=input_dim, num_heads=num_heads, batch_first=True)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.q_norm = nn.LayerNorm(input_dim)
        self.k_norm = nn.LayerNorm(input_dim)
        self.v_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # Attention part
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        output, attention_weights = self.cross_attention(q, k, v)
        x = q + self.dropout(output)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class CorrespondenceDetection():
    def __init__(self, window_szie, spatial_resolution=14, output_resolution=96) -> None:
        self.window_size = window_szie
        self.neihbourhood = self.restrict_neighborhood(spatial_resolution, spatial_resolution, self.window_size)
        self.output_resolution = output_resolution

    
    def restrict_neighborhood(self, h, w, size_mask_neighborhood):
        # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
        mask = torch.zeros(h, w, h, w)
        for i in range(h):
            for j in range(w):
                for p in range(2 * size_mask_neighborhood + 1):
                    for q in range(2 * size_mask_neighborhood + 1):
                        if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                            continue
                        if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                            continue
                        mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

        # mask = mask.reshape(h * w, h * w)
        return mask

    def __call__(self, features1, features2, crops):
        with torch.no_grad():
            bs, spatial_resolution, spatial_resolution, d_model = features1.shape
            _, h, w = crops.shape
            patch_size = h // spatial_resolution
            crops = crops.reshape(bs, h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 1, 3, 2, 4)
            crops = crops.flatten(3, 4)
            cropped_feature_mask = crops.sum(-1) > 0 ## size (bs, spatial_resolution, spatial_resolution)
            ## find the idx of the croped features_mask
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)
            similarities = torch.einsum('bxyd,bkzd->bxykz', features1, features2)
            most_similar_features_mask = torch.zeros(bs, spatial_resolution, spatial_resolution)
            revised_crop = torch.zeros(bs, spatial_resolution, spatial_resolution)
            self.neihbourhood = self.neihbourhood.to(features1.device)
            similarities = similarities * self.neihbourhood.unsqueeze(0)
            similarities = similarities.flatten(3, 4)
            most_similar_cropped_patches_list = []
            # for i, crp_feature_mask in enumerate(cropped_feature_mask): 
            crp_feature_mask = cropped_feature_mask[0]
            true_coords  = torch.argwhere(crp_feature_mask)
            min_coords = true_coords.min(0).values
            max_coords = true_coords.max(0).values
            rectangle_shape = max_coords - min_coords + 1
            crop_h, crop_w = rectangle_shape
            most_similar_patches = similarities.argmax(-1)
            most_similar_cropped_patches = most_similar_patches[cropped_feature_mask]
            most_similar_cropped_patches = most_similar_cropped_patches.reshape(bs, crop_h, crop_w)
            # most_similar_cropped_patches = F.interpolate(most_similar_cropped_patches.float().unsqueeze(0).unsqueeze(0), size=(self.output_resolution, self.output_resolution), mode='nearest').squeeze(0).squeeze(0)
            # most_similar_cropped_patches_list.append(most_similar_cropped_patches)

            # for i, similarity in enumerate(similarities):
            #     croped_feature_idx = croped_feature_mask[i].nonzero()
            #     for j, mask_idx in enumerate(croped_feature_idx):
            #         # print(mask_idx)
            #         revised_crop[i, mask_idx[0], mask_idx[1]] = 1
            #         min_x, max_x = max(0, mask_idx[0] - self.window_size), min(spatial_resolution, mask_idx[0] + self.window_size)
            #         min_y, max_y = max(0, mask_idx[1] - self.window_size), min(spatial_resolution, mask_idx[1] + self.window_size)
            #         neiborhood_similarity = similarity[mask_idx[0], mask_idx[1], min_x:max_x, min_y:max_y]
            #         max_value = neiborhood_similarity.max()
            #         indices = (neiborhood_similarity == max_value).nonzero()[0]
            #         label_patch_number = (indices[0] + min_x) * spatial_resolution + (indices[1] + min_y)
            #         most_similar_features_mask[i, mask_idx[0], mask_idx[1]] = label_patch_number
            
            # most_similar_cropped_patches = torch.stack(most_similar_cropped_patches_list)
            revised_crop = cropped_feature_mask.float() 
            return most_similar_cropped_patches, revised_crop
    

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, use_gru=False, use_kv=False, eps = 1e-8, hidden_dim = 128, output_dim = 256):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.use_gru = use_gru
        self.use_kv = use_kv
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            torch.nn.Linear(dim, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, output_dim),
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)      
        if self.use_kv:  
            k, v = self.to_k(inputs), self.to_v(inputs)
        else:
            k = inputs
            v = inputs
        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = slots

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)
            if self.use_gru:
                slots = self.gru(
                    updates.reshape(-1, d),
                    slots_prev.reshape(-1, d)
                )
                slots = updates.reshape(-1, d) + slots
            else:
                slots = updates
            slots = slots.reshape(b, -1, d)
        slots = self.mlp(self.norm_pre_ff(slots))

        return updates, slots

class DistributedDataParallelModel(nn.Module):
    def __init__(self, model, gpu):
        super(DistributedDataParallelModel, self).__init__()
        self.model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    def forward(self, *input):
        return self.model(*input)
    def get_non_ddp_model(self):
        return self.model.module
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)
         
if __name__ == "__main__":

    img = torch.randn(1, 3, 224, 224)
    dino_vit_s16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino_vit_s16.eval()
    feature_extractor = FeatureExtractor(dino_vit_s16)
    feats, attentions = feature_extractor.get_intermediate_layer_feats(img, feat="k", layer_num=-1)
    print(f"Feats shape : {feats.shape}")
    print(f"Attentions shape : {attentions.shape}")