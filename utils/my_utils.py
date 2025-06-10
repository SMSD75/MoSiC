import imageio
import timm
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from torchvision.transforms import GaussianBlur
from typing import List
from IPython.display import display, display_markdown
import io
import os, sys
import requests
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import glob
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from typing import List
from torchvision.utils import draw_segmentation_masks
import cv2
from PIL import Image
import matplotlib
import numpy as np
import wandb
import torch.distributed as dist
import torchvision

from utils.clustering import PerDatasetClustering




from typing import Optional, List, Tuple

def process_and_save_gif(
    normalized_video: torch.Tensor,
    grid_size: int = 8,
    lk_params: Optional[dict] = None,
    draw_trails: bool = False,
    trail_length: int = 5,
    circle_radius: int = 3,
    circle_color: Tuple[int,int,int] = (0,255,0),
    trail_color: Tuple[int,int,int] = (0,128,255),
    gif_path_template: str = "tracked_{batch}.gif",
    fps: int = 20,
) -> List[str]:
    """
    Full pipeline:
      1. Denormalize
      2. Track a grid via Lucas–Kanade
      3. Draw points (and optional trails) on frames
      4. Save animated GIF(s)

    Args:
        normalized_video    (B, T, C, H, W): input in [0,1]
        grid_size           int: grid resolution per axis
        lk_params           dict|None: overrides for cv2.calcOpticalFlowPyrLK
        draw_trails         bool: draw past positions
        trail_length        int: how many frames back to draw trails
        circle_radius       int: radius of current-point marker
        circle_color        BGR tuple: marker color
        trail_color         BGR tuple: trail color
        gif_path_template   str: "tracked_{batch}.gif" → tracked_0.gif, etc.
        fps                 int: GIF playback speed
        
    Returns:
        List of file paths for the created GIFs (one per batch element).
    """
    # 1) Denormalize
    video = denormalize_video_cotracker_compatible(normalized_video)
    B, T, C, H, W = video.shape
    device = video.device

    # 2) Track grid points (inlined)
    # — build initial grid
    ys = torch.linspace(0, H-1, grid_size, device=device)
    xs = torch.linspace(0, W-1, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    init_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (N,2): (x,y)
    N = init_pts.shape[0]

    # LK defaults
    if lk_params is None:
        lk_params = dict(
            winSize  = (21,21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    # prepare outputs
    all_tracks = torch.zeros((B, T, N, 2), dtype=torch.float32, device=device)
    all_vis    = torch.zeros((B, T, N), dtype=torch.uint8, device=device)

    for b in range(B):
        # initial positions
        all_tracks[b,0,:,0] = init_pts[:,1]  # y
        all_tracks[b,0,:,1] = init_pts[:,0]  # x
        all_vis[b,0,:] = 1

        # prepare numpy frames
        clip_np = video[b].permute(0,2,3,1).cpu().numpy()
        if clip_np.dtype != np.uint8:
            clip_np = (clip_np).clip(0,255).astype(np.uint8)
        def to_gray(f): return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if C==3 else f[...,0]

        prev_gray = to_gray(clip_np[0])
        pts_np     = init_pts.cpu().numpy().reshape(-1,1,2).astype(np.float32)

        for t in range(1, T):
            curr_gray = to_gray(clip_np[t])
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts_np, None, **lk_params
            )
            next_pts = next_pts.reshape(-1,2)
            status   = status.reshape(-1)

            # keep lost points at last known pos
            prev_pts = pts_np.reshape(-1,2)
            lost = (status==0)
            next_pts[lost] = prev_pts[lost]

            # clamp
            next_pts[:,0] = np.clip(next_pts[:,0], 0, W-1)  # x
            next_pts[:,1] = np.clip(next_pts[:,1], 0, H-1)  # y

            # store
            all_tracks[b,t,:,0] = torch.from_numpy(next_pts[:,1]).to(device)
            all_tracks[b,t,:,1] = torch.from_numpy(next_pts[:,0]).to(device)
            all_vis[b,t]        = torch.from_numpy(status.astype(np.uint8)).to(device)

            pts_np     = next_pts.reshape(-1,1,2).astype(np.float32)
            prev_gray  = curr_gray

    # 3) Visualize + 4) Save GIF
    gif_paths = []
    for b in range(B):
        # convert to list of BGR frames
        frames = []
        clip_np = video[b].permute(0,2,3,1).cpu().numpy()
        if clip_np.dtype != np.uint8:
            clip_np = (clip_np).clip(0,255).astype(np.uint8)
        if C==3: clip_np = clip_np[..., ::-1]  # RGB→BGR
        frames = [f.copy() for f in clip_np]

        # optional trail buffer
        if draw_trails:
            trail_buf = all_tracks[b].clone()

        for t in range(T):
            img = frames[t]
            # draw trails
            if draw_trails and t>0:
                start = max(0, t - trail_length)
                for tt in range(start, t):
                    p0 = trail_buf[tt].long().cpu().numpy()
                    p1 = trail_buf[tt+1].long().cpu().numpy()
                    for (y0,x0),(y1,x1) in zip(p0,p1):
                        cv2.line(img, (int(x0),int(y0)), (int(x1),int(y1)), trail_color, 1)
            # draw current points
            pts = all_tracks[b,t].long().cpu().numpy()
            vis = all_vis[b,t].cpu().numpy()
            for (y,x),v in zip(pts, vis):
                if v:
                    cv2.circle(img, (int(x),int(y)), circle_radius, circle_color, -1)

        # save GIF (convert BGR→RGB)
        rgb_frames = [f[..., ::-1] for f in frames]
        path = gif_path_template.format(batch=b)
        imageio.mimsave(path, rgb_frames, duration=1/fps)
        gif_paths.append(path)

    return gif_paths


def show_trainable_paramters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)




def overlay_overclustering_maps(data, cluster_maps, save_dir=None, model="DINO", prefix="cluster", nrow=8, colors=None):
    """
    Overlays clustering maps on the input data with consistent colors and saves them as a grid.
    
    Args:
        data: [B, c, h, w] tensor of normalized images (with ImageNet mean and std)
        cluster_maps: [B, 1, h, w] tensor of cluster assignments
        save_dir: directory to save the grid visualizations
        prefix: prefix for saved image filenames
        nrow: number of images per row in the grid
        colors: optional list of colors to use. If None, uses a predefined set
    
    Returns:
        Tensor of overlaid images [B, c, h, w]
    """
    if colors is None:
        colors = [
            "black", "blue", "red", "yellow", "green", "purple", "cyan", 
            "orange", "lime", "pink", "teal", "skyblue", "navy", "white",
            "maroon", "coral", "gold", "indigo", "silver", "turquoise"
        ]
        while len(colors) < 500:
            colors.extend(colors)
    
    # Convert to CPU for processing
    maps = cluster_maps.cpu()
    images = data.cpu()
    
    # Denormalize images
    IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMGNET_STD = torch.tensor([0.228, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * IMGNET_STD + IMGNET_MEAN
    
    # Get unique cluster IDs
    cluster_ids = torch.unique(maps)
    
    # Create boolean masks for each cluster
    masks = []
    for cluster_map in maps:
        mask = torch.zeros((len(cluster_ids), cluster_map.shape[1], cluster_map.shape[2]))
        mask = mask.type(torch.bool)
        for i, cluster_id in enumerate(cluster_ids):
            boolean_mask = (cluster_map[0] == cluster_id)
            mask[i] = boolean_mask
        masks.append(mask)
    
    masks = torch.stack(masks)
    
    # Create overlaid images
    overlaid = [
        draw_segmentation_masks(
            (img * 255).to(torch.uint8), 
            masks=mask,
            alpha=0.5,
            colors=colors[:len(cluster_ids)]
        )
        for img, mask in zip(images, masks)
    ]
    
    overlaid = torch.stack(overlaid)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert cluster maps to RGB visualization
        cluster_viz = []
        for cluster_map in maps:
            # Normalize cluster IDs to [0, 1]
            normalized_map = (cluster_map[0] - cluster_map[0].min()) / (cluster_map[0].max() - cluster_map[0].min() + 1e-8)
            # Convert to colormap (using plasma colormap)
            cm = plt.get_cmap('plasma')
            colored_map = cm(normalized_map.numpy())
            # Convert to torch tensor
            colored_map = torch.from_numpy(colored_map[..., :3]).permute(2, 0, 1)
            cluster_viz.append(colored_map)
        cluster_viz = torch.stack(cluster_viz)

        # Create grids
        original_grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
        cluster_grid = torchvision.utils.make_grid(cluster_viz, nrow=nrow, padding=2)
        overlaid_grid = torchvision.utils.make_grid(overlaid.float() / 255.0, nrow=nrow, padding=2)
        
        # Create a combined figure
        fig, axes = plt.subplots(3, 1, figsize=(20, 20))
        
        # Plot original images
        axes[0].imshow(original_grid.permute(1, 2, 0).numpy())
        # axes[0].set_title(f"Model: {model}")
        axes[0].axis('off')
        
        # Plot cluster maps
        axes[1].imshow(cluster_grid.permute(1, 2, 0).numpy())
        # axes[1].set_title(f"Cluster Maps (Number of clusters: {len(cluster_ids)})")
        axes[1].axis('off')
        
        # Plot overlaid results
        axes[2].imshow(overlaid_grid.permute(1, 2, 0).numpy())
        # axes[2].set_title("Overlaid Results")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_grid.png"), 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()
        
        # Save individual grids
        torchvision.utils.save_image(original_grid, os.path.join(save_dir, f"{prefix}_original_grid.png"))
        torchvision.utils.save_image(cluster_grid, os.path.join(save_dir, f"{prefix}_cluster_grid.png"))
        torchvision.utils.save_image(overlaid_grid, os.path.join(save_dir, f"{prefix}_overlaid_grid.png"))
    
    return overlaid




def gather_in_chunks(data_list, chunk_size=10000):
    chunks = torch.cat(data_list).split(chunk_size)
    gathered_chunks = []
    for chunk in chunks:
        gathered = all_gather_concat(chunk)
        gathered_chunks.append(gathered.detach().cpu())
        del gathered
        torch.cuda.empty_cache()
    return gathered_chunks


def all_gather_concat(tensor, group=None):
    """
    Gathers `tensor` from all ranks onto every rank and returns the concatenated result.
    
    Args:
        tensor (torch.Tensor): Local tensor to gather. Must be same shape on all ranks.
        group (optional): The process group to work on. Defaults to the global default group.
    
    Returns:
        Torch tensor of shape [world_size * local_batch, ...] on every rank.
    """
    if group is None:
        group = dist.group.WORLD
        
    world_size = dist.get_world_size(group=group)

    # Make a list of placeholders (same shape as 'tensor') for each rank
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor, group=group)

    return torch.cat(gather_list, dim=0)  # concatenates on the first dimension


def process_attentions(attn_batch, spatial_res, threshold = 0.5, blur_sigma = 0.6):
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    # attns_processed = torch.cat(attns_group, dim = 0)
    attns_processed = sum(attn_batch[:, i] * 1 / attn_batch.size(1) for i in range(attn_batch.size(1)))
    attentions = attns_processed.reshape(-1, 1, spatial_res, spatial_res)
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask] = 0
    return th_attn.detach()



def preprocess(imgs):
    img_group = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = T.ToPILImage()(img.cpu())
        target_image_size = 224
        s = min(img.size)
        
        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')
            
        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        img_group.append(map_pixels(img))
    return torch.cat(img_group, dim = 0)



def cosine_scheduler(base_value: float, final_value: float, max_iters: int):
    # Construct cosine schedule starting at base_value and ending at final_value with epochs * niter_per_ep values.
    iters = np.arange(max_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule


def denormalize_video(video):
    """
    video: [1, nf, c, h, w]
    """
    IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    IMGNET_STD = torch.tensor([0.228, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    denormalized_video = video.cpu().detach() * IMGNET_STD + IMGNET_MEAN
    # denormalized_video = (denormalized_video * 255).type(torch.uint8)
    denormalized_video = denormalized_video.squeeze(0)
    return denormalized_video


def denormalize_video_cotracker_compatible(video):
    """
    video: [1, nf, c, h, w]
    """
    IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=video.device).view(1, 1, 3, 1, 1)
    IMGNET_STD = torch.tensor([0.228, 0.224, 0.225], device=video.device).view(1, 1, 3, 1, 1)
    denormalized_video = video * IMGNET_STD + IMGNET_MEAN
    denormalized_video = (denormalized_video * 255).float()
    return denormalized_video


def overlay_video_cmap(cluster_maps, denormalized_video):
    """
    cluster_maps: [nf, h, w]
    denormalized_video: [nf, c, h, w]
    """
    ## generate 12 distinguishable colors
    colors = ["orange", "blue", "red", "yellow", "white", "green", "brown", "purple", "gold", "black", "pink", "cyan", "magenta", "lime", "teal", "maroon", "navy", "olive", "gray", "silver", "indigo"]
        ## convert cluster_maps to [num_maps, h, w]
    masks = []
    cluster_ids = torch.unique(cluster_maps)
    for cluster_map in cluster_maps:
        mask = torch.zeros((cluster_ids.shape[0], cluster_map.shape[0], cluster_map.shape[1])) 
        mask = mask.type(torch.bool)
        for i, cluster_id in enumerate(cluster_ids):
                ## make a boolean mask for each cluster
                ## make a boolean mask for each cluster if cluster_map == cluster_id
            boolean_mask = (cluster_map == cluster_id)
            mask[i, :, :] = boolean_mask
        masks.append(mask)
    cluster_maps = torch.stack(masks)
            
    overlayed = [
                draw_segmentation_masks(img, masks=mask, alpha=0.5, colors=colors)
                for img, mask in zip(denormalized_video, cluster_maps)
            ]
    overlayed = torch.stack(overlayed)
    return cluster_maps,overlayed




def make_seg_maps(data, cluster_map, logging_directory, name, w_featmap=28, h_featmap=28):
    bs, fs, c, h, w = data.shape
    # cluster_map = torch.Tensor(cluster_map.reshape(bs, fs, w_featmap, h_featmap))
    # cluster_map = nn.functional.interpolate(cluster_map.type(torch.DoubleTensor), scale_factor=8, mode="nearest").detach().cpu()
    cluster_map = cluster_map
    for i, datum in enumerate(data):
        frame_buffer = []
        for j, frame in enumerate(datum):
            frame_buffer.append(localize_objects(frame.permute(1, 2, 0).detach().cpu(), cluster_map[i, j]))
        convert_list_to_video(frame_buffer, name + "_" + str(i), speed=1000/ datum.size(0), directory=logging_directory, wdb_log=False)
    

def visualize_sampled_videos(samples, path, name):
    # os.system(f'rm -r {path}')
    scale_255 = lambda x : (x * 255).astype('uint8')
    layer, height, width = samples[0].shape[-3:]
    if not os.path.isdir(path):
        os.mkdir(path)
    video = cv2.VideoWriter(path + name, 0, 1, (width,height))
    if len(samples.shape) == 4: ## sampling a batch of images and not clips
        frames = samples
    else: ## clip-wise sampling
        frames = samples[0][0]  

    for frame in frames:
        if len(frame.shape) == 3:
            frame_1 = frame.permute(1, 2, 0).numpy()
        else:
            frame_1 = frame[..., None].repeat(1, 1, 3).numpy()
        temp = scale_255(frame_1)
        # temp = frame_1
        video.write(temp)
    video.release()
    cv2.destroyAllWindows()


## read 00018.jpg and return the cluster map
def get_cluster_map(input_img_path, model_name, device, num_classes):

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
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=224, pretrained=False)
        state_dict = torch.load("MoSiC_dinov2-s.pth")
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
    input_imgs = []
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
    ])
    if os.path.isfile(input_img_path):
        input_imgs.append(transform(Image.open(input_img_path)).to(device))
        ## load all the images in the directory
    elif os.path.isdir(input_img_path):
        # Get list of files and sort them alphabetically
        files = sorted(os.listdir(input_img_path))
        for file in files: # Iterate over the sorted list
            img_path = os.path.join(input_img_path, file)
            # Ensure it's a file before trying to open
            if os.path.isfile(img_path):
                try:
                    input_imgs.append(transform(Image.open(img_path)).to(device))
                except PIL.UnidentifiedImageError:
                    print(f"Warning: Could not open or identify image file: {img_path}. Skipping.")
            else:
                 print(f"Warning: Found non-file item in directory: {img_path}. Skipping.")
        # Check if any images were loaded
        if len(input_imgs) == 0:
             raise ValueError(f"No valid image files found in directory: {input_img_path}")
    
    feature_maps = []
    for input_img in input_imgs:
        spatial_feature_dim = 50  # or model.get_dino_feature_spatial_dim()
        spatial_features = model.forward_features(input_img.unsqueeze(0))[:, 1:]  # shape (B, np, dim)
        B, npix, dim = spatial_features.shape
        spatial_features = spatial_features.reshape(
            B,
            16,
            16,
            dim
        )
        spatial_features = spatial_features.permute(0, 3, 1, 2).contiguous()
        spatial_features = F.interpolate(
            spatial_features, 
            size=(224, 224),
            mode="bilinear"
        )
        spatial_features = spatial_features.reshape(B, dim, -1).permute(0, 2, 1)
        # shape [B, HW, dim]
        spatial_features = spatial_features.detach().cpu().unsqueeze(1)
        # shape [B, 1, HW, dim]
        feature_maps.append(spatial_features)

    spatial_features = torch.cat(feature_maps, dim=0)
    B = spatial_features.shape[0]
    clustering_method = PerDatasetClustering(spatial_feature_dim, num_classes)
    cluster_maps = clustering_method.cluster(spatial_features)
    cluster_maps = cluster_maps.reshape(B, 224, 224).unsqueeze(1)
    for i, cluster_map in enumerate(cluster_maps):
        input_img = input_imgs[i]
        overlaid_images = overlay_overclustering_maps(
            input_img.unsqueeze(0),
            cluster_map.unsqueeze(0),
            model="DINO",
            save_dir="./davis",
            prefix=f"overclustering_single_DINO_{i}",
            nrow=1  # Number of images per row in the grid
        )
    return overlaid_images




def track_grid_points_batch(
    video: torch.Tensor,
    grid_size: int,
    lk_params: dict = None
) -> (torch.Tensor, torch.Tensor):
    """
    Track a regular grid of points through a batch of videos using Lucas–Kanade,
    and record visibility (status) for each point.

    Args:
        video    (B, T, C, H, W): float32 tensor in [0,1] or [0,255].
        grid_size         int:   number of points per axis (e.g. 8→8×8 grid).
        lk_params        dict:   overrides for cv2.calcOpticalFlowPyrLK.

    Returns:
        pred_tracks      (B, T, N, 2) float tensor: (y, x) positions.
        pred_visibility  (B, T, N)    uint8 tensor: 1 if tracked, 0 if lost.
    """
    B, T, C, H, W = video.shape
    device = video.device

    # Lucas–Kanade defaults
    if lk_params is None:
        lk_params = dict(
            winSize  = (21, 21),
            maxLevel = 3,
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            ),
        )

    # initial grid (x, y) points
    ys = torch.linspace(0, H - 1, grid_size, device=device)
    xs = torch.linspace(0, W - 1, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (N,2) as (x,y)
    N = pts.shape[0]

    # outputs
    pred_tracks     = torch.zeros((B, T, N, 2),  dtype=torch.float32, device=device)
    pred_visibility = torch.zeros((B, T, N),     dtype=torch.uint8,  device=device)

    for b in range(B):
        # frame 0: all points start visible
        pred_tracks[b, 0, :, 0] = pts[:, 1]  # y
        pred_tracks[b, 0, :, 1] = pts[:, 0]  # x
        pred_visibility[b, 0, :] = 1

        # prepare frames as uint8 numpy
        vid_np = video[b].permute(0, 2, 3, 1).cpu().numpy()
        if vid_np.dtype != np.uint8:
            vid_np = (vid_np * 255).clip(0,255).astype(np.uint8)

        def to_gray(f):
            return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if C == 3 else f[...,0]

        prev_gray = to_gray(vid_np[0])
        pts_np    = pts.cpu().numpy().reshape(-1,1,2).astype(np.float32)

        for t in range(1, T):
            curr_gray = to_gray(vid_np[t])
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts_np, None, **lk_params
            )

            next_pts = next_pts.reshape(-1, 2)   # (N,2): (x,y)
            status   = status.reshape(-1)        # (N,)

            # keep lost points at their last good position
            prev_pts = pts_np.reshape(-1, 2)
            lost = (status == 0)
            next_pts[lost] = prev_pts[lost]

            # clamp to [0…W-1] and [0…H-1]
            next_pts[:, 0] = np.clip(next_pts[:, 0], 0, W - 1)  # x
            next_pts[:, 1] = np.clip(next_pts[:, 1], 0, H - 1)  # y

            # write out
            pred_tracks[b, t, :, 0]     = torch.from_numpy(next_pts[:, 1]).to(device)
            pred_tracks[b, t, :, 1]     = torch.from_numpy(next_pts[:, 0]).to(device)
            pred_visibility[b, t, :]    = torch.from_numpy(status.astype(np.uint8)).to(device)

            # prep next iter
            pts_np    = next_pts.reshape(-1, 1, 2).astype(np.float32)
            prev_gray = curr_gray

    return pred_tracks, pred_visibility



def localize_objects(input_img, cluster_map):

    colors = ["orange", "blue", "red", "yellow", "white", "green", "brown", "purple", "gold", "black"]
    ticks = np.unique(cluster_map.flatten()).tolist()

    dc = np.zeros(cluster_map.shape)
    for i in range(cluster_map.shape[0]):
        for j in range(cluster_map.shape[1]):
            dc[i, j] = ticks.index(cluster_map[i, j])

    colormap = matplotlib.colors.ListedColormap(colors)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 3))
    # plt.figure(figsize=(5,3))
    im = axes[0].imshow(dc, cmap=colormap, interpolation="none", vmin=-0.5, vmax=len(colors) - 0.5)
    cbar = fig.colorbar(im, ticks=range(len(colors)))
    axes[1].imshow(input_img)
    axes[2].imshow(dc, cmap=colormap, interpolation="none", vmin=-0.5, vmax=len(colors) - 0.5)
    axes[2].imshow(input_img, alpha=0.5)
    # plt.show(block=True)
    # plt.close()
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return np.asarray(Image.open(buffer))


def convert_list_to_video(frames_list, name, speed, directory="", wdb_log=False):
    frames_list = [Image.fromarray(frame) for frame in frames_list]
    frames_list[0].save(f"{directory}{name}.gif", save_all=True, append_images=frames_list[1:], duration=speed, loop=0)
    if wdb_log:
        wandb.log({name: wandb.Video(f"{directory}{name}.gif", fps=4, format="gif")})


@torch.no_grad()
def sinkhorn(Q: torch.Tensor, nmb_iters: int, world_size=1) -> torch.Tensor:
    with torch.no_grad():
        Q = Q.detach().clone()
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        K, B = Q.shape
        u = torch.zeros(K).to(Q.device)
        r = torch.ones(K).to(Q.device) / K
        c = torch.ones(B).to(Q.device) / (B * world_size)

        if world_size > 1:
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

        for _ in range(nmb_iters):
            if world_size > 1:
                u = curr_sum
            else:
                u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            if world_size > 1:
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def find_optimal_assignment(scores, epsilon, sinkhorn_iterations, world_size=1):
    """
    Computes the Sinkhorn matrix Q.
    :param scores: similarity matrix
    :return: Sinkhorn matrix Q
    """
    with torch.no_grad():
        q = torch.exp(scores / epsilon).t()
        q = sinkhorn(q, sinkhorn_iterations, world_size=world_size)
        # q = torch.softmax(scores / epsilon, dim=0)
        # q = q / q.sum(dim=1, keepdim=True)
    return q




import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import softmax



def merge_models(model_1, model_2, coeff=0.5):
    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
        param_1.data = param_1.data * coeff + param_2.data * (1 - coeff)
    return model_1


def test_sinkhorn_single_process():
    scores = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]])
    epsilon = 0.1
    sinkhorn_iterations = 5

    # Compute Sinkhorn
    q = find_optimal_assignment(scores, epsilon, sinkhorn_iterations, world_size=1)

    # Expected output: Row and column sums are close to 1
    assert torch.allclose(torch.sum(q, dim=0), torch.ones(q.shape[1]), atol=1e-5), "Column sums are incorrect"
    assert torch.allclose(torch.sum(q, dim=1), torch.ones(q.shape[0]), atol=1e-5), "Row sums are incorrect"
    print("Single process test passed!")



def distributed_sinkhorn_test(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    scores = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]])
    scores = scores.to(rank)  # Move to the appropriate device for each process

    epsilon = 0.1
    sinkhorn_iterations = 5

    # Compute Sinkhorn
    q = find_optimal_assignment(scores, epsilon, sinkhorn_iterations, world_size)

    # Validate output in rank 0
    if rank == 0:
        assert torch.allclose(torch.sum(q, dim=0), torch.ones(q.shape[1]), atol=1e-5), "Column sums are incorrect"
        assert torch.allclose(torch.sum(q, dim=1), torch.ones(q.shape[0]), atol=1e-5), "Row sums are incorrect"
        print("Distributed test passed!")

    dist.destroy_process_group()

def run_distributed_test():
    world_size = 2
    mp.spawn(distributed_sinkhorn_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    # Test in single-process mode
    test_sinkhorn_single_process()

    # Test in distributed mode
    run_distributed_test()