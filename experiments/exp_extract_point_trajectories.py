import numpy as np
import torch
from dataset.data_loader import PascalVOCDataModule, SamplingMode, VideoDataModule
from utils.my_utils import denormalize_video, denormalize_video_cotracker_compatible
from cotracker.utils.visualizer import Visualizer
import shutil  # Add this import at the top
import imageio.v3 as iio
import video_transformations
import os
import argparse


def extract_trajectories(dataset_name, grid_size):
    if dataset_name != "ytvos":
        raise ValueError(f"Dataset {dataset_name} not supported")
    rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    data_transform = video_transformations.Compose(data_transform_list)
    video_transform_list = [video_transformations.Resize((224, 224)), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])] #video_transformations.RandomResizedCrop((224, 224))
    video_transform = video_transformations.Compose(video_transform_list)
    num_clips = 1
    num_clip_frames = 32
    regular_step = 5
    batch_size = 1
    transformations_dict = {"data_transforms": None, "target_transforms": None, "shared_transforms": video_transform}
    # prefix = "/ssdstore/ssalehi/dataset"
    prefix = "/ssdstore/ssalehi/EpicKitchens/"
    data_path = os.path.join(prefix, "")
    annotation_path = os.path.join(prefix, "Annotations/")
    trajectory_path = os.path.join(prefix, "train1/trajectories/")
    # if not os.path.exists(trajectory_path):
    #     os.makedirs(trajectory_path)
    # else:
    #     shutil.rmtree(trajectory_path, ignore_errors=True)  # Replace os.rmdir line
    #     os.makedirs(trajectory_path)
    meta_file_path = os.path.join(prefix, "train1/meta.json")
    path_dict = {"class_directory": data_path, "annotation_directory": "", "meta_file_path": None}
    sampling_mode = SamplingMode.DENSE
    num_workers = 16
    video_data_module = VideoDataModule("ytvos_test", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers)
    video_data_module.setup(transformations_dict)
    video_data_module.make_data_loader()

    data_loader = video_data_module.get_data_loader()

    device = 'cuda:0'
    grid_size = grid_size
# video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# Run Offline CoTracker:
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

    for i, batch in enumerate(data_loader):
        datum, annotations, video_path = batch
        video_dir = video_path[0].split("/")[-1]
        annotations = annotations.squeeze(1)
        datum = datum.squeeze(1)
        datum = datum[:, 0:32]
        denormalized_video = denormalize_video_cotracker_compatible(datum)
        datum = datum.to(device)
        denormalized_video = denormalized_video.to(device)
        pred_tracks, pred_visibility = cotracker(denormalized_video, grid_size=grid_size) # B T N 2,  B T N 1
        # pred_tracks_np = pred_tracks.squeeze(0).cpu().numpy()
        # pred_visibility_np = pred_visibility.squeeze(0).cpu().numpy()
        # os.makedirs(os.path.join(trajectory_path, video_dir), exist_ok=True)
        # np.save(os.path.join(trajectory_path, video_dir, f"pred_tracks_grid{grid_size}.npy"), pred_tracks_np)
        # np.save(os.path.join(trajectory_path, video_dir, f"pred_visibility_grid{grid_size}.npy"), pred_visibility_np)
        vis = Visualizer(save_dir=f"./co_tracker_saved_videos/{i}", pad_value=120, linewidth=3)
        pred_tracks = pred_tracks
        pred_visibility = pred_visibility
        vis.visualize(denormalized_video, pred_tracks, pred_visibility)

extract_trajectories("ytvos", grid_size=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ytvos")
    parser.add_argument("--grid_size", type=int, default=16)
    args = parser.parse_args()
    extract_trajectories(args.dataset_name, args.grid_size)