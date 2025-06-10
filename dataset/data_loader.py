from collections import OrderedDict
import torch
import torchvision.transforms as trn
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import wandb
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
from typing import Tuple, Any
from pathlib import Path
from typing import Optional, Callable
from torchvision.datasets import VisionDataset
from dataset.image_transformations import RandomResizedCrop, RandomHorizontalFlip, Compose
import random
import json
from enum import Enum
from utils.my_utils import denormalize_video, make_seg_maps, visualize_sampled_videos
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import dataset.video_transformations as video_transformations
import torch.nn.functional as F

import torch
import torchvision as tv
from abc import ABC, abstractmethod
import os
import scipy.io as sio
from torchvision.datasets import CIFAR10
from pycocotools.coco import COCO
from typing import Any, Callable, List, Optional, Tuple
from zipfile import ZipFile
from collections import defaultdict
import io


project_name = "MoSiC"





# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# random.seed(1)
# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)




class CO3DDataset(Dataset):
    def __init__(
        self,  
        sampling_mode, 
        num_clips, 
        num_frames, 
        frame_transform=None, 
        target_transform=None, 
        video_transform=None, 
        mapping_path="./all_frames_complete.json", #modify based on location of all_frames_complete.json
        zip_mapping_path = "./zip_mapping.json", #modify based on location of zip_mapping.json
        regular_step=0,
        grid_size=16
    ):
        """
        CO3D Dataset loader.

        Args:
            subset_name (str): Subset name (e.g., "manyview_dev_0").
            sampling_mode (str): Sampling mode (e.g., "dense").
            num_clips (int): Number of clips per sequence.
            num_frames (int): Number of frames per clip.
            frame_transform (callable): Transform to apply to each frame.
            target_transform (callable): Transform to apply to targets.
            video_transform (callable): Transform to apply to video sequences.
            mapping_path (str): Path to save/load the frame-to-zip mapping.
        """
        super().__init__()

        if not os.path.exists(zip_mapping_path):
            raise FileNotFoundError(f"zip_mapping.json not found at {zip_mapping_path}")
        with open(zip_mapping_path, "r") as f:
            self.zip_mapping = json.load(f)
        #self.subset_name = subset_name
        self.sampling_mode = sampling_mode
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frame_transform = frame_transform
        self.target_transform = target_transform
        self.video_transform = video_transform
        self.regular_step = regular_step
        #self.dataset_structure = self.prepare_data(mapping_path=mapping_path)
        self.dataset_structure = self.create_frame_to_zip_mapping(mapping_path=mapping_path)
        self.grid_size = grid_size
    
    def create_frame_to_zip_mapping(self, mapping_path, min_frames=10):
        """
        Create a mapping of frames to their corresponding zip files.

        Args:
            mapping_path (str): Path to save the resulting frame-to-zip mapping.
            min_frames (int): Minimum number of frames required for a sequence to be included.
        """
        if os.path.exists(mapping_path):
            print(f"Loading frame-to-zip mapping from {mapping_path}")
            with open(mapping_path, "r") as f:
                return json.load(f)

        print("Processing categories from zip mapping...")
        frame_to_zip = defaultdict(lambda: defaultdict(list))

        # Process each category
        for category, zip_files in self.zip_mapping.items():
            print(f"Processing category: {category}")

            # Track frames across zips for each sequence
            sequence_frames = defaultdict(list)

            # Process each zip file in the category
            for zip_file in zip_files:
                print(f"Processing zip: {zip_file}")
                with ZipFile(zip_file, 'r') as zf:
                    for file_path in zf.namelist():
                        # Only consider paths inside the 'images/' folder of the category
                        if file_path.startswith(f"{category}/") and "images/" in file_path:
                            parts = file_path.split("/")
                            if len(parts) >= 3:
                                sequence = parts[1]  # Extract the sequence folder name
                                if file_path.endswith(".jpg") or file_path.endswith(".png"):
                                    frame_path = "/".join(parts[1:])  # Full relative frame path
                                    sequence_frames[sequence].append((frame_path, zip_file))

            # Add frames to the mapping, sorted by filename, only if enough frames exist
            for sequence, frames in sequence_frames.items():
                if len(frames) >= min_frames:
                    frame_to_zip[category][sequence] = sorted(frames, key=lambda x: x[0])
                else:
                    print(f"Skipping sequence '{sequence}' in category '{category}' (only {len(frames)} frames).")

        # Save the mapping to a JSON file
        with open(mapping_path, "w") as f:
            json.dump(frame_to_zip, f, indent=4)

        print(f"Frame-to-zip mapping saved to {mapping_path}")
        return frame_to_zip



    def prepare_data2(self, mapping_path):
        """Parse zips, locate set_lists, and save frame-to-zip mapping."""
        # Check if the mapping already exists
        if os.path.exists(mapping_path):
            print(f"Loading frame-to-zip mapping from {mapping_path}")
            with open(mapping_path, "r") as f:
                return json.load(f)
        
        print("Creating frame-to-zip mapping...")
        structure = defaultdict(lambda: defaultdict(list))  # {category: {sequence: [(frame_path, zip_file)]}}

        # Process each category and its associated zips
        for category, zips in self.zip_mapping.items():
            print(f"Processing category: {category}")
            
            # First, locate and process the set_lists files across all zips
            combined_set_list = []
            for zip_file in zips:
                with ZipFile(zip_file, 'r') as zf:
                    # Locate set_list files for the subset
                    set_lists_files = [
                        f for f in zf.namelist() if f"set_lists/set_lists_{self.subset_name}.json" in f
                    ]
                    for set_lists_file in set_lists_files:
                        print(f"Found set_list file: {set_lists_file} in zip: {zip_file}")
                        with zf.open(set_lists_file) as f:
                            set_list = json.load(f)
                        
                        # Append entries from the 'train' split
                        if 'train' in set_list:
                            combined_set_list.extend(set_list['train'])  # Each entry: [sequence_name, frame_index, relative_frame_path]
            
            # Locate the frames across all zips
            for entry in combined_set_list:
                sequence_name, frame_index, relative_frame_path = entry
                if sequence_name not in structure[category]:
                    structure[category][sequence_name] = []

                print(f"relative_frame_path: {relative_frame_path}")
                normalized_frame_path = relative_frame_path
                found = False

                # Search for the frame in all zips of the category
                for zip_file in zips:
                    print(f"Searching for frame '{normalized_frame_path}' in zip '{zip_file}'")
                    with ZipFile(zip_file, 'r') as zf:
                        #print(f'namelist in zips {zf.namelist()}')
                        if normalized_frame_path in zf.namelist():
                            #print(f'namelist in zips {zf.namelist()}')
                            structure[category][sequence_name].append((relative_frame_path, zip_file))
                            found = True
                            print(f"Frame '{normalized_frame_path}' found in zip '{zip_file}'")
                            break  # No need to check other zips once found
                
                if not found:
                    print(f"Frame '{normalized_frame_path}' NOT found in any zips for category '{category}'")
                    raise ValueError("Frame not found in any zips")
        
        # Save the mapping for future use
        with open(mapping_path, "w") as f:
            json.dump(structure, f, indent=4)
        print(f"Frame-to-zip mapping saved to {mapping_path}")
        
        return structure

    def load_image2(self, zip_path, file_path):
        """Load an image directly from a zip."""
        file_data = self.stream_file(zip_path, file_path)
        return Image.open(file_data)

    def stream_file(self, zip_path, file_path):
        """Stream a file from a zip."""
        with ZipFile(zip_path, 'r') as zf:
            with zf.open(file_path) as f:
                return io.BytesIO(f.read())

    def load_image(self, zip_path, file_path):
        """Load a JPG image directly from a zip."""
        try:
            # Stream the file and open as an image
            file_data = self.stream_file(zip_path, file_path)
            return Image.open(file_data).convert("RGB")  # Convert to RGB
        except UnidentifiedImageError:
            print(f"UnidentifiedImageError: Could not load image {file_path} in {zip_path}.")
            return None
        except Exception as e:
            print(f"Error loading image {file_path} from {zip_path}: {e}")
            return None

    def __len__(self):
        """Return the total number of category-sequence pairs."""
        return sum(len(sequences) for sequences in self.dataset_structure.values())
    
    def generate_indices(self, size, sampling_num):
        indices = []
        for i in range(self.num_clips):
            if self.sampling_mode == SamplingMode.UNIFORM:
                    if size < sampling_num:
                        ## sample repeatly
                        idx = random.choices(range(0, size), k=sampling_num)
                    else:
                        idx = random.sample(range(0, size), sampling_num)
                    idx.sort()
                    indices.append(idx)
            elif self.sampling_mode == SamplingMode.DENSE:
                    base = random.randint(0, size - sampling_num)
                    idx = range(base, base + sampling_num)
                    indices.append(idx)
            elif self.sampling_mode == SamplingMode.Full:
                    indices.append(range(0, size))
            elif self.sampling_mode == SamplingMode.Regular:
                if size < sampling_num * self.regular_step:
                    step = size // sampling_num
                else:
                    step = self.regular_step
                if step == 0:
                    base = 0
                    step = 1
                else:
                    base = random.randint(0, size - (sampling_num * step))
                idx = range(base, base + (sampling_num * step), step)
                ## convert the indices larger than size to be size - 1
                idx = [i if i < size else size - 1 for i in idx]
                indices.append(idx)
        return indices

    def __getitem__(self, index):
        # Retrieve the category and sequence based on the index
        flat_structure = [
            (category, sequence_name)
            for category, sequences in self.dataset_structure.items()
            for sequence_name in sequences
        ]
        category, sequence_name = flat_structure[index]
        frame_info = self.dataset_structure[category][sequence_name]
        total_frames = len(frame_info)
        #print(f"Category: {category}, Sequence: {sequence_name}, Num frames: {total_frames}")
        
        # Generate indices for frame sampling
        indices = self.generate_indices(total_frames, self.num_frames)
        indices = indices[0]  # Since we generate multiple clips, take the first set of indices

        # Load the sampled frames from their respective zips
        frame_images = []
        for idx in indices:
            frame_path, zip_path = frame_info[idx]
            full_frame_path = f"{category}/{frame_path}"
            try:
                img = self.load_image(zip_path, full_frame_path)
                if img is None:
                    print(f"Warning: Failed to load image at {full_frame_path} in {zip_path}.")
                    continue
                frame_images.append(img)
            except FileNotFoundError:
                print(f"Frame '{full_frame_path}' not found in '{zip_path}'. Skipping...")
                continue

        if not frame_images:
            raise ValueError(f"No frames could be loaded for sequence '{sequence_name}' in category '{category}'.")

        # Apply frame-level transformations
        frame_images = [img for img in frame_images if img is not None]
        if self.frame_transform:
            #print("Applying frame-level transformations...")
            frame_images = self.frame_transform(frame_images)

        # Apply video-level transformations
        if self.video_transform:
            #print("Applying video-level transformations...")
            frame_images = self.video_transform(frame_images)

        # Convert the list of images into a tensor
        frame_images = torch.stack(
            [trn.ToTensor()(img) if isinstance(img, Image.Image) else img for img in frame_images]
        )
        sampled_pred_tracks = torch.zeros((self.num_frames, self.grid_size, self.grid_size))
        sampled_pred_visibility = torch.zeros((self.num_frames, self.grid_size, self.grid_size))

        return frame_images.unsqueeze(0), torch.empty(0), sampled_pred_tracks, sampled_pred_visibility



class Dataset(torch.nn.Module):
    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    @abstractmethod
    def get_val_loader(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def get_val_dataset(self):
        pass
    
    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass


class NormalSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, subset_indices):
        self.dataset = dataset
        self.subset_indices = subset_indices

    def __getitem__(self, index):
        return self.dataset[self.subset_indices[index]]

    def __len__(self):
        return len(self.subset_indices)


class Cifar10_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "Cifar10"
        self.num_classes = 10
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = CIFAR10(root="/ssdstore/ssalehi/cifar", train=True, download=True, transform=self.transform)
        self.test_dataset = CIFAR10(root="/ssdstore/ssalehi/cifar", train=False, download=True, transform=self.val_transform )
        self.binarize_test_labels()
        ## split the dataset to train and validation
        normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes

    def get_normal_sebset_indices(self):
        normal_subset_indices = [i for i, (data, label) in enumerate(self.train_dataset) if label in self.normal_classes]
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        for i, label in enumerate(self.test_dataset.targets):
            if label in self.normal_classes:
                self.test_dataset.targets[i] = 0
            else:
                self.test_dataset.targets[i] = 1
   
    



class ImangeNet_100_Handler(Dataset):
    def __init__(self, batch_size, dataset_path, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "ImageNet_100"
        self.num_classes = 100
        self.transform = transformations
        self.val_transform = val_transformations
        self.image_train_transform = transformations["img"]
        self.image_val_transform = transformations["img"]
        self.image_test_transform = transformations["img"]
        self.target_train_transform = None
        self.target_val_transform = None
        self.target_test_transform = None
        self.shared_train_transform = transformations["shared"]
        self.shared_val_transform = val_transformations["shared"]
        self.shared_test_transform = val_transformations["shared"]


    def setup(self):
        self.train_dataset = ImageFolder(root=f"{self.dataset_path}/train", transform=self.image_train_transform)
        self.test_dataset = ImageFolder(root=f"{self.dataset_path}/val", transform=self.val_transform )
        ## split the dataset to train and validation
        print("Normal Subset Size: ", len(self.train_dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [int(len(self.train_dataset)*0.9), int(len(self.train_dataset)*0.1)])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_dataloader(self):
        return self.train_loader

    def get_test_dataloader(self):
        return self.test_loader

    def get_val_dataloader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes



class SamplingMode(Enum):
    UNIFORM = 0
    DENSE = 1
    Full = 2
    Regular = 3


def get_file_path(classes_directory):
    ## find all the folders and add all the files in the folders to a dict with keys are name of the file and values are the path to the file

    folder_file_path = {} ## key is the directory_path and value are the files in the directory
    for root, dirs, files in os.walk(classes_directory):
        for file in sorted(files):
            if "depths" in root or "depth_masks" in root or "masks" in root: ## skip depth and mask directories for EpicKitchens
                continue
            if file.endswith(".jpg") or file.endswith(".png"):
                if root not in folder_file_path:
                    folder_file_path[root] = []
                folder_file_path[root].append(file)
    
    return dict(sorted(folder_file_path.items()))



def make_categories_dict(meta_dict, name):
    category_list = []
    if "ytvos" in name:
        video_name_list = meta_dict["videos"].keys()
        for name in video_name_list:
            obj_list = meta_dict["videos"][name]["objects"].keys()
            for obj in obj_list:
                if meta_dict["videos"][name]["objects"][obj]["category"] not in category_list:
                    category_list.append(meta_dict["videos"][name]["objects"][obj]["category"])
        category_list = sorted(list(OrderedDict.fromkeys(category_list)))
        category_ditct = {k: v+1 for v, k in enumerate(category_list)} ## zero is always for the background
    return category_ditct



def map_instances(data, meta, category_dict):
    bs, fs, h, w = data.shape
    for i, datum in enumerate(data):
        for j, frame in enumerate(data):
            objects = torch.unique(frame)
            for k, obj in enumerate(objects):
                if int(obj.item()) == 0:
                    continue
                frame[frame == obj] = category_dict[meta[str(int(obj.item()))]["category"]]
    return data


class VideoDataset(torch.utils.data.Dataset):
    ## The data loader gets training sample and annotations direcotories, sampling mode, number of clips that is being sampled of each training video, number of frames in each clip
    ## and number of labels for each training clip. 
    ## Note that the number of annotations should be exactly similar to the number of frames existing in the training path.
    ## Frame_transform is a function that transforms the frames of the video. It is applied to each frame of the video.
    ## Target_transform is a function that transforms the annotations of the video. It is applied to each annotation of the video.
    ## Video_transform is a function that transforms the whole video. It is applied to both frames and annotations of the video.
    ## The same set of transformations is applied to the clips of the video.

    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__()
        self.train_dict = get_file_path(classes_directory)
        self.train_dict_lenghts = {}
        self.find_directory_length()
        if (annotations_directory != "") and (os.path.exists(annotations_directory)):
            self.train_annotations_dict = get_file_path(annotations_directory)
            self.use_annotations = True
        else:
            self.use_annotations = False
            print("Because there is no annotation directory, only training samples have been loaded.")
        if (meta_file_directory is not None):
            if (os.path.exists(meta_file_directory)):
                print("Meta file has been read.")
                file = open(meta_file_directory)
                self.meta_dict = json.load(file)
            else:
                self.meta_dict = None
                print("There is no meta file.")
        else:
            print("Meta option is off.")
            self.meta_dict = None
         
        self.sampling_mode = sampling_mode
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frame_transform = frame_transform
        self.target_transform = target_transform
        self.video_transform = video_transform
        self.regular_step = regular_step
        self.keys = list(self.train_dict.keys())
        if self.use_annotations:
            self.annotation_keys = list(self.train_annotations_dict.keys())
        
    def __len__(self):
        return len(self.keys)

    
    def find_directory_length(self):
        for key in self.train_dict:
            self.train_dict_lenghts[key] = len(self.train_dict[key])

    
    def read_clips(self, path, clip_indices):
        clips = []
        files = sorted(glob.glob(path + "/" + "*.jpg"))
        if len(files) == 0:
            files = sorted(glob.glob(path + "/" + "*.png"))
        for i in range(len(clip_indices)):
            images = []
            for j in clip_indices[i]:
                # frame_path = path + "/" + f'{j:05d}' + ".jpg"
                frame_path = files[j]
                if not os.path.exists(frame_path):
                    frame_path = path + "/" + f'{j:05d}' + ".png"
                if not os.path.exists(frame_path): ## This is for kinetics dataset
                    frame_path = path + "/" + f'img_{(j + 1):05d}' + ".jpg" 
                if not os.path.exists(frame_path): ## This is for kinetics dataset
                    frame_path = path + "/" + f'frame_{(j + 1):010d}' + ".jpg" 

                images.append(Image.open(frame_path))
            clips.append(images)
        return clips
    
    
    def generate_indices(self, size, sampling_num):
        indices = []
        for i in range(self.num_clips):
            if self.sampling_mode == SamplingMode.UNIFORM:
                    if size < sampling_num:
                        ## sample repeatly
                        idx = random.choices(range(0, size), k=sampling_num)
                    else:
                        idx = random.sample(range(0, size), sampling_num)
                    idx.sort()
                    indices.append(idx)
            elif self.sampling_mode == SamplingMode.DENSE:
                    if size < sampling_num:
                        ## sample repeatly
                        idx = range(0, sampling_num)
                        idx = [i if i < size else size - 1 for i in idx]
                    else:
                        base = random.randint(0, size - sampling_num)
                        idx = range(base, base + sampling_num)
                    indices.append(idx)
            elif self.sampling_mode == SamplingMode.Full:
                    indices.append(range(0, size))
            elif self.sampling_mode == SamplingMode.Regular:
                if size < sampling_num * self.regular_step:
                    step = size // sampling_num
                else:
                    step = self.regular_step
                if step == 0:
                    base = 0
                    step = 1
                else:
                    base = random.randint(0, size - (sampling_num * step))
                idx = range(base, base + (sampling_num * step), step)
                ## convert the indices larger than size to be size - 1
                idx = [i if i < size else size - 1 for i in idx]
                indices.append(idx)
        return indices
    

    def read_batch(self, path, annotation_path=None, frame_transformation=None, target_transformation=None, video_transformation=None):
        size = self.train_dict_lenghts[path]
        # sampling_num = size if self.num_frames > size else self.num_frames
        clip_indices = self.generate_indices(self.train_dict_lenghts[path], self.num_frames)
        sampled_clips = self.read_clips(path, clip_indices)
        annotations = []
        sampled_clip_annotations = []
        if annotation_path is not None:
            sampled_clip_annotations = self.read_clips(annotation_path, clip_indices)
            if target_transformation is not None:
                for i in range(len(sampled_clip_annotations)):
                    sampled_clip_annotations[i] = target_transformation(sampled_clip_annotations[i])
        if frame_transformation is not None:
            for i in range(len(sampled_clips)):
                try:
                    sampled_clips[i] = frame_transformation(sampled_clips[i])
                except:
                    print("Error in frame transformation")
        if video_transformation is not None:
            for i in range(len(sampled_clips)):
                if len(sampled_clip_annotations) != 0:
                    sampled_clips[i], sampled_clip_annotations[i] = video_transformation(sampled_clips[i], sampled_clip_annotations[i])
                else:
                    sampled_clips[i] = video_transformation(sampled_clips[i])
        sampled_data = torch.stack(sampled_clips)
        if len(sampled_clip_annotations) != 0:
            sampled_annotations = torch.stack(sampled_clip_annotations)
            if sampled_annotations.size(0) != 0:
                sampled_annotations = (255 * sampled_annotations).type(torch.uint8) 
                if sampled_annotations.shape[2] == 1:
                    sampled_annotations = sampled_annotations.squeeze(2)
        else:
            sampled_annotations = torch.empty(0)
        ## squeezing the annotations to be in the shape of (num_sample, num_clips, num_frames, height, width)
        return sampled_data, sampled_annotations


    def read_batch_with_new_transforms(self, path, annotation_path=None, frame_transformation=None, target_transformation=None, video_transformation=None):
        size = self.train_dict_lenghts[path]
        # sampling_num = size if self.num_frames > size else self.num_frames
        clip_indices = self.generate_indices(self.train_dict_lenghts[path], self.num_frames)
        sampled_clips = self.read_clips(path, clip_indices)
        annotations = []
        sampled_clip_annotations = []
        labels_dict = {}
        if annotation_path is not None:
            sampled_clip_annotations = self.read_clips(annotation_path, clip_indices)
            if target_transformation is not None:
                for i in range(len(sampled_clip_annotations)):
                    sampled_clip_annotations[i] = target_transformation(sampled_clip_annotations[i])
        if frame_transformation is not None:
            for i in range(len(sampled_clips)):
                # try:
                multi_crops, labels_dict = frame_transformation(sampled_clips[i])
                # except:
                #     print("Error in frame transformation")
        if video_transformation is not None:
            for i in range(len(sampled_clips)):
                if len(sampled_clip_annotations) != 0:
                    sampled_clips[i], sampled_clip_annotations[i] = video_transformation(sampled_clips[i], sampled_clip_annotations[i])
                else:
                    sampled_clips[i] = video_transformation(sampled_clips[i])
        # sampled_data = torch.stack(sampled_clips)
        if len(sampled_clip_annotations) != 0:
            sampled_annotations = torch.stack(sampled_clip_annotations)
            if sampled_annotations.size(0) != 0:
                sampled_annotations = (255 * sampled_annotations).type(torch.uint8) 
                if sampled_annotations.shape[2] == 1:
                    sampled_annotations = sampled_annotations.squeeze(2)
        else:
            sampled_annotations = torch.empty(0)
        ## squeezing the annotations to be in the shape of (num_sample, num_clips, num_frames, height, width)
        return multi_crops, labels_dict, sampled_annotations
    

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        # idx = 0  ## This is a hack to make the code work with the dataloader.
        # idx = random.randint(0, 5)
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            category_dict = make_categories_dict(self.meta_dict, "davis")
            meta_dict = self.meta_dict["videos"][dir_name]["objects"]
            annotations = map_instances(annotations, meta_dict, category_dict)

        return data, annotations



class YVOSDataset(VideoDataset):

    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__(classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)
        if self.meta_dict is not None:
            self.category_dict = make_categories_dict(self.meta_dict, "ytvos")

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, annotations


class YTVOSDatasetTest(YVOSDataset):
    def __getitem__(self, idx):
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, annotations, video_path


class YTVOSTrajectoryDataset(YVOSDataset):
    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1, grid_size=16) -> None:
        super().__init__(classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)
        if self.meta_dict is not None:
            self.category_dict = make_categories_dict(self.meta_dict, "ytvos")
        self.trajectory_path = classes_directory.replace("JPEGImages", "trajectories")
        self.grid_size = grid_size

    def __getitem__(self, idx):
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, annotations, pred_tracks, pred_visibility = self.read_batch_with_trajectories(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, annotations, pred_tracks, pred_visibility

    def read_batch_with_trajectories(self,  path, annotation_path=None, frame_transformation=None, target_transformation=None, video_transformation=None):
        size = self.train_dict_lenghts[path]
        dir_name = path.split("/")[-1]
        # sampling_num = size if self.num_frames > size else self.num_frames
        clip_indices = self.generate_indices(self.train_dict_lenghts[path], self.num_frames)
        sampled_clips = self.read_clips(path, clip_indices)

        sampled_pred_tracks = torch.zeros((self.num_frames, self.grid_size, self.grid_size))
        sampled_pred_visibility = torch.zeros((self.num_frames, self.grid_size, self.grid_size))
        ##
        sampled_clip_annotations = []
        if annotation_path is not None:
            sampled_clip_annotations = self.read_clips(annotation_path, clip_indices)
            if target_transformation is not None:
                for i in range(len(sampled_clip_annotations)):
                    sampled_clip_annotations[i] = target_transformation(sampled_clip_annotations[i])
        if frame_transformation is not None:
            for i in range(len(sampled_clips)):
                try:
                    sampled_clips[i] = frame_transformation(sampled_clips[i])
                except:
                    print("Error in frame transformation")
        if video_transformation is not None:
            for i in range(len(sampled_clips)):
                if len(sampled_clip_annotations) != 0:
                    sampled_clips[i], sampled_clip_annotations[i] = video_transformation(sampled_clips[i], sampled_clip_annotations[i])
                else:
                    sampled_clips[i] = video_transformation(sampled_clips[i])
        sampled_data = torch.stack(sampled_clips)
        if len(sampled_clip_annotations) != 0:
            sampled_annotations = torch.stack(sampled_clip_annotations)
            if sampled_annotations.size(0) != 0:
                sampled_annotations = (255 * sampled_annotations).type(torch.uint8) 
                if sampled_annotations.shape[2] == 1:
                    sampled_annotations = sampled_annotations.squeeze(2)
        else:
            sampled_annotations = torch.empty(0)
        ## squeezing the annotations to be in the shape of (num_sample, num_clips, num_frames, height, width)
        return sampled_data, sampled_annotations, sampled_pred_tracks, sampled_pred_visibility


class TimeTYVOSDataset(VideoDataset):

    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__(classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)
        if self.meta_dict is not None:
            self.category_dict = make_categories_dict(self.meta_dict, "timetytvos")

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, labels, annotations = self.read_batch_with_new_transforms(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, labels, annotations


class Kinetics(VideoDataset):

    def __init__(self, classes_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__(classes_directory, "", sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)    
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, annotations


class VOCDataset(Dataset):

    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        super(VOCDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.image_set = image_set
        if "trainaug" in self.image_set or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        # else:
        #     raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'images')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'sets')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]
        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.return_masks:
            mask = Image.open(self.masks[index])
        if self.image_set == "val":
            if self.transform:
                img = self.transform(img)
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask
        elif "train" in self.image_set:
            if self.transform:
                img = self.transform(img)
            if self.transforms:
                res = self.transforms(img, mask)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)



class VOCDataModule_HB():

    CLASS_IDX_TO_NAME = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                         'train', 'tvmonitor']

    def __init__(self,
                 data_dir: str,
                 train_split: str,
                 val_split: str,
                 train_image_transform: Optional[Callable],
                 batch_size: int,
                 num_workers: int,
                 val_image_transform: Optional[Callable]=None,
                 val_target_transform: Optional[Callable]=None,
                 val_transforms: Optional[Callable]=None,
                 shuffle: bool = False,
                 return_masks: bool = False,
                 drop_last: bool = True):
        """
        Data module for PVOC data. "trainaug" and "train" are valid train_splits.
        If return_masks is set train_image_transform should be callable with imgs and masks or None.
        """
        super().__init__()
        self.root = os.path.join(data_dir, "VOCSegmentation")
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_transform = train_image_transform
        self.val_transforms = val_transforms
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.return_masks = return_masks

        # Set up datasets in __init__ as we need to know the number of samples to init cosine lr schedules
        # assert train_split == "trainaug" or train_split == "train"
        self.voc_train = VOCDataset(root=self.root, image_set=train_split, transforms=self.train_image_transform,
                                    return_masks=self.return_masks)
        self.voc_val = VOCDataset(root=self.root, image_set=val_split, transform=self.val_image_transform,
                                  target_transform=self.val_target_transform, transforms=self.val_transforms, return_masks=self.return_masks)

    def __len__(self):
        return len(self.voc_train)

    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]

    def setup(self, stage: Optional[str] = None):
        print(f"Train size {len(self.voc_train)}")
        print(f"Val size {len(self.voc_val)}")

    def train_dataloader(self):
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.voc_train,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=self.shuffle,
                drop_last=self.drop_last
            )
        
        return DataLoader(self.voc_train, batch_size=self.batch_size,
                          shuffle=(sampler is None),
                          sampler=sampler,
                          num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    # def val_dataloader(self):
    #     return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
    #                       drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        """
        Return a DataLoader for the test/validation dataset,
        with a DistributedSampler if dist is initialized.
        """
        # This is the typical approach for distributed validation.
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.voc_val,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=False
            )
        test_loader = DataLoader(
            self.voc_val,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        return test_loader

    def get_train_dataset_size(self):
        return len(self.voc_train)

    def get_val_dataset_size(self):
        return len(self.voc_val)
    
    def get_num_classes(self):
        return len(self.CLASS_IDX_TO_NAME)
    
    def get_name(self):
        return "VOCDataModule_HB"
    

class PascalVOCDataModule():
    """ 
    DataModule for Pascal VOC dataset

    Args:
        batch_size (int): batch size
        train_transform (torchvision.transforms): transform for training set
        val_transform (torchvision.transforms): transform for validation set
        test_transform (torchvision.transforms): transform for test set
        dir (str): path to dataset
        year (str): year of dataset
        split (str): split of dataset
        num_workers (int): number of workers for dataloader

    """

    def __init__(self, batch_size, train_transform, val_transform, test_transform,  dir="/ssdstore/ssalehi/VOCSegmentation", num_workers=0) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dir = dir
        self.image_train_transform = train_transform["img"]
        self.image_val_transform = val_transform["img"]
        self.image_test_transform = test_transform["img"]
        self.target_train_transform = None
        self.target_val_transform = None
        self.target_test_transform = None
        self.shared_train_transform = train_transform["shared"]
        self.shared_val_transform = val_transform["shared"]
        self.shared_test_transform = test_transform["shared"]

    def setup(self):
        download = False
        if os.path.isdir(self.dir) == False:
            download = True
        self.train_dataset = VOCDataset(self.dir, image_set="trainaug", transform=self.image_train_transform, target_transform=self.target_train_transform, transforms=self.shared_train_transform, return_masks=True)
        self.val_dataset = VOCDataset(self.dir, image_set="val", transform=self.image_val_transform, target_transform=self.target_val_transform, transforms=self.shared_val_transform, return_masks=True)
        self.test_dataset = VOCDataset(self.dir, image_set="val", transform=self.image_test_transform, target_transform=self.target_test_transform, transforms=self.shared_test_transform, return_masks=True)
        print(f"Train size : {len(self.train_dataset)}")
        print(f"Val size : {len(self.val_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def get_test_dataloader(self):
        """
        Return a DataLoader for the test/validation dataset,
        with a DistributedSampler if dist is initialized.
        """
        # This is the typical approach for distributed validation.
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=False
            )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        return test_loader
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def get_module_name(self):
        return "PascalVOCDataModule"
    
    def get_num_classes(self):
        return 21
    


class MixedDataset(Dataset):
    def __init__(self, datasets, sampling_ratios=None):
        """
        Sample from multiple datasets with ratios
        Args:
            datasets (list): A list of dataset instances (CO3D, YTVOS, DL3DV).
            sampling_ratios (list, optional): List of ratios for sampling from each dataset.
                                              If None, datasets are sampled uniformly.
        """
        super().__init__()
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_dataset_lengths = [0] + [sum(self.dataset_lengths[:i+1]) for i in range(self.num_datasets)]
        # If no ratios are provided, sample each dataset equally
        if sampling_ratios is None:
            self.sampling_ratios = [1 / self.num_datasets] * self.num_datasets
        else:
            self.sampling_ratios = sampling_ratios

        # Make sure ratios sum to 1
        total_ratio = sum(self.sampling_ratios)
        self.sampling_ratios = [r / total_ratio for r in self.sampling_ratios]

        # Compute cumulative probabilities for weighted sampling
        self.cumulative_ratios = [sum(self.sampling_ratios[:i+1]) for i in range(self.num_datasets)]

    def __len__(self):
        """ For simplicity just return default len"""
        return self.cumulative_dataset_lengths[-1]


    def __getitem__(self, index):
        """
        Randomly samples from one of the datasets based on predefined ratios.
        """
        rand_val = random.random()

        # Determine which dataset to sample from
        # for i, ratio in enumerate(self.cumulative_ratios):
        #     if rand_val <= ratio:
        #         dataset_idx = i
        #         break
        for i, cumulative_length in enumerate(self.cumulative_dataset_lengths):
            if index < cumulative_length:
                dataset_idx = i
                break

        # Sample from the chosen dataset
        # sample_index = random.randint(0, len(self.datasets[dataset_idx]) - 1)
        data = self.datasets[dataset_idx - 1][index - self.cumulative_dataset_lengths[dataset_idx - 1]]
        return data
    


class VideoDataModule():

    def __init__(self, name, path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers=0, world_size=1, rank=0, grid_size=16, mixed_datasets=None, sampling_ratios=None):
        super().__init__()
        self.name = name
        self.path_dict = path_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.name != "mixed":
            if self.name != "co3d":
                self.class_directory = self.path_dict["class_directory"]
                self.annotations_directory = self.path_dict["annotation_directory"]
                self.meta_file_path = self.path_dict["meta_file_path"]
            elif self.name == "co3d":
                self.mapping_path = self.path_dict["mapping_path"]
                self.zip_mapping_path = self.path_dict["zip_mapping_path"]
        self.num_clip_frames = num_clip_frames
        self.sampling_mode = sampling_mode
        self.regular_step = regular_step
        self.num_clips = num_clips
        self.world_size = world_size
        self.rank = rank
        self.grid_size = grid_size  
        self.sampler = None
        self.data_loader = None
        self.mixed_datasets = mixed_datasets
        self.sampling_ratios = sampling_ratios

    
    def setup(self, transforms_dict):
        data_transforms = transforms_dict["data_transforms"]
        target_transforms = transforms_dict["target_transforms"]
        shared_transforms = transforms_dict["shared_transforms"]
        if self.name == "mixed":
            dataset_instances = []
            for dataset_name in self.mixed_datasets:
                if dataset_name == "mose":
                    regular_step = 1
                else:
                    regular_step = self.regular_step
                if dataset_name == "co3d":
                    zip_mapping_path = self.path_dict[dataset_name]["zip_mapping_path"]
                    mapping_path = self.path_dict[dataset_name]["mapping_path"]
                    dataset_instances.append(CO3DDataset(self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, mapping_path, zip_mapping_path, regular_step))
                else:
                    path_dict = self.path_dict[dataset_name]
                    class_directory = path_dict["class_directory"]
                    annotations_directory = path_dict["annotation_directory"]
                    meta_file_path = path_dict["meta_file_path"]
                    dataset_instances.append(YTVOSTrajectoryDataset(class_directory, annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, meta_file_path, regular_step, self.grid_size))
            self.dataset = MixedDataset(dataset_instances, self.sampling_ratios)
        elif self.name == "timetytvos":
            self.dataset = TimeTYVOSDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "ytvos":
            self.dataset = YVOSDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "ytvos_test":
            self.dataset = YTVOSDatasetTest(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "ytvos_trj":
            self.dataset = YTVOSTrajectoryDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step, self.grid_size)
        elif self.name == "kinetics":
            self.dataset = Kinetics(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "co3d":
            self.dataset = CO3DDataset(self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.mapping_path, self.zip_mapping_path, self.regular_step)
        else:
            self.dataset = VideoDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        print(f"Dataset size : {len(self.dataset)}")
    
    def make_data_loader(self, shuffle=True):
        """
        Create a DataLoader that uses a DistributedSampler when in multi-GPU mode.
        If world_size > 1, each rank gets a subset of the dataset.
        """
        if self.world_size > 1:
            # Create a DistributedSampler so each GPU sees a disjoint split
            self.sampler = DistributedSampler(
                self.dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=shuffle
            )
            # shuffle=False here because the sampler will shuffle internally if shuffle=True
            self.data_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=self.sampler
            )
        else:
            # Single GPU or single process
            self.data_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True
            )
    def get_data_loader(self):
        return self.data_loader


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transform is not None:
            image = self.transform(image)
            target = torch.Tensor([0])

        elif self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

    

class CocoDataModule():

    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 data_dir: str,
                 train_transforms,
                 val_transforms,
                 file_list: List[str],
                 mask_type: str = None,
                 file_list_val: List[str] = None,
                 val_target_transforms=None,
                 shuffle: bool = True,
                 size_val_set: int = 10):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size_val_set = size_val_set
        self.file_list = file_list
        self.file_list_val = file_list_val
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.file_list_val = file_list_val
        self.val_target_transforms = val_target_transforms
        self.mask_type = mask_type
        self.coco_train = None
        self.coco_val = None

    def __len__(self):
        return len(self.file_list)

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        if self.mask_type is None:
            self.coco_train = UnlabelledCoco(self.file_list,
                                             self.train_transforms,
                                             os.path.join(self.data_dir, "train2017"))
            self.coco_val = UnlabelledCoco(self.file_list[:self.size_val_set * self.batch_size],
                                           self.val_transforms,
                                           os.path.join(self.data_dir, "val2017"))
        else:
            self.coco_train = COCOSegmentation(self.data_dir,
                                               self.file_list,
                                               self.mask_type,
                                               image_set="train",
                                               transforms=self.train_transforms)
            self.coco_val = COCOSegmentation(self.data_dir,
                                             self.file_list_val,
                                             self.mask_type,
                                             image_set="val",
                                             transform=self.val_transforms,
                                             target_transform=self.val_target_transforms)

        print(f"Train size {len(self.coco_train)}")
        print(f"Val size {len(self.coco_val)}")
        print(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.coco_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.coco_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)


class COCOSegmentation(VisionDataset):

    def __init__(
            self,
            root: str,
            file_names: List[str],
            mask_type: str,
            image_set: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(COCOSegmentation, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        self.file_names = file_names
        self.mask_type = mask_type
        assert self.image_set in ["train", "val"]
        assert mask_type in ["stuff", "thing"]

        # Set mask folder depending on mask_type
        if mask_type == "thing":
            seg_folder = "annotations/{}2017/"
            json_file = "annotations/panoptic_annotations/panoptic_val2017.json"
        elif mask_type == "stuff":
            seg_folder = "annotations/stuff_annotations/stuff_{}2017_pixelmaps/"
            json_file = "annotations/stuff_annotations/stuff_val2017.json"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_folder = seg_folder.format(image_set)

        # Load categories to category to id map for merging to coarse categories
        with open(os.path.join(root, json_file)) as f:
            an_json = json.load(f)
            all_cat = an_json['categories']
            if mask_type == "thing":
                all_thing_cat_sup = set(cat_dict["supercategory"] for cat_dict in all_cat if cat_dict["isthing"] == 1)
                super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(all_thing_cat_sup))}
                self.cat_id_map = {}
                for cat_dict in all_cat:
                    if cat_dict["isthing"] == 1:
                        self.cat_id_map[cat_dict["id"]] = super_cat_to_id[cat_dict["supercategory"]]
                    elif cat_dict["isthing"] == 0:
                        self.cat_id_map[cat_dict["id"]] = 255
            else:
                super_cats = set([cat_dict['supercategory'] for cat_dict in all_cat])
                super_cats.remove("other")  # remove others from prediction targets as this is not semantic
                super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(super_cats))}
                super_cat_to_id["other"] = 255  # ignore_index for CE
                self.cat_id_map = {cat_dict['id']: super_cat_to_id[cat_dict['supercategory']] for cat_dict in all_cat}

        # Get images and masks fnames
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, "images", f"{image_set}2017")
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir):
            print(seg_dir)
            print(image_dir)
            raise RuntimeError('Dataset not found or corrupted.')
        self.images = [os.path.join(image_dir, x) for x in self.file_names]
        self.masks = [os.path.join(seg_dir, x.replace("jpg", "png")) for x in self.file_names]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        if self.transforms:
            img, mask = self.transforms(img, mask)

        if self.mask_type == "stuff":
            # move stuff labels from {0} U [92, 183] to [0,15] and [255] with 255 == {0, 183}
            # (183 is 'other' and 0 is things)
            mask *= 255
            assert torch.max(mask).item() <= 183
            mask[mask == 0] = 183  # [92, 183]
            assert torch.min(mask).item() >= 92
            for cat_id in torch.unique(mask):
                mask[mask == cat_id] = self.cat_id_map[cat_id.item()]

            assert torch.max(mask).item() <= 255
            assert torch.min(mask).item() >= 0
            mask /= 255
            return img, mask
        elif self.mask_type == "thing":
            mask *= 255
            assert torch.max(mask[mask!=255]).item() <= 200
            mask[mask == 0] = 200  # map unlabelled to stuff
            merged_mask = mask.clone()
            for cat_id in torch.unique(mask):
                cid = int(cat_id.item())
                if cid in self.cat_id_map and cid <= 200:
                    merged_mask[mask == cat_id] = self.cat_id_map[cid]  # [0, 11] + {255}
                else:
                    merged_mask[mask == cat_id] = 255

            assert torch.max(merged_mask).item() <= 255
            assert torch.min(merged_mask).item() >= 0
            merged_mask /= 255
            return img, merged_mask
        return img, mask


class UnlabelledCoco(Dataset):

    def __init__(self, file_list, transforms, data_dir):
        self.file_names = file_list
        self.transform = transforms
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        image = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class Ade20kDataModule():

    def __init__(self,
                 root,
                 train_transforms,
                 val_transforms,
                 shuffle,
                 num_workers,
                 batch_size,
                 val_file_set=None,
                 train_file_set=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.val_file_set = val_file_set
        self.train_file_set = train_file_set

    def setup(self, stage: Optional[str] = None):
        self.val = ADE20K(self.root, self.val_transforms, split='val', file_set=None)
        self.train = ADE20K(self.root, self.train_transforms, split='train', file_set=self.train_file_set)
        print(f"Train size : {len(self.train)}")
        print(f"Val size : {len(self.val)}")
    

    def get_name(self):
        return "Ade20kDataModule"


    def train_dataloader(self):

        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.train,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=self.shuffle,
                drop_last=False
            )
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True, sampler=sampler)

    def val_dataloader(self):
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.val,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=self.shuffle,
                drop_last=False
            )
        return DataLoader(self.val, batch_size=self.batch_size // 4, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True, sampler=sampler)
    
    def test_dataloader(self):
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.val,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=self.shuffle,
                drop_last=False
            )
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True, sampler=sampler)

    def get_train_dataset_size(self):
        return len(self.train)

    def get_val_dataset_size(self):
        return len(self.val)
    
    def get_num_classes(self):
        return 151


class ADE20K(Dataset):
    split_to_dir = {
        'train': 'training',
        'val': 'validation'
    }

    def __init__(self, root, transforms, split='train', skip_other_class=False, file_set=None):
        super().__init__()
        self.transforms = transforms
        self.split = split
        self.root = root
        self.skip_other_class = skip_other_class
        self.file_set = file_set
        self.file_names = []

        if file_set is not None:
            with open(file_set, "r") as f:
                self.file_names = [x.strip() for x in f.readlines()]

        # Collect the data
        self.data = self.collect_data()

    def collect_data(self):
        # Get the image and annotation dirs
        image_dir = os.path.join(self.root, f'images/{self.split_to_dir[self.split]}')
        annotation_dir = os.path.join(self.root, f'annotations/{self.split_to_dir[self.split]}')

        # Collect the filepaths
        if self.file_set is None:
            image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
            annotation_paths = [os.path.join(annotation_dir, f) for f in sorted(os.listdir(annotation_dir))]
        else:
            image_paths = [os.path.join(image_dir, f'{f}.jpg') for f in sorted(self.file_names)]
            annotation_paths = [os.path.join(annotation_dir, f'{f}.png') for f in sorted(self.file_names)]

        data = list(zip(image_paths, annotation_paths))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the  paths
        image_path, annotation_path = self.data[index]

        # Load
        image = Image.open(image_path).convert("RGB")
        target = Image.open(annotation_path)

        # Augment
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            # Convert unwanted class to the class to skip
            # which in our case we always skip the class of 255
        else:
            target = F.pil_to_tensor(target)

        if self.skip_other_class == True:
            target = target * 255.0
            target[target.type(torch.int64)==0]=255.0
            target /= 255.0

        if self.transforms is None:
            target = F.to_pil_image(target)
        
        return image, target
    


def test_pascal_data_module(logger):
    min_scale_factor = 0.5
    max_scale_factor = 2.0
    brightness_jitter_range = 0.1
    contrast_jitter_range = 0.1
    saturation_jitter_range = 0.1
    hue_jitter_range = 0.1

    brightness_jitter_probability = 0.5
    contrast_jitter_probability = 0.5
    saturation_jitter_probability = 0.5
    hue_jitter_probability = 0.5

    # Create the transformation
    image_train_transform = trn.Compose([
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    shared_transform = Compose([
        RandomResizedCrop(size=(448, 448), scale=(min_scale_factor, max_scale_factor)),
        # RandomHorizontalFlip(probability=0.1),
    ])
        
    
    # image_train_transform = trn.Compose([trn.Resize((448, 448)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    # target_train_transform = trn.Compose([trn.Resize((448, 448), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_transform}
    dataset = PascalVOCDataModule(batch_size=4, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms)
    dataset.setup()
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    test_dataloader = dataset.get_test_dataloader()
    print(f"Train size : {len(dataset.train_dataset)}")
    print(f"Val size : {len(dataset.val_dataset)}")
    print(f"Test size : {len(dataset.test_dataset)}")
    print(f"Train dataloader size : {len(train_dataloader)}")
    print(f"Val dataloader size : {len(val_dataloader)}")
    print(f"Test dataloader size : {len(test_dataloader)}")
    for i, (x, y) in enumerate(val_dataloader):
        print(f"Train batch {i} : {x.shape}, {y.shape}")
        ## log image
        classes = torch.unique((y * 255).long())
        print(f"Number of classes : {classes}")
        logger.log({"train_batch": [wandb.Image(x[0]), wandb.Image(y[0])]})
        if i == 10:
            break



def test_video_data_module(logger):
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
    video_transform_list = [
        video_transformations.Resize((224, 224)),
        video_transformations.RandomHorizontalFlip(),
        video_transformations.ClipToTensor(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
        )
    ]
    video_transform = video_transformations.Compose(video_transform_list)
    transformations_dict = {
        "data_transforms": None,
        "target_transforms": None,
        "shared_transforms": video_transform
    }


    num_clips = 1
    batch_size = 1
    num_workers = 4
    num_clip_frames = 4
    regular_step = 1
    grid_size = 16
    prefix = "/ssdstore/ssalehi/"
    data_path = os.path.join(prefix, "co3d_images")
    annotation_path = os.path.join(prefix, "train1/Annotations/")
    meta_file_path = os.path.join(prefix, "train1/meta.json")
    path_dict = {"class_directory": data_path, "annotation_directory": "", "meta_file_path": ""}
    sampling_mode = SamplingMode.Full
    video_data_module = VideoDataModule( #"ytvos_trj", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers, grid_size=args.grid_size
    "ytvos_trj", path_dict,
    num_clips=num_clips,
    num_clip_frames=num_clip_frames,
    sampling_mode=sampling_mode,
    regular_step=regular_step,
    batch_size=batch_size,
    num_workers=num_workers,
    grid_size=grid_size
    )  
    video_data_module.setup(transformations_dict)
    video_data_module.make_data_loader()  # Creates .data_loader
    data_loader = video_data_module.data_loader
    logging_directory = "data_loader_log/"

    if os.path.exists(logging_directory):
        os.system(f'rm -r {logging_directory}')
    os.makedirs(logging_directory)

    for i, train_data in enumerate(data_loader):
        datum, annotations, pred_tracks, pred_visibility = train_data
        print("===========================")
        print("")
        annotations = annotations.squeeze(1)
        datum = datum.squeeze(1)
        datum = denormalize_video(datum)
        print((torch.unique(annotations)))
        print(datum.shape)
        print(annotations.shape)
        visualize_sampled_videos(datum, "data_loader_log/", f"test_{i}.avi")
        # visualize_sampled_videos(annotations, "data_loader_log/", f"test_anotations_{i}.avi")
        # make_seg_maps(datum, annotations, logging_directory, f"test_seg_maps_{i}.avi")



def test_video_dataloader_with_MoSiC_transforms(logger):
    sampling = "dense"
    num_clips = 1
    batch_size = 1
    num_workers = 4
    num_clip_frames = 8
    regular_step = 25
    num_crops = 4
    logging_directory = "data_loader_log/"
    video_transform_list = [video_transformations.RandomResizedCrop((224, 224)), video_transformations.ClipToTensor()] #video_transformations.RandomResizedCrop((224, 224))
    target_transform = video_transformations.Compose(video_transform_list)
    video_transform = video_transformations.TimeTTransform([224, 96], [1, num_crops], [0.35, 0.25], [1., 0.4], 1, 0.01, 1)
    world_size = 1
    transformations_dict = {"data_transforms": video_transform, "target_transforms": target_transform, "shared_transforms": None}
    prefix = "/ssdstore/ssalehi/dataset"
    data_path = os.path.join(prefix, "all_frames/train_all_frames/JPEGImages/")
    annotation_path = "" # os.path.join(prefix, "train1/Annotations/")
    meta_file_path = "" # os.path.join(prefix, "train1/meta.json")
    path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    if sampling == "dense":
        sampling_mode = SamplingMode.DENSE
    elif sampling == "uniform":
        sampling_mode = SamplingMode.UNIFORM
    elif sampling == "full":
        sampling_mode = SamplingMode.FULL
    else:
        raise ValueError("Sampling mode is not valid")
    video_data_module = VideoDataModule("timetytvos", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers, world_size=world_size)
    video_data_module.setup(transformations_dict)
    video_data_module.make_data_loader()
    data_loader = video_data_module.get_data_loader()
    for i, batch in enumerate(data_loader):
        batch_crop_list, label, annotations = batch
        global_crops_1 = batch_crop_list[0]
        annotations = annotations.squeeze(1)
        datum = global_crops_1
        annotations = annotations.squeeze(1)
        datum = datum.squeeze(1)
        # datum = denormalize_video(datum)
        print((torch.unique(annotations)))
        print(datum.shape)
        print(annotations.shape)
        visualize_sampled_videos(datum[0], "data_loader_log/", f"test_{i}.avi")
        local_crop = batch_crop_list[1]
        local_crop = local_crop.squeeze(1)
        print(local_crop.shape)
        # local_crop = denormalize_video(local_crop)
        visualize_sampled_videos(local_crop[0], "data_loader_log/", f"test_local_{i}.avi")
        # visualize_sampled_videos(annotations, "data_loader_log/", f"test_anotations_{i}.avi")
        # make_seg_maps(datum, annotations, logging_directory, f"test_seg_maps_{i}.avi")


def test_co3d_data_module(logger):
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
    video_transform_list = [
        video_transformations.Resize((224, 224)),
        video_transformations.RandomHorizontalFlip(),
        video_transformations.ClipToTensor(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
        )
    ]
    video_transform = video_transformations.Compose(video_transform_list)
    transformations_dict = {
        "data_transforms": None,
        "target_transforms": None,
        "shared_transforms": video_transform
    }


    num_clips = 1
    batch_size = 16
    num_workers = 16
    num_clip_frames = 12
    regular_step = 6
    grid_size = 16
    prefix = "/ssdstore/ssalehi/"
    data_path = os.path.join(prefix, "co3d_images")
    annotation_path = os.path.join(prefix, "train1/Annotations/")
    meta_file_path = os.path.join(prefix, "train1/meta.json")
    path_dict = {"mapping_path": "./all_frames_complete.json", "zip_mapping_path": "./zip_mapping.json"}
    sampling_mode = SamplingMode.Regular
    video_data_module = VideoDataModule( #"ytvos_trj", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers, grid_size=args.grid_size
    "co3d", path_dict,
    num_clips=num_clips,
    num_clip_frames=num_clip_frames,
    sampling_mode=sampling_mode,
    regular_step=regular_step,
    batch_size=batch_size,
    num_workers=num_workers,
    grid_size=grid_size
    )  
    video_data_module.setup(transformations_dict)
    video_data_module.make_data_loader()  # Creates .data_loader
    data_loader = video_data_module.data_loader
    logging_directory = "data_loader_log/"

    if os.path.exists(logging_directory):
        os.system(f'rm -r {logging_directory}')
    os.makedirs(logging_directory)

    for i, train_data in enumerate(data_loader):
        datum, annotations, pred_tracks, pred_visibility = train_data
        print("===========================")
        print("")
        annotations = annotations.squeeze(1)
        datum = datum.squeeze(1)
        datum = denormalize_video(datum)
        print((torch.unique(annotations)))
        print(datum.shape)
        print(annotations.shape)
        visualize_sampled_videos(datum, "data_loader_log/", f"test_{i}.avi")
        # visualize_sampled_videos(annotations, "data_loader_log/", f"test_anotations_{i}.avi")
        # make_seg_maps(datum, annotations, logging_directory, f"test_seg_maps_{i}.avi")
    

if __name__ == "__main__":
    ## init wandb
    logger = wandb.init(project=project_name, mode="offline", group="data_loader", tags="PascalVOCDataModule", job_type="eval")
    ## test data module
    # test_pascal_data_module(logger)
    ## finish wandb
    # logger.finish()

    # get_file_path("/ssdstore/ssalehi/dataset/val1/JPEGImages/")
    test_co3d_data_module(logger)