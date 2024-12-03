import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class FullDatasetCache:
    def __init__(self):
        self.cache = {}

    def __getitem__(self, idx):
        return self.cache.get(idx, (None, None))

    def __setitem__(self, idx, item):
        self.cache[idx] = item

    def clear(self):
        self.cache.clear()

class VideoDataset(Dataset):
    def __init__(self, dataset_folder, transform=None, seed=None, device=torch.device('cpu')):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.device = device
        self.class_folders = sorted([folder for folder in os.listdir(self.dataset_folder) if os.path.isdir(os.path.join(self.dataset_folder, folder))])
        self.class_to_idx = {class_folder: i for i, class_folder in enumerate(self.class_folders)}
        self.video_files, self.labels, self.subjects = self._find_video_files_and_labels()
        self.class_names = self.class_folders

        # Use FullDatasetCache for the entire dataset
        self._cached_videos = FullDatasetCache()

        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Create a mapping from subjects to classes
        self.subject_to_class = self._create_subject_to_class_mapping()

        # Unpack transform and brightness if provided
        if self.transform is not None:
            self.transform, self.consistent_brightness = self.transform
        else:
            self.transform, self.consistent_brightness = None, None

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        self.reset_brightness()  # Reset brightness for each video
        if self._cached_videos[idx][0] is None:
            video_file = self.video_files[idx]
            frames, label = self.load_video(video_file)
            if frames is None or label is None:
                return None, None

            frames_pil = [Image.fromarray(frame) for frame in frames]

             # Apply consistent brightness to all frames
            frames_brightened = self.consistent_brightness(frames_pil)

             # Apply other transformations if available
            if self.transform:
                frames_transformed = [self.transform(frame) for frame in frames_brightened]
            else:
                frames_transformed = frames_brightened


            video_tensor = torch.stack([frame for frame in frames_transformed])

            if label not in self.class_to_idx:
                print(f"Warning: Label '{label}' not found in class folders.")
                return None, None

            label_tensor = torch.tensor(self.class_to_idx[label], dtype=torch.long).to(self.device)

            self._cached_videos[idx] = (video_tensor, label_tensor)

        return self._cached_videos[idx]

    def reset_brightness(self):
        self.consistent_brightness.reset()

    def _find_video_files_and_labels(self):
        video_files = []
        labels = []
        subjects = []
        for class_folder in self.class_folders:
            class_folder_path = os.path.join(self.dataset_folder, class_folder)
            if not os.path.isdir(class_folder_path):
                continue
            for root, _, files in os.walk(class_folder_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() == '.avi':  # Ensure correct extension
                        video_files.append(os.path.join(root, file))
                        labels.append(class_folder)
                        subject = self.extract_subject_from_file(file)
                        subjects.append(subject)
        return video_files, labels, subjects

    def extract_subject_from_file(self, filename):
        """
        Extracts the subject name from the video filename.
        Assumes the subject name is the part after the last underscore.
        """
        subject = filename.split('_')[-1]
        return subject

    def load_video(self, video_file):
        frames = []
        label = None

        try:
            cap = cv2.VideoCapture(video_file)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if len(frames) == 0:
                print(f"No frames found in video {video_file}")
                return None, None
        except Exception as e:
            print(f"An error occurred while loading the video {video_file}: {e}")
            return None, None

        try:
            label = os.path.basename(os.path.dirname(video_file))
            if label not in self.class_folders:
                print(f"Label '{label}' not recognized for video {video_file}")
                return None, None
        except Exception as e:
            print(f"An error occurred while getting the label for the video {video_file}: {e}")
            return None, None

        return frames, label
    def _create_subject_to_class_mapping(self):
        """
        Create a mapping from subjects to their respective class indices.
        """
        subject_to_class = {}
        for idx, subject in enumerate(self.subjects):
            label = self.labels[idx]
            subject_to_class[subject] = self.class_to_idx[label]
        return subject_to_class

    def get_subject_class(self, subject):
        """
        Get the class index for a given subject.
        """
        return self.subject_to_class.get(subject, None)

    def collate_wrapper_with_mask(self, batch):
        videos, labels = zip(*batch)

        if any(video is None for video in videos):
            videos = [video for video in videos if video is not None]
            labels = [label for label in labels if label is not None]

        if not videos:
            return torch.empty(0, 0, 0, 0), torch.empty(0, 0), torch.empty(0, dtype=torch.long)

        max_frames = max(video.shape[0] for video in videos)

        padded_videos = []
        masks = []
        for video in videos:
            pad_size = max_frames - video.shape[0]
            if pad_size > 0:
                padding = torch.zeros((pad_size, *video.shape[1:]), dtype=video.dtype, device=video.device)
                padded_video = torch.cat((video, padding), dim=0)
                mask = torch.cat((torch.ones(video.shape[0], dtype=torch.float32, device=video.device),
                                torch.zeros(pad_size, dtype=torch.float32, device=video.device)), dim=0)
            else:
                padded_video = video
                mask = torch.ones(video.shape[0], dtype=torch.float32, device=video.device)
            padded_videos.append(padded_video)
            masks.append(mask)

        videos_tensor = torch.stack(padded_videos)
        masks_tensor = torch.stack(masks)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return videos_tensor, masks_tensor, labels_tensor
