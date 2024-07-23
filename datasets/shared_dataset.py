import os
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.general_utils import matrix_to_quaternion, compose_input_view


class SharedDataset(Dataset):
    """
    Parent dataset class with shared functions
    """

    def __init__(self) -> None:
        super().__init__()

    def make_poses_relative_to_first(self, images_and_camera_poses):
        inverse_first_camera = (
            images_and_camera_poses["world_view_transforms"][0].inverse().clone()
        )
        for c in range(images_and_camera_poses["world_view_transforms"].shape[0]):
            images_and_camera_poses["world_view_transforms"][c] = torch.bmm(
                inverse_first_camera.unsqueeze(0),
                images_and_camera_poses["world_view_transforms"][c].unsqueeze(0),
            ).squeeze(0)
            images_and_camera_poses["view_to_world_transforms"][c] = torch.bmm(
                images_and_camera_poses["view_to_world_transforms"][c].unsqueeze(0),
                inverse_first_camera.inverse().unsqueeze(0),
            ).squeeze(0)
            images_and_camera_poses["full_proj_transforms"][c] = torch.bmm(
                inverse_first_camera.unsqueeze(0),
                images_and_camera_poses["full_proj_transforms"][c].unsqueeze(0),
            ).squeeze(0)
            images_and_camera_poses["camera_centers"][c] = images_and_camera_poses[
                "world_view_transforms"
            ][c].inverse()[3, :3]
        return images_and_camera_poses

    def get_source_cw2wT(self, source_cameras_view_to_world):
        # Compute view to world transforms in quaternion representation.
        # Used for transforming predicted rotations
        qs = []
        for c_idx in range(source_cameras_view_to_world.shape[0]):
            qs.append(
                matrix_to_quaternion(
                    source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)
                )
            )
        return torch.stack(qs, dim=0)


class WholePartialDataset(SharedDataset):
    def __init__(
        self,
        cfg,
        dataset,
        return_input_view=False,
        shuffle=True,
        empty_overlay=False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.return_composed_input_view: bool = return_input_view
        self.no_overlay: bool = empty_overlay

        self.data = dataset
        self._shuffle = shuffle
        self.shuffle()

    def shuffle(self, seed=None):
        if self._shuffle:
            if seed is not None:
                rng = np.random.default_rng(seed=seed)
                self.map = torch.from_numpy(rng.permutation(len(self.data)))
                rng = np.random.default_rng(seed=rng.integers(2147483647, size=1))
                self.overlay_map = torch.from_numpy(rng.permutation(len(self.data)))
            else:
                self.map = torch.randperm(len(self.data))
                self.overlay_map = torch.randperm(len(self.data))
        else:
            self.map = torch.arange(start=0, end=len(self.data))
            rng = np.random.default_rng(seed=32 if seed is None else seed)
            self.overlay_map = torch.from_numpy(rng.permutation(len(self.data)))

    def __len__(self):
        return len(self.data)

    def get_example_id(self, index):
        return self.data.get_example_id(self.map[index])

    def __getitem__(self, index):
        object_data = self.data[self.map[index]]
        if not self.no_overlay:
            occlusion_data = self.data[self.overlay_map[index]]["gt_images"]

        if self.return_composed_input_view and not self.no_overlay:
            if len(object_data["gt_images"].shape) == 5:
                input_view_data = compose_input_view(
                    object_data["gt_images"][:, : self.cfg.data.input_images, ...],
                    occlusion_data[:, : self.cfg.data.input_images, ...],
                )
            elif len(object_data["gt_images"].shape) == 4:
                input_view_data = compose_input_view(
                    object_data["gt_images"][None, : self.cfg.data.input_images, ...],
                    occlusion_data[None, : self.cfg.data.input_images, ...],
                )[0, ...]
            else:
                raise RuntimeError("Expected data dimension to be 4 or 5")
            return object_data, occlusion_data, input_view_data
        elif self.return_composed_input_view and self.no_overlay:
            if len(object_data["gt_images"].shape) == 5:
                return (
                    object_data,
                    torch.zeros_like(object_data["gt_images"]),
                    object_data["gt_images"][:, : self.cfg.data.input_images, ...],
                )
            elif len(object_data["gt_images"].shape) == 4:
                return (
                    object_data,
                    torch.zeros_like(object_data["gt_images"]),
                    object_data["gt_images"][: self.cfg.data.input_images, ...],
                )
            else:
                raise RuntimeError("Expected data dimension to be 4 or 5")

        elif not self.return_composed_input_view and not self.no_overlay:
            return object_data, occlusion_data
        elif not self.return_composed_input_view and self.no_overlay:
            return object_data, torch.zeros_like(object_data["gt_images"])
