import os

import torch
from torch.utils.data import Dataset

from utils.general_utils import matrix_to_quaternion, superimpose_overlay


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


class MaskedDataset(SharedDataset):
    def __init__(self, cfg, dataset, return_superimposed_input=False) -> None:
        super().__init__()
        self.cfg = cfg
        self._ret_input_data = return_superimposed_input

        self.data = dataset
        self.reshuffle()

    def reshuffle(self):
        self.overlay_map = torch.randperm(len(self.data))

    def __len__(self):
        return len(self.data)

    def get_example_id(self, index):
        return self.data.get_example_id(index)

    def __getitem__(self, index):
        object_data = self.data[index]
        overlay_data = self.data[self.overlay_map[index]]

        if self._ret_input_data:
            if len(object_data["gt_images"].shape) == 4:
                input_data = superimpose_overlay(
                    object_data["gt_images"][None, : self.cfg.data.input_images, ...],
                    overlay_data["gt_images"][None, : self.cfg.data.input_images, ...],
                )[0, ...]
            else:
                input_data = superimpose_overlay(
                    object_data["gt_images"][:, : self.cfg.data.input_images, ...],
                    overlay_data["gt_images"][:, : self.cfg.data.input_images, ...],
                )
            return object_data, overlay_data, input_data
        else:
            return object_data, overlay_data
