import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cotracker.models.build_cotracker import build_cotracker
from cotracker.models.core.cotracker.cotracker import get_points_on_a_grid

from sam_pt.point_tracker.tracker import PointTracker
from sam_pt.utils.util import log_video_to_wandb


class CoTrackerPointTracker(PointTracker):
    """
    The class implements a Point Tracker using the CoTracker model from https://arxiv.org/abs/2307.07635.
    """

    def __init__(self, checkpoint_path, interp_shape, visibility_threshold,
                 support_grid_size, support_grid_every_n_frames, add_debug_visualisations):
        """
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file of the pre-trained model.
        interp_shape : int or tuple
            The shape of the interpolation kernel used in the tracker.
        visibility_threshold : float
            The visibility threshold. Points with a visibility score below this threshold are marked as occluded.
        support_grid_size : int or tuple
            The size of the support grid for the tracker.
        support_grid_every_n_frames : int
            Add a support grid every n frames.
        add_debug_visualisations : bool
            If True, debug visualisations will be added to the output.
        """

        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.interp_shape = interp_shape
        self.visibility_threshold = visibility_threshold
        self.support_grid_size = support_grid_size
        self.support_grid_every_n_frames = support_grid_every_n_frames
        self.add_debug_visualisations = add_debug_visualisations

        print(f"Loading CoTracker model from {self.checkpoint_path}")
        self.model = build_cotracker(self.checkpoint_path)

        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

    @property
    def device(self):
        return self.model.norm.weight.device

    def forward(self, rgbs, query_points):
        query_points = query_points.float()
        rgbs = rgbs.float()

        n_masks, n_points, _ = query_points.shape
        batch_size, n_frames, channels, height, width = rgbs.shape
        assert query_points.shape[2] == 3

        if self.interp_shape is None:
            self.interp_shape = (height, width)

        rgbs = rgbs.reshape(batch_size * n_frames, channels, height, width)
        rgbs = F.interpolate(rgbs, tuple(self.interp_shape), mode="bilinear").to(self.device)
        rgbs = rgbs.reshape(batch_size, n_frames, channels, self.interp_shape[0], self.interp_shape[1]).to(self.device)

        query_points = query_points.clone()
        query_points[:, :, 1] *= self.interp_shape[1] / width
        query_points[:, :, 2] *= self.interp_shape[0] / height

        for i in range(0, n_frames, self.support_grid_every_n_frames):
            grid_pts = get_points_on_a_grid(self.support_grid_size, self.interp_shape)
            grid_pts = torch.cat([i * torch.ones_like(grid_pts[:, :, :1]), grid_pts], dim=2)
            query_points = torch.cat([query_points, grid_pts], dim=1)

        raw_trajectories, _, raw_visibilities, _ = self.model(rgbs=rgbs, queries=query_points, iters=6)
        raw_trajectories, raw_visibilities = \
            self._compute_backward_tracks(rgbs, query_points, raw_trajectories, raw_visibilities)

        if self.add_debug_visualisations:
            video_idx = 0
            fps = 5
            annot_size = 6
            annot_line_width = 2
            print(f"n_points={n_points}")
            print(f"self.visibility_threshold={self.visibility_threshold}")
            print(f"raw_trajectories.shape={raw_trajectories.shape}")
            print(f"raw_visibilities.shape={raw_visibilities.shape}")
            frames_with_trajectories = rgbs[video_idx].permute(0, 2, 3, 1).cpu().numpy()
            frames_with_trajectories = np.ascontiguousarray(frames_with_trajectories, dtype=np.uint8)
            for frame_idx in range(n_frames):
                for i, (point, vis) in enumerate(zip(
                        raw_trajectories[video_idx, frame_idx],
                        raw_visibilities[video_idx, frame_idx] > self.visibility_threshold,
                )):
                    x, y = int(point[0]), int(point[1])
                    c = (0, 255, 0) if vis else (255, 0, 0)
                    frames_with_trajectories[frame_idx] = cv2.circle(
                        frames_with_trajectories[frame_idx], (x, y), annot_size, c, annot_line_width,
                    )
                    frames_with_trajectories[frame_idx] = cv2.putText(
                        frames_with_trajectories[frame_idx], f"{i:03}", (int(point[0]), int(point[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                        color=(250, 225, 100)
                    )
            log_video_to_wandb("debug/cotracker-trajectories", frames_with_trajectories, fps=fps)

        trajectories = raw_trajectories[:, :, :n_points].clone()
        visibilities = raw_visibilities[:, :, :n_points].clone()

        visibilities = visibilities > self.visibility_threshold

        trajectories[:, :, :, 0] *= width / float(self.interp_shape[1])
        trajectories[:, :, :, 1] *= height / float(self.interp_shape[0])

        return trajectories, visibilities

    def _compute_backward_tracks(self, rgbs, query_points, trajectories, visibilities):
        rgbs_flipped = rgbs.flip(1).clone()
        query_points_flipped = query_points.clone()
        query_points_flipped[:, :, 0] = rgbs_flipped.shape[1] - query_points_flipped[:, :, 0] - 1

        trajectories_flipped, _, visibilities_flipped, _ = self.model(
            rgbs=rgbs_flipped, queries=query_points_flipped, iters=6
        )

        trajectories_flipped = trajectories_flipped.flip(1)
        visibilities_flipped = visibilities_flipped.flip(1)

        mask = trajectories == 0

        trajectories[mask] = trajectories_flipped[mask]
        visibilities[mask[:, :, :, 0]] = visibilities_flipped[mask[:, :, :, 0]]
        return trajectories, visibilities
