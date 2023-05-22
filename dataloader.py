import os
import torch
import numpy as np 
from tqdm import tqdm
from torch.utils.data import Dataset

'''
Note for the current data loader I am using the discrete window approach, however
for future and real time data this need to be implemented as a sliding window approach
'''
class framesDataLoader(Dataset):
    def __init__(self, frames_path, tf_matrices_pth, window_size = 10) -> None:
        super().__init__()
        self.window_size = window_size
        self.transformed_frames = []
        
        frames = sorted(os.listdir(frames_path))    
        tf_matrices  = sorted(os.listdir(tf_matrices_pth))

        assert len(frames) == len(tf_matrices), "frames and tf_matrices have different lengths"

        # To reduce the number of frames for testing 
        # frames = frames[0:50]

        # Create a tqdm progress bar
        for i in tqdm(range(len(frames))):
            frame_pth = os.path.join(frames_path, frames[i])
            traj_pth = os.path.join(tf_matrices_pth, tf_matrices[i])

            points = np.loadtxt(frame_pth)
            transformation_matrix = np.loadtxt(traj_pth, delimiter=',')
            points[:, :3] = self.transform(points[:, :3], transformation_matrix)

            self.transformed_frames.append(points)
        
        print(f"Total samples {len(self.transformed_frames) // self.window_size}.")

    def __getitem__(self, idx: int):
        points_mos = [
            torch.hstack([torch.tensor(self.transformed_frames[ idx * self.window_size + i])])
            for i in range(self.window_size)
        ]
        points_mos = torch.cat(points_mos)

        return points_mos

    def __len__(self) -> int:
        return len(self.transformed_frames) // self.window_size
    
    def transform(self, point_cloud, transformation_matrix):
        # Add homogeneous coordinates to the point cloud
        homogeneous_coordinates = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

        # Apply the transformation
        transformed_points = np.dot(transformation_matrix, homogeneous_coordinates.T).T[:, :3]

        return transformed_points
