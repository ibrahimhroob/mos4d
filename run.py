#!/usr/bin/env python3
# @file      run.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2023 Benedikt Mersch, all rights reserved
#!/usr/bin/env python

import os
import click
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from mos4d import MOS4DNet
from dataloader import framesDataLoader as fdl

# define some constants
DATASET = 'june_1_ABEF'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option(
    "--path",
    "-p",
    type=str,
    help="Path to checkpoint",
    default="./checkpoints/10_scans.ckpt",
)
@click.option(
    "--voxel_size",
    "-v",
    type=float,
    help="Voxel size for 4DMOS",
    default=0.1,
)
@click.option(
    "--visualize",
    "-vis",
    help="visualize",
    default=True,
    is_flag=True,
)
@click.option(
    "--window_size",
    "-w",
    type=int,
    help="Frames registration window size for 4DMOS",
    default=10,
)
def main(path, voxel_size=0.1, visualize=True, window_size=10):
    ############ IH dataloader ############
    dataset_dir = os.path.join(BASE_DIR, 'data', DATASET)
    frames_path = os.path.join(dataset_dir, 'frames')
    tf_matrices_pth = os.path.join(dataset_dir, 'tf_matrices')
    
    print("Start loading test data ...")
    TEST_DATA = fdl(frames_path, tf_matrices_pth, window_size)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATA, batch_size=1, shuffle=False, num_workers=2,
                                                 pin_memory=True, drop_last=False)

    # Load model
    state_dict = {
        k.replace("model.MinkUNet.", ""): v
        for k, v in torch.load(path)["state_dict"].items()
    }
    state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
    model = MOS4DNet(voxel_size)
    model.MinkUNet.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    model.freeze()

    # # create vis dir
    vis_dir = os.path.join(BASE_DIR, 'vis')

    # # Create the directory
    Path(vis_dir).mkdir(parents=True, exist_ok=True)

    for i, (points) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0):
        points = points.squeeze().to(torch.float32)
        # Add batch index and pass through model
        coordinates = torch.hstack(
            [
                torch.zeros(len(points)).reshape(-1, 1).type_as(points),
                points,
            ]
        ).cuda()
        predicted_logits = model.forward(coordinates)

        points[:,3] = predicted_logits > 0

        '''
        For testing and ease of debugging, the frames are saved for visualization later.
        To view the frames you can use cloudCompare, where the data structure is as follows:
            X Y Z Scalar 
        '''
        np.savetxt(os.path.join(vis_dir, str(i) + '.asc'), points.cpu().numpy(), fmt='%.3f')

if __name__ == "__main__":
    main()
