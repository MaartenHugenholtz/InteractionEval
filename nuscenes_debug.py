# Init NuScenes. Requires the dataset to be stored on disk.
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import sys
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

DATAROOT = '/home/maarten/Documents/NuScenes_mini'

sys.path.append(DATAROOT)

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

# Render ego poses.
nusc_map_bos = NuScenesMap(dataroot=DATAROOT, map_name='boston-seaport')
# map_poses, fig, ax = nusc_map_bos.render_egoposes_on_fancy_map(nusc, scene_tokens=[nusc.scene[1]['token']], verbose=False)
lane_id = nusc_map_bos.get_closest_lane(x = 500, y = 1740, radius  =500)
print(lane_id)

# plt.show()