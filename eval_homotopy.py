import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
sys.path.append(os.getcwd())
from data.dataloader_debug import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from eval_utils import *
from utils.homotopy import *
import plotly.graph_objects as go
import plotly.express as px
from agent_class import Agent
import time

start_time = time.time()

""" setup """
cfg = Config('nuscenes_5sample_agentformer' )
epochs = [cfg.get_last_epoch()]
epoch = epochs[0]


torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=0) if 0 >= 0 and torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available(): torch.cuda.set_device(0)
torch.set_grad_enabled(False)
log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')


model_id = cfg.get('model_id', 'agentformer')
model = model_dict[model_id](cfg)
model.set_device(device)
model.eval()
cp_path = cfg.model_path % epoch
print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
model_cp = torch.load(cp_path, map_location='cpu')
model.load_state_dict(model_cp['model_dict'], strict=False)

""""  #################  """


def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D




""" Get predictions and compute metrics """

split = 'train'
scene_name = 'scene-1077'
plot = False
use_crossing_pairs = False


generator = data_generator(cfg, log, split=split, phase='testing')
scene_preprocessors = generator.sequence
df_modemetrics = pd.DataFrame()

for scene in scene_preprocessors:

    if scene.seq_name == scene_name:

        gt = scene.gt
        pred_frames = scene.pred_frames
        df_scene = Agent.process_data(gt)
        vmax_scene = df_scene.v.max()

        fig = px.line_3d(df_scene, x = 'x', y = 'y', z = 'frame', color = 'agent_id')
        fig.show()

        # find intersecting agent trajectory pairs
        if use_crossing_pairs:
            raise NotImplementedError
        else:
            pass

        # init agent modification classes here; init interpolation functions, and allow for accel/decel rollouts
        agent_dict = {}
        agents_scene = list(df_scene.agent_id.unique())
        for agent in agents_scene:
            df_agent = df_scene[df_scene.agent_id == agent]
            agent_class = Agent(df_agent, v_max = vmax_scene) # impose vmax based on gt scene velocities
            agent_dict.update({agent: agent_class})

        # initialize homotopy matrix for whole scene
        N_features = 15
        homotopy_scene_tensor = torch.full((N_features, len(pred_frames), len(agents_scene), len(agents_scene)), float('nan')) # N_features x frames x agents x agents

        for frame_idx, frame in enumerate(pred_frames):
            # frame corresponds to the current timestep, i.e. the last of pre_motion
            data = scene(frame)
            if data is None:
                print('Frame skipped in loop')
                continue

            seq_name, frame = data['seq'], data['frame']
            frame = int(frame)
            sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
            sys.stdout.flush()

            with torch.no_grad():
                recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
            recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

            # calculate roll-outs
            fut_mod_decel_list = []
            fut_mod_accel_list = []
            for agent_id in data['valid_id']:
                agent = agent_dict[str(int(agent_id))]
                fut_rollout_decel = agent.rollout_future(frame_curr = frame, direction = 'decel')
                fut_rollout_accel = agent.rollout_future(frame_curr = frame, direction = 'accel')
                fut_mod_decel_list.append(fut_rollout_decel)
                fut_mod_accel_list.append(fut_rollout_accel)
            
            fut_mod_decel = torch.from_numpy(np.stack(fut_mod_decel_list)).unsqueeze(0)
            fut_mod_accel = torch.from_numpy(np.stack(fut_mod_accel_list)).unsqueeze(0)

            # get roll-out combinations and check for collisions
            fut_mod_rollout_combinations = get_rollout_combinations(fut_mod_decel, fut_mod_accel)
            collision_matrix, min_distances, _, _ = calc_collision_matrix(fut_mod_rollout_combinations)

            # get gt and distances
            fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
            fut_motion_batch = torch.from_numpy(fut_motion).unsqueeze(0)
            _, min_distances_gt, min_indices_gt, distances_gt = calc_collision_matrix(fut_motion_batch)
            converging_trajectories_bool = (min_indices_gt > 0).squeeze(0) # minimum distance not at current frame -> converging trajectory


            # calculate homotopy_classes: gt, pred, roll-outs
            interaction_bool, gt_distance_matrix = calc_intersections(fut_motion_batch)
            homotopy_gt = calc_path_homotopy(fut_motion_batch, agents_scene)
            homotopy_pred = calc_path_homotopy(sample_motion_3D, agents_scene)
            homotopy_mod = calc_path_homotopy(fut_mod_rollout_combinations, agents_scene)

            # calculate mode correctness/coverage predictions:
            modes_correct_matrix = (homotopy_gt==homotopy_pred[0,:,:]) # first entry corresponds to ML mode.
            k_sample = homotopy_pred.shape[0]
            modes_covered_matrix = (homotopy_gt.repeat(k_sample, 1, 1)==homotopy_pred[:,:,:]).max(axis=0).values
            # modes_correct = modes_correct_matrix.all().item()
            # modes_covered = modes_covered_matrix.all().item()

            # claculate feasible homotopy classes
            homotopy_mod_feasible = homotopy_mod.clone()
            homotopy_mod_feasible[collision_matrix] = np.nan
            homotopy_mod_feasible_0 = (homotopy_mod_feasible == 0.0).any(dim=0)
            homotopy_mod_feasible_1 = (homotopy_mod_feasible == 1.0).any(dim=0)
            homotopy_mod_feasible_2 = (homotopy_mod_feasible == 2.0).any(dim=0)
            homotopy_mod_feasible_Nclasses = homotopy_mod_feasible_0*0 + homotopy_mod_feasible_1*1 + homotopy_mod_feasible_2*1

            # calculate predited homotopy classes
            homotopy_pred_0 = (homotopy_pred == 0.0).any(dim=0)
            homotopy_pred_1 = (homotopy_pred == 1.0).any(dim=0)
            homotopy_pred_2 = (homotopy_pred == 2.0).any(dim=0)
            homotopy_pred_Nclasses = homotopy_pred_0*0 + homotopy_pred_1*1 + homotopy_pred_2*1

            # mode-collapse == a possible (a2a) homotopy class is feasible, but not predicted. Tested by checking the number of predicted/feasible classes
            # TODO: properly check if feasible class is predicted
            a2a_mode_collapse = (homotopy_pred_Nclasses < homotopy_mod_feasible_Nclasses) * converging_trajectories_bool
            # TODO: properly check if final class is gt class
            homotopy_class_final = homotopy_mod_feasible_Nclasses <= 1

            assert( len(homotopy_class_final) > 0)

            # append al information to big matrix:
            agent_idx = torch.tensor([agents_scene.index(str(int(agent_id))) for agent_id in data['valid_id']])
            homotopy_scene_tensor[0, frame_idx, agent_idx[:, None], agent_idx] = a2a_mode_collapse.to(torch.float32)
            homotopy_scene_tensor[1, frame_idx, agent_idx[:, None], agent_idx] = homotopy_class_final.to(torch.float32)
            homotopy_scene_tensor[2, frame_idx, agent_idx[:, None], agent_idx] = modes_correct_matrix.to(torch.float32)
            homotopy_scene_tensor[3, frame_idx, agent_idx[:, None], agent_idx] = modes_covered_matrix.to(torch.float32)
            homotopy_scene_tensor[4, frame_idx, agent_idx[:, None], agent_idx] = converging_trajectories_bool.to(torch.float32)
            homotopy_scene_tensor[5, frame_idx, agent_idx[:, None], agent_idx] = interaction_bool.to(torch.float32)
            homotopy_scene_tensor[6, frame_idx, agent_idx[:, None], agent_idx] = gt_distance_matrix.to(torch.float32)

            # add gt, ml, covered and feasible modes
            homotopy_scene_tensor[7, frame_idx, agent_idx[:, None], agent_idx] = homotopy_gt.to(torch.float32)
            homotopy_scene_tensor[8, frame_idx, agent_idx[:, None], agent_idx] = homotopy_pred[0,:,:].to(torch.float32) #ML
            homotopy_scene_tensor[9, frame_idx, agent_idx[:, None], agent_idx] = homotopy_pred_0.to(torch.float32)
            homotopy_scene_tensor[10, frame_idx, agent_idx[:, None], agent_idx] = homotopy_pred_1.to(torch.float32)
            homotopy_scene_tensor[11, frame_idx, agent_idx[:, None], agent_idx] = homotopy_pred_2.to(torch.float32)
            homotopy_scene_tensor[12, frame_idx, agent_idx[:, None], agent_idx] = homotopy_mod_feasible_0.to(torch.float32)
            homotopy_scene_tensor[13, frame_idx, agent_idx[:, None], agent_idx] = homotopy_mod_feasible_1.to(torch.float32)
            homotopy_scene_tensor[14, frame_idx, agent_idx[:, None], agent_idx] = homotopy_mod_feasible_2.to(torch.float32)




            if plot:
                data['scene_map'].visualize_trajs(data, sample_motion_3D)
                # also plot roll outs here
                data['scene_map'].visualize_trajs(data, fut_mod_rollout_combinations)


        df_modemetrics_scene = calc_scene_mode_metrics(homotopy_scene_tensor, agents_scene)
        df_modemetrics_scene['scene'] = seq_name
        # append to main df:
        df_modemetrics = pd.concat([df_modemetrics, df_modemetrics_scene])


end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")

# save result:
print(df_modemetrics)
df_modemetrics.to_csv(f'mode_metrics_{split}.csv', index = False)