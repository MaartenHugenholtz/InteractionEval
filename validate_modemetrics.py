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
import plotly.io as pio
from agent_class import Agent
import time
from tqdm import tqdm


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



df_modemetrics = pd.read_csv('mode_metrics_val.csv')
# calculate metrics:
df_modemetrics['r_t2cor'] = 100 * df_modemetrics['t2cor'] / df_modemetrics['time_pred']
df_modemetrics['r_t2cov'] = 100 * df_modemetrics['t2cov'] / df_modemetrics['time_pred']
df_modemetrics['r_mode_collapse'] = 100 * df_modemetrics['r_mode_collapse']
df_modemetrics['r_t2cor'] = df_modemetrics['r_t2cor'].round(1)
df_modemetrics['r_t2cov'] = df_modemetrics['r_t2cov'].round(1)
df_modemetrics['r_mode_collapse'] = df_modemetrics['r_mode_collapse'].round(1)


split = 'val'
save_pred_imgs_path = f'pred_imgs_{split}'
plot = True
save_imgs = False

scene_focus_name = 'scene-0103'

generator = data_generator(cfg, log, split=split, phase='testing')
scene_preprocessors = generator.sequence
scene_names = [scene_preprocessors[s].seq_name for s in range(len(scene_preprocessors))]


for idx, row in df_modemetrics.iterrows():

    # get relevant interaction variables
    scene_name = row['scene']
    scene = scene_preprocessors[scene_names.index(scene_name)]
    h_final_frame = row['h_final_frame']
    cmp_frame = h_final_frame - row['t2cor']*2 # correct mode prediction frame
    
    focus_agents = (row['agent1'], row['agent2'])

    if scene_name == scene_focus_name:

        gt = scene.gt
        pred_frames = scene.pred_frames
        df_scene = Agent.process_data(gt)
        vmax_scene = df_scene.v.max()

        # px.scatter(df_scene, x = 'x', y = 'y', hover_data = ['frame', 'lane_num'], color = 'agent_id').show()
        # fig = px.line_3d(df_scene, x = 'x', y = 'y', z = 'frame', color = 'agent_id')
        # fig.show()

        # init agent modification classes here; init interpolation functions, and allow for accel/decel rollouts
        agent_dict = {}
        agents_scene = list(df_scene.agent_id.unique()) # definitive order for agent ids in all tensors
        for agent in agents_scene:
            df_agent = df_scene[df_scene.agent_id == agent]
            agent_class = Agent(df_agent, v_max = vmax_scene) # impose vmax based on gt scene velocities
            agent_dict.update({agent: agent_class})

        # path_intersection_bool, inframes_bool,_ = calc_path_intersections(df_scene, agents_scene, pred_frames)
        focus_frame = np.arange(cmp_frame - 1, h_final_frame + 1 + 1+5, 1)
        # focus_frame = [h_final_frame] # just visualize final frame

        for frame_idx, frame in enumerate(pred_frames):
            if frame in focus_frame:
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

                # calculate roll-outs for possible agent pairs:

                fut_mod_decel_list = []
                fut_mod_accel_list = []
                for agent_id in focus_agents:
                    agent = agent_dict[str(int(agent_id))]
                    fut_rollout_decel = agent.rollout_future(frame_curr = frame, direction = 'decel')
                    fut_rollout_accel = agent.rollout_future(frame_curr = frame, direction = 'accel')
                    fut_mod_decel_list.append(fut_rollout_decel)
                    fut_mod_accel_list.append(fut_rollout_accel)
                
                fut_mod_decel = torch.from_numpy(np.stack(fut_mod_decel_list)).unsqueeze(0)
                fut_mod_accel = torch.from_numpy(np.stack(fut_mod_accel_list)).unsqueeze(0)

                # get roll-out combinations and check for collisions
                fut_mod_rollout_combinations = get_rollout_combinations(fut_mod_decel, fut_mod_accel)
                fut_mod_rollout_combinations = fut_mod_rollout_combinations[0:2] # first two entries already contain all possible combinations (only for 2 agent case!)
                collision_matrix, min_distances, _, _ = calc_collision_matrix(fut_mod_rollout_combinations)

                fig = data['scene_vis_map'].visualize_interactionpair(data, sample_motion_3D, fut_mod_rollout_combinations, collision_matrix, focus_agents)

                # add title and save:
                title = f"{row['scene']} frame {frame}, h_final_frame: {row['h_final_frame']}, t2cor: {row['t2cor']}s, t2cov: {row['t2cov']}s, r_t2cor: {row['r_t2cor']}%, r_t2cov: {row['r_t2cov']}%, pred_consistency: {row['pred_consistency']}, r_mode_collapse: {row['r_mode_collapse']}%"
                fig.update_layout(
                    title=dict(text = title),
                )
                if plot:
                    fig.show()
                if save_imgs:
                    pio.write_image(fig, save_pred_imgs_path + f'/{scene_name}_{focus_agents[0]}_{focus_agents[1]}.png',width=1200, height=1000)
                


end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")
