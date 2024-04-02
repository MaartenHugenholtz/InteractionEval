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

split = 'val'
plot = False
use_crossing_pairs = False


generator = data_generator(cfg, log, split=split, phase='testing')
scene_preprocessors = generator.sequence

for scene in scene_preprocessors:

    gt = scene.gt
    pred_frames = scene.pred_frames
    df_scene = Agent.process_data(gt)

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
        agent_class = Agent(df_agent)
        agent_dict.update({agent: agent_class})

    # initialize homotopy matrix for whole scene
    N_features = 6
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
        converging_trajectories_bool = min_indices_gt > 0 # minimum distance not at current frame -> converging trajectory
        calc_path_homotopy(fut_motion_batch)


        # calculate homotopy_classes: gt, pred, roll-outs
        angle_diff_gt, homotopy_gt = identify_pairwise_homotopy(fut_motion_batch)
        angle_diff_pred, homotopy_pred = identify_pairwise_homotopy(sample_motion_3D)
        angle_diff_mod, homotopy_mod = identify_pairwise_homotopy(fut_mod_rollout_combinations)

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
        homotopy_mod_feasible_Nclasses = homotopy_mod_feasible_0*1 + homotopy_mod_feasible_1*1 + homotopy_mod_feasible_2*1

        # calculate predited homotopy classes
        homotopy_pred_0 = (homotopy_pred == 0.0).any(dim=0)
        homotopy_pred_1 = (homotopy_pred == 1.0).any(dim=0)
        homotopy_pred_2 = (homotopy_pred == 2.0).any(dim=0)
        homotopy_pred_Nclasses = homotopy_pred_0*1 + homotopy_pred_1*1 + homotopy_pred_2*1

        # mode-collapse == a possible (a2a) homotopy class is feasible, but not predicted. Tested by checking the number of predicted/feasible classes
        # TODO: properly check if feasible class is predicted
        a2a_mode_collapse = (homotopy_pred_Nclasses < homotopy_mod_feasible_Nclasses) * converging_trajectories_bool
        # TODO: properly check if final class is gt class
        homotopy_class_final = homotopy_mod_feasible_Nclasses == 1

        # append al information to big matrix:
        agent_idx = torch.tensor([agents_scene.index(str(int(agent_id))) for agent_id in data['valid_id']])
        homotopy_scene_tensor[0, frame_idx, agent_idx[:, None], agent_idx] = a2a_mode_collapse[0,...].to(torch.float32)
        homotopy_scene_tensor[1, frame_idx, agent_idx[:, None], agent_idx] = homotopy_class_final[0,...].to(torch.float32)
        homotopy_scene_tensor[2, frame_idx, agent_idx[:, None], agent_idx] = modes_correct_matrix[0,...].to(torch.float32)
        homotopy_scene_tensor[3, frame_idx, agent_idx[:, None], agent_idx] = modes_covered_matrix[0,...].to(torch.float32)
        homotopy_scene_tensor[4, frame_idx, agent_idx[:, None], agent_idx] = converging_trajectories_bool[0,...].to(torch.float32)



        if plot:
            data['scene_map'].visualize_trajs(data, sample_motion_3D)
            # also plot roll outs here
            data['scene_map'].visualize_trajs(data, fut_mod_rollout_combinations)


    calc_scene_mode_metrics(homotopy_scene_tensor)


        # if a2a_mode_collapse.any().item():
        #     # show mode-collapse:
        #     # indices, possible classes, show predictions and collapsed mode:
        #     collapsed_pairs = torch.nonzero(torch.triu(a2a_mode_collapse)[0,...], as_tuple = True)
        #     for pair in collapsed_pairs:
        #         i0, i1 = pair[0].item(), pair[1].item()
        #         print(f"Mode-collapse at frame {frame}, between agents {data['valid_id'][i0]}, {data['valid_id'][i1]}")
        #         print(f"Predicted classes (0, 1, 2): {homotopy_pred_0[i0,i1].item(), homotopy_pred_1[i0,i1].item(), homotopy_pred_2[i0,i1].item()}")
        #         print(f"Feasible classes (0, 1, 2): {homotopy_mod_feasible_0[i0,i1].item(), homotopy_mod_feasible_1[i0,i1].item(), homotopy_mod_feasible_2[i0,i1].item()}")


            # data['scene_map'].visualize_trajs(data, sample_motion_3D)
            # # also plot roll outs here
            # data['scene_map'].visualize_trajs(data, fut_mod_rollout_combinations)

        #NEXT: put every thing in giant matrix with nans and post process results?

        # pred_frame_agents = data['valid_id']