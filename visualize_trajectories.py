import numpy as np
import argparse
import os
import sys
import subprocess
import shutil

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

DATAROOT = '/home/maarten/Documents/NuScenes_mini'

sys.path.append(DATAROOT)

def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

def save_prediction(pred, data, suffix, save_dir):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = pred[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num

def modify_traj(traj, dv, dt):
    pass

def modify_scene(data, dv = 0, dt = 0):
    """
    keep paths, change velocity / position (with time shift, more intuitive) of ego agent only

    TODO:
    - function to actually change a trajectory, given a trajectory and a dv/dt 
    - account for traj_scale  
    - change all fields of data 
        - recalculate heading at prediction time 
    """
    idx_ego = data['valid_id'].index(99)

    # transform data to real scale and get full trajectory:
    pre_motion = np.stack(data['pre_motion_3D']) * data['traj_scale']
    fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
    all_motion = np.concatenate((pre_motion, fut_motion), axis=1)

def test_model(generator, save_dir, cfg):
    total_num_pred = 0
    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        sys.stdout.flush()


        #### put stuff here to get lane ids: # make seperate data modify function!!!
        # nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
        # scene_token = nuscenes.field2token('scene', 'name', seq_name)[0]
        # scene = nuscenes.get('scene', scene_token)
        # scene_log_token = scene['log_token']
        # log_data = nuscenes.get('log', scene_log_token)
        # location = log_data['location']

        # nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=location)
        # lane_ids = nusc_map.get_closest_lane(x = 500, y = 1740, radius  =5)


        # modify_scene(data)

        if seq_name in SCENES:
            # gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
            with torch.no_grad():
                recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
                assert (sample_motion_3D[0] == recon_motion_3D).min().item() # check that ML is at idx 0 
            recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

            # implement plotting here
            # if data['seq'] == 'scene-0553':
            data['scene_map'].visualize_trajs(data, sample_motion_3D)
            # print()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    args = parser.parse_args()

    

    """ setup """
    cfg = Config('nuscenes_5sample_agentformer' )
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=0) if 0 >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = ['train']
        SCENES = ['scene-1100']

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples'
            if not args.cached:
                test_model(generator, save_dir, cfg)

            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
            subprocess.run(cmd.split(' '))



