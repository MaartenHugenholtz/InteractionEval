import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from AF_model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing



""" MODEL """
cfg = Config('nuscenes_5sample_agentformer' )

H_PRED = 12 # frames (at 2 Hz)
cfg.future_frames  = H_PRED  # overwrite H_pred in config!


""""""" SETUP """""""
torch.set_default_dtype(torch.float32)
device =  torch.device('cpu')
torch.set_grad_enabled(False)
log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

epochs = [cfg.get_last_epoch()]
epoch = epochs[0]
model_id = cfg.get('model_id', 'agentformer')
model = model_dict[model_id](cfg)
model.set_device(device)
model.eval()
cp_path = cfg.model_path % epoch
print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
model_cp = torch.load(cp_path, map_location='cpu')
model.load_state_dict(model_cp['model_dict'], strict=False)

def get_model_prediction(data, sample_k = 5):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D