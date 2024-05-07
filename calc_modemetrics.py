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


df_modemetrics = pd.DataFrame()
df_interactions = pd.read_csv('interaction_metrics_val.csv')


split = 'val'
save_pred_imgs_path = f'pred_imgs_{split}'
plot = False
plot_all = False
save_imgs = True

focus_scene_bool = True
scene_focus_name = 'scene-0103'

generator = data_generator(cfg, log, split=split, phase='testing')
scene_preprocessors = generator.sequence
scene_names = [scene_preprocessors[s].seq_name for s in range(len(scene_preprocessors))]


for idx, row in df_interactions.iterrows():

    try:
        # get relevant interaction variables
        scene_name = row['scene']
        scene = scene_preprocessors[scene_names.index(scene_name)]
        common_start_frame = row['common_start_frame']
        common_end_frame = row['common_end_frame']
        interaction_frames = np.arange(common_start_frame, common_end_frame+1)
        
        focus_agents = (row['agent1'], row['agent2'])

        if not focus_scene_bool or scene_name == scene_focus_name: # settings above to just plot one scene for debugging

            gt = scene.gt
            pred_frames = scene.pred_frames
            df_scene = Agent.process_data(gt)
            vmax_scene = df_scene.v.max()

            # # plot overview scene:
            # px.line_3d(df_scene, x = 'x', y = 'y', z = 'frame', color = 'agent_id').show()


            # init agent modification classes here; init interpolation functions, and allow for accel/decel rollouts
            agent_dict = {}
            agents_scene = list(df_scene.agent_id.unique()) # definitive order for agent ids in all tensors
            for agent in agents_scene:
                df_agent = df_scene[df_scene.agent_id == agent]
                agent_class = Agent(df_agent, v_max = vmax_scene) # impose vmax based on gt scene velocities
                agent_dict.update({agent: agent_class})


            figs_scene = []
            modes_scene = []
            for frame_idx, frame in enumerate(interaction_frames):
                # frame corresponds to the current timestep, i.e. the last of pre_motion
                data = scene(frame)
                if data is None:
                    print('Frame skipped in loop')
                    continue

                if not (focus_agents[0] in data['valid_id'])*(focus_agents[1] in data['valid_id']):
                    continue # need both agents to be valid in order to make predicitons

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
                
                # length and widths for collision calculation
                widths = [df_scene[df_scene['agent_id'] == str(agent_id)].width.values[0] for agent_id in focus_agents]
                lengths = [df_scene[df_scene['agent_id'] == str(agent_id)].length.values[0] for agent_id in focus_agents]

                # get roll-out combinations and check for collisions
                fut_mod_rollout_combinations = get_rollout_combinations(fut_mod_decel, fut_mod_accel)
                fut_mod_rollout_combinations = fut_mod_rollout_combinations[0:2] # first two entries already contain all possible combinations (only for 2 agent case!)
                fut_mod_rollout_combinations_motion = fut_mod_rollout_combinations[...,0:2]
                fut_mod_rollout_combinations_heading = fut_mod_rollout_combinations[...,2].unsqueeze(-1)
                collision_margins, collision_bool = calc_collision_matrix_agentpair(fut_mod_rollout_combinations_motion, fut_mod_rollout_combinations_heading, lengths, widths)

                # visualize interaction pair and calculate modes
                fig, scene_mode_dict = data['scene_vis_map'].visualize_interactionpair_splitplot(data, sample_motion_3D, fut_mod_rollout_combinations_motion, collision_bool, focus_agents)
                figs_scene.append(fig)
                modes_scene.append(scene_mode_dict)

                if frame == 11:
                    fig.update_layout(
                        margin=dict(
                            l=0,  # left margin
                            r=0,  # right margin
                        )
                    )
                    pio.write_image(fig, 'example_vis_method.png',width=0.8*1700/1.1, height=0.8*800/1.2)


                if plot_all:
                    fig.show()

                if scene_mode_dict['h_final']:
                    df_modes_pair = pd.DataFrame(modes_scene)
                    # at what point to cut the data? pred horizon? gt mode? 
                    df_modes_pair_filt = df_modes_pair[df_modes_pair.gt_mode == df_modes_pair.gt_mode.values[-1]] # cut data with other modes
                    df_modes_pair_filt = df_modes_pair_filt[~df_modes_pair_filt['h_final']] # only look at predictions before the homotoyp class is inevitable
                    df_modes_pair_filt = df_modes_pair_filt.tail(PRED_FRAMES) # limit to number of prediciton frames
                    # df_modes_pair_filt[df_modes_pair_filt.keys()[[0,1,2,3,4,5,6,7,10]]]
                    # calc metrics:
                    if len(df_modes_pair_filt) > 0:
                        prediction_consistentcy = check_consistency(df_modes_pair_filt['ml_mode'])
                        t2cor, pred_time = calc_time_based_metric(df_modes_pair_filt['mode_correct'])
                        t2cov, _ = calc_time_based_metric(df_modes_pair_filt['mode_covered'])
                        mode_collapse = df_modes_pair_filt['N_modes_covered'] < df_modes_pair_filt['N_feasible_rollouts']
                        r_mode_collapse = round(100*(sum(mode_collapse)/len(mode_collapse)), 1)

                        # save metrics to existing df
                        df_interactions.loc[idx, 'pred_time'] = pred_time
                        df_interactions.loc[idx, 't2cor'] = t2cor
                        df_interactions.loc[idx, 't2cov'] = t2cov
                        df_interactions.loc[idx, 'r_mode_collapse'] = r_mode_collapse
                        df_interactions.loc[idx, 'prediction_consistency'] = prediction_consistentcy

                        # save figure and metrics:
                        fig = figs_scene[-2]
                        vis_start_frame = df_modes_pair_filt.iloc[-1].frame
                        vis_end_frame = vis_start_frame + df_modes_pair_filt.iloc[-1].Npred_frames
                        title = f"{row['scene']} frame {vis_start_frame}-{vis_end_frame}, t2cor: {t2cor}s, t2cov: {t2cov}s, pred_time: {pred_time}s, pred_consistency: {prediction_consistentcy}, r_mode_collapse: {r_mode_collapse}%"
                        fig.update_layout(
                                title=dict(text = title),
                            )
                        if plot:
                            fig.show()
                        if save_imgs:
                            pio.write_image(fig, save_pred_imgs_path + f'/{scene_name}_{focus_agents[0]}_{focus_agents[1]}.png',width=1200, height=1000)
                    else:
                        print('prediction length too short for mode evaluation')
                    break # break for loop 

    except Exception as e:
        print(e)
# save df
if not focus_scene_bool:
    df_interactions.to_csv(f'interaction_mode_metrics_{split}.csv', index = False)

end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")
