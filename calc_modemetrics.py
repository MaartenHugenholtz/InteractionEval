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
from models.ctt_model import get_model_prediction as get_model_prediction_ctt
from models.cv_model import get_model_prediction as get_model_prediction_cv
from models.oracle_model import get_model_prediction as get_model_prediction_oracle
import logging
logging.basicConfig(filename='calc_modemetric.log', 
                    level = logging.DEBUG,    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Log message format
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('calc_modemetric')
start_time = time.time()


""" MODEL """
K_Modes = 5
if K_Modes == 5:
    cfg = Config('nuscenes_5sample_agentformer' )
else:
    cfg = Config('nuscenes_10sample_agentformer' )

############################################
H_PRED = 12 # frames (at 2 Hz)
cfg.future_frames  = H_PRED  # overwrite H_pred in config!
MODEL = 'AF'
# MODEL = 'CTT'
# MODEL = 'cv'
# MODEL = 'oracle'

# only used for oracle/cv
if MODEL == 'oracle':
    use_gt_path = True
else:
    use_gt_path = False

"""" DATA"""
interaction_scenes_input = 'interaction_scenes/interaction_metrics_val_all.csv'

split = 'val'
save_pred_imgs_path = f'pred_imgs/{MODEL}_{split}_{H_PRED}f_{K_Modes}samples'
mkdir_if_missing(save_pred_imgs_path)
mode_metrics_path = f'mode_metric_results/interaction_mode_metrics_{MODEL}_{split}_Tpred_{H_PRED}f_{K_Modes}samples.csv'
mode_metrics_data_path = f'mode_metric_results/interaction_mode_metrics_data_{MODEL}_{split}_Tpred_{H_PRED}f_{K_Modes}samples.csv'

plot_mode_overview = True
plot_all_modes = False
plot_all_scenes = False

save_modes_plots = True
save_modes_csv = True

focus_scene_bool = True
scene_focus_name = 'scene-0104'

""""""" SETUP """""""
torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=0) if 0 >= 0 and torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available(): torch.cuda.set_device(0)
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

def get_model_prediction_af(data, sample_k = K_Modes):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

""""  #################  """


""" Get predictions and compute metrics """
df_interactions_in = pd.read_csv(interaction_scenes_input) # input
df_interactions_in = df_interactions_in[df_interactions_in.interaction_bool] # new format contains all possible interactions --> filter on interaction bool
df_interactions_out = df_interactions_in.copy() # save output in similar df
dfs_data = []

generator = data_generator(cfg, log, split=split, phase='testing')
scene_preprocessors = generator.sequence
scene_names = [scene_preprocessors[s].seq_name for s in range(len(scene_preprocessors))]


for idx, row in df_interactions_in.iterrows():

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
                agent_class = Agent(df_agent, v_max = vmax_scene, fut_steps=H_PRED) # impose vmax based on gt scene velocities
                agent_dict.update({agent: agent_class})

            path_intersection_bool, _, _ = calc_path_intersections(df_scene, agents_scene, pred_frames)

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

                # path_intersction for current agents in frame (for cv/oracle)
                idx_scene_agents = [agents_scene.index(str(int(agent_id))) for agent_id in data['valid_id']]
                path_intersection_bool_frame = path_intersection_bool[idx_scene_agents,:][:,idx_scene_agents] # re-order according to data[valid_id]
                
                ### GET MODEL PREDICTIONS ###
                if MODEL == 'AF':
                    with torch.no_grad():
                        recon_motion_3D, sample_motion_3D = get_model_prediction_af(data, cfg.sample_k)
                    recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale
                elif MODEL == 'CTT':
                    assert H_PRED == 6, 'CTT not made for H_pred other than 6'
                    try:
                        recon_motion_3D, sample_motion_3D = get_model_prediction_ctt(data, cfg.sample_k)
                    except ValueError:
                        continue
                elif MODEL == 'cv':
                    recon_motion_3D, sample_motion_3D = get_model_prediction_cv(data, cfg.sample_k, agent_dict, path_intersection_bool_frame, use_gt_path = use_gt_path)
                elif MODEL == 'oracle':
                    recon_motion_3D, sample_motion_3D = get_model_prediction_oracle(data, cfg.sample_k, agent_dict, path_intersection_bool_frame, use_gt_path = use_gt_path)
                else:
                    raise NameError
                
                sample_motion_3D = sample_motion_3D[:,:,:H_PRED,:]
                recon_motion_3D = recon_motion_3D[:,:H_PRED,:]

                if plot_all_scenes:
                    data['scene_vis_map'].visualize_trajs(data, sample_motion_3D)

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

                # if frame == 11:
                #     fig.update_layout(
                #         margin=dict(
                #             l=0,  # left margin
                #             r=0,  # right margin
                #         )
                #     )
                #     fig.show()
                #     pio.write_image(fig, 'example_vis_method.png',width=0.8*1700/1.1, height=0.8*800/1.2)


                if plot_all_modes:
                    fig.show()

                if scene_mode_dict['h_final']:
                    df_modes_pair = pd.DataFrame(modes_scene)
                    # at what point to cut the data? pred horizon? gt mode? 
                    df_modes_pair_filt = df_modes_pair[df_modes_pair.gt_mode == df_modes_pair.gt_mode.values[-1]] # cut data with other modes
                    df_modes_pair_filt = df_modes_pair_filt[~df_modes_pair_filt['h_final']] # only look at predictions before the homotoyp class is inevitable
                    df_modes_pair_filt = df_modes_pair_filt.tail(H_PRED) # limit to number of prediciton frames
                    # df_modes_pair_filt[df_modes_pair_filt.keys()[[0,1,2,3,4,5,6,7,10]]]

                    # make new df and store: correct, covered, collapse, v1, h1, v2, h2, dv, dh, frame
                    for key, value in df_interactions_in.loc[idx].items():
                        df_modes_pair_filt[key] = value                        

                    # first filter agent dfs on frame and id
                    df_agent1 = df_scene[(df_scene.agent_id == str(df_interactions_in.loc[idx, 'agent1']))*(df_scene.frame.isin(list(df_modes_pair_filt.frame.values)))]
                    df_agent2 = df_scene[(df_scene.agent_id == str(df_interactions_in.loc[idx, 'agent2']))*(df_scene.frame.isin(list(df_modes_pair_filt.frame.values)))]
                    
                    # assign data:
                    df_modes_pair_filt['v1'] = df_agent1.v.values
                    df_modes_pair_filt['heading1'] = df_agent1.heading.values
                    df_modes_pair_filt['v2'] = df_agent2.v.values
                    df_modes_pair_filt['heading2'] = df_agent2.heading.values

                    dfs_data.append(df_modes_pair_filt)


                    # calc metrics:
                    if len(df_modes_pair_filt) > 0:
                        prediction_consistentcy = check_consistency(df_modes_pair_filt['ml_mode'])
                        t2cor, pred_time = calc_time_based_metric(df_modes_pair_filt['mode_correct'], Hpred = H_PRED)
                        t2cov, _ = calc_time_based_metric(df_modes_pair_filt['mode_covered'], Hpred = H_PRED)
                        mode_collapse = df_modes_pair_filt['N_modes_covered'] < df_modes_pair_filt['N_feasible_rollouts']
                        r_mode_collapse = round(100*(sum(mode_collapse)/len(mode_collapse)), 1)

                        # save metrics to existing df
                        df_interactions_out.loc[idx, 'pred_time'] = pred_time
                        df_interactions_out.loc[idx, 't2cor'] = t2cor
                        df_interactions_out.loc[idx, 't2cov'] = t2cov
                        df_interactions_out.loc[idx, 'r_mode_collapse'] = r_mode_collapse
                        df_interactions_out.loc[idx, 'prediction_consistency'] = prediction_consistentcy

                        # save figure and metrics:
                        fig = figs_scene[-2]
                        vis_start_frame = df_modes_pair_filt.iloc[-1].frame
                        vis_end_frame = vis_start_frame + df_modes_pair_filt.iloc[-1].Npred_frames
                        title = f"{row['scene']} frame {vis_start_frame}-{vis_end_frame}, t2cor: {t2cor}s, t2cov: {t2cov}s, pred_time: {pred_time}s, pred_consistency: {prediction_consistentcy}, r_mode_collapse: {r_mode_collapse}%"
                        fig.update_layout(
                                title=dict(text = title),
                            )
                        if plot_mode_overview:
                            fig.show()
                        if save_modes_plots:
                            fig.update_layout(margin=dict(l=0, r=0, t=100, b=0))
                            pio.write_image(fig, save_pred_imgs_path + f'/{scene_name}_{focus_agents[0]}_{focus_agents[1]}.png',width=1200, height=1200/2.4)
                    else:
                        raise ValueError('prediction length too short for mode evaluation')
                        # print('prediction length too short for mode evaluation')

                    break # break for loop 

    except Exception as e:
        print(e)
        logger.error(f'Error for {MODEL} model (Tpred {H_PRED} frames) in {scene_name} for agents: {str(focus_agents)}: {e}')

# save df
if save_modes_csv:
    df_interactions_out.to_csv(mode_metrics_path, index = False)
    pd.concat(dfs_data,ignore_index = True).to_csv(mode_metrics_data_path, index = False)

end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")
