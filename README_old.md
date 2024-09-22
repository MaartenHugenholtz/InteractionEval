# Evaluating interaction mode-collapse in joint VTP models
This repo contains the code to evaluate critical interactions in joint VTP models.
For our algorithm we use [AgentFormers](https://github.com/Khrylx/AgentFormer) preprocessing functions as backbone, see their [README](README_AF.md) for more information.


## Installation 

### Environment
* **Tested OS:** MacOS, Linux
* Python >= 3.7
* PyTorch == 1.8.0
### Dependencies:
1. Install [PyTorch 1.8.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

### Datasets
* This resposistory already contains the preprocessed scenes of the NuScenes dataset [here](datasets/nuscenes_pred).
* To perform the preprocessing for the nuScenes dataset from scratch, the following steps are required:
  1. Download the orignal [nuScenes](https://www.nuscenes.org/nuscenes) dataset. Checkout the instructions [here](https://github.com/nutonomy/nuscenes-devkit).
  2. Follow the [instructions](https://github.com/nutonomy/nuscenes-devkit#prediction-challenge) of nuScenes prediction challenge. Download and install the [map expansion](https://github.com/nutonomy/nuscenes-devkit#map-expansion).
  3. Run our [script](data/process_nuscenes.py) to obtain a processed version of the nuScenes dataset under [datasets/nuscenes_pred](datasets/nuscenes_pred):
      ```
      python data/process_nuscenes.py --data_root <PATH_TO_NUSCENES>
      ``` 

## Evaluation scripts
We provide the following evaluation scripts:
- [calc_interaction_scenes.py](calc_interaction_scenes.py) can be used to find the safety-critical interaction in the nuScenes dataset
- [calc_modemetrics.py](calc_modemetrics.py) is used to calculate the interaciton mode metrics
- [plot_modemetrics_data_misc.py](plot_modemetrics_data_misc.py), [plot_modemetrics_rates_dist.py](plot_modemetrics_rates_dist.py), [plot_modemetrics_time_hist.py](plot_modemetrics_time_hist.py) are used to visualized the various metrics and statistics

Next to these scripts, we provide various useful classes and functions for evaluating interaction modes:
- [data/map.py](data/map.py) is adopted form AgentFormers implementation, but extended with various visualization functions.
- [agent_class.py](agent_class.py) can be used to perform constant velocity, acceleration, or deceleration roll-outs for an agent.
- [eval_utils.py](eval_utils.py) contains functions to combine roll-outs, check collisions and more. 
- [utils/homotopy.py](utils/homotopy.py) is taken from [CTT](https://github.com/NVlabs/diffstack/tree/CTT_release) and contains the functions for calculating the homotopy angles and classes

## Models
This reposistory contains four baseline models:
- **AgentFormer**: AgentFormer (AF) is a multi-agent trajectory prediction model. They utilize a transformer-based architecture, that simultaneously models the social and time dimension of agents. Their prediction framework jointly models the agents' intentions, to predict diverse and socially-aware future trajectories. We build upon their code stack for this project.
- **Categorical Traffic Transformer**: Categorical Traffic Transformer (CTT) is a multi-agent trajectory prediction model, with an interpretable latent space consisting of agent-to-agent and agent-to-lane modes. CTT generates diverse behaviors by conditioning the trajectory prediction on different modes. Unfortunately, we did not manage to reproduce the numbers reported in their paper and uncovered various issues, making direct comparison with the other models difficult.
Firstly, their pre-trained model is trained for a prediction horizon of 3 seconds, whereas AF is trained for 6 seconds, as dictated by the NuScenes benchmark. 
To match the varying prediction horizons, the 6-second predictions from AF are simply cut to 3 seconds. 
Secondly, all predicted modes and trajectories are identical, making the model effectively unimodal. 
Finally, whereas AF predicts for all vehicles in the scenes, CTT predicts only for the road-users within a certain attention radius of the ego-vehicle, but it does include pedestrians whereas AF does not.
We use AF's data preprocessing backbone and match CTT's predictions to the corresponding agents. However, due to aforementioned attention radius used in CTT, many predictions are missing for certain agents. In these cases, the current ground truth position is kept static and used as prediction instead.
Due to these issues, we are not able to report the real performance of CTT on interaction prediction. However, we still report the metrics and compare to the other models, to set a baseline and show that our methodology generalizes to other models.
*NOTE*: Due to hardware limitations, I could not run CTT on my own laptop, which is why I did not include their code in this repository. Instead, I ran it on another machine and saved the predictions to a folder on this repository. To evaluate CTT's interaction performance, I use AF's preprocessing functions and try to match these with the saved predictions from CTT. Due to the issues with CTT, some predictions and files are missing. 
- **Oracle model**: The cardinality of the space of interaction modes grows exponentially with the number of agent in the scene. Because trajectory prediction models usually predict a fixed set of $K$ modes, covering all feasible modes becomes infeasible in scenes with many agents. To test this limitation, we propose a multimodal oracle model. The oracle's goal is to predict a set of K multimodal trajectories that cover all feasible modes of the interacting agents. 
The oracle will be given access to the agent's ground truth paths, so it knows which agents will be interacting, i.e. crossing the same path, in the near future. However, the trajectories are unknown, i.e. it does not know the velocity profiles along the path, so the interaction class is still to be determined by the model. The oracle's goal is to cover all feasible interaction modes between the path-crossing agent-pairs. 
For the roll-outs, we keep the agents' ground truth paths and simulate future roll-outs with a constant velocity, deceleration, or acceleration profile. 
Firstly, all agents are initialized with their constant velocity profile. 
Next, we calculate all combinations of constant velocity, acceleration, and deceleration profiles between the interacting agents and reject the combinations with collisions. 
Finally, we must assign each joint prediction a likelihood. We argue that the likelihood of a joint scene prediction is proportional to the overall utility in the scene, where the average speed of a roll-out combination can be used as a utility measure. Therefore, to get a finite set of $K$ joint predictions, we calculate the average speed of the roll-outs and output the top-$K$ trajectory combinations with the highest average speed. 
- **Constant Velocity model**: The constant velocity (CV) model is a simplistic unimodal model that assumes the vehicle will remain its current heading and velocity. Because it produces a single mode, it inherently suffers from mode collapse. 
However, it is an interesting baseline to compare to, because it tells us in how many scenarios we can correctly assess the vehicle pair's interaction class by simply extrapolating their current trajectories.
