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
* The perform the preprocessing for the nuScenes dataset from scratch, the following steps are required:
  1. Download the orignal [nuScenes](https://www.nuscenes.org/nuscenes) dataset. Checkout the instructions [here](https://github.com/nutonomy/nuscenes-devkit).
  2. Follow the [instructions](https://github.com/nutonomy/nuscenes-devkit#prediction-challenge) of nuScenes prediction challenge. Download and install the [map expansion](https://github.com/nutonomy/nuscenes-devkit#map-expansion).
  3. Run our [script](data/process_nuscenes.py) to obtain a processed version of the nuScenes dataset under [datasets/nuscenes_pred](datasets/nuscenes_pred):
      ```
      python data/process_nuscenes.py --data_root <PATH_TO_NUSCENES>
      ``` 

### Models
This reposistory contains four baseline models:
- AgentFormer
- Categorical Traffic Transformer
- Constant Velocity model
- Oracle model

## Evaluation scripts
We provide the following evaluation scripts:
- [calc_interaction_scenes.py](calc_interaction_scenes.py) can be used to find the safety-critical interaction in the nuScenes dataset
- [calc_modemetrics.py](calc_modemetrics.py) is used to calculate the interaciton mode metrics
- [plot_modemetrics_data_misc.py](plot_modemetrics_data_misc.py), [plot_modemetrics_rates_dist.py](plot_modemetrics_rates_dist.py), [plot_modemetrics_time_hist.py](plot_modemetrics_time_hist.py) are used to visualized the various metrics and statistics

Next to these scripts, we provide various useful classes and functions for evaluating interaction modes:
- [data/map.py](data/map.py) is adopted form AgentFormers implementation, but extended with various visualization functions.
- [agent_class.py](agent_class.py) can be used to perform constant velocity, acceleration, or deceleration roll-outs for an agent.

