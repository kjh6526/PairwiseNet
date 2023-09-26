# PairwiseNet: Pairwise Collision Distance Learning for High-dof Robot Systems
The official repository for \<A PairwiseNet: Pairwise Collision Distance Learning for High-dof Robot Systems\> (Jihwan Kim* and Frank C. Park, CoRL 2023).

> PairwiseNet is a novel method that estimates the pairwise collision distance between pairs of elements in a robot system, providing an alternative approach to data-driven methods that estimate the global collision distance.

- *[Paper](https://openreview.net/forum?id=Id4b5SY1Y8)* 
<!-- - *[Poster](https://drive.google.com/file/d/1NuyQaG-g3zwWQEl6qS5tl5_H3oKsEzcK/view?usp=sharing)*   -->

## Preview

### Collision Distance Estimation of PairwiseNet

### Pairwise Collision Distance Learning

## Requirements

### Environment

We recommend using Anaconda for environment management. To set up the Python environment, simply run:
```bash
conda env create -f environment.yml
conda activate PairwiseNet
```
Adjust as needed based on the specifics of your setup.

### Datasets

You need to generate the pairwise collision distance dataset for the multi-arm robot system by run:
```bash
python generate_dataset_multiarm_pairwise.py --config configs/pairwise_dataset_generation/data_config_multiarm.yml
```

The detailed generation setting can be modified by adjusting `data_config_multiarm.yml`.

You also need to generate the global collision distance dataset (for the test dataset) by run:
```bash
python generate_dataset_multiarm_global.py --env configs/env/{env_config} --n_data {N}
```
- `env_config` is the config file (.yml) of the target robot system, `env_config_multipanda.yml` for example. 
- `N` is the number of data points (joint configurations) in the dataset.

## Training

The training procedure of PairwiseNet can be started by
```bash
python train.py --config configs/training/{training_config} --device {device} --run {run_id}
```
- `training_config` is the config file (.yml) of the training, `config_PairwiseNet.yml` for example. 
- `device` is the index of GPU device, either `0` or `1`, `cpu` for the use of CPU.
- `run_id` is the string identifier of the training, whatever you want. 

## Citation
If you found this repository useful in your research, please consider citing:
```
@inproceedings{kim2023pairwisenet,
  title={PairwiseNet: Pairwise Collision Distance Learning for High-dof Robot Systems},
  author={Kim, Jihwan and Park, Frank C},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```