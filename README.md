# TNT-VectorNet-and-HOME-Trajectory-Forecasting

Master thesis (text is on Serbian) - Trajectory Forecasting on scenes with multiple moving objects

## Setup

Install `argoverse-api` from [Argoverse API GitHub](https://github.com/argoai/argoverse-api).

Install packages:

```
pip3 install -r src/requirements.txt
```

## Configuration

Example of config files can be found in `src/configs`. Config `src/configs/vectornet.yaml` gives best results for `TNT-Vectornet` and `src/configs/home.yaml` gives
best results for `HOME`. 

All training scripts have Tensorboard logging support.

## Common Data Preparation

Check config examples in `configs` directory.

First step for both approaches is HD map vectorization (config section: `data_process`). Note: Set `visualize: True` to visualize output.

```
python3 common_data_processing/script_vectorize_hd_maps.py --cfg [cfg]
```

### Data structure

Path where all data are stored (input data, intermediate data and results) is `global_path` defined in config yaml file. Relative to that
directory path this structure can be found:

```
dataset/  # Add argoverse raw csv files here
  train/*
    ... csv files
  val/*
  test/*
internal/*  # Generated after running common_data_processing/script_vectorize_hd_maps.py
  train/*
  val/*
  test/*
internal_graph/*  # Generated after running vectornet/script_transform_to_polylines.py
  train/*
  val/*
  test/*
```

## TNT-VectorNet

Original paper can be found [here](https://arxiv.org/abs/2008.08294).



### Usage

To train `TNT-Vectornet` it is also required to transform vectorized HD maps (acquired from previous step) into polylines structure (config section: `graph/data_process`)
Note: Set `visualize: True` to visualize output.

```
python3 vectornet/script_transform_to_polylines.py --cfg [cfg]
```

To train a model run (50 epochs - 4h 45m):

```
python3 vectornet/script_train_vectornet.py --cfg [cfg]
```

To evaluate model run (requires trained `TNT-Vectornet` model):

```
python3 vectornet/script_evaluate_vectornet.py --cfg [cfg]
```

Estimated training time is for `RTX 3070`

### Results

![vectornet-example](https://github.com/Robotmurlock/TNT-VectorNet-and-HOME-Trajectory-Forecasting/blob/main/thesis/images/result_MIA_10454.png)

Current results:

- `minADE`: 
    - custom (val): 1.03
    - original (val): 0.73
    - original (test): 0.91
- `minFDE`:
    - custom (val): 1.91
    - original (val): 1.29
    - original (test): 1.45
- `MissRate`:
    - custom (val): 0.30
    - original (val): 0.09
    - original (test): 0.22

## HOME

Original paper can be found [here](https://arxiv.org/abs/2105.10968)

### Usage
Transformation (rasterization) for `HOME` model is run during training.

To train `HOME: Heatmap Estimation` run (12 epochs - 4h 45m):

```
python3 home/script_train_heatmap.py --cfg [cfg]
```

To train `HOME: Trajectory Forecaster` run (30 epochs - 45m):

```
python3 home/script_train_trajectory_forecaster.py --cfg [cfg]
```

Estimated training time is for `RTX 3070`

To evaluate the model run (requires to be trained `HOME-Heatmap` and `HOME-Forecaster` model):

```
python3 home/script_evaluate_home.py --cfg [cfg]
```

### Results

![vectornet-example](https://github.com/Robotmurlock/TNT-VectorNet-and-HOME-Trajectory-Forecasting/blob/main/thesis/images/home_MIA_10454.png)

Current results (optimal MR model):

- `minADE`: 
    - custom (val): 0.97
    - original (val): -
    - original (test): 0.92
- `minFDE`:
    - custom (val): 1.71
    - original (val): 1.28
    - original (test): 1.45
- `MissRate`:
    - custom (val): 0.15
    - original (val): 0.07
    - original (test): 0.10

## Citation

```
@misc{madzemovic_home_tnt-vectornet,
  author = {Adzemovic, Momir},
  title = {Trajectory Forecasting on scenes with multiple moving objects},
  year = {2022},
  publisher = {GitHub, MATF},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Robotmurlock/TNT-VectorNet-and-HOME-Trajectory-Forecasting}},
}
```