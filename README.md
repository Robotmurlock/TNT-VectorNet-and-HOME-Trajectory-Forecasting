# TNT-VectorNet-and-HOME-Trajectory-Forecasting

Master thesis (text is on Serbian) - Trajectory Forecasting on scenes with multiple moving objects

Work in progress... (code refactor)

## Setup

Install `argoverse-api` from [Argoverse API Github](https://github.com/argoai/argoverse-api).

Install packages:

```
pip3 install -r src/requirements.txt
```

## Configuration

Example of config files can be found in `src/configs`. Config `src/configs/vectornet.yaml` gives best results for `TNT-Vectornet` and `src/configs/home.yaml` gives
best results for `HOME`. 

## Data Preparataion

Check config examples in `configs` directory.

First step for both approaches is HD map vectorization (config section: `data_process`). Note: Set `visualize: True` to visualize output.

```
python3 data_processing/offline/script_vectorize_hd_maps.py --cfg [cfg]
```

To train `TNT-Vectornet` it is also required to transform vectorized HD maps (acquired from previous step) into polylines structure (config section: `graph/data_process`)
Note: Set `visualize: True` to visualize output.

```
python3 data_processing/offline/script_graph.py --cfg [cfg]
```

Transformation (rasterization) for `HOME` model is run during training.

## Training

To train `TNT-Vectornet` run (50 epochs - 4h 45m):

```
python3 training/vectornet/script_train_vectornet.py --cfg [cfg]
```

To train `HOME: Heatmap Estimation` run (12 epochs - 4h 45m):

```
python3 training/heatmap/script_train_heatmap.py --cfg [cfg]
```

To train `HOME: Trajectory Forecaster` run (30 epochs - 45m):

```
python3 training/heatmap/script_train_trajectory_forecaster.py --cfg [cfg]
```

Note: All training scripts have Tensorboard logging support.

Estimated training time is for `RTX 3070` 

## Evaluation

To evaluate `TNT-Vectornet` run (requires trained `TNT-Vectornet` model):

```
python3 evaluation/script_evaluate_vectornet.py --cfg [cfg]
```

To evaluate `HOME` run (requires trained `HOME-Heatmap` and `HOME-Forecaster` model):

```
python3 evaluation/script_evaluate_home.py --cfg [cfg]
```

## Results

### TNT-Vectornet

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

### HOME

![vectornet-example](https://github.com/Robotmurlock/TNT-VectorNet-and-HOME-Trajectory-Forecasting/blob/main/thesis/images/home_MIA_10454.png)

Current results (optimal MR model):

- `minADE`: 
    - custom (val): 0.97
    - original (val): -
    - original (test): 0.92
- `minFDE`:
    - custom (val): 1.71
    - original (val): 1.45
    - original (test): 1.71
- `MissRate`:
    - custom (val): 0.15
    - original (val): 0.07
    - original (test): 0.10
