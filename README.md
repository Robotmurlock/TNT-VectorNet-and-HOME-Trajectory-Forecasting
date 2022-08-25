# TNT-VectorNet-and-HOME-Trajectory-Forecasting

Master thesis (Serbian) - Trajectory Forecasting on scenes with multiple moving objects

Work in progress...

## TNT-Vectornet

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
