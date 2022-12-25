# SPGCL

This repository contains the complete contents of *Mining Spatio-Temporal Relations via Self-Paced Graph Contrastive Learning* [DOI: https://doi.org/10.1145/3534678.3539422](https://doi.org/10.1145/3534678.3539422), including appendix and poster. I am sorry that the code is not well organized, because I was busy looking for a job during this time, apologize again. 

---

This is a Pytorch implementation of SPGCL:  Mining Spatio-temporal Relations via Self-paced Graph Contrastive Learning.

# Mining Spatio-temporal Relations via Self-paced Graph Contrastive Learning

More details of the paper and dataset will be released after it is published.

# The Code

## Requirements

Following is the suggested way to install the dependencies:

    conda install --file SPGCL.yaml

Note that ``pytorch >=1.10``.

## Folder Structure

```tex
└── code-and-data
    ├── datasets                 # Including datasets
    ├── CONFIG                   # Parameter settings
    ├── lib                      # Contains self-defined modules for our work
    │   |──  adj_from_loc.py     # Build adjacency matrix by KNN
    │   |──  args.py             # Settings about models and loading configure files
    │   |──  dataloader.py       # Dataloaders 
    │   |──  normalization.py    # Data normalizations
    │   |──  utils.py            # Defination of auxiliary functions for running
    │   ├──  layers
    |	│	├──  SPGCL.py    # The core source code of our model SPGCL
    ├── results                  # Saved model and evaluation results are here
    ├── prepare_data.py          # Prepare data before training (run DTW)
    ├── baselines.py             # Some baselines including SVR and ARIMA
    ├── Trainers.py              # Trainer
    ├── train.py                 # Train model
    ├── test.py                  # Test model
    ├── SPGCL.yaml               # The python environment needed for SPGCL
    └── README.md                # This document
```



## Datasets

Step 1:  Download PEMS04 and PEMS08 datasets provided by [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). 

Step 2:  Before you start running this model, please do:

```
python prepare_data.py
```

to prepare DTW matrixs which may cost some time.

## Configuration

Step 1:  Parameter settings for different datasets are all in  `./CONFIG/` 

Step 2:  Important parameters in the configuration are as follows (take PEMS03 as example):

```tex
delta = 0.9             # The threshold of positive edges, i.e., \delta^+
delta_negative = 0.2    # The threshold of negative edges, i.e., \delta^-
gamma_1 = 10            # Reweighting hyperparmeter for PU-loss, i.e., \gamma_1
gamma_2 = 20            # Reweighting hyperparmeter for predict loss, i.e., \gamma_2
eta = 0.1               # Prior probablity, i.e., \eta
Q = 0.3                 # Weighting parameters, i.e., \Gamma
```

##  Train and Test

Replace `PEMS03` with your own dataset name and you can start training and testing your model.

- For train your own model:

  ```
  python train.py --data="PEMS03" --data_path="./datasets/PEMS03"
  ```

- To test the results on the test set:

  ```
  python test.py --data="PEMS03" --data_path="./datasets/PEMS03" --conti=True
  ```

All the parameter settings are in `lib.args.py` and `CONFIG/` 



## Baselines

Our baselines included: 

1. Autoregressive Integrated Moving Average model (ARIMA)
2. Support Vector Regression model (SVR)
3. Gated Recurrent Unit model (GRU) (Chung et al.2014)
4. Iterative Deep Graph Learning(IDGL and IDGL-ANCH)(Yu Chen et al.2020), code links [IDGL](https://github.com/hugochan/IDGL).
5. Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting(STGCN) (Wu et al.2020),  code links [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18).

6. Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting(ASTGCN and MSTGCN)(Shengnan et al.2019), code links [ASTGCN](https://github.com/guoshnBJTU/ASTGCN-r-pytorch).

7. Land Deformation Predict via Slope-Aware Graph Neural Networks(SA-GNN)(Fan et al.2021)
8. Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting(STGODE)(Zheng et al.2019), code links [STGODE](https://github.com/square-coder/STGODE).

9. Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting(AGCRN)(Lei et al.2020), code links [AGCRN](https://github.com/LeiBAI/AGCRN ).


The python implementations of ARIMA/SVR models are in the `baselines.py`. Code of other baselines (IDGL, IDGL-ANCH, STGCN, ASTGCN, MSTGCN, STGODE, AGCRN)  can be found in the corresponding papers and Github links.

