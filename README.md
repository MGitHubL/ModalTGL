# ModalTGL

Code and data of the manuscript: Dictionary Multi-Modal Temporal Graph Learning (IEEE T-PAMI 2026).

If you find any problems, feel free to contact us: ```mengliuedu@163.com```.

## Dataset

### Download

In the paper, we bulit 5 new datasets and compare 3 public datasets.

The new datasets can be downloaded from the repository.

**Amazon**: It has been uploaded to this repository.

**DBLP**: It has been uploaded to this repository.

(We have only uploaded two datasets and promise to add two more after the paper is accepted.)

The public datasets can be downloaded from:

**Yelp** and **Stack**: https://github.com/zjs123/DTGB

**TemFin**: https://github.com/FDUDSDE/SEAN


### Use your own data
Put your data under `processed` folder. The required input data includes `ml_${DATA_NAME}.csv`, `ml_${DATA_NAME}.npy` and `ml_${DATA_NAME}_node.npy`. They store the edge linkages, edge features and node features respectively.

The `.csv` file has following columns
```
u, i, ts, label, idx
```
, which represents source node index, target node index, time stamp, edge label and the edge index.

`ml_${DATA_NAME}.npy` has shape of [#temporal edges + 1, edge features dimention]. Similarly, `ml_${DATA_NAME}_node.npy` has shape of [#nodes + 1, node features dimension].


All node index starts from `1`. The zero index is reserved for `null` during padding operations. So the maximum of node index equals to the total number of nodes. Similarly, maxinum of edge index equals to the total number of temporal edges. The padding embeddings or the null embeddings is a vector of zeros.


## Code

### Requirements
* `python >= 3.7`, `PyTorch >= 1.4`, please refer to their official websites for installation details.
* Other dependencies:
```{bash}
pandas==1.4.3
tqdm==4.41.1
numpy==1.23.1
scikit_learn==1.1.2
```

### Training Commands

#### Examples:

* To train **ModalTGL** with Amazon dataset in inductive training:
```bash
python main.py -d amazon --mode t
```


### Usage Summary

* Most parameters have already been set to default values.
```
usage: Interface for Neighbourhood-aware Scalable Learning for Temporal Networks
       [-h]
       [-d {amazon, google, ml1m, brain, dblp}]
       [-m {t,i}] [--n_degree [N_DEGREE [N_DEGREE ...]]] [--n_hop N_HOP]
       [--bias BIAS] [--pos_dim POS_DIM] [--self_dim SELF_DIM]
       [--ngh_dim NGH_DIM] [--linear_out] [--attn_n_head ATTN_N_HEAD]
       [--time_dim TIME_DIM] [--n_epoch N_EPOCH] [--bs BS] [--lr LR]
       [--drop_out DROP_OUT] [--replace_prob REPLACE_PROB]
       [--tolerance TOLERANCE] [--seed SEED] [--verbosity VERBOSITY]
       [--run RUN]
```

### optional arguments:
```
  -h, --help            show this help message and exit
  -d {amazon, google, ml1m, brain, dblp}
  -m {t,i}, --mode {t,i}
                        transductive (t) or inductive (i)
  --n_degree [N_DEGREE [N_DEGREE ...]]
                        a list of neighbor sampling numbers for different
                        hops, when only a single element is input n_layer will
                        be activated
  --n_hop N_HOP         number of hops the N-cache is used
  --bias BIAS           the hyperparameter alpha controlling sampling
                        preference with time closeness, default to 0 which is
                        uniform sampling
  --pos_dim POS_DIM     dimension of the positional embedding
  --self_dim SELF_DIM   dimension of the self representation
  --ngh_dim NGH_DIM     dimension of the neighborhood representation
  --linear_out          whether to linearly project each node's
  --attn_n_head ATTN_N_HEAD
                        number of heads used in tree-shaped attention layer,
                        we only use the default here
  --time_dim TIME_DIM   dimension of the time embedding
  --n_epoch N_EPOCH     number of epochs
  --bs BS               batch_size
  --lr LR               learning rate
  --drop_out DROP_OUT   dropout probability for all dropout layers
  --replace_prob REPLACE_PROB
                        probability for storing new neighbors to N-cache
                        replacing old ones
  --tolerance TOLERANCE
                        toleratd margainal improvement for early stopper
  --seed SEED           random seed for all randomized algorithms
  --verbosity VERBOSITY
                        verbosity of the program output
  --run RUN             number of model runs
```

## Acknowledge
Codes and model implementations are referred to several projects: [NAT](https://github.com/Graph-COM/Neighborhood-Aware-Temporal-Network), [DyGlib](https://github.com/yule-BUAA/DyGLib), [DTGB](https://github.com/zjs123/DTGB). Thanks for their great contributions!

## Cite us

If you feel our work has been helpful, thank you for the citation.

```
@ARTICLE{ModalTGL_ML_TPAMI,
  author={Liu, Meng and Liang, Ke and Li, Miaomiao and Zhu, Xueling and Liu, Xinwang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Dictionary Multi-Modal Temporal Graph Learning}, 
  year={2026}
}

```
