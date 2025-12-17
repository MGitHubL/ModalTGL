# ModalTGL

Code and data of the manuscript: Dictionary Multi-Modal Temporal Graph Learning.

If you find any problems, feel free to contact us: ```mengliuedu@163.com```.

## Dataset

### Download

In the paper, we bulit 4 new datasets and compare 4 public datasets.

The new datasets can be downloaded from the repository.

**Amazon**: It has been uploaded to this repository.

**DBLP**: It has been uploaded to this repository.

(We have only uploaded two datasets and promise to add two more after the paper is accepted.)

The public datasets can be downloaded from:

**Yelp** and **Stack**: https://github.com/zjs123/DTGB

**TemFin**: https://github.com/FDUDSDE/SEAN

### Data Format

#### Use your own data
Put your data under `processed` folder. The required input data includes `ml_${DATA_NAME}.csv`, `ml_${DATA_NAME}.npy` and `ml_${DATA_NAME}_node.npy`. They store the edge linkages, edge features and node features respectively.

The `.csv` file has following columns
```
u, i, ts, label, idx
```
, which represents source node index, target node index, time stamp, edge label and the edge index.

`ml_${DATA_NAME}.npy` has shape of [#temporal edges + 1, edge features dimention]. Similarly, `ml_${DATA_NAME}_node.npy` has shape of [#nodes + 1, node features dimension].


All node index starts from `1`. The zero index is reserved for `null` during padding operations. So the maximum of node index equals to the total number of nodes. Similarly, maxinum of edge index equals to the total number of temporal edges. The padding embeddings or the null embeddings is a vector of zeros.


## Code

## Training Commands

#### Examples:

* To train **NAT** with Wikipedia dataset in transductive training, using 32 1-hop N-cache, 16 2-hop N-cache, neighborhood representation of 4 dimensions , and with overriding probability alpha = 0.7:
```bash
python main.py -d wikipedia --pos_dim 16 --bs 100 --n_degree 32 16 --n_hop 2 --mode t --bias 1e-5 --seed 2 --verbosity 1 --drop_out 0.1 --attn_n_head 2 --ngh_dim 4 --self_dim 72 --replace_prob 0.7 --run 5
```
To train in inductive training, change `mode` from `t` to `i`. Here is an example of inductive training on Wiki-talk, using 16 1-hop neighbors with neighborhood representation size 4 and 72 dimensions of self representations.
```bash
python main.py -d wiki-talk-temporal --pos_dim 16 --bs 100 --n_degree 16 --n_hop 1 --mode i --bias 1e-5 --seed 2 --verbosity 1 --drop_out 0.1 --attn_n_head 1 --ngh_dim 4 --self_dim 72 --run 5
```

## Usage Summary
```
usage: Interface for Neighbourhood-aware Scalable Learning for Temporal Networks
       [-h]
       [-d {wikipedia,reddit,socialevolve,uci,enron,socialevolve_1month,socialevolve_2weeks,sx-askubuntu,sx-superuser,wiki-talk-temporal,mooc}]
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
  -d {wikipedia,reddit,socialevolve,uci,enron,socialevolve_1month,socialevolve_2weeks,sx-askubuntu,sx-superuser,wiki-talk-temporal,mooc}, --data {wikipedia,reddit,socialevolve,uci,enron,socialevolve_1month,socialevolve_2weeks,sx-askubuntu,sx-superuser,wiki-talk-temporal,mooc}
                        data sources to use, try wikipedia or reddit
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

## Instructions on Acquiring Datasets
Preprocessed datasets: Reddit, Wikipedia, SocialEvolve, Enron, and UCI can be downloaded from [here](https://drive.google.com/drive/folders/1umS1m1YbOM10QOyVbGwtXrsiK3uTD7xQ?usp=sharing) to `processed/`. Then run the following:
```{bash}
cd processed/
unzip data.zip
```
You may check that each dataset corresponds to three files: one `.csv` containing timestamped links, and two ``.npy`` as node & link features. Note that some datasets do not have node & link features, in which case the `.npy` files will be all zeros.

Raw data for Ubuntu can be downloaded using this [link](https://snap.stanford.edu/data/sx-askubuntu.txt.gz) and Wiki-talk using this [link](https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz) to `processed`. Run the following to preprocess the data.
```{bash}
python process.py --dataset sx-askubuntu
python process.py --dataset wiki-talk-temporal
```



## Acknowledge
Codes and model implementations are referred to several projects: [NAT](https://github.com/Graph-COM/Neighborhood-Aware-Temporal-Network), [DyGlib](https://github.com/yule-BUAA/DyGLib), [DTGB](https://github.com/zjs123/DTGB). Thanks for their great contributions!
