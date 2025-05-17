import argparse
import sys


def get_args():
  parser = argparse.ArgumentParser('Interface for Neighbourhood-aware Scalable Learning for Temporal Networks')
  
  data = 'amazon'
  tunning_weight = {'amazon': 1, 'ml1m': 0.001, 'brain': 1, 'dblp':0.1, 'google':0.001}
  time_encoding = {'amazon': 'gaussian', 'ml1m': 'gaussian', 'brain': 'wavelet', 'dblp':'cosine', 'google':'cosine'}
  time_dim =  {'amazon': 32, 'ml1m': 16, 'brain': 2, 'dblp': 32, 'google': 2}
  # select dataset and training mode
  parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default=data)
  parser.add_argument('--bs', type=int, default=256, help='batch_size')
  parser.add_argument('--time_fun', type=str, default=time_encoding[data], help='time encoding function')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
  parser.add_argument('-m', '--mode', type=str, default='i', choices=['i', 't'], help='transductive (t) or inductive (i)')
  parser.add_argument('--time_dim', type=int, default=time_dim[data], help='dimension of the time embedding')
  parser.add_argument('--tunning_weight', type=float, default=tunning_weight[data])

  # methodology-related hyper-parameters
  parser.add_argument('--n_degree', nargs='*', default=['32', '16'],
            help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
  parser.add_argument('--n_hop', type=int, default=2, help='number of hops the N-cache is used')
  parser.add_argument('--bias', default=0.0, type=float, help='the hyperparameter alpha controlling sampling preference with time closeness, default to 0 which is uniform sampling')
  parser.add_argument('--pos_dim', type=int, default=16, help='dimension of the positional embedding')
  parser.add_argument('--self_dim', type=int, default=72, help='dimension of the self representation')
  parser.add_argument('--ngh_dim', type=int, default=4, help='dimension of the neighborhood representation')
  parser.add_argument('--linear_out', action='store_true', default=False, help="whether to linearly project each node's ")

  parser.add_argument('--attn_n_head', type=int, default=2, help='number of heads used in tree-shaped attention layer, we only use the default here')
  # general training hyper-parameters
  parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
  parser.add_argument('--drop_out', type=float, default=0.2, help='dropout probability for all dropout layers')
  parser.add_argument('--replace_prob', type=float, default=0.9, help='probability for storing new neighbors to N-cache replacing old ones')
  parser.add_argument('--tolerance', type=float, default=1e-3,
            help='toleratd margainal improvement for early stopper')

  # parameters controlling computation settings but not affecting results in general
  seed = {'amazon': 3407, 'ml1m': 3407, 'brain': 42, 'temfin': 0, 'wiki': 0, 'dblp':42, 'google':0}
  parser.add_argument('--seed', type=int, default=seed[data], help='random seed for all randomized algorithms')
  # parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
  # parser.add_argument('--cpu_cores', type=int, default=1, help='number of cpu_cores used for position encoding')
  parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')
  parser.add_argument('--run', type=int, default=3, help='number of model runs')


  try:
    args = parser.parse_args()
  except:
    parser.print_help()
    sys.exit(0)
  return args, sys.argv