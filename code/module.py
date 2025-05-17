import torch
import torch.nn as nn
import logging
import time
import numpy as np
import random
import math
from GAT import GAT
import pywt
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F

class ModalTGL(torch.nn.Module):
  def __init__(self, args, n_feat, e_feat, memory_dim, total_nodes, get_checkpoint_path=None, get_ngh_store_path=None, get_self_rep_path=None, get_prev_raw_path=None, time_dim=2, pos_dim=0, n_head=4, num_neighbors=['1', '32'],
      dropout=0.1, linear_out=False, verbosity=1, seed=1, n_hops=2, replace_prob=0.9, self_dim=100, ngh_dim=8, device=None):
    super(ModalTGL, self).__init__()
    self.logger = logging.getLogger(__name__)
    self.dropout = dropout
    self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
    self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
    self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
    self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
    self.time_dim = time_dim  # default to be time feature dimension
    self.self_dim = self_dim
    self.ngh_dim = ngh_dim
    # embedding layers and encoders
    self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
    self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
    self.time_encoder = self.init_time_encoder(args.time_fun) # fourier
    self.device = device

    self.pos_dim = pos_dim
    self.trainable_embedding = nn.Embedding(num_embeddings=64, embedding_dim=self.pos_dim) # position embedding
    
    # final projection layer
    self.linear_out = linear_out
    self.affinity_score = MergeLayer(self.feat_dim + self_dim, self.feat_dim + self_dim, self.feat_dim + self_dim, 1, non_linear=not self.linear_out) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
    self.out_layer = OutLayer(self.feat_dim + self_dim + self_dim, self.feat_dim + self_dim + self_dim, 1)
    self.get_checkpoint_path = get_checkpoint_path
    self.get_ngh_store_path = get_ngh_store_path
    self.get_self_rep_path = get_self_rep_path
    self.get_prev_raw_path = get_prev_raw_path
    self.src_idx_l_prev = self.tgt_idx_l_prev = self.cut_time_l_prev = self.e_idx_l_prev = None
    self.num_neighbors = num_neighbors
    self.n_hops = n_hops
    self.ngh_id_idx = 0
    self.e_raw_idx = 1
    self.ts_raw_idx = 2
    self.num_raw = 3

    self.ngh_rep_idx = [self.num_raw, self.num_raw + self.ngh_dim]

    self.memory_dim = memory_dim
    self.verbosity = verbosity
    
    self.attn_dim = self.feat_dim + self.ngh_dim + self.pos_dim
    self.gat = GAT(1, [n_head], [self.attn_dim, self.feat_dim], add_skip_connection=False, bias=True,
                 dropout=dropout, log_attention_weights=False)
    self.total_nodes = total_nodes
    self.replace_prob = replace_prob
    self.self_rep_linear = nn.Linear(self.self_dim + self.time_dim + self.e_feat_dim, self.self_dim, bias=False)
    self.ngh_rep_linear = nn.Linear(self.self_dim + self.time_dim + self.e_feat_dim, self.ngh_dim, bias=False)
    self.self_aggregator = self.init_self_aggregator() # RNN
    self.ngh_aggregator = self.init_ngh_aggregator() # RNN
    self.film_dim = self.feat_dim + self.self_dim + self.self_dim
    self.film_generator = FiLMGenerator(self.film_dim)
    self.W = nn.Parameter(torch.ones(*[1, self.film_dim]))
    torch.nn.init.kaiming_normal_(self.W)
    self.b = nn.Parameter(torch.zeros(1))
    self.tunning_weight = args.tunning_weight
    
  def set_seed(self, seed):
    self.seed = seed

  def clear_store(self):
    self.neighborhood_store = None

  def reset_store(self):
    ngh_stores = []
    for i in self.num_neighbors:
      max_e_idx = self.total_nodes * i
      raw_store = torch.zeros(max_e_idx, self.num_raw)
      hidden_store = torch.empty(max_e_idx, self.ngh_dim)
      ngh_store = torch.cat((raw_store, nn.init.xavier_uniform_(hidden_store)), -1).to(self.device)
      ngh_stores.append(ngh_store)
    self.neighborhood_store = ngh_stores
    self.self_rep = torch.zeros(self.total_nodes, self.self_dim).to(self.device)
    self.prev_raw = torch.zeros(self.total_nodes, 3).to(self.device)
  
  def get_neighborhood_store(self):
    return self.neighborhood_store

  def set_neighborhood_store(self, neighborhood_store):
    self.neighborhood_store = neighborhood_store

  def set_num_neighbors_stored(self, num_neighbors_stored):
    self.num_neighbors_stored = num_neighbors_stored

  def clear_self_rep(self):
    self.self_rep = None
    self.prev_raw = None

  def reset_self_rep(self):
    self.self_rep = torch.zeros_like(self.self_rep)
    self.prev_raw = torch.zeros_like(self.prev_raw)

  def set_self_rep(self, self_rep, prev_raw):
    self.self_rep = self_rep
    self.prev_raw = prev_raw

  def set_device(self, device):
    self.device = device

  def log_time(self, desc, start, end):
    if self.verbosity > 1:
      self.logger.info('{} for the minibatch, time eclipsed: {} seconds'.format(desc, str(end-start)))
  
  def position_bits(self, bs, hop):
    # return torch.zeros(bs * self.num_neighbors[hop], device=self.device) << hop
    return torch.log2(torch.ones(bs * self.num_neighbors[hop], device=self.device)) * hop

  def contrast(self, src_l_cut, tgt_l_cut, bad_l_cut, cut_time_l, e_idx_l=None, test=False):
    start = time.time()
    start_t = time.time()
    batch_size = len(src_l_cut)
    
    # Move data to the GPU
    src_th = torch.from_numpy(src_l_cut).to(dtype=torch.long, device=self.device)  # node id
    tgt_th = torch.from_numpy(tgt_l_cut).to(dtype=torch.long, device=self.device)
    bad_th = torch.from_numpy(bad_l_cut).to(dtype=torch.long, device=self.device)
    
    idx_th = torch.cat((src_th, tgt_th, bad_th), 0)
    cut_time_th = torch.from_numpy(cut_time_l).to(dtype=torch.float, device=self.device)
    e_idx_th = torch.from_numpy(e_idx_l).to(dtype=torch.long, device=self.device)
    end = time.time()
    batch_idx = torch.arange(batch_size * 3, device=self.device)
    start = time.time()

    self.neighborhood_store[0][idx_th, 0] = idx_th.float()
    # n_id is the node idx of neighbors of query node
    # dense_idx is the position of each neighbors in the batch*nngh tensor
    # sprase_idx is a tensor of batch idx repeated with ngh_n timesfor each node
   
    h0_pos_bit = self.position_bits(3 * batch_size, hop=0)
    updated_mem_h0 = self.batch_fetch_ncaches(idx_th, cut_time_th.repeat(3), hop=0)
    updated_mem_h0_with_pos = torch.cat((updated_mem_h0, h0_pos_bit.unsqueeze(1)), -1)
    feature_dim = self.memory_dim + 1
    updated_mem = updated_mem_h0_with_pos.view(3 * batch_size, self.num_neighbors[0], -1)
    updated_mem_h1 = None
    if self.n_hops > 0:
      h1_pos_bit = self.position_bits(3 * batch_size, hop=1)
      updated_mem_h1 = self.batch_fetch_ncaches(idx_th, cut_time_th.repeat(3), hop=1)
      updated_mem_h1_with_pos = torch.cat((updated_mem_h1, h1_pos_bit.unsqueeze(1)), -1)
      updated_mem = torch.cat((
        updated_mem,
        updated_mem_h1_with_pos.view(3 * batch_size, self.num_neighbors[1], -1)), 1)
    if self.n_hops > 1:
      # second-hop N-cache access
      h2_pos_bit = self.position_bits(3 * batch_size, hop=2)
      updated_mem_h2 = torch.cat((self.batch_fetch_ncaches(idx_th, cut_time_th.repeat(3), hop=2), h2_pos_bit.unsqueeze(1)), -1)
      updated_mem = torch.cat((updated_mem, updated_mem_h2.view(3 * batch_size, self.num_neighbors[2], -1)), 1)

    updated_mem = updated_mem.view(-1, feature_dim)
    ngh_id = updated_mem[:, self.ngh_id_idx].long()
    ngh_exists = torch.nonzero(ngh_id, as_tuple=True)[0]
    ngh_count = torch.count_nonzero(ngh_id.view(3, batch_size, -1), dim=-1)

    ngh_id = ngh_id.index_select(0, ngh_exists)
    updated_mem = updated_mem.index_select(0, ngh_exists)
    src_ngh_n_th, tgt_ngh_n_th, bad_ngh_n_th = ngh_count[0], ngh_count[1], ngh_count[2]
    ngh_n_th = torch.cat((src_ngh_n_th, tgt_ngh_n_th, bad_ngh_n_th), 0)
    ori_idx = torch.repeat_interleave(idx_th, ngh_n_th)
    sparse_idx = torch.repeat_interleave(batch_idx, ngh_n_th).long()
    src_nghs = torch.sum(src_ngh_n_th)
    tgt_nghs = torch.sum(tgt_ngh_n_th)
    bad_nghs = torch.sum(bad_ngh_n_th)

    node_features = self.node_raw_embed(ngh_id)

    pos_raw = updated_mem[:, -1]
    src_pos_raw = pos_raw[0:src_nghs]
    # for the target nodes, shift all the bits by 3 to differentiate from the source nodes
    tgt_pos_raw = torch.log2(pos_raw[src_nghs:src_nghs + tgt_nghs]) * 3
    bad_pos_raw = torch.log2(pos_raw[src_nghs + tgt_nghs:]) * 3
    pos_raw = torch.cat((src_pos_raw, tgt_pos_raw, bad_pos_raw), -1)
    hidden_states = torch.cat((node_features, updated_mem[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]], pos_raw.unsqueeze(1)), -1)
    
    src_prev_f = hidden_states[0:src_nghs]
    tgt_prev_f = hidden_states[src_nghs:src_nghs + tgt_nghs]
    bad_prev_f = hidden_states[src_nghs + tgt_nghs:]

    src_ngh_id = ngh_id[0:src_nghs]
    tgt_ngh_id = ngh_id[src_nghs:src_nghs + tgt_nghs]
    bad_ngh_id = ngh_id[src_nghs + tgt_nghs:]
    src_sparse_idx = sparse_idx[0:src_nghs]
    src_n_sparse_idx = src_sparse_idx + batch_size
    tgt_bad_sparse_idx = sparse_idx[src_nghs:] - batch_size
    tgt_sparse_idx = sparse_idx[src_nghs:src_nghs + tgt_nghs] - batch_size
    bad_sparse_idx = sparse_idx[src_nghs + tgt_nghs:] - batch_size
    
    # joint features construction
    joint_p, ngh_and_batch_id_p = self.get_joint_feature(src_sparse_idx, tgt_sparse_idx, src_ngh_id, tgt_ngh_id, src_prev_f, tgt_prev_f)
    joint_n, ngh_and_batch_id_n = self.get_joint_feature(src_n_sparse_idx, bad_sparse_idx, src_ngh_id, bad_ngh_id, src_prev_f, bad_prev_f)
    joint_p = self.get_position_encoding(joint_p)
    joint_n = self.get_position_encoding(joint_n)

    features = torch.cat((joint_p, joint_n), 0)

    src_self_rep = self.updated_self_rep(src_th)
    tgt_self_rep = self.updated_self_rep(tgt_th)
    bad_self_rep = self.updated_self_rep(bad_th)

    p_score, n_score, attn_score, tunning_loss = self.forward(ngh_and_batch_id_p, ngh_and_batch_id_n, features, batch_size, src_self_rep, tgt_self_rep, bad_self_rep)
    end = time.time()
    self.log_time('attention', start, end)
    
    self.self_rep[src_th] = src_self_rep.detach()
    self.self_rep[tgt_th] = tgt_self_rep.detach()

    self.prev_raw[src_th] = torch.stack([tgt_th, e_idx_th, cut_time_th], dim = 1)
    self.prev_raw[tgt_th] = torch.stack([src_th, e_idx_th, cut_time_th], dim = 1)

    # N-cache update
    self.update_memory(src_th, tgt_th, e_idx_th, cut_time_th, updated_mem_h0, updated_mem_h1, batch_size)
    return p_score.sigmoid(), n_score.sigmoid(), tunning_loss
  
  def get_position_encoding(self, joint):
    if self.pos_dim == 0:
      return joint[:, :-1]
    pos_raw = joint[:, -1]
    pos_encoding = self.trainable_embedding(pos_raw.long())
    return torch.cat((joint[:, :-1], pos_encoding), -1)
    

  def updated_self_rep(self, node_id):
    self_store = self.prev_raw[node_id]
    oppo_id = self_store[:, self.ngh_id_idx].long()
    e_raw = self_store[:,self.e_raw_idx].long()
    ts_raw = self_store[:,self.ts_raw_idx]
    e_feat = self.edge_raw_embed(e_raw)
    # ts_feat = self.time_encoder(ts_raw)
    ts_feat = self.time_encoder(ts_raw) * 0
    prev_self_rep = self.self_rep[node_id]
    prev_oppo_rep = self.self_rep[oppo_id]
    updated_self_rep = self.self_aggregator(self.self_rep_linear(torch.cat((prev_oppo_rep, e_feat, ts_feat), -1)), prev_self_rep)
    return updated_self_rep

  def update_memory(self, src_th, tgt_th, e_idx_th, cut_time_th, updated_mem_h0, updated_mem_h1, batch_size):
    ori_idx = torch.cat((src_th, tgt_th), 0)
    cut_time_th = cut_time_th.repeat(2)
    opp_th = torch.cat((tgt_th, src_th), 0)
    e_idx_th = e_idx_th.repeat(2)
    # Update neighbors
    batch_id = torch.arange(batch_size * 2, device=self.device)
    if self.n_hops > 0:
      updated_mem_h1 = updated_mem_h1.detach()[:2 * batch_size * self.num_neighbors[1]]
      # Update second hop neighbors
      if self.n_hops > 1:
        ngh_h1_id = updated_mem_h1[:, self.ngh_id_idx].long()
        ngh_exists = torch.nonzero(ngh_h1_id, as_tuple=True)[0]
        updated_mem_h2 = updated_mem_h1.index_select(0, ngh_exists)
        ngh_count = torch.count_nonzero(ngh_h1_id.view(2 * batch_size, self.num_neighbors[1]), dim=-1)
        opp_expand_th = torch.repeat_interleave(opp_th, ngh_count, dim=0)
        self.update_ncaches(opp_expand_th, updated_mem_h2, 2)
      updated_mem_h1 = updated_mem_h1[(batch_id * self.num_neighbors[1] + self.ncache_hash(opp_th, 1))]
      ngh_id_is_match = (updated_mem_h1[:, self.ngh_id_idx] == opp_th).unsqueeze(1).repeat(1, self.memory_dim)
      updated_mem_h1 = updated_mem_h1 * ngh_id_is_match

      candidate_ncaches = torch.cat((opp_th.unsqueeze(1), e_idx_th.unsqueeze(1), cut_time_th.unsqueeze(1), updated_mem_h1[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]), -1)
      self.update_ncaches(ori_idx, candidate_ncaches, 1)
    # Update self
    updated_mem_h0 = updated_mem_h0.detach()[:batch_size * self.num_neighbors[0] * 2]
    candidate_ncaches = torch.cat((ori_idx.unsqueeze(1), e_idx_th.unsqueeze(1), cut_time_th.unsqueeze(1), updated_mem_h0[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]), -1)
    self.update_ncaches(ori_idx, candidate_ncaches, 0)

  def ncache_hash(self, ngh_id, hop):
    ngh_id = ngh_id.long()
    return ((ngh_id * (self.seed % 100) + ngh_id * ngh_id * ((self.seed % 100) + 1)) % self.num_neighbors[hop]).int()

  def update_ncaches(self, self_id, candidate_ncaches, hop):
    if self.num_neighbors[hop] == 0:
      return
    ngh_id = candidate_ncaches[:, self.ngh_id_idx]
    idx = self_id * self.num_neighbors[hop] + self.ncache_hash(ngh_id, hop)
    is_occupied = torch.logical_and(self.neighborhood_store[hop][idx,self.ngh_id_idx] != 0, self.neighborhood_store[hop][idx,self.ngh_id_idx] != ngh_id)
    should_replace =  (is_occupied * torch.rand(is_occupied.shape[0], device=self.device)) < self.replace_prob
    idx *= should_replace
    idx *= ngh_id != 0
    self.neighborhood_store[hop][idx] = candidate_ncaches

  def store_memory(self, n_id, e_pos_th, ts_th, e_th, agg_p):
    prev_data = torch.cat((n_id.unsqueeze(1), e_th.unsqueeze(1), ts_th.unsqueeze(1), agg_p), -1)
    self.neighborhood_store[0][e_pos_th.long()] = prev_data

  def get_joint_neighborhood(self, src_sparse_idx, tgt_sparse_idx, src_n_id, tgt_n_id, src_hidden, tgt_hidden):
    sparse_idx = torch.cat((src_sparse_idx, tgt_sparse_idx), 0)
    n_id = torch.cat((src_n_id, tgt_n_id), 0)
    all_hidden = torch.cat((src_hidden, tgt_hidden), 0)
    feat_dim = src_hidden.shape[-1]
    key = torch.cat((sparse_idx.unsqueeze(1), n_id.unsqueeze(1)), -1) # tuple of (idx in the current batch, n_id)
    unique, inverse_idx = key.unique(return_inverse=True, dim=0)
    # SCATTER ADD FOR TS WITH INV IDX
    relative_ts = torch.zeros(unique.shape[0], feat_dim, device=self.device)
    relative_ts.scatter_add_(0, inverse_idx.unsqueeze(1).repeat(1,feat_dim), all_hidden)
    relative_ts = relative_ts.index_select(0, inverse_idx)
    assert(relative_ts.shape[0] == sparse_idx.shape[0] == all_hidden.shape[0])

    return relative_ts

  def get_joint_feature(self, src_sparse_idx, tgt_sparse_idx, src_n_id, tgt_n_id, src_hidden, tgt_hidden):
    sparse_idx = torch.cat((src_sparse_idx, tgt_sparse_idx), 0)
    n_id = torch.cat((src_n_id, tgt_n_id), 0)
    all_hidden = torch.cat((src_hidden, tgt_hidden), 0)
    feat_dim = src_hidden.shape[-1]
    key = torch.cat((n_id.unsqueeze(1), sparse_idx.unsqueeze(1)), -1) # tuple of (idx in the current batch, n_id)
    unique, inverse_idx = key.unique(return_inverse=True, dim=0)
    # SCATTER ADD FOR TS WITH INV IDX
    relative_ts = torch.zeros(unique.shape[0], feat_dim, device=self.device)
    relative_ts.scatter_add_(0, inverse_idx.unsqueeze(1).repeat(1,feat_dim), all_hidden)
    return relative_ts, unique

  def batch_fetch_ncaches(self, ori_idx, curr_time, hop):
    ngh = self.neighborhood_store[hop].view(self.total_nodes, self.num_neighbors[hop], self.memory_dim)[ori_idx].view(ori_idx.shape[0] * (self.num_neighbors[hop]), self.memory_dim)
    curr_time = curr_time.repeat_interleave(self.num_neighbors[hop])
    ngh_id = ngh[:,self.ngh_id_idx].long()
    ngh_e_raw = ngh[:,self.e_raw_idx].long()
    ngh_ts_raw = ngh[:,self.ts_raw_idx]
    prev_ngh_rep = ngh[:,self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]
    e_feat = self.edge_raw_embed(ngh_e_raw)
    # ts_feat = self.time_encoder(ngh_ts_raw)
    ts_feat = self.time_encoder(ngh_ts_raw) * 0
    ngh_self_rep = self.self_rep[ngh_id]
    updated_self_rep = self.ngh_aggregator(self.ngh_rep_linear(torch.cat((ngh_self_rep, e_feat, ts_feat), -1)), prev_ngh_rep)
    updated_self_rep *= (ngh_ts_raw != 0).unsqueeze(1).repeat(1, self.ngh_dim)
    ori_idx = torch.repeat_interleave(ori_idx, self.num_neighbors[hop])
    # msk = ngh_ts_raw <= curr_time
    updated_mem = torch.cat((ngh[:, :self.num_raw], updated_self_rep), -1)
    # updated_mem *= msk.unsqueeze(1).repeat(1, self.memory_dim)
    return updated_mem


  def forward(self, ngh_and_batch_id_p, ngh_and_batch_id_n, feat, bs, src_self_rep=None, tgt_self_rep=None, bad_self_rep=None):
    edge_idx = torch.cat((ngh_and_batch_id_p, ngh_and_batch_id_n), dim=0).T
    embed, _, attn_score = self.gat((feat, edge_idx.long(), 2*bs))
    # feat include pos_feat + neg_feat 
    p_embed = embed[:bs]  # [b,out_d]
    n_embed = embed[bs:2*bs]
    p_feat = feat[:bs]  # [b,in_d]
    n_feat = feat[bs:2*bs]
    if src_self_rep is not None:
      assert(tgt_self_rep is not None)
      assert(bad_self_rep is not None)
      p_embed = torch.cat((p_embed, src_self_rep, tgt_self_rep), -1)
      n_embed = torch.cat((n_embed, src_self_rep, bad_self_rep), -1)
        
    new_p_embed, p_loss = self.film(p_embed)
    new_n_embed, n_loss = self.film(n_embed)
    # tunning_loss = (p_loss + n_loss) * self.tunning_weight
    tunning_loss = 0
    
    # p_embed = (new_p_embed * self.tunning_weight + p_embed)
    # n_embed = (new_n_embed * self.tunning_weight + n_embed)
    
    p_score = self.out_layer(p_embed).squeeze_(dim=-1)
    n_score = self.out_layer(n_embed).squeeze_(dim=-1)
    return p_score, n_score, attn_score, tunning_loss
    
  def film(self, emb):
    alpha, beta = self.film_generator(emb)
    W = self.W
    b = self.b
    film_W = torch.mul(W, alpha[0]) + beta[0]
    film_b = torch.mul(b, alpha[1]) + beta[1]
    output = emb * film_W + film_b
    tunning_loss = [torch.norm(s, dim=1) for s in alpha]
    tunning_loss = [torch.mean(s) for s in tunning_loss]
    tunning_loss = torch.sum(torch.stack(tunning_loss))
    tunning_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=1)) for s in beta]))
    return output, tunning_loss

  def init_time_encoder(self, time_fun):
    return TimeEncode(time_fun, expand_dim=self.time_dim)

  def init_self_aggregator(self):
    return FeatureEncoderGRU(self.self_dim, self.self_dim, self.dropout)

  def init_ngh_aggregator(self):
    return FeatureEncoderGRU(self.ngh_dim, self.ngh_dim, self.dropout)

class FeatureEncoderGRU(torch.nn.Module):
  def __init__(self, input_dim, ngh_dim, dropout_p=0.1):
    super(FeatureEncoderGRU, self).__init__()
    self.gru = nn.GRUCell(input_dim, ngh_dim)
    self.dropout = nn.Dropout(dropout_p)
    self.ngh_dim = ngh_dim

  def forward(self, input_features, hidden_state, use_dropout=False):
    encoded_features = self.gru(input_features, hidden_state)
    if use_dropout:
      encoded_features = self.dropout(encoded_features)
    
    # return input_features
    return encoded_features

class TimeEncode(nn.Module):
    def __init__(self, time_fun, expand_dim, factor=5, t_min=0.0, t_max=1.0, learnable=True):
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.factor = factor
        self.mode = time_fun  # 'cosine', 'gaussian', or 'wavelet'

        # Cosine encoding parameters
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())

        # Gaussian encoding parameters
        mu = torch.linspace(t_min, t_max, expand_dim).float()
        sigma = torch.full((expand_dim,), (t_max - t_min) / expand_dim).float()
        if learnable:
            self.mu = nn.Parameter(mu)
            self.sigma = nn.Parameter(sigma)
        else:
            self.register_buffer('mu', mu)
            self.register_buffer('sigma', sigma)

        # Wavelet encoding parameters (similar to cosine but with envelope)
        self.wavelet_freq = nn.Parameter(torch.linspace(1.0, 10.0, expand_dim))  # or logspace
        self.wavelet_sigma = nn.Parameter(torch.full((expand_dim,), 1.0))  # initial envelope width

    def forward(self, ts):
        if self.mode == 'cosine':
            return self.cosine_time_encoding(ts)
        elif self.mode == 'gaussian':
            return self.gaussian_time_encoding(ts)
        elif self.mode == 'wavelet':
            return self.wavelet_time_encoding(ts)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def cosine_time_encoding(self, ts):
        batch_size = ts.size(0)
        ts_fourier = ts.view(batch_size, 1)  # [N, 1]
        map_ts = ts_fourier * self.basis_freq.view(1, -1)  # [N, D]
        map_ts += self.phase.view(1, -1)
        time_emb = torch.cos(map_ts)
        return time_emb

    def gaussian_time_encoding(self, ts):
        ts = ts.view(-1, 1)  # [N, 1]
        mu = self.mu.view(1, -1)
        sigma = self.sigma.view(1, -1) + 1e-6
        time_emb = torch.exp(- ((ts - mu) ** 2) / (2 * sigma ** 2))
        return time_emb

    def wavelet_time_encoding(self, ts):
        """
        Morlet-like wavelet: cos(wt) * exp(-t^2 / (2 * sigma^2))
        """
        ts = ts.view(-1, 1)  # [N, 1]
        freq = self.wavelet_freq.view(1, -1)  # [1, D]
        sigma = self.wavelet_sigma.view(1, -1) + 1e-6  # [1, D]

        # Time shift for centering wavelets around zero
        centered_ts = ts - ts.mean()

        wave = torch.cos(2 * np.pi * centered_ts * freq)
        envelope = torch.exp(- (centered_ts ** 2) / (2 * sigma ** 2))
        time_emb = wave * envelope  # Element-wise product

        return time_emb


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
    super().__init__()
    #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

    # special linear layer for motif explainability
    self.non_linear = non_linear
    if not non_linear:
      assert(dim1 == dim2)
      self.fc = nn.Linear(dim1, 1)
      torch.nn.init.xavier_normal_(self.fc1.weight)

  def forward(self, x1, x2):
    z_walk = None
    if self.non_linear:
      x = torch.cat([x1, x2], dim=-1)
      #x = self.layer_norm(x)
      h = self.act(self.fc1(x))
      z = self.fc2(h)
    else: # for explainability
      # x1, x2 shape: [B, M, F]
      x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
      z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
      z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
    return z, z_walk

class OutLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.fc2 = torch.nn.Linear(dim2, dim3)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    h = self.act(self.fc1(x))
    z = self.fc2(h)
    return z

class FiLMGenerator(nn.Module):
  def __init__(self, output_dim):
    super(FiLMGenerator, self).__init__()
    self.output_dim = output_dim
    self.vars_alpha = nn.ParameterList()
    w1 = nn.Parameter(torch.ones(*[output_dim+1, output_dim]))
    torch.nn.init.kaiming_normal_(w1)
    self.vars_alpha.append(w1)
    self.vars_alpha.append(nn.Parameter(torch.zeros(output_dim+1)))
    
    self.vars_beta = nn.ParameterList()
    w2 = nn.Parameter(torch.ones(*[output_dim+1, output_dim]))
    torch.nn.init.kaiming_normal_(w2)
    self.vars_beta.append(w2)
    self.vars_beta.append(nn.Parameter(torch.zeros(output_dim+1)))

  def forward(self, emb):
    
    vars_alpha = self.vars_alpha
    a = F.linear(emb, vars_alpha[0], vars_alpha[1])
    a = F.leaky_relu(a)
    a = a.T
    a1 = a[:self.output_dim].T #.view(x.size(0), self.args.out_dim)
    a2 = a[self.output_dim:].T #.view(x.size(0), 1)
    alpha_list = [a1, a2]
    
    vars_beta = self.vars_beta
    b = F.linear(emb, vars_beta[0], vars_beta[1])
    b = F.leaky_relu(b)
    b = b.T
    b1 = b[:self.output_dim].T #.view(x.size(0), self.args.out_dim)
    b2 = b[self.output_dim:].T #.view(x.size(0), 1)
    beta_list = [b1, b2]
    return alpha_list, beta_list