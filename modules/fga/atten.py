#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product, permutations, combinations_with_replacement, chain


class Unary(nn.Module):
    def __init__(self, embed_size):
        """
            Captures local entity information
        :param embed_size:  the embedding dimension
        """
        super(Unary, self).__init__()
        self.embed = nn.Conv1d(embed_size, embed_size, 1)
        self.feature_reduce = nn.Conv1d(embed_size, 1, 1)

    def forward(self, X):
        X = X.transpose(1, 2)

        X_embed = self.embed(X)

        X_nl_embed = F.dropout(F.relu(X_embed))
        X_poten = self.feature_reduce(X_nl_embed)
        return X_poten.squeeze(1)


class Pairwise(nn.Module):
    def __init__(self, embed_x_size, x_spatial_dim=None, embed_y_size=None, y_spatial_dim=None):
        """
            Captures interaction between utilities or entities of the same utility
        :param embed_x_size: the embedding dimension of the first utility
        :param x_spatial_dim: the spatial dimension of the first utility for batch norm and weighted marginalization
        :param embed_y_size: the embedding dimension of the second utility (none for self-interactions)
        :param y_spatial_dim: the spatial dimension of the second utility for batch norm and weighted marginalization
        """

        super(Pairwise, self).__init__()
        embed_y_size = embed_y_size if y_spatial_dim is not None else embed_x_size
        self.y_spatial_dim = y_spatial_dim if y_spatial_dim is not None else x_spatial_dim

        self.embed_size = max(embed_x_size, embed_y_size)
        self.x_spatial_dim = x_spatial_dim

        self.embed_X = nn.Conv1d(embed_x_size, self.embed_size, 1)
        self.embed_Y = nn.Conv1d(embed_y_size, self.embed_size, 1)
        if x_spatial_dim is not None:
            self.normalize_S = nn.BatchNorm1d(self.x_spatial_dim * self.y_spatial_dim)

            self.margin_X = nn.Conv1d(self.y_spatial_dim, 1, 1)
            self.margin_Y = nn.Conv1d(self.x_spatial_dim, 1, 1)

    def forward(self, X, Y=None):

        X_t = X.transpose(1, 2)
        Y_t = Y.transpose(1, 2) if Y is not None else X_t


        X_embed = self.embed_X(X_t)
        Y_embed = self.embed_Y(Y_t)

        X_norm = F.normalize(X_embed)
        Y_norm = F.normalize(Y_embed)

        S = X_norm.transpose(1, 2).bmm(Y_norm)
        if self.x_spatial_dim is not None:
            S = self.normalize_S(S.view(-1, self.x_spatial_dim * self.y_spatial_dim)) \
                .view(-1, self.x_spatial_dim, self.y_spatial_dim)

            X_poten = self.margin_X(S.transpose(1, 2)).transpose(1, 2).squeeze(2)
            Y_poten = self.margin_Y(S).transpose(1, 2).squeeze(2)
        else:
            X_poten = S.mean(dim=2, keepdim=False)
            Y_poten = S.mean(dim=1, keepdim=False)

        if Y is None:
            return X_poten
        else:
            return X_poten, Y_poten


class Atten(nn.Module):
    def __init__(self, util_e, sharing_factor_weights=[], prior_flag=False,
                 sizes=[], size_force=False, pairwise_flag=True,
                 unary_flag=True, self_flag=True):
        """
            The class performs an attention on a given list of utilities representation.
        :param util_e: the embedding dimensions
        :param sharing_factor_weights:  To share weights, provide a dict of tuples:
         {idx: (num_utils, connected utils)
         Note, for efficiency, the shared utils (i.e., history, are connected to ans
          and question only.
         TODO: connections between shared utils
        :param prior_flag: is prior factor provided
        :param sizes: the spatial simension (used for batch-norm and weighted marginalization)
        :param size_force: force spatial size with adaptive avg pooling.
        :param pairwise_flag: use pairwise interaction between utilities
        :param unary_flag: use local information
        :param self_flag: use self interactions between utilitie's entities
        """
        super(Atten, self).__init__()
        self.util_e = util_e

        self.prior_flag = prior_flag

        self.n_utils = len(util_e)

        self.spatial_pool = nn.ModuleDict()

        self.un_models = nn.ModuleList()

        self.self_flag = self_flag
        self.pairwise_flag = pairwise_flag
        self.unary_flag = unary_flag
        self.size_force = size_force

        if len(sizes) == 0:
            sizes = [None for _ in util_e]

        self.sharing_factor_weights = sharing_factor_weights

        #force the provided size
        for idx, e_dim in enumerate(util_e):
            self.un_models.append(Unary(e_dim))
            if self.size_force:
                self.spatial_pool[str(idx)] = nn.AdaptiveAvgPool1d(sizes[idx])

        #Pairwise
        self.pp_models = nn.ModuleDict()
        for ((idx1, e_dim_1), (idx2, e_dim_2)) \
                in combinations_with_replacement(enumerate(util_e), 2):
            # self
            if self.self_flag and idx1 == idx2:
                self.pp_models[str(idx1)] = Pairwise(e_dim_1, sizes[idx1])
            else:
                if pairwise_flag:
                    if idx1 in self.sharing_factor_weights:
                        # not connected
                        if idx2 not in self.sharing_factor_weights[idx1][1]:
                           continue
                    if idx2 in self.sharing_factor_weights:
                        # not connected
                        if idx1 not in self.sharing_factor_weights[idx2][1]:
                            continue
                    self.pp_models[str((idx1, idx2))] = Pairwise(e_dim_1, sizes[idx1], e_dim_2, sizes[idx2])

        # Handle reduce potentials (with scalars)
        self.reduce_potentials = nn.ModuleList()

        self.num_of_potentials = dict()

        self.default_num_of_potentials = 0

        if self.self_flag:
            self.default_num_of_potentials += 1
        if self.unary_flag:
            self.default_num_of_potentials += 1
        if self.prior_flag:
            self.default_num_of_potentials += 1
        for idx in range(self.n_utils):
            self.num_of_potentials[idx] = self.default_num_of_potentials

        '''
         All other utilities
        '''
        if pairwise_flag:
            for idx, (num_utils, connected_utils) in sharing_factor_weights:
                for c_u in connected_utils:
                    self.num_of_potentials[c_u] += num_utils
                    self.num_of_potentials[idx] += 1
            for k in self.num_of_potentials:
                if k not in self.sharing_factor_weights:
                    self.num_of_potentials[k] += (self.n_utils - 1) \
                                                 - len(sharing_factor_weights)

        for idx in range(self.n_utils):
            self.reduce_potentials.append(nn.Conv1d(self.num_of_potentials[idx],
                                                    1, 1, bias=False))

    def forward(self, utils, priors=None):
        assert self.n_utils == len(utils)
        assert (priors is None and not self.prior_flag) \
               or (priors is not None
                   and self.prior_flag
                   and len(priors) == self.n_utils)
        b_size = utils[0].size(0)
        util_factors = dict()
        attention = list()

        #Force size, constant size is used for pairwise batch normalization
        if self.size_force:
            for i, (num_utils, _) in self.sharing_factor_weights.items():
                if str(i) not in self.spatial_pool.keys():
                    continue
                else:
                    high_util = utils[i]
                    high_util = high_util.view(num_utils * b_size, high_util.size(2), high_util.size(3))
                    high_util = high_util.transpose(1, 2)
                    utils[i] = self.spatial_pool[str(i)](high_util).transpose(1, 2)

            for i in range(self.n_utils):
                if i in self.sharing_factor_weights \
                        or str(i) not in self.spatial_pool.keys():
                    continue
                utils[i] = utils[i].transpose(1, 2)
                utils[i] = self.spatial_pool[str(i)](utils[i]).transpose(1, 2)
                if self.prior_flag and priors[i] is not None:
                    priors[i] = self.spatial_pool[str(i)](priors[i].unsqueeze(1)).squeeze(1)

        # handle Shared weights
        for i, (num_utils, connected_list) in self.sharing_factor_weights:
            if self.unary_flag:
                util_factors.setdefault(i, []).append(self.un_models[i](utils[i]))

            if self.self_flag:
                util_factors.setdefault(i, []).append(self.pp_models[str(i)](utils[i]))

            if self.pairwise_flag:
                for j in connected_list:
                    other_util = utils[j]
                    expanded_util = other_util.unsqueeze(1).expand(b_size,
                                                                   num_utils,
                                                                   other_util.size(1),
                                                                   other_util.size(2)).contiguous().view(
                        b_size * num_utils,
                        other_util.size(1),
                        other_util.size(2))

                    if i < j:
                        factor_ij, factor_ji = self.pp_models[str((i, j))](utils[i], expanded_util)
                    else:
                        factor_ji, factor_ij = self.pp_models[str((j, i))](expanded_util, utils[i])
                    util_factors[i].append(factor_ij)
                    util_factors.setdefault(j, []).append(factor_ji.view(b_size, num_utils, factor_ji.size(1)))

        # handle local factors
        for i in range(self.n_utils):
            if i in self.sharing_factor_weights:
                continue
            if self.unary_flag:
                util_factors.setdefault(i, []).append(self.un_models[i](utils[i]))
            if self.self_flag:
                util_factors.setdefault(i, []).append(self.pp_models[str(i)](utils[i]))

        # joint
        if self.pairwise_flag:
            for (i, j) in combinations_with_replacement(range(self.n_utils), 2):
                if i in self.sharing_factor_weights \
                        or j in self.sharing_factor_weights:
                    continue
                if i == j:
                    continue
                else:
                    factor_ij, factor_ji = self.pp_models[str((i, j))](utils[i], utils[j])
                    util_factors.setdefault(i, []).append(factor_ij)
                    util_factors.setdefault(j, []).append(factor_ji)

        # perform attention
        for i in range(self.n_utils):
            if self.prior_flag:
                prior = priors[i] \
                    if priors[i] is not None \
                    else torch.zeros_like(util_factors[i][0], requires_grad=False).cuda()

                util_factors[i].append(prior)

            util_factors[i] = torch.cat([p if len(p.size()) == 3 else p.unsqueeze(1)
                                       for p in util_factors[i]], dim=1)
            util_factors[i] = self.reduce_potentials[i](util_factors[i]).squeeze(1)
            util_factors[i] = F.softmax(util_factors[i], dim=1).unsqueeze(2)
            attention.append(torch.bmm(utils[i].transpose(1, 2), util_factors[i]).squeeze(2))

        return attention


class NaiveAttention(nn.Module):
    def __init__(self):
        """
            Used for ablation analysis - removing attention.
        """
        super(NaiveAttention, self).__init__()

    def forward(self, utils, priors):
        atten = []
        spatial_atten = []
        for u, p in zip(utils, priors):
            if type(u) is tuple:
                u = u[1]
                num_elements = u.shape[0]
                if p is not None:
                    u = u.view(-1, u.shape[-2], u.shape[-1])
                    p = p.view(-1, p.shape[-2], p.shape[-1])
                    spatial_atten.append(
                        torch.bmm(p.transpose(1, 2), u).squeeze(2).view(num_elements, -1, u.shape[-2], u.shape[-1]))
                else:
                    spatial_atten.append(u.mean(2))
                continue
            if p is not None:
                atten.append(torch.bmm(u.transpose(1, 2), p.unsqueeze(2)).squeeze(2))
            else:
                atten.append(u.mean(1))
        return atten, spatial_atten