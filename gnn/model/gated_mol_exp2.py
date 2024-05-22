import logging
from dgl import readout
import torch.nn as nn
from gnn.layer.gatedconv_exp2 import GatedGCNConv, GatedGCNConv1, GatedGCNConv2
from gnn.layer.readout_exp2 import Set2SetThenCat,Set2SetThenCat_atom
from gnn.layer.utils import UnifySize

logger = logging.getLogger(__name__)


import torch_scatter
from torch_geometric.utils import get_embeddings
import torch
from torch import nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25,
                 delta: float = 1):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.delta = delta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents_mean):
        latents_shape = latents_mean.shape
        flat_latents = latents_mean.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents_mean and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents_mean.device
        encoding_one_hot_mean = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot_mean.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents_mean
        quantized_latents_mean = torch.matmul(encoding_one_hot_mean, self.embedding.weight)  # [BHW, D]
        quantized_latents_mean = quantized_latents_mean.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents_mean.detach(), latents_mean)
        # 改成两个高斯分布的loss
        embedding_loss = F.mse_loss(quantized_latents_mean, latents_mean.detach())
        #embedding_loss = cal_kl_loss(latents_mean, latents_std, quantized_latents_mean.detach(), quantized_latents_std.detach())

        vq_loss = commitment_loss * self.beta + self.delta * embedding_loss


        # Add the residue back to the latents
        quantized_latents_mean = latents_mean + (quantized_latents_mean - latents_mean).detach()
        avg_probs_mean = torch.mean(encoding_one_hot_mean, dim=0)
        perplexity_mean = torch.exp(-torch.sum(avg_probs_mean * torch.log(avg_probs_mean + 1e-10)))
        # print('perplexity_mean: ',perplexity_mean)


        return quantized_latents_mean, vq_loss, perplexity_mean,encoding_inds,dist

    def sample(self, latents_mean):
        # Convert to one-hot encodings
        device = latents_mean.device    
        latents_shape = latents_mean.shape
        flat_latents = latents_mean.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents_mean and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
               
        # encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        # 从距离矩阵中随机选择一个索引
        encoding_inds = torch.randint(dist.size(1), (dist.size(0), 1), dtype=torch.long, device=device)  # [BHW, 1]

        
        encoding_one_hot_mean = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot_mean.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents_mean
        quantized_latents_mean = torch.matmul(encoding_one_hot_mean, self.embedding.weight)  # [BHW, D]
        quantized_latents_mean = quantized_latents_mean.view(latents_shape)  # [B x H x W x D]

    

        # Add the residue back to the latents
        quantized_latents_mean = latents_mean + (quantized_latents_mean - latents_mean).detach()
        avg_probs_mean = torch.mean(encoding_one_hot_mean, dim=0)
        perplexity_mean = torch.exp(-torch.sum(avg_probs_mean * torch.log(avg_probs_mean + 1e-10)))
        # print('perplexity_mean: ',perplexity_mean)

    

        return quantized_latents_mean

def cal_kl_loss(mu_poster=None, sigma_poster=None, mu_prior=None, sigma_prior=None):
    eps = 10 ** -8
    sigma_poster = sigma_poster ** 2
    sigma_prior = sigma_prior ** 2
    sigma_poster_matrix_det = torch.prod(sigma_poster, dim=1)
    sigma_prior_matrix_det = torch.prod(sigma_prior, dim=1)

    sigma_prior_matrix_inv = 1.0 / sigma_prior
    delta_u = (mu_prior - mu_poster)
    term1 = torch.sum(sigma_poster / sigma_prior, dim=1)
    term2 = torch.sum(delta_u * sigma_prior_matrix_inv * delta_u, 1)
    term3 = - mu_poster.shape[-1]
    term4 = torch.log(sigma_prior_matrix_det + eps) - torch.log(
        sigma_poster_matrix_det + eps)
    kl_loss = 0.5 * (term1 + term2 + term3 + term4)
    kl_loss = torch.clamp(kl_loss, 0, 10)

    return torch.mean(kl_loss)




class GatedGCNMol(nn.Module):
    """
    Gated graph neural network model to predict molecular property.
    This model is similar to most GNN for molecular property such as MPNN and MEGNet.
    It iteratively updates atom, bond, and global features, then aggregates the
    features to form a representation of the molecule, and finally map the
    representation to a molecular property.

    Args:
        in_feats (dict): input feature size.
        embedding_size (int): embedding layer size.
        gated_num_layers (int): number of graph attention layer
        gated_hidden_size (list): hidden size of graph attention layers
        gated_num_fc_layers (int):
        gated_graph_norm (bool):
        gated_batch_norm(bool): whether to apply batch norm to gated layer.
        gated_activation (torch activation): activation fn of gated layers
        gated_residual (bool, optional): [description]. Defaults to False.
        gated_dropout (float, optional): dropout ratio for gated layer.
        fc_num_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_batch_norm (bool): whether to apply batch norm to fc layer
        fc_activation (torch activation): activation fn of fc layers
        fc_dropout (float, optional): dropout ratio for fc layer.
        outdim (int): dimension of the output. For regression, choose 1 and for
            classification, set it to the number of classes.
    """

    def __init__(
        self,
        solute_in_feats,
        solvent_in_feats,
        embedding_size=32,
        gated_num_layers=2,
        gated_hidden_size=[64, 64, 32],
        gated_num_fc_layers=1,
        gated_graph_norm=False,
        gated_batch_norm=True,
        gated_activation="ReLU",
        gated_residual=True,
        gated_dropout=0.0,
        num_lstm_iters=6,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        fc_num_layers=2,
        fc_hidden_size=[32, 16],
        fc_batch_norm=False,
        fc_activation="ReLU",
        fc_dropout=0.0,
        outdim=1,
        conv="GatedGCNConv",
    ):
        super().__init__()

        if isinstance(gated_activation, str):
            gated_activation = getattr(nn, gated_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()
        # embedding layer
        self.solute_embedding = UnifySize(solute_in_feats, embedding_size)
        #self.solvent_embedding = UnifySize(solvent_in_feats, embedding_size)
        
        # gated layer
        if conv == "GatedGCNConv":
            conv_fn = GatedGCNConv
        elif conv == "GatedGCNConv1":
            conv_fn = GatedGCNConv1
        elif conv == "GatedGCNConv2":
            conv_fn = GatedGCNConv2
        else:
            raise ValueError()

        in_size = embedding_size
        self.gated_layers = nn.ModuleList()
        for i in range(gated_num_layers):
            self.gated_layers.append(
                conv_fn(
                    input_dim=in_size,
                    output_dim=gated_hidden_size[i],
                    num_fc_layers=gated_num_fc_layers,
                    graph_norm=gated_graph_norm,
                    batch_norm=gated_batch_norm,
                    activation=gated_activation,
                    residual=gated_residual,
                    dropout=gated_dropout,
                )
            )
            in_size = gated_hidden_size[i]

        # set2set readout layer
        ntypes = ["atom","atom2"]
        self.ntype = ['atom', 'atom2', 'bond', 'bond2', 'global']
        in_size = [gated_hidden_size[-1]] * len(ntypes)

        self.readout_layer2 = Set2SetThenCat_atom(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )
        
        self.readout_layer = Set2SetThenCat(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )

        # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
        # feature twice the the size of in feature)
        readout_out_size = gated_hidden_size[-1] * 2 + gated_hidden_size[-1] * 2
        # for global feat
        if set2set_ntypes_direct is not None:
            readout_out_size += gated_hidden_size[-1] * len(set2set_ntypes_direct)

        readout_out_size *= 2

        #vq_num_embeddings HUANJINGSHULIANG  600 ENMBDING
        #self.vq = VectorQuantizer(vq_num_embeddings, 600, vq_beta, vq_delta)
        self.vq = VectorQuantizer(num_embeddings=10, embedding_dim=400, beta=1, delta=0.1)
        
        # need dropout?
        delta = 1e-3
        if fc_dropout < delta:
            apply_drop = False
        else:
            apply_drop = True

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = 1600

        for i in range(fc_num_layers):
            out_size = fc_hidden_size[i]

            self.fc_layers.append(nn.Linear(in_size, out_size))
            # batch norm
            if fc_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))
            # activation
            self.fc_layers.append(fc_activation)
            # dropout
            if apply_drop:
                self.fc_layers.append(nn.Dropout(fc_dropout))

            in_size = out_size

        # final output layer, mapping feature to the corresponding shape
        self.fc_layers.append(nn.Linear(in_size, outdim))
        self.MLP = nn.Sequential(
            nn.Linear(400, 200), 
            nn.BatchNorm1d(200), 
            nn.ReLU(True),
            nn.Linear(200, 1), 
            nn.Sigmoid())

    def forward(self, graph, feats, norm_atom, norm_bond):
        """
        Args:
            feats (dict)
            norm_atom (2D tensor)
            norm_bond (2D tensor)
        Returns:
            2D tensor: of shape (N, ft_size)
        """

        # embedding
        feats = self.embedding(feats)

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # readout layer
        feats = self.readout_layer(graph, feats)

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats


class AttentionGCN(nn.Module):
    """
    Gated graph neural network model to predict molecular property.
    This model is similar to most GNN for molecular property such as MPNN and MEGNet.
    It iteratively updates atom, bond, and global features, then aggregates the
    features to form a representation of the molecule, and finally map the
    representation to a molecular property.

    Args:
        in_feats (dict): input feature size.
        embedding_size (int): embedding layer size.
        gated_num_layers (int): number of graph attention layer
        gated_hidden_size (list): hidden size of graph attention layers
        gated_num_fc_layers (int):
        gated_graph_norm (bool):
        gated_batch_norm(bool): whether to apply batch norm to gated layer.
        gated_activation (torch activation): activation fn of gated layers
        gated_residual (bool, optional): [description]. Defaults to False.
        gated_dropout (float, optional): dropout ratio for gated layer.
        fc_num_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_batch_norm (bool): whether to apply batch norm to fc layer
        fc_activation (torch activation): activation fn of fc layers
        fc_dropout (float, optional): dropout ratio for fc layer.
        outdim (int): dimension of the output. For regression, choose 1 and for
            classification, set it to the number of classes.
    """

    def __init__(
        self,
        solute_in_feats,
        solvent_in_feats,
        embedding_size=32,
        gated_num_layers=2,
        gated_hidden_size=[64, 64, 32],
        gated_num_fc_layers=1,
        gated_graph_norm=False,
        gated_batch_norm=True,
        gated_activation="ReLU",
        gated_residual=True,
        gated_dropout=0.0,
        attention=True,
        num_lstm_iters=6,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        fc_num_layers=2,
        fc_hidden_size=[32, 16],
        fc_batch_norm=False,
        fc_activation="ReLU",
        fc_dropout=0.0,
        outdim=1,
        conv="GatedGCNConv",
    ):
        super().__init__()

        if isinstance(gated_activation, str):
            gated_activation = getattr(nn, gated_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        # embedding layer
        self.solute_embedding = UnifySize(solute_in_feats, embedding_size)
        self.solvent_embedding = UnifySize(solvent_in_feats, embedding_size)
        self.attention = attention
        
        # gated layer
        if conv == "GatedGCNConv":
            conv_fn = GatedGCNConv
        elif conv == "GatedGCNConv1":
            conv_fn = GatedGCNConv1
        elif conv == "GatedGCNConv2":
            conv_fn = GatedGCNConv2
        else:
            raise ValueError()

        in_size = embedding_size
        self.gated_layers = nn.ModuleList()
        for i in range(gated_num_layers):
            self.gated_layers.append(
                conv_fn(
                    input_dim=in_size,
                    output_dim=gated_hidden_size[i],
                    num_fc_layers=gated_num_fc_layers,
                    graph_norm=gated_graph_norm,
                    batch_norm=gated_batch_norm,
                    activation=gated_activation,
                    residual=gated_residual,
                    dropout=gated_dropout,
                )
            )
            in_size = gated_hidden_size[i]

        #Attention map layer
        self.solute_W_a = nn.Linear(gated_hidden_size[-1], gated_hidden_size[-1])
        self.solvent_W_a = nn.Linear(gated_hidden_size[-1], gated_hidden_size[-1])
        self.W_activation = fc_activation

        # set2set readout layer
        ntypes = ["atom", "bond"]
        in_size = [gated_hidden_size[-1]] * len(ntypes)

        self.readout_layer = Set2SetThenCat(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )

        # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
        # feature twice the the size of in feature)
        readout_out_size = gated_hidden_size[-1] * 2 + gated_hidden_size[-1] * 2
        # for global feat
        if set2set_ntypes_direct is not None:
            readout_out_size += gated_hidden_size[-1] * len(set2set_ntypes_direct)

        readout_out_size *= 2

        # need dropout?
        delta = 1e-3
        if fc_dropout < delta:
            apply_drop = False
        else:
            apply_drop = True

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = readout_out_size

        for i in range(fc_num_layers):
            out_size = fc_hidden_size[i]

            self.fc_layers.append(nn.Linear(in_size, out_size))
            # batch norm
            if fc_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))
            # activation
            self.fc_layers.append(fc_activation)
            # dropout
            if apply_drop:
                self.fc_layers.append(nn.Dropout(fc_dropout))

            in_size = out_size

        # final output layer, mapping feature to the corresponding shape
        self.fc_layers.append(nn.Linear(in_size, outdim))

    def forward(self, graph, feats, norm_atom, norm_bond):
        """
        Args:
            feats (dict)
            norm_atom (2D tensor)
            norm_bond (2D tensor)
        Returns:
            2D tensor: of shape (N, ft_size)
        """

        # embedding
        feats = self.embedding(feats)

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # readout layer
        feats = self.readout_layer(graph, feats)

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats
