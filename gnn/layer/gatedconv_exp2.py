import torch
from torch import nn
import logging
import torch.nn.functional as f
from dgl import function as fn
from gnn.layer.hgatconv import NodeAttentionLayer
from gnn.layer.utils import LinearN
from typing import Callable, Union, Dict
import dgl


logger = logging.getLogger(__name__)


import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

'''


class GatedGCNConv(nn.Module):
    """
    Gated GCN layer.
    It update bond, atom, and global features in sequence. See the BonDNet paper for
    details. This is a direct extension of the Residual Gated Graph ConvNets
    (https://arxiv.org/abs/1711.07553) by adding global features.
    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        num_fc_layers: number of NN layers to transform input to output. In `Residual
            Gated Graph ConvNets` the number of layers is set to 1. Here we make it a
            variable to accept any number of layers.
        graph_norm: whether to apply the graph norm proposed in
            Benchmarking Graph Neural Networks (https://arxiv.org/abs/2003.00982)
        batch_norm: whether to apply batch normalization
        activation: activation function
        residual: whether to add residual connection as in the ResNet:
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        dropout: dropout ratio. Note, dropout is applied after residual connection.
            If `None`, do not apply dropout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc_layers: int = 1,
        graph_norm: bool = False,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU(),
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super().__init__()
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers

        # A, B, ... I are phi_1, phi_2, ..., phi_9 in the BonDNet paper
        self.A = LinearN(input_dim, out_sizes, acts, use_bias)
        self.B = LinearN(input_dim, out_sizes, acts, use_bias)
        self.C = LinearN(input_dim, out_sizes, acts, use_bias)
        self.D = LinearN(input_dim, out_sizes, acts, use_bias)
        self.E = LinearN(input_dim, out_sizes, acts, use_bias)
        self.F = LinearN(input_dim, out_sizes, acts, use_bias)
        self.G = LinearN(output_dim, out_sizes, acts, use_bias)
        self.H = LinearN(output_dim, out_sizes, acts, use_bias)
        self.I = LinearN(input_dim, out_sizes, acts, use_bias)
        self.w = LinearN(400, out_sizes, acts, use_bias)
        self.z = LinearN(400, out_sizes, acts, use_bias)
        
        self.attn_fc = nn.Linear(2 * output_dim, 1, bias=False)
        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout is None or dropout < delta:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    @staticmethod
    def reduce_fn_a2b(nodes):
        """
        Reduce `Eh_j` from atom nodes to bond nodes.
        Expand dim 1 such that every bond has two atoms connecting to it.
        This is to deal with the special case of single atom graph (e.g. H+).
        For such graph, an artificial bond is created and connected to the atom in
        `grapher`. Here, we expand it to let each bond connecting to two atoms.
        This is necessary because, otherwise, the reduce_fn wil not work since
        dimension mismatch.
        """
        x = nodes.mailbox["Eh_j"]
        if x.shape[1] == 1:
            x = x.repeat_interleave(2, dim=1)

        return {"Eh_j": x}

    @staticmethod
    def message_fn(edges):
        return {"Eh_j": edges.src["Eh_j"], "e": edges.src["e"]}

    @staticmethod
    def reduce_fn(nodes):
        Eh_i = nodes.data["Eh"]
        e = nodes.mailbox["e"]
        Eh_j = nodes.mailbox["Eh_j"]

        # TODO select_not_equal is time consuming; it might be improved by passing node
        #  index along with Eh_j and compare the node index to select the different one
        Eh_j = select_not_equal(Eh_j, Eh_i)
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)

        # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
        h = torch.sum(sigma_ij * Eh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)

        return {"h": h}

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        norm_atom: torch.Tensor = None,
        norm_bond: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: the graph
            feats: node features. Allowed node types are `atom`, `bond` and `global`.
            norm_atom: values used to normalize atom features as proposed in graph norm.
            norm_bond: values used to normalize bond features as proposed in graph norm.
        Returns:
            updated node features.
        """
        g = g.to('cuda:3')
        g = g.local_var()
        #print(feats)
        h = feats["atom"]
        e = feats["bond"]
        h2 = feats["atom2"]
        e2 = feats["bond2"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        h_in2 = h2
        e_in2 = e2
        u_in = u
        g.nodes["atom"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["bond"].data.update({"Be": self.B(e)})
        g.nodes["atom2"].data.update({"Ah": self.A(h2), "Dh": self.D(h2), "Eh": self.E(h2)})
        g.nodes["bond2"].data.update({"Be": self.B(e2)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        g.multi_update_all(
            {
                "1a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "1b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "1g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )

        e2 = g.nodes["bond2"].data["e"]
        if self.graph_norm:
            e2 = e2 * norm_bond
        if self.batch_norm:
            e2 = self.bn_node_e(e2)
        e2 = self.activation(e2)
        if self.residual:
            e2 = e_in2 + e2
        g.nodes["bond2"].data["e"] = e2
        
        e = g.nodes["bond"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="1a2b")

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        
        
        g.multi_update_all(
            {
                "1a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "1b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "1g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        
        
        
        
        
        
        #两个节点直接联系的传递
        #------------------------------------
        '''
        def u2v_message_fn(edges):
             z2 = torch.cat([edges.src["Dh"], edges.dst["Dh"]], dim=1)
             a = self.attn_fc(z2)
             return {'m': F.leaky_relu(a),'z': edges.src["Dh"]}
        def reduce_func1(nodes):
            # reduce UDF for equation (3) & (4)
            # equation (3)
            alpha = F.softmax(nodes.mailbox['m'], dim=1)
            print(alpha.size())
            # 获取前10%的alpha值的阈值
            k = int(alpha.size()[1] * 0.1)
            topk_values, _ = torch.topk(alpha, k, dim=1)
            threshold = topk_values[:, -1].unsqueeze(1)
            
            # 创建掩码矩阵
            mask = alpha >= threshold
            
            # 应用掩码
            alpha = alpha * mask.float()
            
            # 重新归一化注意力系数
            alpha = alpha / alpha.sum(dim=1, keepdim=True)
            
            # equation (4)
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
            return {'agg': h}
        def add(nodes):
            #print("--------------------")
            #print(nodes.data['h'])
            #print("***************")
            #print( nodes.data['agg'])
            #print("--------------------")
            return {'h': nodes.data['h']*0.8 + nodes.data['agg']*0.2}
        g.multi_update_all(
            {
                "u2v": (u2v_message_fn,reduce_func1,add),  # D * h_i
                "v2u": (u2v_message_fn,reduce_func1,add),
            },
            "sum",
        )
        '''
        
        
        
        
        
        
        
        
        def u2v_message_fn(edges):
            z2 = torch.cat([edges.src["Dh"], edges.dst["Dh"]], dim=1)
            a = self.attn_fc(z2)
            return {'m': F.leaky_relu(a), 'z': edges.src["Dh"]}


        
        def reduce_func1(nodes):
            global global_alpha

            alpha = F.softmax(nodes.mailbox['m'], dim=1)
            
            # 计算全局 alpha 阈值
            k = int(global_alpha.size(0) * 0.1)
            threshold = global_alpha[k]

            # 创建掩码矩阵，只保留前10%的 alpha
            mask = alpha >= threshold

            # 应用掩码
            alpha = alpha * mask.float()

            # 检查 alpha 的总和，防止除以零
            alpha_sum = alpha.sum(dim=1, keepdim=True)
            if torch.any(alpha_sum == 0):
                #print("Warning: alpha sum is zero, causing NaN values")
                alpha_sum = alpha_sum + 1e-10  # 防止除以零

            # 重新归一化注意力系数
            alpha = alpha / alpha_sum

            # 聚合特征
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
            
            # 如果 alpha 全被掩码掉，则保持原始特征不变
            no_message_mask = (alpha_sum == 1e-10).squeeze()
            h[no_message_mask] = nodes.data['h'][no_message_mask]

            return {'agg': h}

        def add(nodes):
            #return {'h': nodes.data['h']*1}
            return {'h': nodes.data['h']*0.8 + nodes.data['agg']*0.2}

        def compute_global_alpha(g, etype):
            global global_alpha
            with g.local_scope():
                g.apply_edges(u2v_message_fn, etype=etype)
                with torch.no_grad():
                    all_alpha = g.edges[etype].data['m'].view(-1)
                    #print(all_alpha)
                    global_alpha, _ = torch.sort(all_alpha, descending=True)

        # 计算全局 alpha 阈值，针对每种边类型
        edge_types = ['u2v', 'v2u']
        for etype in edge_types:
            compute_global_alpha(g, etype)

        # 进行多重更新操作
        g.multi_update_all(
            {
                "u2v": (u2v_message_fn, reduce_func1, add),
                "v2u": (u2v_message_fn, reduce_func1, add),
            },
            "sum",
        )
        '''
        def u2v_message_fn(edges):
            z2 = torch.cat([edges.src["Dh"], edges.dst["Dh"]], dim=1)
            a = self.attn_fc(z2)
            
            return {'m': F.leaky_relu(a), 'z': edges.src["Dh"]}

        def reduce_func1(nodes):
            global global_alpha

            alpha = F.softmax(nodes.mailbox['m'], dim=1)
            
            # 计算全局 alpha 阈值
            k = int(global_alpha.size(0) * 0.1)
            threshold = global_alpha[k]

            # 创建掩码矩阵，只保留前10%的 alpha
            mask = alpha >= threshold

            # 应用掩码
            alpha = alpha * mask.float()

            # 检查 alpha 的总和，防止除以零
            alpha_sum = alpha.sum(dim=1, keepdim=True)
            if torch.any(alpha_sum == 0):
                print("Warning: alpha sum is zero, causing NaN values")
                alpha_sum = alpha_sum + 1e-10  # 防止除以零

            # 重新归一化注意力系数
            alpha = alpha / alpha_sum

            # 聚合特征
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
            
            # 如果 alpha 全被掩码掉，则保持原始特征不变
            no_message_mask = (alpha_sum == 0).squeeze()
            h[no_message_mask] = nodes.data['h'][no_message_mask]

            return {'agg': h}

        def add(nodes):
            return {'h': nodes.data['agg']*1}

        def compute_global_alpha(g, etype):
            global global_alpha
            with g.local_scope():
                g.apply_edges(u2v_message_fn, etype=etype)
                with torch.no_grad():
                    all_alpha = g.edges[etype].data['m'].view(-1)
                    #print(all_alpha)
                    global_alpha, _ = torch.sort(all_alpha, descending=True)

        # 计算全局 alpha 阈值，针对每种边类型
        edge_types = ['u2v', 'v2u']
        for etype in edge_types:
            compute_global_alpha(g, etype)

        # 进行多重更新操作
        g.multi_update_all(
            {
                "u2v": (u2v_message_fn, reduce_func1, add),
                "v2u": (u2v_message_fn, reduce_func1, add),
            },
            "sum",
        )
        '''

        
        
        
        
        
        
        
        
        #u2v = g.edges["u2v"].data["m"]
        #v2u = g.edges["v2u"].data["m"]
        h = g.nodes["atom"].data["h"]
        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom"].data["h"] = h
        
        
        
        
        
        
        
        
        
        
        
        h2 = g.nodes["atom2"].data["h"]
        if self.graph_norm:
            h2 = h2 * norm_atom
        if self.batch_norm:
            h2 = self.bn_node_h(h2)
        h2 = self.activation(h2)
        if self.residual:
            h2 = h_in2 + h2
        g.nodes["atom2"].data["h"] = h2
        
        
        #--------------------------------------
        
        # update global feature u
        g.nodes["atom"].data.update({"Gh": self.G(h)})
        g.nodes["bond"].data.update({"He": self.H(e)})
        g.nodes["atom2"].data.update({"Gh": self.G(h2)})
        g.nodes["bond2"].data.update({"He": self.H(e2)})
        g.nodes["global"].data.update({"Iu": self.I(u)})
        g.multi_update_all(
            {
                "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "1a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "1b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
            },
            "sum",
        )
        u = g.nodes["global"].data["u"]
        # do not apply batch norm if it there is only one graph
        if self.batch_norm and u.shape[0] > 1:
            u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        h2 = self.dropout(h2)
        e2 = self.dropout(e2)
        u = self.dropout(u)

        #feats = {"atom2": h2, "bond2": e2, "global": u,"atom": h, "bond": e,"u2v":u2v, "v2u":v2u}
        feats = {"atom2": h2, "bond2": e2, "global": u,"atom": h, "bond": e}

        return feats
    
    def get_emb(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        norm_atom: torch.Tensor = None,
        norm_bond: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: the graph
            feats: node features. Allowed node types are `atom`, `bond` and `global`.
            norm_atom: values used to normalize atom features as proposed in graph norm.
            norm_bond: values used to normalize bond features as proposed in graph norm.
        Returns:
            updated node features.
        """
        g = g.to('cuda:3')
        g = g.local_var()
        #print(feats)
        h = feats["atom"]
        e = feats["bond"]
        h2 = feats["atom2"]
        e2 = feats["bond2"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        h_in2 = h2
        e_in2 = e2
        u_in = u
        g.nodes["atom"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["bond"].data.update({"Be": self.B(e)})
        g.nodes["atom2"].data.update({"Ah": self.A(h2), "Dh": self.D(h2), "Eh": self.E(h2)})
        g.nodes["bond2"].data.update({"Be": self.B(e2)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        g.multi_update_all(
            {
                "1a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "1b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "1g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )

        e2 = g.nodes["bond2"].data["e"]
        if self.graph_norm:
            e2 = e2 * norm_bond
        if self.batch_norm:
            e2 = self.bn_node_e(e2)
        e2 = self.activation(e2)
        if self.residual:
            e2 = e_in2 + e2
        g.nodes["bond2"].data["e"] = e2
        
        e = g.nodes["bond"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="1a2b")

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        
        
        g.multi_update_all(
            {
                "1a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "1b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "1g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        #两个节点直接联系的传递
        #------------------------------------
        h = g.nodes["atom"].data["h"]
        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom"].data["h"] = h
        
        
        h2 = g.nodes["atom2"].data["h"]
        if self.graph_norm:
            h2 = h2 * norm_atom
        if self.batch_norm:
            h2 = self.bn_node_h(h2)
        h2 = self.activation(h2)
        if self.residual:
            h2 = h_in2 + h2
        g.nodes["atom2"].data["h"] = h2
        
        
        #--------------------------------------
        
        # update global feature u
        g.nodes["atom"].data.update({"Gh": self.G(h)})
        g.nodes["bond"].data.update({"He": self.H(e)})
        g.nodes["atom2"].data.update({"Gh": self.G(h2)})
        g.nodes["bond2"].data.update({"He": self.H(e2)})
        g.nodes["global"].data.update({"Iu": self.I(u)})
        g.multi_update_all(
            {
                "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "1a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "1b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
            },
            "sum",
        )
        u = g.nodes["global"].data["u"]
        # do not apply batch norm if it there is only one graph
        if self.batch_norm and u.shape[0] > 1:
            u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        h2 = self.dropout(h2)
        e2 = self.dropout(e2)
        u = self.dropout(u)

        feats = {"atom2": h2, "bond2": e2, "global": u,"atom": h, "bond": e}

        return feats


class GatedGCNConv1(GatedGCNConv):
    """
    Compared with GatedGCNConv, we use hgat attention layer to update global feature.  
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_fc_layers=1,
        graph_norm=True,
        batch_norm=True,
        activation=nn.ELU(),
        residual=False,
        dropout=0.0,
    ):
        super().__init__(
            input_dim,
            output_dim,
            num_fc_layers,
            graph_norm,
            batch_norm,
            activation,
            residual,
            dropout,
        )

        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.A = LinearN(input_dim, out_sizes, acts, use_bias)
        self.B = LinearN(input_dim, out_sizes, acts, use_bias)
        self.C = LinearN(input_dim, out_sizes, acts, use_bias)
        self.D = LinearN(input_dim, out_sizes, acts, use_bias)
        self.E = LinearN(input_dim, out_sizes, acts, use_bias)
        self.F = LinearN(input_dim, out_sizes, acts, use_bias)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout >= delta:
            self.dropout = nn.Dropout(dropout)
        else:
            logger.warning(f"dropout ({dropout}) smaller than {delta}. Ignored.")
            self.dropout = nn.Identity()

        self.node_attn_layer = NodeAttentionLayer(
            master_node="global",
            attn_nodes=["atom2", "bond2", "global"],
            attn_edges=["a2g", "b2g", "g2g"],
            in_feats={"atom2": output_dim, "bond2": output_dim, "global": input_dim},
            out_feats=output_dim,
            num_heads=1,
            num_fc_layers=num_fc_layers,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            batch_norm=False,
        )

    def forward(self, g, feats, norm_atom, norm_bond):
        g = g.to('cuda:3')
        g = g.local_var()

        h = feats["atom2"]
        e = feats["bond2"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        u_in = u

        g.nodes["atom2"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["bond2"].data.update({"Be": self.B(e)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        e = g.nodes["bond2"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond2"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        h = g.nodes["atom2"].data["h"]
        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom2"].data["h"] = h

        u = self.node_attn_layer(g, u, [h, e, u]).flatten(start_dim=1)
        if self.batch_norm:
            u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        u = self.dropout(u)

        feats = {"atom2": h, "bond2": e, "global": u}

        return feats


class GatedGCNConv2(GatedGCNConv):
    """
    Compared with GatedGCNConv, global feature is not used here.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_fc_layers=1,
        graph_norm=True,
        batch_norm=True,
        activation=nn.ELU(),
        residual=False,
        dropout=0.0,
    ):
        super(GatedGCNConv, self).__init__()
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.A = LinearN(input_dim, out_sizes, acts, use_bias)
        self.B = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.C = LinearN(input_dim, out_sizes, acts, use_bias)
        self.D = LinearN(input_dim, out_sizes, acts, use_bias)
        self.E = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.F = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.G = LinearN(output_dim, out_sizes, acts, use_bias)
        # self.H = LinearN(output_dim, out_sizes, acts, use_bias)
        # self.I = LinearN(input_dim, out_sizes, acts, use_bias)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
        # self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout >= delta:
            self.dropout = nn.Dropout(dropout)
        else:
            logger.warning(f"dropout ({dropout}) smaller than {delta}. Ignored.")
            self.dropout = nn.Identity()

    def forward(self, g, feats, norm_atom, norm_bond):
        g = g.to('cuda:3')
        g = g.local_var()

        h = feats["atom2"]
        e = feats["bond2"]
        # u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        # u_in = u

        g.nodes["atom2"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["bond2"].data.update({"Be": self.B(e)})
        # g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                # "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        e = g.nodes["bond2"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond2"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                # "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        h = g.nodes["atom2"].data["h"]
        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom2"].data["h"] = h

        # # update global feature u
        # g.nodes["atom2"].data.update({"Gh": self.G(h)})
        # g.nodes["bond2"].data.update({"He": self.H(e)})
        # g.nodes["global"].data.update({"Iu": self.I(u)})
        # g.multi_update_all(
        #     {
        #         "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
        #         "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
        #         "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
        #     },
        #     "sum",
        # )
        # u = g.nodes["global"].data["u"]
        # if self.batch_norm:
        #     u = self.bn_node_u(u)
        # u = self.activation(u)
        # if self.residual:
        #     u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        # u = self.dropout(u)

        # feats = {"atom2": h, "bond2": e, "global": u}
        feats = {"atom2": h, "bond2": e}

        return feats


def select_not_equal(x, y):
    """Subselect an array from x, which is not equal to the corresponding element
    in y.
    Args:
        x (4D tensor): shape are d0: node batch dim, d1: number of edges dim,
            d2: selection dim, d3: feature dim
        y (2D tensor): shape are 0: nodes batch dim, 1: feature dim
    For example:
    >>> x =[[ [ [0,1,2],
    ...         [3,4,5] ],
    ...       [ [0,1,2],
    ...         [6,7,8] ]
    ...     ],
    ...     [ [ [0,1,2],
    ...         [3,4,5] ],
    ...       [ [3,4,5],
    ...         [6,7,8] ]
    ...     ]
    ...    ]
    >>>
    >>> y = [[0,1,2],
    ...      [3,4,5]]
    >>>
    >>> select_no_equal(x,y)
    ... [[[3,4,5],
    ...   [6,7,8]],
    ...  [[0,1,2],
    ...   [6,7,8]]
    Returns:
        3D tensor: of shape (d0, d1, d3)
    """
    d0, d1, d2, d3 = x.shape
    assert d2 == 2, f"Expect x.shape[2]==2, got {d2}"

    ## method 1, slow
    # rst = []
    # for x1, y1 in zip(x, y):
    #     xx = [x2[0] if not torch.equal(y1, x2[0]) else x2[1] for x2 in x1]
    #     rst.append(torch.stack(xx))
    # rst = torch.stack(rst)

    # method 2, a much faster version
    y = torch.repeat_interleave(y, d1 * d2, dim=0).view(x.shape)
    any_not_equal = torch.any(x != y, dim=3)
    # bool index
    idx1 = any_not_equal[:, :, 0].view(d0, d1, 1)
    idx2 = ~idx1
    idx = torch.cat([idx1, idx2], dim=-1)
    # select result
    rst = x[idx].view(d0, d1, -1)

    return rst