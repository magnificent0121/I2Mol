"""Torch modules for GAT for heterograph."""
import logging
import torch
from torch import nn
from dgl import function as fn
from gnn.layer.utils import LinearN

logger = logging.getLogger(__name__)


class NodeAttentionLayer(nn.Module):
    

    def __init__(
        self,
        master_node,
        attn_nodes,
        attn_edges,
        in_feats,
        out_feats,
        num_heads,
        num_fc_layers=3,
        fc_activation=nn.ELU(),
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        batch_norm=False,
        activation=None,
    ):

        super(NodeAttentionLayer, self).__init__()
        self.master_node = master_node
        self.attn_nodes = attn_nodes
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.activation = activation
        self.edge_types = [(n, e, master_node) for n, e in zip(attn_nodes, attn_edges)]

        self.fc_layers = nn.ModuleDict()
        for nt, sz in in_feats.items():
            if num_fc_layers > 0:
                # last layer does not use bias and activation
                out_sizes = [out_feats] * (num_fc_layers - 1) + [out_feats * num_heads]
                act = [fc_activation] * (num_fc_layers - 1) + [nn.Identity()]
                use_bias = [True] * (num_fc_layers - 1) + [False]
                self.fc_layers[nt] = LinearN(sz, out_sizes, act, use_bias)
            else:
                self.fc_layers[nt] = nn.Identity()

        # parameters for attention
        self.attn_l = nn.Parameter(torch.zeros(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.zeros(1, num_heads, out_feats))
        self.reset_parameters()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # here we use different dropout for each node type
        delta = 1e-3
        if feat_drop >= delta:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            logger.warning(
                "`feat_drop = {}` provided for {} smaller than {}. "
                "Ignore dropout.".format(feat_drop, self.__class__.__name__, delta)
            )
            self.feat_drop = nn.Identity()

        # attn_drop is used for edge_types, not node types, so here we create dropout
        # using attn_nodes instead of in_feats as for feat_drop
        if attn_drop >= delta:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            logger.warning(
                "`attn_drop = {}` provided for {} smaller than {}. "
                "Ignore dropout.".format(attn_drop, self.__class__.__name__, delta)
            )
            self.attn_drop = nn.Identity()

        if residual:
            if in_feats[master_node] != out_feats:
                self.res_fc = nn.Linear(
                    in_feats[master_node], num_heads * out_feats, bias=False
                )
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        # batch normalization
        if batch_norm:
            self.bn_layer = nn.ModuleList(
                [nn.BatchNorm1d(num_features=out_feats) for _ in range(num_heads)]
            )
        else:
            self.register_buffer("bn_layer", None)

    def reset_parameters(self):
        """Reinitialize parameters."""
        gain = nn.init.calculate_gain("relu")
        # for nt, layer in self.fc_layers.items():
        #     nn.init.xavier_normal_(layer.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, master_feats, attn_feats):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): The graph.
            master_feats (torch.Tensor): Feature of the master node that
                will be updated. It is of shape :math:`(N, D_{in})` where :math:`N` is
                the number of nodes and :math:`D_{in}` is the feature size.
            attn_feats (list of torch.Tensor): Features of the attention nodes.
                Each element is of shape :math:`(N, D_{in})` where :math:`N` is the
                number of nodes and :math:`D_{in}` is the feature size.
                Note that the order of the features should corresponds to that of the
                `attn_nodes` provided at instantiation.
        Returns:
            torch.Tensor: The updated master feature of shape :math:`(N, H, D_{out})`
            where :math:`N` is the number of nodes, :math:`H` is the number of heads,
            and :math:`D_{out}` is the size of the output master feature.
        """
        graph = graph.local_var()

        N = master_feats.shape[0]

        # assign data
        # master node
        master_feats = self.feat_drop(master_feats)  # (N, in)
        feats = self.fc_layers[self.master_node](master_feats).view(
            N, -1, self.out_feats
        )  # (N, H, out)
        er = (feats * self.attn_r).sum(dim=-1).unsqueeze(-1)  # (N, H, 1)
        graph.nodes[self.master_node].data.update({"ft": feats, "er": er})

        # attention node
        for ntype, feats in zip(self.attn_nodes, attn_feats):
            feats = self.feat_drop(feats)
            feats = self.fc_layers[ntype](feats).view(-1, self.num_heads, self.out_feats)
            el = (feats * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.nodes[ntype].data.update({"ft": feats, "el": el})

        # compute edge attention
        e = []  # each component is of shape(Ne, H, 1)
        for etype in self.edge_types:
            graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
            e.append(self.leaky_relu(graph.edges[etype].data.pop("e")))

        # softmax, each component is of shape(Ne, H, 1)
        softmax = heterograph_edge_softmax(graph, self.edge_types, e)

        # apply attention dropout
        for etype, a in zip(self.edge_types, softmax):
            graph.edges[etype].data["a"] = self.attn_drop(a)

        # message passing, "ft" is of shape(H, out), and "a" is of shape(H, 1)
        # computing the part inside the parenthesis of eq. 4 of the GAT paper
        graph.multi_update_all(
            {
                etype: (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
                for etype in self.edge_types
            },
            "sum",
        )
        rst = graph.nodes[self.master_node].data["ft"]  # shape(N, H, out)

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(master_feats).view(N, -1, self.out_feats)
            rst = rst + resval

        # batch normalization
        if self.bn_layer is not None:
            rst_bn = [layer(rst[:, i, :]) for i, layer in enumerate(self.bn_layer)]
            rst = torch.stack(rst_bn, dim=1)

        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst

class HGATConv(nn.Module):
    """
    Graph attention convolution layer for hetero graph that attends between different
    (and the same) type of nodes.
    Args:
        attn_mechanism (dict of dict): The attention mechanism, i.e. how the node
            features will be updated. The outer dict has `node types` as its key
            and the inner dict has keys `nodes` and `edges`, where the values (list)
            of `nodes` are the `node types` that the master node will attend to,
            and the corresponding `edges` are the `edge types`.
        attn_order (list): `node type` string that specify the order to attend the node
            features.
        in_feats (list): input feature size for the corresponding (w.r.t. index) node
            in `attn_order`.
        out_feats (int): output feature size, the same for all nodes
        num_heads (int): number of attention heads, the same for all nodes
        num_fc_layers (int): number of fully-connected layer before attention
        feat_drop (float, optional): [description]. Defaults to 0.0.
        attn_drop (float, optional): [description]. Defaults to 0.0.
        negative_slope (float, optional): [description]. Defaults to 0.2.
        residual (bool, optional): [description]. Defaults to False.
        batch_norm(bool): whether to apply batch norm to the output
        activation (nn.Moldule or str): activation fn
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        out_feats,
        num_heads=4,
        num_fc_layers=3,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        batch_norm=False,
        activation=None,
    ):

        super(HGATConv, self).__init__()

        self.attn_mechanism = attn_mechanism
        self.master_nodes = attn_order

        in_feats_map = dict(zip(attn_order, in_feats))

        self.layers = nn.ModuleDict()

        for ntype in self.master_nodes:
            self.layers[ntype] = NodeAttentionLayer(
                master_node=ntype,
                attn_nodes=self.attn_mechanism[ntype]["nodes"],
                attn_edges=self.attn_mechanism[ntype]["edges"],
                in_feats=in_feats_map,
                out_feats=out_feats,
                num_heads=num_heads,
                num_fc_layers=num_fc_layers,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation,
                batch_norm=batch_norm,
            )
            in_feats_map[ntype] = num_heads * out_feats

    def forward(self, graph, feats):
        """
        Args:
            graph (dgl heterograph): the graph
            feats (dict): node features with node type as key and the corresponding
            features as value.
        Returns:
            dict: updated node features with the same keys as in `feats`.
                Each feature value has a shape of `(N, out_feats*num_heads)`, where
                `N` is the number of nodes (different for different key) and
                `out_feats` and `num_heads` are the out feature size and number
                of heads specified at instantiation (the same for different keys).
        """
        updated_feats = {k: v for k, v in feats.items()}
        for ntype in self.master_nodes:
            master_feats = updated_feats[ntype]
            attn_feats = [updated_feats[t] for t in self.attn_mechanism[ntype]["nodes"]]
            ft = self.layers[ntype](graph, master_feats, attn_feats)
            updated_feats[ntype] = ft.flatten(start_dim=1)  # flatten the head dimension
        return updated_feats


def heterograph_edge_softmax(graph, edge_types, edge_data):
    r"""Edge softmax for heterograph.
     For a node :math:`i`, edge softmax is an operation of computing
    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}
    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also called logits
    in the context of softmax. :math:`\mathcal{N}(i)` is the set of nodes that have an
    edge to :math:`i`. The type of j is ignored, i.e. it runs over all j that directs
    to i, no matter what the node type of j is.
    .. code:: python
        score = dgl.EData(g, score)
        score_max = score.dst_max()  # of type dgl.NData
        score = score - score_max  # edge_sub_dst, ret dgl.EData
        score_sum = score.dst_sum()  # of type dgl.NData
        out = score / score_sum    # edge_div_dst, ret dgl.EData
        return out.data
    ""[summary]
    Returns:
        [type]: [description]
    """
    g = graph.local_var()

    #####################################################################################
    ## assign data
    # max_e = []
    # min_e = []
    # for etype, edata in zip(edge_types, edge_data):
    #    g.edges[etype].data["e"] = edata
    #    max_e.append(torch.max(edata))
    #    min_e.append(torch.min(edata))
    # max_e = max(max_e)
    # min_e = min(min_e)

    ## The softmax trick, making the exponential stable.
    ## see https://stackoverflow.com/questions/42599498/numercially-stable-softmax
    ## max_e > 64 to prevent overflow; min_e<-64 to prevent underflow
    ##
    ## Of course, we can apply the trick all the time, but here we choose to apply only
    ## in some conditions to save some time, since multi_update_all is really expensive.
    # if max_e > 64.0 or min_e < -64.0:
    #    # e max (fn.max operates on the axis of features from different nodes)
    #    g.multi_update_all(
    #        {etype: (fn.copy_e("e", "m"), fn.max("m", "emax")) for etype in edge_types},
    #        "max",
    #    )
    #    # subtract max and compute exponential
    #    for etype in edge_types:
    #        g.apply_edges(fn.e_sub_v("e", "emax", "e"), etype=etype)

    #####################################################################################
    for etype, edata in zip(edge_types, edge_data):
        g.edges[etype].data["e"] = edata

    g.multi_update_all(
        {etype: (fn.copy_e("e", "m"), fn.max("m", "emax")) for etype in edge_types}, "max"
    )
    # subtract max and compute exponential
    for etype in edge_types:
        g.apply_edges(fn.e_sub_v("e", "emax", "e"), etype=etype)
        g.edges[etype].data["out"] = torch.exp(g.edges[etype].data["e"])

    #####################################################################################

    # e sum
    g.multi_update_all(
        {etype: (fn.copy_e("out", "m"), fn.sum("m", "out_sum")) for etype in edge_types},
        "sum",
    )

    a = []
    for etype in edge_types:
        g.apply_edges(fn.e_div_v("out", "out_sum", "a"), etype=etype)
        a.append(g.edges[etype].data["a"])

    return a