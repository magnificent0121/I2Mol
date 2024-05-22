from typing import ForwardRef
import torch
import itertools
import numpy as np
import dgl
from torch._C import device
from gnn.model.gated_mol import GatedGCNMol, AttentionGCN
import torch.nn.functional as F
import copy









class GatedGCNSolvationNetwork(GatedGCNMol):
    def forward(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
     solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None,data_point_embedding_list=None,data_point_cookbook_idx_list=None):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding features as value
            norm_atom (2D tensor or None): graph norm for atom
            norm_bond (2D tensor or None): graph norm for bond

        Returns:
            2D tensor: of shape(N, M), where M = outdim.
        """
        # embed the solute and solvent
        #print(solute_feats)
        solute_feats = self.solute_embedding(solute_feats)
        #solvent_feats = self.solvent_embedding(solvent_feats)
        
        #先让他进行一下分子内传播

        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            
        xunhuan_solute_feats = solute_feats.copy()
        
        
        atom1_pro = self.MLP(solute_feats["atom"])
        atom2_pro = self.MLP(solute_feats["atom2"])
        #print(atom2_pro.size())
        #然后在此处对solute_feat_copy进行操作，
        
        # pass through gated layer
        node_nation1 = solute_feats["atom"] *atom1_pro
        node_nation2 = solute_feats["atom2"] *atom2_pro
        
        atom1_neg = 1-atom1_pro
        atom2_neg = 1-atom2_pro
        
        node_env1 = solute_feats["atom"] *atom1_neg
        node_env2 = solute_feats["atom2"] *atom2_neg
        
        orinal_feats1=solute_feats["atom"]*1
        orinal_feats2=solute_feats["atom2"]*1
    
    
        solute_feats["atom"]=node_env1
        solute_feats["atom2"]=node_env2
        
        
        env_feats = self.readout_layer2(solute_graph, solute_feats)
        # env_feats 
        quantized_graph_env1, vq_loss_1, perplexity_mean_1k,encoding_inds = self.vq(env_feats)
       
        #问题

        # 假设 quantized_graph_env1 是一个形状为 (batch_size, feature_dim) 的张量
        # 假设 solute_graph 是一个包含多个 DGL 图的列表
        
        
        solute_graph=solute_graph.to('cuda:2')
        
        solute_feats["atom"]=node_nation1
        solute_feats["atom2"]=node_nation2
        # 遍历每个图以及对应的池化向量
        
        copied_solute_feats = solute_feats.copy()
        copied_solute_feats_2 = solute_feats.copy()
        #print(dgl.broadcast_nodes(solute_graph, quantized_graph_env1, ntype=self.ntype))
        

        # 将原始图的特征复制到复制的图中
        '''
        solute_graph_1 = dgl.heterograph({etype: solute_graph.edges(etype) for etype in solute_graph.canonical_etypes})


        # 将原始图的特征复制到复制的图中
        for ntype in solute_graph.ntypes:
            print("+++++++++")
            solute_graph_1.nodes[ntype].data.update(solute_graph.nodes[ntype].data)
        for etype in solute_graph.etypes:
            solute_graph_1.edges[etype].data.update(solute_graph.edges[etype].data)
        '''
        ls=[]
        with solute_graph.local_scope():
        
            copied_solute_feats_2["atom"]+=dgl.broadcast_nodes(solute_graph, quantized_graph_env1, ntype="atom")
            copied_solute_feats_2["atom2"]+=dgl.broadcast_nodes(solute_graph, quantized_graph_env1, ntype="atom2")

            # 现在，每个图的节点都有了对应的池化向量作为特征
            
            final_feats1 = self.readout_layer(solute_graph, copied_solute_feats_2)
            for layer in self.fc_layers:
                final_feats1 = layer(final_feats1)
                
            ls.append(final_feats1)
            

        #这里应该有个平衡因子
        
        #loss_1 = loss_1+vq_loss_1+vq_loss_2
        
        #for layer in self.gated_layers:
        #    solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            #solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        # readout layer - set2set
        #solvent_feats = self.readout_layer(solvent_graph, solvent_feats) 
        #print(solute_feats)
        # concatenate
        feats_list = []
        #predictions_list.append(predictions)
        
        for i in range(1):
            '''
            solute_graph_2 = dgl.heterograph({etype: solute_graph.edges(etype) for etype in solute_graph.canonical_etypes})

        # 将原始图的特征复制到复制的图中
            for ntype in solute_graph.ntypes:
                solute_graph_2.nodes[ntype].data.update(solute_graph.nodes[ntype].data)
            for etype in solute_graph.etypes:
                solute_graph_2.edges[etype].data.update(solute_graph.edges[etype].data)
            '''
            with solute_graph.local_scope():
                quantized_graph_env1, vq_loss_1, perplexity_mean_1,_ = self.vq(env_feats)
                #问题
                
                copied_solute_feats_1=xunhuan_solute_feats.copy()
                copied_solute_feats_1["atom"]=node_nation1
                copied_solute_feats_1["atom2"]=node_nation2
                
                copied_solute_feats_1["atom"]+=dgl.broadcast_nodes(solute_graph, quantized_graph_env1, ntype="atom")
                copied_solute_feats_1["atom2"]+=dgl.broadcast_nodes(solute_graph, quantized_graph_env1, ntype="atom2")
                
                    #solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

                # readout layer - set2set
                solute_feats = self.readout_layer(solute_graph, copied_solute_feats_1) 
                #solvent_feats = self.readout_layer(solvent_graph, solvent_feats) 
                #print(solute_feats)
                # concatenate
                feats=solute_feats
                for layer in self.fc_layers:
                    feats = layer(feats)
                feats_list.append(feats)
            
        return ls[0],feats_list,vq_loss_1
        # return ls[0],feats_list,vq_loss_1,solute_feats
        # return ls[0],feats_list,vq_loss_1,env_feats,encoding_inds
    
    def feature_before_fc(self, solute_graph, solvent_graph, solute_feats, solvent_feats, 
                          solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):
        """
        Get the features before the final fully-connected.
        This is used for feature visualization.
        """
        # embed the solute and solvent
        solute_feats = self.embedding(solute_feats)
        solute_feats["relation"]=solute_graph.nodes["relation"].data["feat"]
        #solvent_feats = self.embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            #solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        # readout layer - set2set
        solute_feats = self.readout_layer(solute_graph, solute_feats) # 100 * hidden_dim
        #solvent_feats = self.readout_layer(solvent_graph, solvent_feats) # 100 * hidden_dim

        # concatenate
        feats=solute_feats
        feats = torch.cat((solute_feats, solvent_feats)) # 200 * hidden_dim

        return solute_feats, solvent_feats, feats
    def visualise_attn_weights(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
     solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):

        solute_feats = self.solute_embedding(solute_feats)
        solute_feats["relation"]=solute_graph.nodes["relation"].data["feat"]
        #solvent_feats = self.solvent_embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            #solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        solute_wts = []
        #solvent_wts = []

        fts_solu = _split_batched_output_atoms(solute_graph, solute_feats["atom"])
        print(len(fts_solu))
        #fts_solv = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"]) 

        for solute_ft in zip(fts_solu):
            pairwise_solute_feature = F.leaky_relu(self.solute_W_a(solute_ft), 0.1)
            #pairwise_solvent_feature = F.leaky_relu(self.solvent_W_a(solvent_ft), 0.1) 
            
            pairwise_pred = torch.sigmoid(torch.matmul(
                pairwise_solute_feature, pairwise_solvent_feature.t()))

            solute_fts_att_w  = torch.matmul(pairwise_pred, pairwise_solvent_feature)       
            solvent_fts_att_w  = torch.matmul(pairwise_pred.t(), pairwise_solute_feature)

            solute_wts.append(solute_fts_att_w)
            solvent_wts.append(solvent_fts_att_w)

        return solute_wts, solvent_wts

    def atom_features_at_each_layer(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
                                    solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):
        """
        Get the atom features at each layer before the final fully-connected layer
        This is used for feature visualisation to see how the model learns.

        Returns:
            dict (layer_idx, feats), each feats is a list of each atom's features.
        """

        layer_idx = 0
        all_feats = dict()

        # embedding
        solute_feats = self.solute_embedding(solute_feats)
        #print(solute_feats)
        #solvent_feats = self.embedding(solvent_feats)

        # store atom features of each molecule
        solute_atom_fts = _split_batched_output_atoms(solute_graph, solute_feats["atom"])
        #print(len(solute_atom_fts))
        #solvent_atom_fts = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"])
        all_feats[layer_idx] = solute_atom_fts
        layer_idx += 1

        # gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            #solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)
            solute_atom_fts_u2v = _split_batched_output_u2v(solute_graph, solute_feats["u2v"])
            solute_atom_fts_v2u = _split_batched_output_v2u(solute_graph, solute_feats["v2u"])
            #solvent_atom_fts = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"])
            all_feats[layer_idx] = [solute_atom_fts_u2v,solute_atom_fts_v2u]
            layer_idx += 1

        return all_feats





class SelfInteractionMap(AttentionGCN):
    def forward(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
     solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding features as value
            norm_atom (2D tensor or None): graph norm for atom
            norm_bond (2D tensor or None): graph norm for bond

        Returns:
            2D tensor: of shape(N, M), where M = outdim.
        """
        # embed the solute and solvent

        solute_feats = self.solute_embedding(solute_feats)
        solvent_feats = self.solvent_embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        fts_solu = _split_batched_output_atoms(solute_graph, solute_feats["atom"]) 
        fts_solv = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"])

        updated_solute_atom_fts = []
        updated_solvent_atom_fts = []

        for layer in self.intmap_layers:
            continue

        for solute_ft, solvent_ft in zip(fts_solu, fts_solv):
            # Effect of the solvent on the solute
            solute_fts_att_w  = torch.matmul(self.intmap_layers[0](solute_ft), solute_ft.t()) 
            solute_fts_att_w = torch.nn.functional.softmax(solute_fts_att_w, dim=0)
            
            solvent_fts_att_w  = torch.matmul(self.intmap_layers[2](solvent_ft), solvent_ft.t()) 
            solvent_fts_att_w = torch.nn.functional.softmax(solvent_fts_att_w, dim=0)

            solute_attn_hiddens = torch.matmul(solute_fts_att_w, solute_ft)
            solute_attn_hiddens = self.W_activation(self.intmap_layers[1](solute_attn_hiddens))

            solvent_attn_hiddens = torch.matmul(solvent_fts_att_w, solvent_ft) 
            solvent_attn_hiddens = self.W_activation(self.intmap_layers[3](solvent_attn_hiddens))

            new_solute_feats = solute_ft + solute_attn_hiddens
            new_solvent_feats = solvent_ft + solvent_attn_hiddens

            updated_solute_atom_fts.append(new_solute_feats)
            updated_solvent_atom_fts.append(new_solvent_feats)

        new_solute_feats = torch.cat(updated_solute_atom_fts)
        new_solvent_feats = torch.cat(updated_solvent_atom_fts)

        solute_feats["atom"] = new_solute_feats
        solvent_feats["atom"] = new_solvent_feats

        # readout layer - set2set
        solute_feats = self.readout_layer(solute_graph, solute_feats) 
        solvent_feats = self.readout_layer(solvent_graph, solvent_feats) 
        # concatenate
        feats = torch.cat([solute_feats, solvent_feats], dim=1) 

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats


class InteractionMap(AttentionGCN):
    def forward(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
     solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding features as value
            norm_atom (2D tensor or None): graph norm for atom
            norm_bond (2D tensor or None): graph norm for bond

        Returns:
            2D tensor: of shape(N, M), where M = outdim.
        """

        solute_feats = self.solute_embedding(solute_feats)
        solvent_feats = self.solvent_embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

       # interaction map - attention mechanism
       # adapted from https://github.com/tbwxmu/SAMPN/blob/7d8db6223e8f6f35f0953310da03fa842187fbcc/mpn.py

        fts_solu = _split_batched_output_atoms(solute_graph, solute_feats["atom"]) 
        fts_solv = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"]) 
        updated_solute_atom_fts = []
        updated_solvent_atom_fts = []


        for solute_ft, solvent_ft in zip(fts_solu, fts_solv):
            pairwise_solute_feature = F.leaky_relu(self.solute_W_a(solute_ft), 0.01) 
            pairwise_solvent_feature = F.leaky_relu(self.solvent_W_a(solvent_ft), 0.01) 
            pairwise_pred = torch.sigmoid(torch.matmul(
                pairwise_solute_feature, pairwise_solvent_feature.t())) 

            new_solvent_feats = torch.matmul(pairwise_pred.t(), pairwise_solute_feature)
            new_solute_feats = torch.matmul(pairwise_pred, pairwise_solvent_feature) 

            # Add the old solute_ft to the new one to get a representation of both inter- and intra-molecular interactions.
            new_solute_feats += solute_ft
            new_solvent_feats += solvent_ft
            updated_solute_atom_fts.append(new_solute_feats)
            updated_solvent_atom_fts.append(new_solvent_feats)

        new_solute_feats = torch.cat(updated_solute_atom_fts)
        new_solvent_feats = torch.cat(updated_solvent_atom_fts)

        solute_feats["atom"] = new_solute_feats
        solvent_feats["atom"] = new_solvent_feats
        
        # readout layer - set2set
        solute_feats_prime = self.readout_layer(solute_graph, solute_feats) 
        solvent_feats_prime = self.readout_layer(solvent_graph, solvent_feats) 

        # concatenate
        feats = torch.cat([solute_feats_prime, solvent_feats_prime], dim=1) 

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats

        
    def visualise_attn_weights(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
     solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):

        solute_feats = self.solute_embedding(solute_feats)
        solvent_feats = self.solvent_embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        solute_wts = []
        solvent_wts = []

        fts_solu = _split_batched_output_atoms(solute_graph, solute_feats["atom"])
        fts_solv = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"]) 

        for solute_ft, solvent_ft in zip(fts_solu, fts_solv):
            pairwise_solute_feature = F.leaky_relu(self.solute_W_a(solute_ft), 0.1)
            pairwise_solvent_feature = F.leaky_relu(self.solvent_W_a(solvent_ft), 0.1) 
            
            pairwise_pred = torch.sigmoid(torch.matmul(
                pairwise_solute_feature, pairwise_solvent_feature.t()))

            solute_fts_att_w  = torch.matmul(pairwise_pred, pairwise_solvent_feature)       
            solvent_fts_att_w  = torch.matmul(pairwise_pred.t(), pairwise_solute_feature)

            solute_wts.append(solute_fts_att_w)
            solvent_wts.append(solvent_fts_att_w)

        return solute_wts, solvent_wts





def _split_batched_output_bonds(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.
    Returns:
        list of tensor.
    """
    nbonds = tuple(graph.batch_num_nodes("bond"))
    return torch.split(value, nbonds)

def _split_batched_output_atoms(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.
    Returns:
        list of tensor.
    """
    natoms = tuple(graph.batch_num_nodes("atom"))
    return torch.split(value, natoms)
def _split_batched_output_u2v(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.
    Returns:
        list of tensor.
    """
    natoms = tuple(graph.batch_num_edges("u2v"))
    return torch.split(value, natoms)
def _split_batched_output_v2u(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.
    Returns:
        list of tensor.
    """
    natoms = tuple(graph.batch_num_edges("v2u"))
    return torch.split(value, natoms)