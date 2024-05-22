
"""
Build molecule graph and then featurize it.
"""
import itertools
import dgl

class BaseGraph:
    """
    Base grapher to build DGL graph and featurizer. Typically should not use this
    directly.
    """
    
    def __init__(self, atom_featurizer=None, bond_featurizer=None, self_loop=False):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.self_loop = self_loop

    def build_graph(self, mol):
        raise NotImplementedError

    def featurize(self, g, mol, **kwargs):
        raise NotImplementedError

    def build_graph_and_featurize(self, mol, **kwargs):
        """
        Build a graph with atoms as the nodes and bonds as the edges and then featurize
        the graph.
        Args:
            mol (rdkit mol): a rdkit molecule
            kwargs: extra keyword arguments needed by featurizer
        Returns:
            (DGLGraph)
        """

        g = self.build_graph(mol)
        g = self.featurize(g, mol, **kwargs)
        return g

    @property
    def feature_size(self):
        res = {}
        if self.atom_featurizer is not None:
            res["atom"] = self.atom_featurizer.feature_size
        if self.bond_featurizer is not None:
            res["bond"] = self.bond_featurizer.feature_size
        if hasattr(self, "global_featurizer") and self.global_featurizer is not None:
            res["global"] = self.global_featurizer.feature_size
        return res

    @property
    def feature_name(self):
        res = {}
        if self.atom_featurizer is not None:
            res["atom"] = self.atom_featurizer.feature_name
        if self.bond_featurizer is not None:
            res["bond"] = self.bond_featurizer.feature_name
        if hasattr(self, "global_featurizer") and self.global_featurizer is not None:
            res["global"] = self.global_featurizer.feature_name
        return res

class HomoBidirectedGraph(BaseGraph):
    """
    Convert a RDKit molecule to a homogeneous bidirected DGLGraph and featurize for it.
    This creates a bidirectional graph. Atom i of the molecule is node i of the graph.
    Bond 0 corresponds to graph edges 0 and 1, bond 1 corresponds to graph edges 2,
    and 3 ... If `self_loop` is `True`, graph edge 2N represents self loop of atom 0,
    edge 2N+1 represents self loop of atom 1... where N is the number of bonds in the
    molecule.
    Notes:
        Make sure your featurizer match the above order, and pay carefully attention
        to bond featurizer.
    """
    def __init__(self, atom_featurizer=None, bond_featurizer=None, self_loop=True):
        super(HomoBidirectedGraph, self).__init__(
            atom_featurizer, bond_featurizer, self_loop
        )

    def build_graph(self, mol):

        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        src_list = []
        dst_list = []
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])
        if self.self_loop:
            src_list.extend(range(num_atoms))
            dst_list.extend(range(num_atoms))

        g = dgl.graph((src_list, dst_list), num_nodes=num_atoms)

        # add name
        g.mol_name = mol.GetProp("_Name")

        return g

    def featurize(self, g, mol, **kwargs):
        if self.atom_featurizer is not None:
            g.ndata.update(self.atom_featurizer(mol, **kwargs))
        if self.bond_featurizer is not None:
            g.edata.update(self.bond_featurizer(mol, **kwargs))
        return g

class HeteroMoleculeGraph(BaseGraph):
    """
    Convert a RDKit molecule to a DGLHeteroGraph and featurize for it.
    Atom, bonds, and global states are all represented as nodes in the graph.
    Atom i corresponds to graph node (of type `atom`) i.
    Bond i corresponds to graph node (of type `bond`) i.
    There is only one global state node 0.
    If no bonds (e.g. H+), create an artifact bond and connect it to the 1st atom
    """

    def __init__(
        self,
        atom_featurizer=None,
        bond_featurizer=None,
        global_featurizer=None,
        self_loop=True,
    ):
        super(HeteroMoleculeGraph, self).__init__(
            atom_featurizer, bond_featurizer, self_loop
        )
        self.global_featurizer = global_featurizer

    def build_graph(self, mol):
        num_atoms = mol.GetNumAtoms()
        # bonds
        num_bonds = mol.GetNumBonds()

        # If no bonds (e.g. H+), create an artifact bond and connect it to the 1st atom
        if num_bonds == 0:
            num_bonds = 1
            a2b = [(0, 0)]
            b2a = [(0, 0)]
        else:
            a2b = []
            b2a = []
            for b in range(num_bonds):
                bond = mol.GetBondWithIdx(b)
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                b2a.extend([[b, u], [b, v]])
                a2b.extend([[u, b], [v, b]])

        a2g = [(a, 0) for a in range(num_atoms)]
        g2a = [(0, a) for a in range(num_atoms)]
        b2g = [(b, 0) for b in range(num_bonds)]
        g2b = [(0, b) for b in range(num_bonds)]

        edges_dict = {
            ("atom", "a2b", "bond"): a2b,
            ("bond", "b2a", "atom"): b2a,
            ("atom", "a2g", "global"): a2g,
            ("global", "g2a", "atom"): g2a,
            ("bond", "b2g", "global"): b2g,
            ("global", "g2b", "bond"): g2b,
            ("atom2", "u2v", "atom"): [],
            ("atom", "v2u", "atom2"): [],
        }
        if self.self_loop:
            a2a = [(i, i) for i in range(num_atoms)]
            b2b = [(i, i) for i in range(num_bonds)]
            g2g = [(0, 0)]
            edges_dict.update(
                {
                    ("atom", "a2a", "atom"): a2a,
                    ("bond", "b2b", "bond"): b2b,
                    ("global", "g2g", "global"): g2g,
                    
                    ("atom2", "1a2b", "bond2"): [],
                    ("bond2", "1b2a", "atom2"): [],
                    ("atom2", "1a2g", "global"): [],
                    ("global", "1g2a", "atom2"): [],
                    ("bond2", "1b2g", "global"): [],
                    ("global", "1g2b", "bond2"): [],
                    ("atom2", "1a2a", "atom2"): [],
                    ("bond2", "1b2b", "bond2"): [],
                }
            )
        g = dgl.heterograph(edges_dict)

        # add name
        g.mol_name = mol.GetProp("_Name")

        return g

    def featurize(self, g, mol, **kwargs):
        if self.atom_featurizer is not None:
            g.nodes["atom"].data.update(self.atom_featurizer(mol, **kwargs))
        if self.bond_featurizer is not None:
            g.nodes["bond"].data.update(self.bond_featurizer(mol, **kwargs))
        if self.global_featurizer is not None:
            g.nodes["global"].data.update(self.global_featurizer(mol, **kwargs))

        return g
