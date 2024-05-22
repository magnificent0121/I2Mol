import logging
import sys, os
import time
import warnings
import torch
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.serialization import save
from gnn.model.metric import EarlyStopping
from gnn.model.gated_solv_network import GatedGCNSolvationNetwork, InteractionMap, SelfInteractionMap
from gnn.data.dataset import SolvationDataset, train_validation_test_split, solvent_split, element_split, substructure_split, stratified_solvent_split, stratified_split
from gnn.data.dataloader import DataLoaderSolvation
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import (
    SolventAtomFeaturizer,
    BondAsNodeFeaturizerFull,
    SolventGlobalFeaturizer,
)
from gnn.data.solvent_graph import HeteroMoleculeGraph2
from gnn.data.dataset import load_mols_labels
from gnn.utils import (
    load_checkpoints,
    pickle_load,
    save_checkpoints,
    seed_torch,
    pickle_dump,
    yaml_dump,
)
from sklearn.metrics import mean_squared_error

ls=[]
p=[]
def parse_args():
    parser = argparse.ArgumentParser(description="GatedSolvationNetwork")

    # input files and global variables
    parser.add_argument('--dataset-file', type=str, default="data/Abraham.csv")
    parser.add_argument('--dataset-pickle', type=str, default="data/Abraham.csv")
    parser.add_argument('--dielectric-constants', type=str, default=None)
    parser.add_argument('--molecular-refractivity', type=bool, default=False)
    parser.add_argument('--molecular-volume', type=bool, default=False)

    # output dir
    parser.add_argument('--save-dir', type=str, default="result/train_file_134667888")

    # training params
    parser.add_argument('--random-seed', type=int, default=50)
    parser.add_argument('--feature-scaling', type=bool, default=True)
    parser.add_argument('--solvent-split', type=str, default=None)
    parser.add_argument('--solvent-stratified-split', type=str, default=None)
    parser.add_argument('--solvent-stratified-frac', type=float, default=0.1)
    parser.add_argument('--stratified-split', type=bool, default=False)
    parser.add_argument('--element-split', type=str, default=None)
    parser.add_argument('--scaffold-split', type=bool, default=False)
    parser.add_argument('--attention-map', type=str, default=None)
    parser.add_argument('--partial-charges', type=str, default=None)


    # embedding layer
    parser.add_argument("--embedding-size", type=int, default=48)

    # gated layer
    parser.add_argument("--gated-num-layers", type=int, default=3)
    parser.add_argument("--gated-hidden-size", type=int, nargs="+", default=[800])
    parser.add_argument("--gated-num-fc-layers", type=int, default=3)
    parser.add_argument("--gated-graph-norm", type=int, default=0)
    parser.add_argument("--gated-batch-norm", type=int, default=0)
    parser.add_argument("--gated-activation", type=str, default="LeakyReLU")
    parser.add_argument("--gated-residual", type=int, default=1)
    parser.add_argument("--gated-dropout", type=float, default=0.0)

    # readout layer
    parser.add_argument(
        "--num-lstm-iters",
        type=int,
        default=6,
        help="number of iterations for the LSTM in set2set readout layer",
    )
    parser.add_argument(
        "--num-lstm-layers",
        type=int,
        default=3,
        help="number of layers for the LSTM in set2set readout layer",
    )

    # fc layer
    parser.add_argument("--fc-num-layers", type=int, default=4)
    parser.add_argument("--fc-hidden-size", type=int, nargs="+", default=[700])
    parser.add_argument("--fc-batch-norm", type=int, default=0)
    parser.add_argument("--fc-activation", type=str, default="LeakyReLU")
    parser.add_argument("--fc-dropout", type=float, default=0.5)

    # training
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=50, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--restore", type=int, default=0, help="read checkpoints")
    parser.add_argument("--load-dataset", type=int, default=0, help="read dataset")
    parser.add_argument(
        "--dataset-state-dict-filename", type=str, default="dataset_state_dict.pkl"
    )
    # gpu
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index. None to use CPU."
    )

    parser.add_argument(
        "--distributed",
        type=int,
        default=0,
        help="DDP training, --gpu is ignored if this is True",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=1,
        help="Number of GPU to use in distributed mode; ignored otherwise.",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://localhost:13456",
        type=str,
        help="url used to set up distributed training",
    )
    
    parser.add_argument("--dist-backend", type=str, default="nccl")

    # output file (needed by hypertunity)
    parser.add_argument("--output_file", type=str, default="results.pkl")

    args = parser.parse_args()
    if len(args.gated_hidden_size) == 1:
        args.gated_hidden_size = args.gated_hidden_size * args.gated_num_layers
    else:
        assert len(args.gated_hidden_size) == args.gated_num_layers, (
            "length of `gat-hidden-size` should be equal to `num-gat-layers`, but got "
            "{} and {}.".format(args.gated_hidden_size, args.gated_num_layers)
        )

    if len(args.fc_hidden_size) == 1:
        val = 2 * args.gated_hidden_size[-1]
        args.fc_hidden_size = [max(val // 2 ** i, 8) for i in range(args.fc_num_layers)]
    else:
        assert len(args.fc_hidden_size) == args.fc_num_layers, (
            "length of `fc-hidden-size` should be equal to `num-fc-layers`, but got "
            "{} and {}.".format(args.fc_hidden_size, args.fc_num_layers)
        )
    return args

def train(optimizer, model, nodes,nodes1, data_loader, loss_fn, metric_fn, device=None):
    """
    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (solute_batched_graph, solvent_batched_graph, label) in enumerate(data_loader):
        solute_feats = {nt: solute_batched_graph.nodes[nt].data["feat"] for nt in nodes1}
        solvent_feats = {nt: solvent_batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = torch.squeeze(label["value"])
        solute_norm_atom = label["solute_norm_atom"]
        solute_norm_bond = label["solute_norm_bond"]
        solvent_norm_atom = label["solvent_norm_atom"]
        solvent_norm_bond = label["solvent_norm_bond"]
        #stdev = label["scaler_stdev"]

        if device is not None:
            solute_feats = {k: v.to(device) for k, v in solute_feats.items()}
            solvent_feats = {k: v.to(device) for k, v in solvent_feats.items()}
            target = target.to(device)
            solute_norm_atom = solute_norm_atom.to(device)
            solute_norm_bond = solute_norm_bond.to(device)
            solvent_norm_atom = solvent_norm_atom.to(device)
            solvent_norm_bond = solvent_norm_bond.to(device)
            #stdev = stdev.to(device)
        #print(solute_feats)
        pred = model(solute_batched_graph, solvent_batched_graph, solute_feats, 
                     solvent_feats, solute_norm_atom, solute_norm_bond, 
                     solvent_norm_atom, solvent_norm_bond)
        pred = pred.view(-1)
        target = target.view(-1)
        #print(pred)
        #print("********")
        #print(target)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, target).detach().item()
        count += len(target)
    
    epoch_loss /= it + 1
    accuracy /= count

    return epoch_loss, accuracy

def evaluate(model, nodes,nodes1, data_loader, metric_fn, scaler = None, device=None, return_preds=False):
    """
    Evaluate the accuracy of a validation set of test set.
    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """
    model.eval()
    with torch.no_grad():
        accuracy = 0.0
        count = 0.0

        preds = []
        y_true = []

        for solute_batched_graph, solvent_batched_graph, label in data_loader:
            solute_feats = {nt: solute_batched_graph.nodes[nt].data["feat"] for nt in nodes1}
            solvent_feats = {nt: solvent_batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = torch.squeeze(label["value"])
            #stdev = label["scaler_stdev"]
            solvent_norm_atom = label["solvent_norm_atom"]
            solvent_norm_bond = label["solvent_norm_bond"]
            solute_norm_atom = label["solute_norm_atom"]
            solute_norm_bond = label["solute_norm_bond"]

            if device is not None:
                solute_feats = {k: v.to(device) for k, v in solute_feats.items()}
                solvent_feats = {k: v.to(device) for k, v in solvent_feats.items()}
                target = target.to(device)
                solute_norm_atom = solute_norm_atom.to(device)
                solute_norm_bond = solute_norm_bond.to(device)
                solvent_norm_atom = solvent_norm_atom.to(device)
                solvent_norm_bond = solvent_norm_bond.to(device)

            pred = model(solute_batched_graph, solvent_batched_graph, solute_feats, 
                     solvent_feats, solute_norm_atom, solute_norm_bond, 
                     solvent_norm_atom, solvent_norm_bond)
            pred = pred.view(-1)
            target = target.view(-1)

            # Inverse scaler
            if scaler is not None:
                pred = scaler.inverse_transform(pred.cpu())
                pred = pred.to(device)

            accuracy += metric_fn(pred, target).detach().item()
            count += len(target)
            #print("----------------------")
            #print(pred)
            #print("===========")
            #print(target)
            #print("----------------------")
            batch_pred = pred.tolist()
            batch_target = target.tolist()
            preds.extend(batch_pred)
            y_true.extend(batch_target)

    if return_preds:
        return y_true, preds

    else:
        return accuracy / count
#from gnn.data.grapher import HeteroMoleculeGraph
#from gnn.data.featurizer import (
#    SolventAtomFeaturizer,
#    BondAsNodeFeaturizerFull,
#    SolventGlobalFeaturizer,
#)
def grapher(dielectric_constant=None, mol_volume=False, mol_refract=False, partial_charges=None,lable=False):
    atom_featurizer = SolventAtomFeaturizer(partial_charges=partial_charges)
    bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
    global_featurizer = SolventGlobalFeaturizer(dielectric_constant=dielectric_constant, mol_volume=mol_volume, mol_refract=mol_refract)
    if lable:
        grapher = HeteroMoleculeGraph(atom_featurizer, bond_featurizer, global_featurizer, self_loop=True)
    else:
        grapher = HeteroMoleculeGraph2(atom_featurizer, bond_featurizer, global_featurizer, self_loop=True)

    return grapher

def main_worker(gpu, world_size, args):
    
    # Explicitly setting seed to ensure the same dataset split and models created in
    # two processes (when distributed) start from the same random weights and biases
    random_seed = args.random_seed
    seed_torch(random_seed)

    args.gpu = gpu

    if not args.distributed or (args.distributed and args.gpu == 0):
        print("\n\nStart training at: ", datetime.now())

    if args.save_dir is None:
        args.save_dir = os.getcwd()

    if args.distributed:
        dist.init_process_group(
            args.dist_backend,
            init_method = args.dist_url,
            world_size = world_size,
            rank = args.gpu
        )
    
    if args.restore:
        dataset_state_dict_filename = args.dataset_state_dict_filename

        if dataset_state_dict_filename is None:
            warnings.warn("Restore with `args.dataset_state_dict_filename` set to None.")
        elif not Path(dataset_state_dict_filename).exists():
            warnings.warn(
                f"`{dataset_state_dict_filename} not found; set "
                f"args.dataset_state_dict_filename` to None"
            )
            dataset_state_dict_filename = None
    else:
        dataset_state_dict_filename = None

    # Load molecules and labels from file
    mols, labels = load_mols_labels(args.dataset_file)

    if args.load_dataset:
        data_dict = args.dataset_pickle
        dataset = pickle_load(data_dict)
    
    else:
        if args.dielectric_constants is not None:
            dc_file = Path(args.dielectric_constants)        
            dataset = SolvationDataset(
                solute_grapher = grapher(mol_volume = args.molecular_volume,
                                        mol_refract = args.molecular_refractivity,
                                        partial_charges=args.partial_charges),
                solvent_grapher = grapher(dielectric_constant=True,
                                        mol_volume = args.molecular_volume,
                                        mol_refract = args.molecular_refractivity,
                                        partial_charges=args.partial_charges),
                molecules = mols,
                labels = labels,
                solute_extra_features = None,
                solvent_extra_features=dc_file,
                feature_transformer = False,
                label_transformer= False,
                state_dict_filename=dataset_state_dict_filename)

        else:
            dataset = SolvationDataset(
                solute_grapher = grapher(mol_volume=args.molecular_volume, mol_refract = args.molecular_refractivity, partial_charges=args.partial_charges),
                solvent_grapher = grapher(mol_volume=args.molecular_volume, mol_refract = args.molecular_refractivity, partial_charges=args.partial_charges,lable=True),
                molecules = mols,
                labels = labels,
                solute_extra_features = None,
                solvent_extra_features = None,
                feature_transformer = False,
                label_transformer= False,
                state_dict_filename=dataset_state_dict_filename
                )

    # Save the solute and solvent graphers for loading datasets later
    pickle_dump([dataset.solute_grapher, dataset.solvent_grapher], os.path.join(args.save_dir,"graphers.pkl"))

    best = np.finfo(np.float32).max
    os.makedirs(args.save_dir, exist_ok=True)

    # Split data: random, solvent-based split, element-based, or scaffold-based split

    possible_solvents = ['hexane', 'water', 'acetone', 'ethanol', 'benzene', 'ethylacetate',
               'dichloromethane', 'acetonitrile', 'thf', 'dmso', 'dmf', 'octanol', 'hexadecane', 'cyclohexane']

    if (args.solvent_split is None) and (args.element_split is None) and (args.solvent_stratified_split is None) and (args.stratified_split is False) and (args.scaffold_split is False):
        print(f'Splitting data using random seed {random_seed}')
        trainset, valset, testset = train_validation_test_split(
            dataset, validation=0.1, test=0.1, random_seed=args.random_seed)
    
    elif args.solvent_split is not None:
        assert args.solvent_split in possible_solvents, "Solvent unavailable! Choose from: hexane, cyclohexane, water, acetone, ethanol, benzene, ethylacetate, dichloromethane, acetonitrile, thf, dmso"
        print(f'Using compounds with {args.solvent_split} solvent as test data.')
        trainset, valset, testset = solvent_split(
            dataset, args.solvent_split, random_seed=args.random_seed)
    elif args.solvent_stratified_split is not None:
        assert args.solvent_stratified_split in possible_solvents, "Solvent unavailable! Choose from: hexane, cyclohexane, water, acetone, ethanol, benzene, ethylacetate, dichloromethane, acetonitrile, thf, dmso"
        print(f'Using {1-args.solvent_stratified_frac}% of {args.solvent_stratified_split} solvent as test data.')
        trainset, valset, testset = stratified_solvent_split(
            dataset, args.solvent_stratified_split, frac=args.solvent_stratified_frac, random_seed=args.random_seed)
    
    elif args.scaffold_split is True:
        trainset, valset, testset = substructure_split(
            dataset, random_seed=args.random_seed)
    
    elif args.stratified_split is True:
        trainset, valset, testset = stratified_split(
            dataset, random_seed=args.random_seed)
    
    elif args.element_split is not None: # element split
        possible_elems = ['Br', 'Cl', 'F', 'I', 'N', 'O', 'S']
        elem = args.element_split
        assert elem in possible_elems, "Element unavailable! Choose from: 'Br', 'Cl', 'F', 'I', 'N', 'O', 'S'"
        print(f'Placing all solutes with {elem} atoms into the test dataset.')
        trainset, valset, testset = element_split(dataset, elem, random_seed=args.random_seed)

    # Scale training dataset features
    if args.feature_scaling:
        solute_features_scaler,solvent_features_scaler= trainset.normalize_features()
        #solute_features_scaler, solvent_features_scaler = trainset.normalize_features()
        valset.normalize_features(solute_features_scaler, solvent_features_scaler)
        testset.normalize_features(solute_features_scaler, solvent_features_scaler)
        #testset.normalize_features(solute_features_scaler)
    else:
        solute_features_scaler, solvent_features_scaler = None, None
    
    label_scaler = trainset.normalize_labels()
    if not args.distributed or (args.distributed and args.gpu == 0):
        torch.save(dataset.state_dict(), os.path.join(args.save_dir, args.dataset_state_dict_filename))
        print(
            "Trainset size: {}, valset size: {}: testset size: {}.".format(
                len(trainset), len(valset), len(testset)
            )
        )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None
    
    train_loader = DataLoaderSolvation(
        trainset,
        batch_size = args.batch_size,
        shuffle = (train_sampler is None),
        sampler = train_sampler
    )
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of val and test set to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoaderSolvation(valset, batch_size=bs, shuffle=False)
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoaderSolvation(testset, batch_size=bs, shuffle=False)
    ### model
    feature_names = ["atom", "bond", "global"]
    solute_feature_names = ["atom", "bond","atom2", "bond2", "global"]
    set2set_ntypes_direct = ["global"]
    solute_feature_size = dataset.feature_sizes[0]
    solute_feature_size ={'bond': 11,'atom': 28,'atom2': 28, 'bond2': 11, 'global': 3}
    solvent_feature_size = dataset.feature_sizes[1]
    args.solute_feature_size = solute_feature_size
    args.solvent_feature_size = solvent_feature_size
    args.set2set_ntypes_direct = set2set_ntypes_direct
    # save args
    if not args.distributed or (args.distributed and args.gpu == 0):
        yaml_dump(args, os.path.join(args.save_dir, "train_args.yaml"))

    if args.attention_map == 'cross':
        model = InteractionMap(
            solute_in_feats=args.solute_feature_size,
            solvent_in_feats=args.solvent_feature_size,
            embedding_size=args.embedding_size,
            gated_num_layers=args.gated_num_layers,
            gated_hidden_size=args.gated_hidden_size,
            gated_num_fc_layers=args.gated_num_fc_layers,
            gated_graph_norm=args.gated_graph_norm,
            gated_batch_norm=args.gated_batch_norm,
            gated_activation=args.gated_activation,
            gated_residual=args.gated_residual,
            gated_dropout=args.gated_dropout,
            num_lstm_iters=args.num_lstm_iters,
            num_lstm_layers=args.num_lstm_layers,
            set2set_ntypes_direct=args.set2set_ntypes_direct,
            fc_num_layers=args.fc_num_layers,
            fc_hidden_size=args.fc_hidden_size,
            fc_batch_norm=args.fc_batch_norm,
            fc_activation=args.fc_activation,
            fc_dropout=args.fc_dropout,
            outdim=1,
            conv="GatedGCNConv",
        )
        
    elif args.attention_map == 'self':
        model = SelfInteractionMap(
            solute_in_feats=args.solute_feature_size,
            solvent_in_feats=args.solvent_feature_size,
            embedding_size=args.embedding_size,
            gated_num_layers=args.gated_num_layers,
            gated_hidden_size=args.gated_hidden_size,
            gated_num_fc_layers=args.gated_num_fc_layers,
            gated_graph_norm=args.gated_graph_norm,
            gated_batch_norm=args.gated_batch_norm,
            gated_activation=args.gated_activation,
            gated_residual=args.gated_residual,
            gated_dropout=args.gated_dropout,
            num_lstm_iters=args.num_lstm_iters,
            num_lstm_layers=args.num_lstm_layers,
            set2set_ntypes_direct=args.set2set_ntypes_direct,
            fc_num_layers=args.fc_num_layers,
            fc_hidden_size=args.fc_hidden_size,
            fc_batch_norm=args.fc_batch_norm,
            fc_activation=args.fc_activation,
            fc_dropout=args.fc_dropout,
            outdim=1,
            conv="GatedGCNConv",
        )
    else:
        model = GatedGCNSolvationNetwork(
            solute_in_feats=args.solute_feature_size,
            solvent_in_feats=args.solvent_feature_size,
            embedding_size=args.embedding_size,
            gated_num_layers=args.gated_num_layers,
            gated_hidden_size=args.gated_hidden_size,
            gated_num_fc_layers=args.gated_num_fc_layers,
            gated_graph_norm=args.gated_graph_norm,
            gated_batch_norm=args.gated_batch_norm,
            gated_activation=args.gated_activation,
            gated_residual=args.gated_residual,
            gated_dropout=args.gated_dropout,
            num_lstm_iters=args.num_lstm_iters,
            num_lstm_layers=args.num_lstm_layers,
            set2set_ntypes_direct=args.set2set_ntypes_direct,
            fc_num_layers=args.fc_num_layers,
            fc_hidden_size=args.fc_hidden_size,
            fc_batch_norm=args.fc_batch_norm,
            fc_activation=args.fc_activation,
            fc_dropout=args.fc_dropout,
            outdim=1,
            conv="GatedGCNConv",
        )
    # if not args.distributed or (args.distributed and args.gpu == 0):
    #     print(model)

    print(f'Model type: {type(model)}')

    if args.gpu is not None:
        model.to(args.gpu)
    if args.distributed:
        ddp_model = DDP(model, device_ids=[args.gpu])
        ddp_model.feature_before_fc = model.feature_before_fc
        model = ddp_model
    ### optimizer, loss, and metric
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_func = MSELoss(reduction="mean")
    metric = L1Loss(reduction="sum")
    ### learning rate scheduler and stopper
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=50, verbose=True
    )
    stopper = EarlyStopping(patience=150)
    # load checkpoint
    state_dict_objs = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
    if args.restore:
        try:
            if args.gpu is None:
                checkpoint = load_checkpoints(state_dict_objs, save_dir=args.save_dir, filename="checkpoint.pkl")
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = load_checkpoints(
                    state_dict_objs, map_location=loc, save_dir=args.save_dir, filename="checkpoint.pkl"
                )
            args.start_epoch = checkpoint["epoch"]
            best = checkpoint["best"]
            print(f"Successfully load checkpoints, best {best}, epoch {args.start_epoch}")
        except FileNotFoundError as e:
            warnings.warn(str(e) + " Continue without loading checkpoints.")
            pass
    # start training
    if not args.distributed or (args.distributed and args.gpu == 0):
        print("\n\n# Epoch     Loss         TrainAcc        ValAcc     Time (s)")
        sys.stdout.flush()
    for epoch in range(args.start_epoch, args.epochs):
        ti = time.time()
        # In distributed mode, calling the set_epoch method is needed to make shuffling
        # work; each process will use the same random seed otherwise.
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train
        loss, train_acc = train(
            optimizer, model, feature_names, solute_feature_names,train_loader, loss_func, metric, args.gpu)
        # bad, we get nan
        if np.isnan(loss):
            print("\n\nBad, we get nan for loss. Exiting")
            sys.stdout.flush()
            sys.exit(1)
        # evaluate
        val_acc = evaluate(model, feature_names, solute_feature_names,val_loader, metric, label_scaler, args.gpu)
        if stopper.step(val_acc):
            pickle_dump(best, os.path.join(args.save_dir, args.output_file))  # save results for hyperparam tune
            break
        scheduler.step(val_acc)
        is_best = val_acc < best
        if is_best:
            best = val_acc
        # save checkpoint
        if not args.distributed or (args.distributed and args.gpu == 0):
            misc_objs = {"best": best, "epoch": epoch}
            scaler_objs = {'label_scaler': {
                            'means': label_scaler.mean,
                            'stds': label_scaler.std
                            } if label_scaler is not None else None,
                            'solute_features_scaler': {
                            'means': solute_features_scaler.mean,
                            'stds': solute_features_scaler.std
                            } if solute_features_scaler is not None else None,
                            'solvent_features_scaler': {
                            'means': solvent_features_scaler.mean,
                            'stds': solvent_features_scaler.std
                            } if solvent_features_scaler is not None else None}
            save_checkpoints(
                state_dict_objs,
                misc_objs,
                scaler_objs,
                is_best,
                msg=f"epoch: {epoch}, score {val_acc}",
                save_dir=args.save_dir)
            tt = time.time() - ti
            print(
                "{:5d}   {:12.6e}   {:12.6e}   {:12.6e}   {:.2f}".format(
                    epoch, loss, train_acc, val_acc, tt
                )
            )
            ls.append( val_acc)
            if epoch % 10 == 0:
                sys.stdout.flush()
    # load best to calculate test accuracy
    if args.gpu is None:
        checkpoint = load_checkpoints(state_dict_objs, args.save_dir, filename="best_checkpoint.pkl")
    else:
        # Map model to be loaded to specified single  gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = load_checkpoints(
            state_dict_objs, map_location=loc, save_dir=args.save_dir, filename="best_checkpoint.pkl"
        )
    
    if not args.distributed or (args.distributed and args.gpu == 0):
        test_acc = evaluate(model, feature_names,solute_feature_names, test_loader, metric, label_scaler, args.gpu)
        y_true, y_pred = evaluate(model, feature_names,solute_feature_names, test_loader, metric, 
                                    label_scaler, args.gpu, return_preds=True)
        
        print(len(y_true))
        print(len(y_pred))
        print("\n#Test MAE: {:12.6e} \n".format(test_acc))
        print("\n#Test RMSE: {:12.6e} \n".format(mean_squared_error(y_true, y_pred, squared=False)))
        print("\nFinish training at:", datetime.now())
        p.append(mean_squared_error(y_true, y_pred, squared=False))
        results_dict = {'y_true': y_true, 'y_pred': y_pred}
        pickle_dump(results_dict, os.path.join(args.save_dir, f'seed_{random_seed}_test_results.pkl'))


def main():
    args = parse_args()
    print(args)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
    filename=os.path.join(args.save_dir, '{}.log'.format(
        datetime.now().strftime("gnn_%Y_%m_%d-%I_%M_%p"))),
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    level=logging.INFO,
    )

    if args.distributed:
        # DDP
        world_size = torch.cuda.device_count() if args.num_gpu is None else args.num_gpu
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

    else:
        # train on CPU or a single GPU
        main_worker(args.gpu, None, args)


if __name__ == "__main__":
    main()
