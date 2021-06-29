import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from datetime import datetime
import time
from Dataloader import get_loader


class MolecularDataset():

    def load(self, filename, subset=1):

        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))


        if filename.endswith('.pkl'):
            self.data = [Chem.MolFromSmiles(line) for line in pickle.load(open(filename, 'rb'))]

        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data   #add H's to molecule and map
        self.data = list(filter(filters, self.data))
        self.data = self.data[:size]

        self.log('Extracted {} out of {} molecules {}adding Hydrogen!'.format(len(self.data),
                                                                              len(Chem.SDMolSupplier(filename)),
                                                                              '' if add_h else 'not '))
        
        self._generate_encoders_decoders()
        self._generate_AX()

        # contains all molecules stored as rdkit.Chem objects
        self.data = np.array(self.data)
        
        # contains all molecules stored as SMILES strings
        self.smiles = np.array(self.smiles)
        
        # (N, L) matrix where N is the length of the dataset and each L-dim vector contains the 
        # indices corresponding to a SMILE sequences with padding w.r.t the max length of the longest 
        # SMILES sequence in the dataset
        self.data_S = np.stack(self.data_S)
        
        # (N, 9, 9) tensor where N is the length of the dataset and each 9x9 matrix contains the 
        # indices of the positions of the ones in the one-hot representation of the adjacency tensor
        self.data_A = np.stack(self.data_A)
        
        # (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the 
        # indices of the positions of the ones in the one-hot representation of the annotation matrix
        self.data_X = np.stack(self.data_X)
        
        # (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the 
        # diagonal of the correspondent adjacency matrix
        self.data_D = np.stack(self.data_D)
        
        # a (N, F) matrix where N is the length of the dataset and each F vector contains features 
        # of the correspondent molecule
        self.data_F = np.stack(self.data_F)
        
        # (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the
        # eigenvalues of the correspondent Laplacian matrix
        self.data_Le = np.stack(self.data_Le)
        
        # (N, 9, 9) matrix where N is the length of the dataset and each 9x9 matrix contains the 
        # eigenvectors of the correspondent Laplacian matrix
        self.data_Lv = np.stack(self.data_Lv)

        self.vertexes = self.data_F.shape[-2]
        self.features = self.data_F.shape[-1]

        self._generate_train_validation_test(validation, test)
    
    def _generate_encoders_decoders(self):
        self.log('Creating atoms encoder and decoder..')
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        self.log('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))

        self.log('Creating bonds encoder and decoder..')
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
                                                                    for mol in self.data
                                                                    for bond in mol.GetBonds())))

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))

        self.log('Creating SMILES encoder and decoder..')
        smiles_labels = ['E'] + list(set(c for mol in self.data for c in Chem.MolToSmiles(mol)))
        self.smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        self.smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        self.smiles_num_types = len(smiles_labels)
        self.log('Created SMILES encoder and decoder with {} types and 1 PAD symbol!'.format(
            self.smiles_num_types - 1))
        
    def _generate_AX(self):
        self.log('Creating features and adjacency matrices..')

        pb = tqdm(len(self.data))

        data = []
        smiles = []
        data_S = []
        data_A = []
        data_X = []
        data_D = []
        data_F = []
        data_Le = []
        data_Lv = []

        max_length = max(mol.GetNumAtoms() for mol in self.data)
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.data)  #max len of smiles

        for i, mol in enumerate(self.data):
            A = self._genA(mol, connected=True, max_length=max_length)
            D = np.count_nonzero(A, -1)
            if A is not None:
                data.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                data_S.append(self._genS(mol, max_length=max_length_s))
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))
                data_D.append(D)
                data_F.append(self._genF(mol, max_length=max_length))

                L = np.diag(D) - A   #Diag mat(non-zero counts in adjacency tensor) - adjacency tensor
                Le, Lv = np.linalg.eigh(L) #get eig values and vectors

                data_Le.append(Le)
                data_Lv.append(Lv)

            pb.update(i + 1)

        self.log(date=False)
        self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(len(data),
                                                                                           len(self.data)))

        self.data = data
        self.smiles = smiles
        self.data_S = data_S
        self.data_A = data_A
        self.data_X = data_X
        self.data_D = data_D
        self.data_F = data_F
        self.data_Le = data_Le
        self.data_Lv = data_Lv
        self.__len = len(self.data)


    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        #get atom index of end atom in the bond
        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
                    max_length - mol.GetNumAtoms()), dtype=np.int32)

    def _genS(self, mol, max_length=None):

        max_length = max_length if max_length is not None else len(Chem.MolToSmiles(mol))

        return np.array([self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)] + [self.smiles_encoder_m['E']] * (
                    max_length - len(Chem.MolToSmiles(mol))), dtype=np.int32)

    def _genF(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        
        features = np.array([[*[a.GetDegree() == i for i in range(5)], #get no. of bonded neighbors in the graph
                              *[a.GetExplicitValence() == i for i in range(9)],  # get explicit valence (including Hs) of this atom
                              *[int(a.GetHybridization()) == i for i in range(1, 7)], #get the hybridization 
                              *[a.GetImplicitValence() == i for i in range(9)],  # get implicit valence of this atom
                              a.GetIsAromatic(),
                              a.GetNoImplicit(),
                              *[a.GetNumExplicitHs() == i for i in range(5)], #get no. of explicit Hs
                              *[a.GetNumImplicitHs() == i for i in range(5)], #get no. of implicit Hs the atom is bonded to
                              *[a.GetNumRadicalElectrons() == i for i in range(5)], #get no. of radical electron of the atom
                              a.IsInRing(),
                              *[a.IsInRingSize(i) for i in range(2, 9)]] for a in mol.GetAtoms()], dtype=np.int32)

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))
    
    def matrices2mol(self, node_labels, edge_labels, strict=False):
        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label])) #add atoms and returns the bonded atom

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def seq2mol(self, seq, strict=False):
        mol = Chem.MolFromSmiles(''.join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def _generate_train_validation_test(self, validation, test):

        self.log('Creating train, validation and test sets..')

        validation = int(validation * len(self))
        test = int(test * len(self))
        train = len(self) - validation - test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[0:train]
        self.validation_idx = self.all_idx[train:train + validation]
        self.test_idx = self.all_idx[train + validation:]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        self.test_count = test

        self.log('Created train ({} items), validation ({} items) and test ({} items) sets!'.format(
            train, validation, test))

    def _next_batch(self, counter, count, idx, batch_size):
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = [obj[idx[counter:counter + batch_size]]
                      for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                  self.data_D, self.data_F, self.data_Le, self.data_Lv)]

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                           self.data_D, self.data_F, self.data_Le, self.data_Lv)]

        return [counter] + output
    
    def next_train_batch(self, batch_size=None):
        out = self._next_batch(counter=self.train_counter, count=self.train_count,
                               idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[0]

        return out[1:]

    def next_validation_batch(self, batch_size=None):
        out = self._next_batch(counter=self.validation_counter, count=self.validation_count,
                               idx=self.validation_idx, batch_size=batch_size)
        self.validation_counter = out[0]

        return out[1:]

    def next_test_batch(self, batch_size=None):
        out = self._next_batch(counter=self.test_counter, count=self.test_count,
                               idx=self.test_idx, batch_size=batch_size)
        self.test_counter = out[0]

        return out[1:]

    # def create_non_uniform_split(args, idxs, client_number, is_train=True):
    # logging.info("create_non_uniform_split------------------------------------------")
    # N = len(idxs)
    # alpha = args.partition_alpha
    # logging.info("sample number = %d, client_number = %d" % (N, client_number))
    # logging.info(idxs)
    # idx_batch_per_client = [[] for _ in range(client_number)]
    # idx_batch_per_client, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_number,
    #                                                                                      idx_batch_per_client, idxs)
    # logging.info(idx_batch_per_client)
    # sample_num_distribution = []

    # for client_id in range(client_number):
    #     sample_num_distribution.append(len(idx_batch_per_client[client_id]))
    #     logging.info("client_id = %d, sample_number = %d" % (client_id, len(idx_batch_per_client[client_id])))
    # logging.info("create_non_uniform_split******************************************")

    # # plot the (#client, #sample) distribution
    # if is_train:
    #     logging.info(sample_num_distribution)
    #     plt.hist(sample_num_distribution)
    #     plt.title("Sample Number Distribution")
    #     plt.xlabel('number of samples')
    #     plt.ylabel("number of clients")
    #     fig_name = "x_hist.png"
    #     fig_dir = os.path.join("./visualization", fig_name)
    #     plt.savefig(fig_dir)
    # return idx_batch_per_client

    # def partition_data_by_sample_size(args, data_dir, client_number, uniform=True):

    #train_data = self.data[self.train_idx]
    #val_data = self.data[self.validation_idx]
    #test_data = self.data[self.test_idx]

    #train_smiles = self.smiles[self.train_idx]
    #val_smiles = self.smiles[self.validation_idx]
    #test_smiles = self.smiles[self.test_idx]

    #S_train = self.S[self.train_idx]
    #S_val = self.S[self.validation_idx]
    #S_test = self.S[self.test_idx]

    #X_train = self.X[self.train_idx]
    #X_val = self.X[self.validation_idx]
    #X_test = self.X[self.test_idx]

    #D_train = self.D[self.train_idx]
    #D_val = self.D[self.validation_idx]
    #D_test = self.D[self.test_idx]

    #F_train = self.F[self.train_idx]
    #F_val = self.F[self.validation_idx]
    #F_test = self.F[self.test_idx]

    #Lv_train = self.data_Lv[self.train_idx]
    #Lv_val = self.data_Lv[self.validation_idx]
    #Lv_test = self.data_Lv[self.test_idx]

    #Le_train = self.data_Le[self.train_idx]
    #Le_val = self.data_Le[self.validation_idx]
    #Le_test = self.data_Le[self.test_idx]

    #train_adj_matrices = self.data_A[self.train_idx]
    #val_adj_matrices = self.data_A[self.validation_idx]
    #test_adj_matrices = self.data_A[self.test_idx]

    #train_feat_matrices = self.data_F[self.train_idx]
    #val_feat_matrices = self.data_F[self.validation_idx]
    #test_feat_matrices = self.data_F[self.test_idx]

    #train_labels = atom_labels[self.train_idx]
    #val_labels = atom_labels[self.validation_idx]
    #test_labels = atom_labels[self.test_idx]

    # num_train_samples = len(train_data)
    # num_val_samples = len(val_data)
    # num_test_samples = len(test_data)

    # train_idxs = list(range(num_train_samples))
    # val_idxs = list(range(num_val_samples))
    # test_idxs = list(range(num_test_samples))

    # random.shuffle(train_idxs)
    # random.shuffle(val_idxs)
    # random.shuffle(test_idxs)

    # partition_dicts = [None] * client_number

    # if uniform:
    #     clients_idxs_train = np.array_split(train_idxs, client_number)
    #     clients_idxs_val = np.array_split(val_idxs, client_number)
    #     clients_idxs_test = np.array_split(test_idxs, client_number)
    # else:
    #     clients_idxs_train = create_non_uniform_split(args, train_idxs, client_number, True)
    #     clients_idxs_val = create_non_uniform_split(args, val_idxs, client_number, False)
    #     clients_idxs_test = create_non_uniform_split(args, test_idxs, client_number, False)

    # labels_of_all_clients = []
    # for client in range(client_number):
    #     client_train_idxs = clients_idxs_train[client]
    #     client_val_idxs = clients_idxs_val[client]
    #     client_test_idxs = clients_idxs_test[client]

    #train_adj_matrices_client = [train_adj_matrices[idx] for idx in client_train_idxs]
    #train_feat_matrices_client = [train_feat_matrices[idx] for idx in client_train_idxs]
    #train_data_client = [train_data[idx] for idx in client_train_idxs]
    #train_smiles_client = [train_smiles[idx] for idx in client_train_idxs]
    #S_train_client = [S_train[idx] for idx in client_train_idxs]
    #X_train_client = [X_train[idx] for idx in client_train_idxs]
    #D_train_client = [D_train[idx] for idx in client_train_idxs]
    #F_train_client = [F_train[idx] for idx in client_train_idxs]
    #Lv_train_client = [Lv_train[idx] for idx in client_train_idxs]
    #Le_train_client = [Le_train[idx] for idx in client_train_idxs]
    #train_labels_client = [train_labels[idx] for idx in client_train_idxs]
    #labels_of_all_clients.append(train_labels_client)

    #val_adj_matrices_client = [val_adj_matrices[idx] for idx in client_val_idxs]
    #val_feat_matrices_client = [val_feat_matrices[idx] for idx in client_val_idxs]
    #val_data_client = [val_data[idx] for idx in client_val_idxs]
    #val_smiles_client = [val_smiles[idx] for idx in client_val_idxs]
    #S_val_client = [S_val[idx] for idx in client_val_idxs]
    #X_val_client = [X_val[idx] for idx in client_val_idxs]
    #D_val_client = [D_val[idx] for idx in client_val_idxs]
    #F_val_client = [F_val[idx] for idx in client_val_idxs]
    #Lv_val_client = [Lv_val[idx] for idx in client_val_idxs]
    #Le_val_client = [Le_val[idx] for idx in client_val_idxs]
    #val_labels_client = [val_labels[idx] for idx in client_val_idxs]

    # test_adj_matrices_client = [test_adj_matrices[idx] for idx in client_test_idxs]
    # test_feat_matrices_client = [test_feat_matrices[idx] for idx in client_test_idxs]
    #test_data_client = [test_data[idx] for idx in client_test_idxs]
    #test_smiles_client = [test_smiles[idx] for idx in client_test_idxs]
    #S_test_client = [S_test[idx] for idx in client_test_idxs]
    #X_test_client = [X_test[idx] for idx in client_test_idxs]
    #D_test_client = [D_test[idx] for idx in client_test_idxs]
    #F_test_client = [F_test[idx] for idx in client_test_idxs]
    #Lv_test_client = [Lv_test[idx] for idx in client_test_idxs]
    #Le_test_client = [Le_test[idx] for idx in client_test_idxs]
    #test_labels_client = [test_labels[idx] for idx in client_test_idxs]

    # train_dataset_client = Molecular(train_data_client, train_smiles_client, S_train_client, train_adj_matrices_client, X_train_client, D_train_client,
    #                                  train_feature_matrices_client, Le_train_client, Lv_train_client)
    # val_dataset_client = Molecular(val_data_client, val_smiles_client, S_val_client, val_adj_matrices_client, X_val_client, D_val_client, 
    #                                  val_feature_matrices_client, Le_val_client, Lv_val_client)
    # test_dataset_client = Molecular(test_data_client, test_smiles_client, S_test_client, test_adj_matrices_client, X_test_client, D_test_client,
    #                                  test_feature_matrices_client, Le_test_client, Lv_test_client)

    # partition_dict = {'train': train_dataset_client,
    #                   'val': val_dataset_client,
    #                   'test': test_dataset_client}

    #partition_dicts[client] = partition_dict

    #plot the label distribution similarity score
    #visualize_label_distribution_similarity_score(labels_of_all_clients)

    # global_data_dict = {
    #     'train': Molecular(train_data, train_smiles, S_train, train_adj_matrices, X_train, D_train, train_feature_matrices, Le_train, Lv_train),
    #     'val': Molecular(val_data, val_smiles, S_val, val_adj_matrices, X_val, D_val, val_feature_matrices, Le_val, Lv_val),
    #     'test': Molecular(test_data, test_smiles, S_test, test_adj_matrices, X_test, D_test, test_feature_matrices, Le_test, Lv_test)}

    # return global_data_dict, partition_dicts

    # def visualize_label_distribution_similarity_score(labels_of_all_clients):
    # label_distribution_clients = []
    # label_num = labels_of_all_clients[0][0]
    # for client_idx in range(len(labels_of_all_clients)):
    #     labels_client_i = labels_of_all_clients[client_idx]
    #     sample_number = len(labels_client_i)
    #     active_property_count = [0.0] * label_num
    #     for sample_index in range(sample_number):
    #         label = labels_client_i[sample_index]
    #         for property_index in range(len(label)):
    #             # logging.info(label[property_index])
    #             if label[property_index] == 1:
    #                 active_property_count[property_index] += 1
    #     active_property_count = [float(active_property_count[i]) for i in range(len(active_property_count))]
    #     label_distribution_clients.append(copy.deepcopy(active_property_count))
    # logging.info(label_distribution_clients)

    # client_num = len(label_distribution_clients)
    # label_distribution_similarity_score_matrix = np.random.random((client_num, client_num))

    # for client_i in range(client_num):
    #     label_distribution_client_i = label_distribution_clients[client_i]
    #     for client_j in range(client_i, client_num):
    #         label_distribution_client_j = label_distribution_clients[client_j]
    #         logging.info(label_distribution_client_i)
    #         logging.info(label_distribution_client_j)
    #         a = np.array(label_distribution_client_i, dtype=np.float32)
    #         b = np.array(label_distribution_client_j, dtype=np.float32)

    #         from scipy.spatial import distance
    #         distance = 1 - distance.cosine(a, b)
    #         label_distribution_similarity_score_matrix[client_i][client_j] = distance
    #         label_distribution_similarity_score_matrix[client_j][client_i] = distance
    #     # break
    # logging.info(label_distribution_similarity_score_matrix)
    # plt.title("Label Distribution Similarity Score")
    # ax = sns.heatmap(label_distribution_similarity_score_matrix, annot=True, fmt='.3f')
    # ax.invert_yaxis()
    # plt.show()

# For centralized training
# def get_dataloader(data_dir):

#     train_dataset = Molecular(train_data, train_smiles, S_train, train_adj_matrices, X_train, D_train, train_feature_matrices, Le_train, Lv_train)
#     vaL_dataset = Molecular(val_data, val_smiles, S_val, val_adj_matrices, X_val, D_val, val_feature_matrices, Le_val, Lv_val)
#     test_dataset = Molecular(test_data, test_smiles, S_test, test_adj_matrices, X_test, D_test, test_feature_matrices, Le_test, Lv_test)
    
#     collator = WalkForestCollator(normalize_features=normalize_features)

#     # IT IS VERY IMPORTANT THAT THE BATCH SIZE = 1. EACH BATCH IS AN ENTIRE MOLECULE.
#     train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator, pin_memory=True)
#     val_dataloader = data.DataLoader(vaL_dataset, batch_size=1, shuffle=False, collate_fn=collator, pin_memory=True)
#     test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator, pin_memory=True)

#     return train_dataloader, val_dataloader, test_dataloader

# # Single process sequential
# def load_partition_data(args, client_number, uniform=True, global_test=True):
#     global_data_dict, partition_dicts = partition_data_by_sample_size(args, client_number, uniform)

#     data_local_num_dict = dict()
#     train_data_local_dict = dict()
#     val_data_local_dict = dict()
#     test_data_local_dict = dict()

#     collator = WalkForestCollator(normalize_features=normalize_features) 

#     # IT IS VERY IMPORTANT THAT THE BATCH SIZE = 1. EACH BATCH IS AN ENTIRE MOLECULE.
#     train_data_global = data.DataLoader(global_data_dict['train'], batch_size=1, shuffle=True, collate_fn=collator,
#                                         pin_memory=True)
#     val_data_global = data.DataLoader(global_data_dict['val'], batch_size=1, shuffle=True, collate_fn=collator,
#                                       pin_memory=True)
#     test_data_global = data.DataLoader(global_data_dict['test'], batch_size=1, shuffle=True, collate_fn=collator,
#                                        pin_memory=True)

#     train_data_num = len(global_data_dict['train'])
#     val_data_num = len(global_data_dict['val'])
#     test_data_num = len(global_data_dict['test'])

    # for client in range(client_number):
    #     train_dataset_client = partition_dicts[client]['train']
    #     val_dataset_client = partition_dicts[client]['val']
    #     test_dataset_client = partition_dicts[client]['test']

    #     data_local_num_dict[client] = len(train_dataset_client)
    #     train_data_local_dict[client] = data.DataLoader(train_dataset_client, batch_size=1, shuffle=True,
    #                                                     collate_fn=collator, pin_memory=True)
    #     val_data_local_dict[client] = data.DataLoader(val_dataset_client, batch_size=1, shuffle=False,
    #                                                   collate_fn=collator, pin_memory=True)
    #     test_data_local_dict[client] = test_data_global if global_test else data.DataLoader(test_dataset_client,
    #                                                                                         batch_size=1, shuffle=False,
    #                                                                                         collate_fn=collator,
    #                                                                                         pin_memory=True)

    #     logging.info("Client idx = {}, local sample number = {}".format(client, len(train_dataset_client)))

    # return train_data_num, val_data_num, test_data_num, train_data_global, val_data_global, test_data_global, \
    #        data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict

# def load_partition_data_distributed(process_id, data_dir, client_number, uniform=True):
#     global_data_dict, partition_dicts = partition_data_by_sample_size(data_dir, client_number, uniform)
#     train_data_num = len(global_data_dict['train'])

#     collator = WalkForestCollator(normalize_features=True)

#     if process_id == 0:
#         train_data_global = data.DataLoader(global_data_dict['train'], batch_size=1, shuffle=True, collate_fn=collator,
#                                             pin_memory=True)
#         val_data_global = data.DataLoader(global_data_dict['val'], batch_size=1, shuffle=True, collate_fn=collator,
#                                           pin_memory=True)
#         test_data_global = data.DataLoader(global_data_dict['test'], batch_size=1, shuffle=True, collate_fn=collator,
#                                            pin_memory=True)

#         train_data_local = None
#         val_data_local = None
#         test_data_local = None
#         local_data_num = 0
#     else:
#         train_dataset_local = partition_dicts[process_id - 1]['train']
#         local_data_num = len(train_dataset_local)
#         train_data_local = data.DataLoader(train_dataset_local, batch_size=1, shuffle=True, collate_fn=collator,
#                                            pin_memory=True)
#         val_data_local = data.DataLoader(partition_dicts[process_id - 1]['val'], batch_size=1, shuffle=True,
#                                          collate_fn=collator, pin_memory=True)
#         test_data_local = data.DataLoader(partition_dicts[process_id - 1]['test'], batch_size=1, shuffle=True,
#                                           collate_fn=collator, pin_memory=True)
#         train_data_global = None
#         val_data_global = None
#         test_data_global = None

#     return train_data_num, train_data_global, val_data_global, test_data_global, local_data_num, \
#            train_data_local, val_data_local, test_data_local


    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len


if __name__ == '__main__':

    data = MolecularDataset()
    data.generate('/Users/daniel/Desktop/PhD materials/Fed-GNN-GAN/fedgan/data_smiles/clintox_smiles.pkl', validation=0.1, test=0.1)
    data.save('/Users/daniel/Desktop/PhD materials/Fed-GNN-GAN/fedgan/data_smiles/clintox.dataset')