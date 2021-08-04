import os
import argparse
from trainer import Trainer
from Dataloader import get_loader
from torch.backends import cudnn
from tqdm import tqdm
from molecular_dataset import MolecularDataset
import numpy as np
import copy
from utils import average_weights
import torch 
import random
import pickle

# PLOTTING LOSS
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def str2bool(v):
    return v.lower() in ('true')

def main(args):

    # set random seed
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
        # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        device = torch.device('cuda:{}'.format(args.cuda_num) if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    # Since graph input sizes remains constant
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    data = MolecularDataset()
    data.load(args.mol_data_dir)
    idxs  = len(str(data))

    # trainer for training and testing StarGAN.
    train_data, test_data, user_groups = get_loader(args)
    # train_data, test_data, user_groups = train_data.to(device), test_data.to(device), user_groups.to(device)
    trainer = Trainer(args, data, idxs)
    
    g_global_model = trainer.g_global_model
    d_global_model = trainer.d_global_model

    g_global_model
    d_global_model
    
    # copy weights
    g_global_weights = g_global_model.state_dict()
    d_global_weights = d_global_model.state_dict()


    g_train_loss, d_train_loss = [], []
    if args.mode == 'train':
        for i in tqdm(range(args.epochs_global)):
            g_local_weights, g_local_losses, d_local_weights, d_local_losses = [], [], [], []
            print(f'\n | Global Training Round : {i+1} |\n')
            # print('\n | Global Training Round: {} |\n'.format(i+1))

            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                local_model = Trainer(args=args, data=train_data, idxs=user_groups[idx])  
                g_weights, d_weights, g_loss, d_loss = local_model.tnr(model=copy.deepcopy(g_global_model), global_round=i)
            
                g_local_weights.append(copy.deepcopy(g_weights))
                g_local_losses.append(copy.deepcopy(g_loss))

                d_local_weights.append(copy.deepcopy(d_weights))
                d_local_losses.append(copy.deepcopy(d_loss))

            # average local weights
            g_global_weights = average_weights(g_local_weights)
            d_global_weights = average_weights(d_local_weights)

            # update global weights
            g_global_model.load_state_dict(g_global_weights)
            d_global_model.load_state_dict(d_global_weights)

            g_loss_avg = sum(g_local_losses) / len(g_local_losses)
            d_loss_avg = sum(d_local_losses) / len(d_local_losses)

            g_train_loss.append(g_loss_avg)
            d_train_loss.append(d_loss_avg)

        # Saving the objects train_loss and train_accuracy:
        file_name = '/home/wej20003/fedgan/objects/{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(data, args.epochs_global, args.frac, args.data_iid,
                args.num_iters_local, args.batch_size)

        with open(file_name, 'wb') as f:
            pickle.dump([g_train_loss, d_train_loss], f)

        # Plot Generator Loss curve
        plt.figure()
        plt.title('Generator Training Loss vs Communication rounds')
        plt.plot(range(len(g_train_loss)), g_train_loss, color='r')
        plt.ylabel('Generator Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig('/home/wej20003/fedgan/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                    format(data, args.g_conv_dim, args.epochs_global, args.frac,
                           args.data_iid, args.num_iters_local, args.batch_size))

        # Plot Discriminator Loss curve
        plt.figure()
        plt.title('Discriminator Training Loss vs Communication rounds')
        plt.plot(range(len(d_train_loss)), d_train_loss, color='b')
        plt.ylabel('Discriminator Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig('/home/wej20003/fedgan/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                    format(data, args.d_conv_dim, args.epochs_global, args.frac,
                           args.data_iid, args.num_iters_local, args.batch_size))
            
    elif args.mode == 'test':
        trainer.test()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=16, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[32, 64, 128], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[64, 128], 64, [128, 1]], help='number of conv filters in the first layer of D') #[128, 64], 128, [128, 64]
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size') #16
    parser.add_argument('--num_iters_local', type=int, default=20000, help='number of total iterations for training D') #200000
    parser.add_argument('--num_iters_decay', type=int, default=10000, help='number of iterations for decaying lr') #100000
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--epochs_global', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=1, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=20000, help='test model from this step') #200000

    # Miscellaneous.
    parser.add_argument('--cuda_num', type=int, default=5, help="GPU number")
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--data_iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    # parser.add_argument('--data_noniid', type=int, default=0, help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')

    # Directories.
    parser.add_argument('--mol_data_dir', type=str, default='/home/daniel/Desktop/fedgan/data_smiles/esol.dataset')
    parser.add_argument('--log_dir', type=str, default='/home/daniel/Desktop/fedgan/logs')
    parser.add_argument('--model_save_dir', type=str, default='/home/daniel/Desktop/fedgan/models')
    parser.add_argument('--sample_dir', type=str, default='/home/daniel/Desktop/fedgan/samples')
    parser.add_argument('--result_dir', type=str, default='/home/daniel/Desktop/fedgan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10) #10
    parser.add_argument('--sample_step', type=int, default=1000)  #1000
    parser.add_argument('--model_save_step', type=int, default=5000) #10000
    parser.add_argument('--lr_update_step', type=int, default=1000)  #1000

    args = parser.parse_args()
    print(args)
    main(args)
