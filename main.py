import copy
import torch
from torch import optim
from torch.utils.data import DataLoader
from datasets import build_dataset
import torch.nn as nn
from base_model import DVIMC
from evaluate import evaluate
from base_fn import vade_trick
import numpy as np
import random
import argparse
from sklearn.cluster import KMeans

import logging
import sys
import os
from datetime import datetime

def setup_logger(args):
    log_dir = "./log"
    dataset_log_dir = os.path.join(log_dir, args.dataset_name)
    os.makedirs(dataset_log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"mr{args.missing_rate}_sr{args.selection_ratio}_seed{args.seed}_{timestamp}.log"
    log_path = os.path.join(dataset_log_dir, log_filename)
    
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(level=logging.INFO, handlers=[])
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return log_path

def log_experiment_info(args, test_times=None):
    logging.info("="*80)
    logging.info("EXPERIMENT CONFIGURATION")
    logging.info("="*80)
    logging.info(f"Dataset          : {args.dataset_name}")
    logging.info(f"Missing Rate     : {args.missing_rate}")
    logging.info(f"Selection Ratio  : {args.selection_ratio}")
    logging.info(f"Number of Views  : {args.num_views}")
    logging.info(f"Number of Classes: {args.class_num}")
    logging.info(f"Data Size        : {args.data_size}")
    logging.info(f"View Dimensions  : {args.multiview_dims}")
    logging.info("-"*40)
    logging.info(f"Epochs           : {args.epochs}")
    logging.info(f"Initial Epochs   : {args.initial_epochs}")
    logging.info(f"Batch Size       : {args.batch_size}")
    logging.info(f"Learning Rate    : {args.learning_rate}")
    logging.info(f"Prior LR         : {args.prior_learning_rate}")
    logging.info(f"Latent Dim       : {args.z_dim}")
    logging.info(f"Alpha            : {args.alpha}")
    logging.info(f"Seed             : {args.seed}")
    logging.info(f"Device           : {args.device}")
    logging.info(f"Likelihood       : {args.likelihood}")
    if test_times:
        logging.info(f"Test Times       : {test_times}")
    logging.info("="*80)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def initialization(model, sv_loaders, cmv_data, args):
    logging.info('Initializing......')
    criterion = nn.MSELoss()
    for v in range(args.num_views):
        optimizer = optim.Adam([{"params": model.encoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                {"params": model.decoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                ])
        for e in range(1, args.initial_epochs + 1):
            for batch_idx, xv in enumerate(sv_loaders[v]):
                optimizer.zero_grad()
                batch_size = xv.shape[0]
                xv = xv.reshape(batch_size, -1).to(args.device)
                _, xvr = model.sv_encode(xv, v)
                view_rec_loss = criterion(xvr, xv)
                view_rec_loss.backward()
                optimizer.step()
    with torch.no_grad():
        initial_data = [torch.tensor(csv_data, dtype=torch.float32).to(args.device) for csv_data in cmv_data]
        latent_representation_list = model.mv_encode(initial_data)
        assert len(latent_representation_list) == args.num_views
        fused_latent_representations = sum(latent_representation_list) / len(latent_representation_list)
        fused_latent_representations = fused_latent_representations.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=args.class_num, n_init=10)
        kmeans.fit(fused_latent_representations)
        model.prior_mu.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(args.device)


def train(model, optimizer, scheduler, imv_loader, args):
    logging.info('Training......')
    eval_data = copy.deepcopy(imv_loader.dataset.data_list)
    eval_mask = copy.deepcopy(imv_loader.dataset.mask_list)
    for v in range(args.num_views):
        eval_data[v] = torch.tensor(eval_data[v], dtype=torch.float32).to(args.device)
        eval_mask[v] = torch.tensor(eval_mask[v], dtype=torch.float32).to(args.device)
    eval_labels = imv_loader.dataset.labels
    # Compute global information matrix
    model.compute_global_information_matrix(eval_data, eval_mask)

    best_acc = 0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        epoch_loss = []
        epoch_rec_loss = []
        epoch_kl_loss = []
        epoch_coherence_loss = []
        for batch_idx, (batch_data, batch_mask, orig_indices) in enumerate(imv_loader):
            optimizer.zero_grad()
            batch_data = [sv_d.to(args.device) for sv_d in batch_data]
            batch_mask = [sv_m.to(args.device) for sv_m in batch_mask]
            
            # Create batch indices for global information matrix lookup
            batch_indices = orig_indices.to(args.device)
            
            aggregated_mu, rec_loss, kl_loss, coherence_loss = model(
                batch_data, batch_mask, batch_indices
            )

            batch_loss = rec_loss + kl_loss + args.alpha * coherence_loss
            epoch_loss.append(batch_loss.item())
            epoch_rec_loss.append(rec_loss.item())
            epoch_kl_loss.append(kl_loss.item())
            epoch_coherence_loss.append(coherence_loss.item())
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            # fix the prior weight
            model.prior_weight.data.clamp_(min=1e-6)
            model.prior_weight.data = model.prior_weight.data / model.prior_weight.data.sum()
        
        scheduler.step()
        overall_loss = sum(epoch_loss) / len(epoch_loss)
        avg_rec_loss = sum(epoch_rec_loss) / len(epoch_rec_loss)
        avg_kl_loss = sum(epoch_kl_loss) / len(epoch_kl_loss)
        avg_coherence_loss = sum(epoch_coherence_loss) / len(epoch_coherence_loss)
        
        if epoch % args.interval == 0 or epoch == args.epochs:
            with torch.no_grad():
                aggregated_mu, _, _, _ = model(eval_data, eval_mask)

                mog_weight = model.prior_weight.data.detach().cpu()
                mog_mu = model.prior_mu.data.detach().cpu()
                mog_var = model.prior_var.data.detach().cpu()
                aggregated_mu = aggregated_mu.detach().cpu()

                c_assignment = vade_trick(aggregated_mu, mog_weight, mog_mu, mog_var)
                predict = torch.argmax(c_assignment, dim=1).numpy()
                acc, nmi, ari, pur = evaluate(eval_labels, predict)
                
                logging.info(f'Train Loss:{overall_loss:.2f} Rec:{avg_rec_loss:.2f}  KL:{avg_kl_loss:.2f}  '
                             f'Coh:{avg_coherence_loss:.2f}')
                
                logging.info(f'Epoch {epoch:>3}/{args.epochs}  Loss:{overall_loss:.2f}  ACC:{acc * 100:.2f}  '
                      f'NMI:{nmi * 100:.2f}  ARI:{ari * 100:.2f}  PUR:{pur * 100:.2f}')
    logging.info('Finish training')
    return acc, nmi, ari, pur


def main(args):
    logging.info('Loading dataset information...')
    np.random.seed(1)
    random.seed(1)
    cmv_data, imv_dataset, sv_datasets = build_dataset(args)
    log_experiment_info(args, args.test_times)
    
    for t in range(1, args.test_times + 1):
        logging.info(f'Test Run {t}/{args.test_times}')
        logging.info("-"*60)
        if t > 1:
            np.random.seed(t)
            random.seed(t)
            cmv_data, imv_dataset, sv_datasets = build_dataset(args)
        setup_seed(args.seed)
        imv_loader = DataLoader(imv_dataset, batch_size=args.batch_size, shuffle=True)
        sv_loaders = [DataLoader(sv_dataset, batch_size=args.batch_size, shuffle=True) for sv_dataset in sv_datasets]
        model = DVIMC(args).to(args.device)

        optimizer = optim.Adam(
            [{"params": model.encoders.parameters(), 'lr': args.learning_rate},
             {"params": model.decoders.parameters(), 'lr': args.learning_rate},
             {"params": model.prior_weight, 'lr': args.prior_learning_rate},
             {"params": model.prior_mu, 'lr': args.prior_learning_rate},
             {"params": model.prior_var_unconstrained, 'lr': args.prior_learning_rate},
             ])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
        initialization(model, sv_loaders, cmv_data, args)
        acc, nmi, ari, pur = train(model, optimizer, scheduler, imv_loader, args)
        test_record["ACC"].append(acc)
        test_record["NMI"].append(nmi)
        test_record["ARI"].append(ari)
        test_record["PUR"].append(pur)
    logging.info('Average ACC {:.2f} std {:.2f} Average NMI {:.2f} std {:.2f} Average ARI {:.2f} std {:.2f} Average PUR {:.2f} std {:.2f}'.format(
        np.mean(test_record["ACC"]) * 100, np.std(test_record["ACC"]) * 100,
        np.mean(test_record["NMI"]) * 100, np.std(test_record["NMI"]) * 100,
        np.mean(test_record["ARI"]) * 100, np.std(test_record["ARI"]) * 100,
        np.mean(test_record["PUR"]) * 100, np.std(test_record["PUR"]) * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--initial_epochs', type=int, default=200, help='initialization epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--prior_learning_rate', type=float, default=0.05, help='initial mixture-of-gaussian learning rate')
    parser.add_argument('--z_dim', type=int, default=10, help='latent dimensions')
    parser.add_argument('--lr_decay_step', type=float, default=10, help='StepLr_Step_size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9, help='StepLr_Gamma')

    parser.add_argument('--dataset', type=int, default=1, choices=range(4), help='0:Caltech7-5v, 1:Scene-15, 2:Multi-Fashion, 3:bbcsport')
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--test_times', type=int, default=5)
    parser.add_argument('--missing_rate', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=5)
    parser.add_argument('--selection_ratio', type=float, default=0.5, help='ratio of selected samples in each view')
    parser.add_argument('--sr_exp', action='store_true', help='whether to run selection ratio experiment')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataset_dir_base = "./npz_data/"

    if args.dataset == 0:
        args.dataset_name = 'Caltech7-5V'
        args.epochs = 200
        args.alpha = 5
        args.seed = 5
        args.likelihood = 'Gaussian'
    elif args.dataset == 1:
        args.dataset_name = 'Scene-15'
        args.alpha = 20
        args.seed = 19
        args.likelihood = 'Gaussian'
    elif args.dataset == 2:
        args.dataset_name = 'Multi-Fashion'
        args.alpha = 10
        args.seed = 12
        args.likelihood = 'Bernoulli'
    elif args.dataset == 3:
        args.dataset_name = 'COIL100'
        args.batch_size = 512
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'


    log_path = setup_logger(args)
    test_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
    main(args)
    logging.info(f"Log saved to: {log_path}")
    
    # for missing_rate in [0.5, 0.4, 0.3, 0.2, 0.1]:
    #     args.missing_rate = missing_rate
    #     args.selection_ratio = 1 - args.missing_rate
    #     logging.info(f"Dataset : {args.dataset_name:<15} Missing rate : {args.missing_rate}")
    #     logging.info(f"Alpha : {args.alpha}, Seed : {args.seed}")
    #     logging.info(f"Selection_ratio : {args.selection_ratio}, Batch_size : {args.batch_size}")
    #     test_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
    #     main(args)