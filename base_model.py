import torch
import torch.nn as nn
import numpy as np
from base_fn import kl_term, vade_trick, coherence_function
from info_prior_latent import InfoPriorCalculator
import logging

class GaussianSampling(nn.Module):
    """Gaussian reparameterization trick for sampling"""
    def forward(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return eps * std + mu


class GaussianPoE(nn.Module):
    """Product of Experts for Gaussian distributions"""
    def forward(self, mu, var, mask=None):
        mask_matrix = torch.stack(mask, dim=0)
        exist_mu = mu * mask_matrix
        precision = 1.0 / var
        exist_precision = precision * mask_matrix
        
        aggregate_precision = torch.sum(exist_precision, dim=0)
        aggregate_var = 1.0 / aggregate_precision
        aggregate_mu = torch.sum(exist_mu * exist_precision, dim=0) / aggregate_precision
        
        return aggregate_mu, aggregate_var


class ViewSpecificEncoder(nn.Module):
    """Encoder for view-specific representations"""
    def __init__(self, view_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(view_dim, 500), nn.ReLU(),
            nn.Linear(500, 500), nn.ReLU(),
            nn.Linear(500, 2000), nn.ReLU()
        )
        self.mu_layer = nn.Linear(2000, latent_dim)
        self.var_layer = nn.Sequential(nn.Linear(2000, latent_dim), nn.Softplus())

    def forward(self, x):
        hidden = self.encoder(x)
        return self.mu_layer(hidden), self.var_layer(hidden)


class ViewSpecificDecoder(nn.Module):
    """Decoder for view-specific reconstructions"""
    def __init__(self, view_dim, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2000), nn.ReLU(),
            nn.Linear(2000, 500), nn.ReLU(),
            nn.Linear(500, 500), nn.ReLU(),
            nn.Linear(500, view_dim)
        )

    def forward(self, z):
        return self.decoder(z)


def compute_wasserstein_distance(mu, var):
    """Compute pairwise Wasserstein-2 distances between Gaussian distributions"""
    # Expand dimensions for broadcasting
    mu_i = mu.unsqueeze(1)  # [batch, 1, dim]
    mu_j = mu.unsqueeze(0)  # [1, batch, dim]
    var_i = var.unsqueeze(1)
    var_j = var.unsqueeze(0)
    
    # Compute mean and variance differences
    mean_diff_sq = torch.sum((mu_i - mu_j)**2, dim=2)
    std_diff_sq = torch.sum((torch.sqrt(var_i) - torch.sqrt(var_j))**2, dim=2)
    
    # Normalize by latent dimension
    return (mean_diff_sq + std_diff_sq) / mu.shape[1]


def kl_divergence(mu1, var1, mu2, var2):
    """Compute KL divergence between two Gaussian distributions"""
    return 0.5 * torch.sum(
        torch.log(var2/var1) + (var1 + (mu1 - mu2)**2) / var2 - 1,
        dim=1
    )


class DVIMC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_dim_list = args.multiview_dims
        self.k = args.class_num
        self.z_dim = args.z_dim
        self.num_views = args.num_views
        self.device = args.device
        self.selection_ratio = args.selection_ratio
        self.gamma = getattr(args, 'gamma', 1.0)

        # Likelihood function
        self.likelihood_fn = (nn.BCEWithLogitsLoss(reduction='none') 
                            if args.likelihood == 'Bernoulli' 
                            else nn.MSELoss(reduction='none'))

        # Prior parameters
        self.prior_weight = nn.Parameter(torch.full((self.k,), 1/self.k))
        self.prior_mu = nn.Parameter(torch.zeros(self.k, self.z_dim))
        # self.prior_var = nn.Parameter(torch.ones(self.k, self.z_dim))
        self.prior_var_unconstrained = nn.Parameter(torch.zeros(self.k, self.z_dim))

        # View-specific encoders and decoders
        self.encoders = nn.ModuleDict({
            f'view_{v}': ViewSpecificEncoder(self.x_dim_list[v], self.z_dim) 
            for v in range(self.num_views)
        })
        self.decoders = nn.ModuleDict({
            f'view_{v}': ViewSpecificDecoder(self.x_dim_list[v], self.z_dim) 
            for v in range(self.num_views)
        })

        self.aggregator = GaussianPoE()
        self.sampler = GaussianSampling()
        
        # Global information matrix storage
        self.global_info_matrix = None
        self.correlation_matrix = None
        self.info_scores_stats = None
    
    @property
    def prior_var(self):
        return torch.nn.functional.softplus(self.prior_var_unconstrained) + 1e-6
    
    def mv_encode(self, x_list):
        latent_representation_list = []
        for v in range(self.num_views):
            latent_representation, _ = self.encoders[f'view_{v}'](x_list[v])
            latent_representation_list.append(latent_representation)
        return latent_representation_list

    def sv_encode(self, x, view_idx):
        latent_representation, _ = self.encoders[f'view_{view_idx}'](x)
        xr = self.decoders[f'view_{view_idx}'](latent_representation)
        return latent_representation, xr
    
    def compute_global_information_matrix(self, data, mask):
        """Compute information matrix for the entire dataset after initialization"""
        print("Computing global information matrix...")
        self.eval()
        with torch.no_grad():
            mask_matrix = torch.stack([m.squeeze() for m in mask], dim=1) #(n_samples, n_views)
            latent_representation_list = self.mv_encode(data)
           
            calculator = InfoPriorCalculator(
                latent_representation_list, mask=mask_matrix, device=self.device
            )
            # _, self.global_info_matrix = calculator.compute_information_matrix()
            self.correlation_matrix, self.global_info_matrix = calculator.compute_information_matrix()
            
           
            self._log_information_statistics(mask_matrix)
        self.train()   
        return self.global_info_matrix

    def _log_information_statistics(self, mask_matrix):
        """Log detailed statistics about information scores and correlation matrix"""
        logging.info("-" * 50)
        logging.info("Information Matrix Statistics:")
        
        info_scores_cpu = self.global_info_matrix.cpu().numpy()
        missing_mask_cpu = (1 - mask_matrix).cpu().numpy()
        
        view_stats = []
        for v in range(self.num_views):
            view_missing_indices = np.where(missing_mask_cpu[:, v] == 1)[0]
            if len(view_missing_indices) > 0:
                view_scores = info_scores_cpu[view_missing_indices, v]
                view_stats.append({
                    'view': v,
                    'missing_count': len(view_missing_indices),
                    'mean_score': np.mean(view_scores),
                    'std_score': np.std(view_scores),
                    'min_score': np.min(view_scores),
                    'max_score': np.max(view_scores)
                })
                
                logging.info(f"View {v+1}: Missing={len(view_missing_indices)}, "
                           f"Score Mean={np.mean(view_scores):.4f}±{np.std(view_scores):.4f}, "
                           f"Range=[{np.min(view_scores):.4f}, {np.max(view_scores):.4f}]")
        
        all_missing_scores = info_scores_cpu[missing_mask_cpu == 1]
        if len(all_missing_scores) > 0:
            logging.info(f"Global Missing: Total={len(all_missing_scores)}, "
                        f"Score Mean={np.mean(all_missing_scores):.4f}±{np.std(all_missing_scores):.4f}")
        
        if self.correlation_matrix is not None:
            corr_cpu = self.correlation_matrix.cpu().numpy()
            logging.info("-" * 50)
            logging.info("Inter-view Correlation Matrix:")
            
            for i in range(self.num_views):
                row_str = f"View {i+1}: "
                for j in range(self.num_views):
                    if i == j:
                        row_str += f"{'1.000':>6} "
                    else:
                        row_str += f"{corr_cpu[i,j]:>6.3f} "
                logging.info(row_str)
            
            off_diagonal_mask = ~np.eye(self.num_views, dtype=bool)
            off_diagonal_corrs = corr_cpu[off_diagonal_mask]

        total_missing = np.sum(missing_mask_cpu)
        if total_missing > 0:
            selection_count = int(total_missing * self.selection_ratio)
            logging.info("-" * 50)
            logging.info(f"Selective Imputation: {selection_count}/{total_missing} "
                        f"({self.selection_ratio*100:.1f}%) positions will be imputed")
        
        self.info_scores_stats = {
            'view_stats': view_stats,
            'global_missing_count': len(all_missing_scores) if len(all_missing_scores) > 0 else 0,
            'global_mean_score': np.mean(all_missing_scores) if len(all_missing_scores) > 0 else 0,
            'correlation_mean': np.mean(off_diagonal_corrs) if self.correlation_matrix is not None else 0
        }
        
        logging.info("=" * 50)

    def _get_batch_info_scores(self, batch_indices):
        """Get information scores for current batch from global matrix"""
        # Directly use batch_indices as they correspond to global dataset indices
        return self.global_info_matrix[batch_indices]
        
    def _compute_imputation_mask(self, mask, batch_info_scores):
        """Compute selective imputation mask using precomputed information scores"""
        missing_info = torch.stack([1 - m.squeeze(-1) for m in mask], dim=1)
        masked_info_scores = batch_info_scores * missing_info
        
        missing_indices = torch.nonzero(missing_info, as_tuple=False)
        if missing_indices.numel() == 0:
            return [m.clone() for m in mask], missing_info, set()
        
        sample_indices = missing_indices[:, 0]
        view_indices = missing_indices[:, 1]
        scores = masked_info_scores[sample_indices, view_indices]
        
        # top-k
        missing_count = len(scores)
        select_count = max(1, int(missing_count * self.selection_ratio))
        # print(f"Selecting {select_count} positions of {missing_count} missing entries")
        _, top_indices = torch.topk(scores, select_count)
        
        selected_positions = set()
        for idx in top_indices:
            sample_idx = sample_indices[idx].item()
            view_idx = view_indices[idx].item()
            selected_positions.add((sample_idx, view_idx))

        return [m.clone() for m in mask], missing_info, selected_positions

    def _impute_missing_data(self, vs_mus, vs_vars, x_list, mask, selected_positions, distance_matrix):
        """Perform KNN-based imputation for selected missing data"""
        missing_info = torch.stack([1 - m.squeeze() for m in mask], dim=1)
        imputed_mus = [mu.clone() for mu in vs_mus]
        imputed_vars = [var.clone() for var in vs_vars]
        imputed_x_list = [x.clone() for x in x_list]

        view_positions = {}
        for sample_idx, view_idx in selected_positions:
            if view_idx not in view_positions:
                view_positions[view_idx] = []
            view_positions[view_idx].append(sample_idx)

        actually_imputed_positions = set()

        for vi, sample_list in view_positions.items():
            if not sample_list:
                continue
                
            selected_indices = torch.tensor(sample_list, device=self.device)
            valid_indices = torch.nonzero(~missing_info[:, vi].bool()).squeeze(1)
            
            if len(valid_indices) == 0:
                continue
            
            distances = distance_matrix[selected_indices][:, valid_indices]
            k = min(10, len(valid_indices))
            if len(valid_indices) < 5:
                continue  
            topk_values, topk_indices = torch.topk(distances, k=k, dim=1, largest=False)
            
            weights = torch.exp(-topk_values/self.gamma)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
            
            global_nn_indices = valid_indices[topk_indices]
            weights_expanded = weights.unsqueeze(-1)
            
            neighbor_mus = vs_mus[vi][global_nn_indices]
            neighbor_vars = vs_vars[vi][global_nn_indices]
            neighbor_inputs = x_list[vi][global_nn_indices]
            
            imputed_mus[vi][selected_indices] = torch.sum(neighbor_mus * weights_expanded, dim=1)
            max_var, _ = torch.max(neighbor_vars, dim=1)
            if neighbor_mus.shape[1] > 1:
                mu_variance = torch.var(neighbor_mus, dim=1)
            else:
                mu_variance = torch.zeros_like(neighbor_mus[:, 0])

            # imputed_vars[vi][selected_indices] = max_var + mu_variance
            imputed_vars[vi][selected_indices] = torch.sum(neighbor_vars * weights_expanded, dim=1) + mu_variance
            imputed_x_list[vi][selected_indices] = torch.sum(neighbor_inputs * weights_expanded, dim=1)

            for sample_idx in sample_list:
                actually_imputed_positions.add((sample_idx, vi))
        return imputed_mus, imputed_vars, imputed_x_list, actually_imputed_positions

    def _create_imputed_mask(self, mask, imputed_positions):
        """Create imputed mask indicating which positions have been imputed"""
        imputed_mask = [m.clone() for m in mask]
        
        # Add imputed positions to the mask
        for sample_idx, view_idx in imputed_positions:
            imputed_mask[view_idx][sample_idx] = 1.0
            
        return imputed_mask

    def _compute_losses(self, vs_mus, vs_vars, aggregated_mu, aggregated_var, 
                       imputed_mus, imputed_vars, x_list, xr_list, mask):
        """Compute all loss components - optimized version"""
        
        # 标准VAE损失
        vade_z_sample = self.sampler(aggregated_mu, aggregated_var)
        qc_x = vade_trick(vade_z_sample, self.prior_weight, self.prior_mu, self.prior_var)
        z_loss, c_loss = kl_term(aggregated_mu, aggregated_var, qc_x, 
                                self.prior_weight, self.prior_mu, self.prior_var)
        
        # 重构损失 - 向量化
        rec_losses = [
            torch.mean(torch.sum(self.likelihood_fn(xr_list[v], x_list[v]), dim=1) * mask[v].squeeze())
            for v in range(self.num_views)
        ]
        rec_loss = sum(rec_losses)
        
        # 一致性损失
        coherence_loss = coherence_function(imputed_mus, imputed_vars, aggregated_mu, aggregated_var, mask)
        
        return rec_loss, z_loss + c_loss, coherence_loss

    def encode_views(self, x_list, mask):
        """Encode all views and aggregate latent representations"""
        vs_mus, vs_vars = [], []
        for v, x in enumerate(x_list):
            mu, var = self.encoders[f'view_{v}'](x)
            vs_mus.append(mu)
            vs_vars.append(var)
        
        mu_stack = torch.stack(vs_mus)
        var_stack = torch.stack(vs_vars)
        aggregated_mu, aggregated_var = self.aggregator(mu_stack, var_stack, mask)
        
        return vs_mus, vs_vars, aggregated_mu, aggregated_var

    def decode_latent(self, z):
        """Decode latent representation to all views"""
        return [decoder(z) for decoder in self.decoders.values()]

    def forward(self, x_list, mask=None, batch_indices=None):
        # Initial encoding
        vs_mus, vs_vars, aggregated_mu, aggregated_var = self.encode_views(x_list, mask)
        
        # Get precomputed information scores for this batch
        if batch_indices is not None:
            batch_info_scores = self._get_batch_info_scores(batch_indices)
        else:
            batch_info_scores = self.global_info_matrix
        
        # Selective imputation using precomputed scores
        _, missing_info, selected_positions = self._compute_imputation_mask(mask, batch_info_scores)
        
        # Only compute distance matrix for imputation if needed
        if len(selected_positions) > 0:
            distance_matrix = compute_wasserstein_distance(aggregated_mu, aggregated_var)
            imputed_mus, imputed_vars, imputed_x_list, imputed_positions = self._impute_missing_data(
                vs_mus, vs_vars, x_list, mask, selected_positions, distance_matrix
            )
        else:
            imputed_mus, imputed_vars, imputed_x_list = vs_mus, vs_vars, x_list
            imputed_positions = set() 
    
        
        # Create imputed mask before re-inference
        imputed_mask = self._create_imputed_mask(mask, imputed_positions)
        
        # Re-inference with imputed data
        mu_stack = torch.stack(imputed_mus)
        var_stack = torch.stack(imputed_vars)
        aggregated_mu, aggregated_var = self.aggregator(mu_stack, var_stack, imputed_mask)
        
        z_sample = self.sampler(aggregated_mu, aggregated_var)
        decoded_x_list = self.decode_latent(z_sample)
        
        # Encode imputed data for consistency loss
        encoded_vs_mus, encoded_vs_vars, _, _ = self.encode_views(imputed_x_list, mask)
        xr_list = decoded_x_list
        
        # Compute all losses
        rec_loss, kl_loss, coherence_loss = self._compute_losses(
            vs_mus, vs_vars, aggregated_mu, aggregated_var,
            imputed_mus, imputed_vars, x_list, xr_list, mask
        )
        
        return aggregated_mu, rec_loss, kl_loss, coherence_loss
