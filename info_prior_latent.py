import torch
import torch.nn.functional as F


class InfoPriorCalculator:
    
    def __init__(self, data_list, mask, distance_matrix=None, view_distance_matrix=None, device='cpu'):
        """
        Optimized version: Information matrix calculator based on missing patterns
        
        Args:
            data_list: list, multi-view data, each element is a feature matrix of a view
            mask: tensor, missing mask matrix, shape is (n_samples, n_views),
                  value 1 indicates observed, 0 indicates missing
            device: str, device type
        """
        self.device = device
        self.data_list = [torch.tensor(data, device=device, dtype=torch.float32) if not isinstance(data, torch.Tensor) 
                          else data.to(device) for data in data_list]
        if not isinstance(mask, torch.Tensor):
            self.mask = torch.tensor(mask, device=device, dtype=torch.bool)
        else:
            self.mask = mask.to(device).bool()  # Explicitly convert to boolean type
    
        n_samples, n_views = self.mask.shape
        self.n_samples = n_samples
        self.n_views = n_views
        
        # Cache structure initialization
        self.mode_templates = {}  # Cache mode templates
        self.mode_key_to_samples = {}  # Mapping from mode to samples
        
        # Pre-compute similarity and correlation matrices
        if distance_matrix is not None and view_distance_matrix is not None:
            self.distance_matrix = torch.tensor(distance_matrix, device=device, dtype=torch.float32)
            self.view_distance_matrix = torch.tensor(view_distance_matrix, device=device, dtype=torch.float32)
            self.sample_similarity = self.compute_with_distance()
        else:  
            self.sample_similarity = self.compute_sample_similarity()
        self.view_corr = self.compute_view_correlation()
    

    def compute_information_matrix(self):
        """
        Information matrix computation: Enhanced version with sample similarity and view weights
        """
        # === Step 1: Parse missing patterns and build mode template index ===
        missing_tasks, unique_modes = self._parse_and_prepare_templates()
        
        # === Step 2: Build and cache information matrix templates ===
        self._build_mode_info_templates(unique_modes)
        
        # === Step 3: Assign weighted information matrices for all missing data ===
        info_dict = self._assign_info_matrices(missing_tasks)
        # Create a matrix of total information scores
        total_info_matrix = torch.zeros((self.n_samples, self.n_views), device=self.device)
        for (sample_idx, view_idx), info in info_dict.items():
            total_info_matrix[sample_idx, view_idx] = info['total_info_score']
        
        return self.view_corr, total_info_matrix
    
    def _parse_and_prepare_templates(self):
        """
        Step 1: Parse missing patterns and build mode template index
        
        Returns:
            missing_tasks: list of tuples, [(sample_idx, view_idx, mode_key), ...]
            unique_modes: set, all unique mode_keys
        """
        missing_tasks = []
        unique_modes = set()

        # Vectorized generation of pattern strings
        mask_int = self.mask.int()
        pattern_strings = [''.join(row.cpu().numpy().astype(str)) for row in mask_int]

        sample_indices, view_indices = torch.where(~self.mask)
        
        # Batch build mode keys and tasks
        for sample_idx, view_idx in zip(sample_indices.cpu().numpy(), view_indices.cpu().numpy()):
            pattern_str = pattern_strings[sample_idx]
            mode_key = f"{pattern_str}-{view_idx}"
            missing_tasks.append((sample_idx, view_idx, mode_key))
            unique_modes.add(mode_key)
            
            # Find support samples for the mode (if not computed yet)
            if mode_key not in self.mode_key_to_samples:
                support_samples = self._find_support_samples(mode_key)
                self.mode_key_to_samples[mode_key] = support_samples
        
        return missing_tasks, unique_modes
    
    def _find_support_samples(self, mode_key):
        """
        Find support samples for a given imputation mode
        """
        # Parse mode_key
        pattern_str, target_view = self._parse_mode_key(mode_key)
        
        # Convert pattern_str to bool tensor
        required_pattern = torch.tensor([c == '1' for c in pattern_str], device=self.device, dtype=torch.bool)
        
        # Condition 1: Samples with observations on target_view
        has_target_view = self.mask[:, target_view]
        
        # Condition 2: Compatible with required_pattern on other views
        other_views = torch.arange(self.n_views, device=self.device) != target_view
        common_views_count = (self.mask[:, other_views] & required_pattern[other_views]).sum(dim=1)
        has_common_views = common_views_count >= 1
        
        support_indices = torch.where(has_target_view & has_common_views)[0]
        
        return support_indices
    
    def _build_mode_info_templates(self, unique_modes):
        """
        Step 2: Build and cache information matrix templates
        """
        for mode_key in unique_modes:     
            pattern_str, target_view = self._parse_mode_key(mode_key)
            
            support_samples = self.mode_key_to_samples[mode_key]
            if len(support_samples) == 0:
                self.mode_templates[mode_key] = torch.zeros((1, self.n_views), device=self.device)
                continue

            pattern = torch.tensor([c == '1' for c in pattern_str], device=self.device, dtype=torch.bool)
            support_mask = self.mask[support_samples].clone()  #
            support_mask[:, ~pattern] = False  

            self.mode_templates[mode_key] = support_mask.float()
    
    def _assign_info_matrices(self, missing_tasks):
        """
        Step 3: Assign information matrices for all missing data
        """
        info_dict = {}
        
        for sample_idx, target_view, mode_key in missing_tasks:
            template = self.mode_templates[mode_key]  # (n_support_samples, n_views)
            support_samples = self.mode_key_to_samples[mode_key] 
            
            if len(support_samples) == 0:
                info_dict[(sample_idx, target_view)] = {
                    'pattern': mode_key,
                    'info_matrix': torch.zeros((1, self.n_views), device=self.device),
                    'intra_info': torch.tensor(0.0, device=self.device),
                    'inter_info': torch.tensor(0.0, device=self.device),
                    'total_info_score': torch.tensor(0.0, device=self.device)
                }
                continue
            
            # Step 1: Initialize sample weight matrix
            specific_weights = torch.zeros_like(template, device=self.device)  # shape: (n_support_samples, n_views)

            for v in range(self.n_views):
                specific_weights[:, v] = self.sample_similarity[v][sample_idx, support_samples]
            specific_weights[:, target_view] = 1.0  
            info_matrix = template * specific_weights
            
            # Step 2: Apply view correlation weights
            view_weights = self.view_corr[target_view, :].clone()  # shape: (n_views,)
            view_weights = view_weights.unsqueeze(0)  # shape: (1, n_views)
            
            info_matrix = info_matrix * view_weights
            
            mask_other_views = torch.arange(self.n_views, device=self.device) != target_view
            target_sample_weights = torch.sum(info_matrix[:, mask_other_views], dim=1)  
            
            base = torch.sum(
                self.view_corr[mask_other_views, target_view].unsqueeze(0) * 
                template[:, mask_other_views], dim=1
            )
            base = torch.clamp(base, min=1e-8)
            target_sample_weights = target_sample_weights / base
            info_matrix[:, target_view] = target_sample_weights
            
            # Step 3: Calculate two parts and total information score
            intra_info = torch.sum(info_matrix[:, target_view]) 
            inter_info = torch.sum(info_matrix) - intra_info 
            total_info_score = torch.sum(info_matrix)
            
            info_dict[(sample_idx, target_view)] = {
                'pattern': mode_key,
                'info_matrix': info_matrix,
                'intra_info': intra_info,
                'inter_info': inter_info,
                'total_info_score': total_info_score,
                'support_samples': support_samples
            }
        
        return info_dict
    
    def _parse_mode_key(self, mode_key):
        parts = mode_key.split('-')
        pattern = parts[0]
        target_view = int(parts[1])
        return pattern, target_view

    def compute_sample_similarity(self):
        """
        Compute sample similarity matrix
        """
        observed_masks = [self.mask[:, v] for v in range(self.n_views)]
        global_sample_weights = []
        
        for v in range(self.n_views):
            observed_mask = observed_masks[v]
            observed_indices = torch.where(observed_mask)[0]
            
            if len(observed_indices) == 0:
                global_sample_weights.append(torch.zeros((self.n_samples, self.n_samples), device=self.device))
                continue
            
            observed_data = self.data_list[v][observed_indices]
            
            distances = torch.cdist(observed_data, observed_data, p=2)
            
            max_distance = torch.max(distances)
            max_distance = max_distance if max_distance > 0 else 1.0
            base_similarity = 1.0 - (distances / max_distance)
            sample_similarity = torch.pow(base_similarity, 2)
            
            global_weight = torch.zeros((self.n_samples, self.n_samples), device=self.device)
            rows, cols = torch.meshgrid(observed_indices, observed_indices, indexing='ij')
            global_weight[rows, cols] = sample_similarity
        
            global_sample_weights.append(global_weight)
        
        return global_sample_weights
    
    def compute_with_distance(self):
        """
        Compute similarity using pre-calculated distance matrix
        """
        global_sample_weights = []
        
        for v in range(self.n_views):
            distances = self.distance_matrix[v]
            max_distance = torch.max(distances)
            max_distance = max_distance if max_distance > 0 else 1.0
            base_similarity = 1.0 - (distances / max_distance)
            sample_similarity = torch.pow(base_similarity, 2)
            global_sample_weights.append(sample_similarity)
        
        return global_sample_weights
    
    def _compute_dcca_correlation(self, view1, view2, reg_param=1e-4):
        if view1.dim() == 1:
            view1 = view1.unsqueeze(0)
        if view2.dim() == 1:
            view2 = view2.unsqueeze(0)

        n_samples = view1.shape[0]
        
        view1_centered = view1 - view1.mean(dim=0, keepdim=True)
        view2_centered = view2 - view2.mean(dim=0, keepdim=True)

        C11 = view1_centered.T @ view1_centered / (n_samples - 1)
        C22 = view2_centered.T @ view2_centered / (n_samples - 1)
        C12 = view1_centered.T @ view2_centered / (n_samples - 1)

        eye1 = torch.eye(C11.size(0), device=view1.device, dtype=view1.dtype)
        eye2 = torch.eye(C22.size(0), device=view2.device, dtype=view2.dtype)
        C11 += reg_param * eye1
        C22 += reg_param * eye2

        try:
            L11 = torch.linalg.cholesky(C11)
            L22 = torch.linalg.cholesky(C22)

            T = torch.linalg.solve(L11, C12)
            T = torch.linalg.solve(L22, T.T).T

            U, S, Vh = torch.linalg.svd(T)
            k = min(3, S.numel())
            if k == 0:
                return 0.0
            
            weights = torch.exp(-torch.arange(k, device=S.device).float())
            weights = weights / weights.sum()
            
            weighted_corr = torch.sum(S[:k] * weights)
            return weighted_corr.item()
            # # Return the sum of correlation coefficients (or the maximum one)
            # return S[0].item() if S.numel() > 0 else 0.0

        except RuntimeError as e:
            print(f"[DCCA Warning] CCA correlation failed: {e}")
            return self._compute_simple_correlation(view1_centered, view2_centered)


    def _compute_simple_correlation(self, view1_centered, view2_centered):
        """
        Compute simple correlation as fallback
        """
        # Compute average correlation
        view1_flat = view1_centered.flatten()
        view2_flat = view2_centered.flatten()
        
        if len(view1_flat) != len(view2_flat):
            min_len = min(len(view1_flat), len(view2_flat))
            view1_flat = view1_flat[:min_len]
            view2_flat = view2_flat[:min_len]
        
        corr = torch.corrcoef(torch.stack([view1_flat, view2_flat]))[0, 1]
        return torch.clamp(corr, 0, 1).item() if not torch.isnan(corr) else 0.0

    def compute_view_correlation(self):
        """
        Compute inter-view correlation matrix
        """
        # Initialize correlation matrix
        view_corr = torch.eye(self.n_views, device=self.device)
        
        for i in range(self.n_views):
            for j in range(i + 1, self.n_views):
                common_mask = self.mask[:, i] & self.mask[:, j]
                common_indices = torch.where(common_mask)[0]
                
                if len(common_indices) < 5:
                    corr = 0.0
                else:
                    data_i = self.data_list[i][common_indices]
                    data_j = self.data_list[j][common_indices]
                    corr = self._compute_dcca_correlation(data_i, data_j)
                view_corr[i, j] = corr
                view_corr[j, i] = corr
        print("view_corr:", view_corr)
        
        return view_corr
