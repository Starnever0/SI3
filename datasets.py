import math
import random
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat


class SingleviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)


class Incomplete_MultiviewDataset(Dataset):
    def __init__(self, data_list, mask_matrix, labels, num_views):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels
        self.mask_list = np.split(mask_matrix, num_views, axis=1)

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32) for v in range(self.num_views)]
        mask = [torch.tensor(self.mask_list[v][index], dtype=torch.float32, requires_grad=False) for v in range(self.num_views)]
        return data, mask, index


def get_mask(view_num, alldata_len, missing_rate, view_weights=None):

    if view_weights is None:
        if view_num == 5:
            view_weights = np.array([1, 1, 2, 4, 8])
        elif view_num == 3:
            view_weights = np.array([1, 3, 9])
        view_weights = view_weights / np.sum(view_weights)
    else:
        view_weights = np.array(view_weights)
        view_weights = view_weights / np.sum(view_weights)
    
    one_rate = 1-missing_rate
    matrix = np.zeros((alldata_len, view_num), dtype=np.int_)
    
    for i in range(alldata_len):
        preserved_view = np.random.choice(range(view_num), p=view_weights)
        matrix[i, preserved_view] = 1
    target_ones = int(view_num * alldata_len * one_rate)
    current_ones = np.sum(matrix)
    remaining_ones = target_ones - current_ones
    if remaining_ones > 0:
        zero_positions = np.where(matrix == 0)
        zero_coords = list(zip(zero_positions[0], zero_positions[1]))
        position_weights = np.array([view_weights[coord[1]] for coord in zero_coords])
        position_weights = position_weights / np.sum(position_weights)
        select_indices = np.random.choice(
            range(len(zero_coords)), 
            size=min(remaining_ones, len(zero_coords)),
            replace=False,
            p=position_weights
        )
        
        for idx in select_indices:
            i, j = zero_coords[idx]
            matrix[i, j] = 1
    return matrix


def load_multiview_data(args):
    data_path = args.dataset_dir_base + args.dataset_name + '.npz'
    data = np.load(data_path)
    num_views = int(data['n_views'])
    data_list = [data[f'view_{v}'].astype(np.float32) for v in range(num_views)]
    labels = data['labels']
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = num_views
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels

# def load_mat(args):
#     # 1*n cell
#     data_path = args.dataset_dir_base + args.dataset_name + '.mat'
#     data = loadmat(data_path)
#     X = data['X']
#     num_views = X.shape[1]
#     data_list = []
#     for v in range(num_views):
#         view_data = X[0, v]
#         data_list.append(view_data.astype(np.float32))
#     labels = data['Y'].flatten()
#     dims = [dv.shape[1] for dv in data_list]
#     data_size = labels.shape[0]
#     class_num = len(np.unique(labels))
#     if np.max(labels) == class_num:
#         labels = labels - 1
#     args.multiview_dims = dims
#     args.num_views = num_views
#     args.class_num = class_num
#     args.data_size = data_size
#     print(f"Loaded {args.dataset_name}: {data_size} samples, {class_num} classes, {num_views} views")
#     print(f"View dimensions: {dims}")   
#     return data_list, labels

def pixel_normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def build_dataset(args):
    if args.dataset_name in ['Caltech7-5V', 'Scene-15', 'Multi-Fashion']:
        data_list, labels = load_multiview_data(args)
    # else:
    #     data_list, labels = load_mat(args)

    if args.dataset_name == 'Caltech7-5V':
        data_list = [pixel_normalize(dv) for dv in data_list]
    elif args.dataset_name == 'Multi-Fashion':
        pass
    else:
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]

    view_weights = args.view_weights if hasattr(args, 'view_weights') else None
    mask = get_mask(args.num_views, args.data_size, args.missing_rate, )
    data_list = [data_list[v] * mask[:, v:v + 1] for v in range(args.num_views)]
    incomplete_multiview_dataset = Incomplete_MultiviewDataset(data_list, mask, labels, args.num_views)

    com_idx = np.sum(mask, axis=1) == args.num_views
    complete_multiview_data = [sv_d[com_idx] for sv_d in data_list]

    exist_singleview_datasets = [SingleviewDataset(data_list[v][mask[:, v] == 1]) for v in range(args.num_views)]

    return complete_multiview_data, incomplete_multiview_dataset, exist_singleview_datasets

