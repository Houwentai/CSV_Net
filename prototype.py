from tqdm import tqdm
import numpy as np
import torch
import time
import os
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import faiss
import numpy as np
import joblib

import csv

def read_first_column_as_list(csv_file_path):
    first_column = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:  # Ensure the row is not empty
                first_column.append(row[0]+".h5")
    return first_column

def cluster(all_tile_features, n_proto, n_iter, n_init=5, feature_dim=1024, n_proto_patches=50000, mode='faiss', use_cuda=False):
    """
    K-Means clustering on embedding space with single GPU
    """
    patches = torch.from_numpy(all_tile_features).to(device)
    n_patches = patches.shape[0]
    print(f"\nTotal of {n_patches} patches aggregated")

    s = time.time()
    if mode == 'faiss':
        assert use_cuda, f"FAISS requires access to GPU. Please enable use_cuda"
        
        numOfGPUs = torch.cuda.device_count()
        print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...")
        print(f"\tNum of clusters {n_proto}, num of iter {n_iter}")

        kmeans = faiss.Kmeans(patches.shape[1], 
                              n_proto, 
                              niter=n_iter, 
                              nredo=n_init,
                              verbose=True, 
                              max_points_per_centroid=n_proto_patches,
                              gpu=numOfGPUs)
        kmeans.train(patches.cpu().numpy())  # Move data to CPU for Faiss
        weight = torch.tensor(kmeans.centroids).to(device)  # Move centroids to GPU

    else:
        raise NotImplementedError(f"Clustering not implemented for {mode}!")

    e = time.time()
    print(f"\nClustering took {e-s} seconds!")

    return n_patches, weight

# 计算惯性（Inertia）和 Silhouette Score 的 GPU 加速版本（分批处理）
def compute_inertia(features, centroids, batch_size=10000):
    # 将数据按批次分割
    n_samples = features.shape[0]
    inertia = 0

    for i in range(0, n_samples, batch_size):
        batch = features[i:i+batch_size]
        
        # 计算每个批次样本到最近簇中心的距离 (Inertia)
        distances = torch.cdist(batch, centroids)  # 使用GPU上的cdist
        inertia_batch = torch.min(distances, dim=1)[0].sum()  # 对每个样本找到最小距离并求和
        inertia += inertia_batch.item()

    return inertia


if __name__ == "__main__":
    # 设置CUDA环境
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用示例
    csv_file_path = '../patient_list.csv'
    patient_list = read_first_column_as_list(csv_file_path)
    patient_list = first_column_list[1:]
    print(patient_list)  
  
    # 加载所有患者的patch特征
    for names in tqdm(len(patient_list)):
        all_tile_features.append(h5py.File(os.path.join(h5dir, names), 'r')["features"][:])
        all_tile_coords.append(h5py.File(os.path.join(h5dir, names), 'r')["coords"][:])
        for i in range(all_tile_features[-1].shape[0]):
            all_tile_names.append(names[:-3])
    all_tile_features = np.vstack(all_tile_features)
    all_tile_coords = np.vstack(all_tile_coords)
    
    # 存储结果的列表
    weights = []
    n_proto = 16
    print("n_proto = {}".format(n_proto))
    
    # 执行聚类
    _, weight = cluster(all_tile_features, n_proto=n_proto, n_iter=50, n_init=5, feature_dim=1024, n_proto_patches=5000000, mode='faiss', use_cuda=True)

    # 存储权重
    weights.append(weight)
    joblib.dump(weights,"../weights_16.pkl")


