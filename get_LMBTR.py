import os
import numpy as np
import h5py
# from rdkit import Chem
# from rdkit.Chem import rdchem
from ase import Atoms
from dscribe.descriptors import LMBTR

# 假设NPZ包含以下键；根据实际文件内容调整
data = np.load('DFT_uniques.npz', allow_pickle=True)
# data = np.load('DMC.npz', allow_pickle=True)
# data = np.load('DFT_all.npz', allow_pickle=True)
# smiles_list = data['graphs']            # 分子SMILES列表
Z_list = data['atoms']                      # 原子序数列表（每项是一个数组，包括氢）
coords_list = data['coordinates']                 # 原子坐标列表（每项形如 N原子×3 的坐标数组）
# energies = data['Etot']                 # DFT总能量列表

# 提取数据集中的所有元素种类（用于初始化SOAP描述符）
all_atomic_numbers = set()
for Z in Z_list:
    # 将每个分子的原子序数加入集合
    all_atomic_numbers.update(Z)
species = sorted(all_atomic_numbers)  # 排序后的原子种类列表

lmbtr = LMBTR(
    species=species,
    geometry={"function": "distance"},
    grid={"min": 0, "max": 5, "n": 100, "sigma": 0.1},
    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
    periodic=False,
    normalization="l2",
)


soap_cache_file = "lmbtr_features_cache.h5"
num_mols = len(Z_list)

if not os.path.exists(soap_cache_file):
    print("Computing SOAP features and caching to", soap_cache_file)
    with h5py.File(soap_cache_file, 'w') as f:
        # 遍历每个分子，逐个计算SOAP描述符后写入HDF5文件
        for i, Z in enumerate(Z_list):
            coords = coords_list[i]
            atoms = Atoms(numbers=Z, positions=coords)
            # 计算该分子的SOAP描述符；输出形状为 (num_atoms, soap_feature_length)
            soap_matrix = lmbtr.create(atoms, n_jobs=-1)
            # 使用分子索引作为数据集名称，将SOAP矩阵保存到文件中，并采用gzip压缩
            f.create_dataset(str(i), data=soap_matrix, compression="gzip")
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} / {num_mols} molecules")
else:
    print("SOAP cache found. Skipping SOAP calculation.")
