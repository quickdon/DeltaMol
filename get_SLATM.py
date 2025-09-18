import os
import numpy as np
import h5py
# from rdkit import Chem
# from rdkit.Chem import rdchem
# from ase import Atoms
from qmllib.representations import generate_slatm, get_slatm_mbtypes

# 假设NPZ包含以下键；根据实际文件内容调整
data = np.load('DFT_uniques.npz', allow_pickle=True)
# data = np.load('DMC.npz', allow_pickle=True)
# data = np.load('DFT_all.npz', allow_pickle=True)
# smiles_list = data['graphs']            # 分子SMILES列表
Z_list = data['atoms']                      # 原子序数列表（每项是一个数组，包括氢）
coords_list = data['coordinates']                 # 原子坐标列表（每项形如 N原子×3 的坐标数组）
# energies = data['Etot']                 # DFT总能量列表

mbtypes = get_slatm_mbtypes(Z_list)

soap_cache_file = "slatm_features_cache.h5"
num_mols = len(Z_list)

if not os.path.exists(soap_cache_file):
    print("Computing SOAP features and caching to", soap_cache_file)
    with h5py.File(soap_cache_file, 'w') as f:
        # 遍历每个分子，逐个计算SOAP描述符后写入HDF5文件
        for i, Z in enumerate(Z_list):
            coords = coords_list[i]
            # 计算该分子的SOAP描述符；输出形状为 (num_atoms, soap_feature_length)
            slatm = generate_slatm(Z, coords, mbtypes, dgrids=[0.5,0.5], rcut=4.0, local=True)
            # 将SLATM list转换为numpy.array
            slatm = np.array(slatm)
            # 使用分子索引作为数据集名称，将SOAP矩阵保存到文件中，并采用gzip压缩
            f.create_dataset(str(i), data=slatm, compression="gzip")
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} / {num_mols} molecules")
else:
    print("SOAP cache found. Skipping SOAP calculation.")
