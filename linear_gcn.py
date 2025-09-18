mport os
import numpy as np
import h5py
from rdkit import Chem
from rdkit.Chem import rdchem
from ase import Atoms
from dscribe.descriptors import SOAP

# 假设NPZ包含以下键；根据实际文件内容调整
data = np.load('DFT_uniques.npz', allow_pickle=True)
smiles_list = data['graphs']            # 分子SMILES列表
Z_list = data['atoms']                      # 原子序数列表（每项是一个数组，包括氢）
coords_list = data['coordinates']                 # 原子坐标列表（每项形如 N原子×3 的坐标数组）
energies = data['Etot']                 # DFT总能量列表

num_mols = len(smiles_list)
print(f"数据集包含 {num_mols} 个分子")

# 2. 确定数据集中出现的原子元素种类，以构建分子式向量
unique_atomic_numbers = sorted({int(z) for atoms in Z_list for z in atoms})
# 建立 原子序号 -> 向量索引 的映射
elem_to_idx = {z: i for i, z in enumerate(unique_atomic_numbers)}
formula_vec_length = len(unique_atomic_numbers)
print(f"元素种类数: {formula_vec_length}, 列表: {unique_atomic_numbers}")


# 提取数据集中的所有元素种类（用于初始化SOAP描述符）
all_atomic_numbers = set()
for Z in Z_list:
    # 将每个分子的原子序数加入集合
    all_atomic_numbers.update(Z)
species = sorted(all_atomic_numbers)  # 排序后的原子种类列表

# 设置SOAP描述符参数（默认典型值，可根据需要调整）
soap = SOAP(species=species, 
            periodic=False, 
            r_cut=6.0,       # 截断半径6Å
            n_max=4,         # 径向基函数最大主量子数
            l_max=4)         # 角度量子数最大值

soap_cache_file = "soap_features_cache.h5"
num_mols = len(Z_list)

if not os.path.exists(soap_cache_file):
    print("Computing SOAP features and caching to", soap_cache_file)
    with h5py.File(soap_cache_file, 'w') as f:
        # 遍历每个分子，逐个计算SOAP描述符后写入HDF5文件
        for i, Z in enumerate(Z_list):
            coords = coords_list[i]
            atoms = Atoms(numbers=Z, positions=coords)
            # 计算该分子的SOAP描述符；输出形状为 (num_atoms, soap_feature_length)
            soap_matrix = soap.create(atoms)
            # 使用分子索引作为数据集名称，将SOAP矩阵保存到文件中，并采用gzip压缩
            f.create_dataset(str(i), data=soap_matrix, compression="gzip")
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} / {num_mols} molecules")
else:
    print("SOAP cache found. Skipping SOAP calculation.")

# 定义获取边特征的辅助函数
from math import dist
def get_bond_features(rdk_mol, coords):
    """
    提取RDKit分子对象中的键特征：
    返回列表，其中每个元素是(edge_index_i, edge_index_j, feature_vector)
    """
    bond_features = []
    for bond in rdk_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 键类型（单键、双键、三键、芳香键）
        bt = bond.GetBondType()
        if bt == rdchem.BondType.SINGLE:
            bond_type = [1, 0, 0, 0]
        elif bt == rdchem.BondType.DOUBLE:
            bond_type = [0, 1, 0, 0]
        elif bt == rdchem.BondType.TRIPLE:
            bond_type = [0, 0, 1, 0]
        elif bt == rdchem.BondType.AROMATIC:
            bond_type = [0, 0, 0, 1]
        else:
            # 其他类型（极少出现），先用零向量表示
            bond_type = [0, 0, 0, 0]
        # 是否共轭键
        conjugated = [1] if bond.GetIsConjugated() else [0]
        # 是否在环中
        in_ring = [1] if bond.IsInRing() else [0]
        # 键长：根据3D坐标计算原子距离
        length = [dist(coords[i], coords[j])]
        # 汇总边特征向量
        feat = bond_type + conjugated + in_ring + length  # 总长度7
        # 无向图：对每条键添加两个方向的边(i->j, j->i)
        bond_features.append((i, j, feat))
        bond_features.append((j, i, feat))
    return bond_features

# 构建PyG的数据列表
import torch
from torch_geometric.data import Data

data_list = []
with h5py.File(soap_cache_file, 'r') as f:
    for i, smiles in enumerate(smiles_list):
        # 使用RDKit解析分子，保证原子顺序与Z_list一致
        mol = Chem.MolFromSmiles(smiles)
        # 添加显式氢（使解析后的分子与Z_list匹配）
        mol = Chem.AddHs(mol)
        coords = coords_list[i]
        # 从HDF5缓存中加载第i个分子的SOAP描述符
        soap_matrix = f[str(i)][()]  # soap_matrix shape: (num_atoms, soap_feature_dim)
        # 获取边及其特征
        bond_info = get_bond_features(mol, coords)
        if len(bond_info) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 7), dtype=torch.float)
        else:
            edge_index = torch.tensor([[i_idx, j_idx] for (i_idx, j_idx, feat) in bond_info],
                                      dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([feat for (_, _, feat) in bond_info], dtype=torch.float)
            
        # 分子式向量（各元素数量），作为图级属性存储（形状 [1, n_elems]）
        formula_counts = [0] * formula_vec_length
        for Z in Z_list[i]:
            if Z in elem_to_idx:
                formula_counts[elem_to_idx[Z]] += 1
        formula_vec = torch.tensor([formula_counts], dtype=torch.float)
        # 节点特征：SOAP描述符
        x = torch.tensor(soap_matrix, dtype=torch.float)
        # 图标签：分子总能量
        y = torch.tensor([energies[i]], dtype=torch.float)
        # 构建PyG Data对象
        data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, formula=formula_vec, y=y)
        data_list.append(data_obj)

print(f"Successfully built {len(data_list)} PyG graph objects.")
# 划分训练集、验证集、测试集 (例如 8:1:1 比例)

np.random.seed(42)
perm = np.random.permutation(len(data_list))
N_train = int(0.8 * len(data_list))
N_val = int(0.1 * len(data_list))
train_idx = perm[:N_train]
val_idx = perm[N_train:N_train+N_val]
test_idx = perm[N_train+N_val:]
train_set = [data_list[i] for i in train_idx]
val_set = [data_list[i] for i in val_idx]
test_set = [data_list[i] for i in test_idx]
print(f"训练集: {len(train_set)} 个, 验证集: {len(val_set)} 个, 测试集: {len(test_set)} 个")

import torch.nn as nn

class LinearFormulaModel(nn.Module):
    def __init__(self, in_dim):
        super(LinearFormulaModel, self).__init__()
        # 线性层: 输入 -> 输出标量
        self.fc = nn.Linear(in_dim, 1, bias=False)
    
    def forward(self, formula_vec):
        # formula_vec: 张量形状 [batch_size, in_dim]
        # 输出层（直接输出一个标量，无激活）
        out = self.fc(formula_vec)
        # 输出形状 [batch_size, 1]，将其压缩为 [batch_size] 方便计算误差
        return out.view(-1)
    
# 实例化线性模型，输入维度为分子式向量长度
linear_model = LinearFormulaModel(in_dim=formula_vec_length)
print(linear_model)

import torch.optim as optim
# 将模型移动到GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
linear_model.to(device)

# 设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(linear_model.parameters(), lr=1e-2)

# 准备线性模型的训练数据加载器（使用TensorDataset）
# 提取训练集的分子式向量和能量
train_formula = torch.cat([data.formula for data in train_set], dim=0)  # [N_train, formula_vec_length]
train_targets = torch.cat([data.y for data in train_set], dim=0)        # [N_train]
val_formula = torch.cat([data.formula for data in val_set], dim=0)
val_targets = torch.cat([data.y for data in val_set], dim=0)

batch_size = 128  # 批大小
train_dataset = torch.utils.data.TensorDataset(train_formula, train_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(val_formula, val_targets)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 训练线性模型
best_val_loss = float('inf')
patience = 25  # 容忍验证集无提升的轮数
stagnant_epochs = 0

for epoch in range(10000):  # 最大训练轮数上限，如1000
    linear_model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        # 前向传播
        pred = linear_model(batch_x)
        loss = criterion(pred, batch_y)
        # 反向传播与参数更新
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)
    
    # 验证集评估
    linear_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = linear_model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}: 训练集Loss = {avg_train_loss:.4f}, 验证集Loss = {avg_val_loss:.4f}")
    
    # 检查验证集误差改善
    if avg_val_loss < best_val_loss - 1e-5:  # 判断相对改善（此处设定阈值1e-4）
        best_val_loss = avg_val_loss
        stagnant_epochs = 0
        # 保存当前最佳模型参数
        torch.save(linear_model.state_dict(), 'linear_model_best_kernel_soap.pth')
    else:
        stagnant_epochs += 1
        # 若验证Loss连续多轮未提升，则提前停止训练
        if stagnant_epochs >= patience:
            print("验证集误差多次未提升，提前停止线性模型训练")
            break
        
# 加载最佳模型参数（确保使用最佳验证性能的模型）
linear_model.load_state_dict(torch.load('linear_model_best_kernel_soap.pth'))
linear_model.eval()

import torch.nn as nn
import torch.nn.functional as F

class KernelMapper(nn.Module):
    def __init__(self, in_dim, out_dim=None,
                 kernel_type='rbf', gamma=1.0, normalize=True,
                 init_centers=None):
        super().__init__()
        self.kernel_type = kernel_type.lower()
        self.normalize = normalize
        self.out_dim = out_dim or in_dim

        # γ ➜ log 参数，softplus 保证正
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)))

        if self.kernel_type in ['rbf', 'gaussian', 'laplacian']:
            if init_centers is None:
                self.centers = nn.Parameter(torch.randn(self.out_dim, in_dim))
            else:
                self.register_buffer('centers', init_centers)  # 固定不训练
        else:
            self.linear = nn.Linear(in_dim, self.out_dim)

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        if self.kernel_type in ['rbf', 'gaussian', 'laplacian']:
            gamma = F.softplus(self.log_gamma)     # 正数
            if self.kernel_type == 'laplacian':
                dist = (x.unsqueeze(1) - self.centers).abs().sum(-1)
                phi = torch.exp(-gamma * dist)
            else:
                dist2 = (x.pow(2).sum(-1, keepdim=True)
                         + self.centers.pow(2).sum(-1)
                         - 2 * x @ self.centers.t())
                phi = torch.exp(-gamma * dist2)
            return phi
        else:
            return self.linear(x)

from torch_geometric.nn import NNConv, global_mean_pool

class EnergyGCN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim,
                 hidden_dim=128, map_type='rbf', map_out_dim=None,
                 gamma=1.0, normalize=True, num_layers=3,
                 init_centers=None, residual=True):
        super().__init__()
        self.kmap = KernelMapper(node_feat_dim, map_out_dim or node_feat_dim,
                                 map_type, gamma, normalize, init_centers)
        mapped = self.kmap.out_dim
        self.residual = residual
        gcn_in = mapped + (node_feat_dim if residual else 0)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(NNConv(gcn_in, hidden_dim,
                                 nn.Sequential(nn.Linear(edge_feat_dim, gcn_in*hidden_dim)),
                                 aggr='mean'))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers-1):
            self.convs.append(NNConv(hidden_dim, hidden_dim,
                                     nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim*hidden_dim)),
                                     aggr='mean'))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.pool = global_mean_pool
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, data):
        x_raw = data.x
        x_k   = self.kmap(x_raw)
        x = torch.cat([x_raw, x_k], dim=-1) if self.residual else x_k

        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index, data.edge_attr)))

        g = self.pool(x, data.batch)
        return self.fc_layers(g).view(-1)

class TotalEnergyModel(nn.Module):
    def __init__(self, linear_model, gcn_model):
        super(TotalEnergyModel, self).__init__()
        self.linear_model = linear_model
        self.gcn_model = gcn_model
        # 冻结线性模型的参数，使其在训练中不更新
        for param in self.linear_model.parameters():
            param.requires_grad = False
    
    def forward(self, data):
        # 从批数据中提取分子式向量 (形状 [batch_size, formula_vec_length])
        formula_vec = data.formula.squeeze(1)  # 原本 shape [batch_size, 1, vec_len]，去掉中间维度
        # 线性模型预测
        linear_out = self.linear_model(formula_vec)
        # 把data里的formula_vec删除，避免传递给GCN
        data.formula = None
        # GCN模型预测
        gcn_out = self.gcn_model(data)
        # 二者相加得到总能量预测
        total_out = linear_out + gcn_out
        return total_out

# 初始化模型和优化器
node_feat_dim = data_list[0].x.shape[1]   # SOAP特征维度
edge_feat_dim = data_list[0].edge_attr.shape[1] if data_list[0].edge_attr is not None and data_list[0].edge_attr.numel() > 0 else 0
from sklearn.cluster import MiniBatchKMeans
k = 256           # 你想要的中心个数
soap_dim = train_set[0].x.size(1)          # 例如 384

# -------- 方法 1：一次性拼接（数据量不大时最简单） --------
all_soap = np.concatenate(
    [d.x.cpu().numpy() for d in train_set],   # 注意转到 CPU
    axis=0                                        # 按行堆叠
)                                                # shape (总原子数, soap_dim)

kmeans = MiniBatchKMeans(n_clusters=k,
                         batch_size=10_000,
                         random_state=0)
kmeans.fit(all_soap)                              # 完成聚类
centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
# 初始化模型和优化器
node_feat_dim = data_list[0].x.shape[1]   # SOAP特征维度
edge_feat_dim = data_list[0].edge_attr.shape[1] if data_list[0].edge_attr is not None and data_list[0].edge_attr.numel() > 0 else 0
gcn_model = EnergyGCN(node_feat_dim=node_feat_dim,
                  edge_feat_dim=edge_feat_dim,
                  hidden_dim=128,
                  map_type='rbf',
                  map_out_dim=k,
                  gamma=1.0,            # 初始 gamma 会被学习调整
                  normalize=True,
                  init_centers=centers,
                  residual=True)
# 构造组合模型实例
combined_model = TotalEnergyModel(linear_model, gcn_model)
print(combined_model)

from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
# 导入DistributedDataParallel
# from torch.nn.parallel import DistributedDataParallel

# 将组合模型移至GPU(主卡)，并封装DataParallel以利用多GPU
combined_model = TotalEnergyModel(linear_model, gcn_model)  # 线性模型已训练好且冻结
device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
combined_model.to(device0)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # 使用两张GPU进行并行
    device_ids = [0, 1]
    combined_model = DataParallel(combined_model, device_ids=device_ids)
    print(f"使用多GPU: {device_ids} 进行训练")
else:
    print("使用单GPU进行训练")

# 准备图数据的 DataLoader
batch_size = 64  # 每批包含的分子数量
if isinstance(combined_model, DataParallel):
    # DataParallel 模式下需使用 DataListLoader，它返回数据列表
    train_loader = DataListLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataListLoader(val_set, batch_size=batch_size, shuffle=False)
else:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器（这里只优化GCN部分参数，线性部分已冻结）
criterion = nn.MSELoss()
# 注意：combined_model 若为DataParallel，需要使用 combined_model.module 获取内部模型
gcn_parameters = combined_model.module.gcn_model.parameters() if isinstance(combined_model, DataParallel) else combined_model.gcn_model.parameters()
optimizer = optim.Adam(gcn_parameters, lr=1e-3)

# 训练组合模型 (主要训练GCN部分)
best_val_loss = float('inf')
patience = 100
stagnant_epochs = 0

for epoch in range(10000):
    combined_model.train()
    total_loss = 0.0
    # 遍历训练集批次
    for batch_data in train_loader:
        optimizer.zero_grad()
        # 将数据搬移到主设备
        if isinstance(combined_model, DataParallel):
            # DataParallel 接受的数据是列表，会自动分发到各GPU
            # 无需手动 .to(device)，DataParallel 内部处理
            out = combined_model(batch_data)  # 列表形式batch由DataParallel处理
            # 提取batch中每个图的y并拼接（DataParallel输出与输入list顺序对应）
            batch_targets = torch.cat([data.y for data in batch_data], dim=0).to(device0)
        else:
            batch_data = batch_data.to(device0)
            out = combined_model(batch_data)
            batch_targets = batch_data.y  # shape [batch_size]
        loss = criterion(out, batch_targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (batch_targets.size(0))
    avg_train_loss = total_loss / len(train_set)
    
    # 验证评估
    combined_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_data in val_loader:
            if isinstance(combined_model, DataParallel):
                out = combined_model(batch_data)
                batch_targets = torch.cat([data.y for data in batch_data], dim=0).to(device0)
            else:
                batch_data = batch_data.to(device0)
                out = combined_model(batch_data)
                batch_targets = batch_data.y
            loss = criterion(out, batch_targets.view(-1))
            val_loss += loss.item() * batch_targets.size(0)
    avg_val_loss = val_loss / len(val_set)
    print(f"Epoch {epoch+1}: 训练Loss = {avg_train_loss:.6f}, 验证Loss = {avg_val_loss:.6f}")
    
    # 检查验证集误差改善
    if avg_val_loss < best_val_loss - 1e-7:
        best_val_loss = avg_val_loss
        stagnant_epochs = 0
        # 保存当前最佳组合模型参数（仅需保存GCN部分，因为线性部分固定）
        if isinstance(combined_model, DataParallel):
            torch.save(combined_model.module.gcn_model.state_dict(), 'gcn_model_best_kernel_soap.pth')
        else:
            torch.save(combined_model.gcn_model.state_dict(), 'gcn_model_best_kernel_soap.pth')
    else:
        stagnant_epochs += 1
        if stagnant_epochs >= patience:
            print("验证集误差无显著改善，提前停止组合模型训练")
            break
    
# 加载最佳GCN模型参数到组合模型中
if isinstance(combined_model, DataParallel):
    combined_model.module.gcn_model.load_state_dict(torch.load('gcn_model_best_kernel_soap.pth'))
else:
    combined_model.gcn_model.load_state_dict(torch.load('gcn_model_best_kernel_soap.pth'))
combined_model.eval()

# 测试集评估
test_loader = DataListLoader(test_set, batch_size=batch_size, shuffle=False)
combined_model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_data in test_loader:
        pred = combined_model(batch_data)
        targets = torch.cat([data.y for data in batch_data], dim=0).to(device0)
        loss = criterion(pred, targets.view(-1))
        test_loss += loss.item() * targets.size(0)
avg_test_loss = test_loss / len(test_set)
print(f"测试集上平均MSE = {avg_test_loss:.6f}")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 计算测试集的MAE、RMSE和R²
predictions = []
targets_list = []
with torch.no_grad():
    for batch_data in test_loader:
        pred = combined_model(batch_data)
        targets = torch.cat([data.y for data in batch_data], dim=0).to(device0)
        targets_list.append(targets.cpu().numpy())
        predictions.append(pred.cpu().numpy())
targets = np.concatenate(targets_list)
predictions = np.concatenate(predictions)
mae = mean_absolute_error(targets, predictions)
rmse = np.sqrt(mean_squared_error(targets, predictions))
r2 = r2_score(targets, predictions)
print(f"测试集上平均MAE = {mae:.6f}")
print(f"测试集上平均RMSE = {rmse:.6f}")
print(f"测试集上平均R² = {r2:.6f}")
# 偏差最大的值和偏差的方差
deviation = targets - predictions
max_deviation = np.max(np.abs(deviation))
print(f"测试集上偏差最大的值 = {max_deviation:.6f}")
# 计算平均绝对误差的标准差
mae_std = np.std(np.abs(deviation))
print(f"测试集上平均绝对误差的标准差 = {mae_std:.6f}")
