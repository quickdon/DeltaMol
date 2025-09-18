import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py

class MoleculeDataset(Dataset):
    def __init__(self, data_path, soap_cache_file):
        # 加载预计算的NPZ文件（需要allow_pickle=True以读取变长数组）
        data = np.load(data_path, allow_pickle=True)
        # 提取原子序号列表、坐标列表、SOAP描述符列表和能量目标值
        self.atom_list = data['atoms'].tolist()         # 每项是一个numpy数组，包含该分子的所有原子序号
        self.coord_list = data['coordinates'].tolist()  # 每项是一个 (N_i, 3) 数组，对应该分子各原子的坐标
        soap_list = []
        with h5py.File(soap_cache_file, 'r') as f:
            for i in range(len(self.atom_list)):
                # 从HDF5文件中读取每个分子的SOAP描述符
                soap_list.append(f[str(i)][()])
        self.soap_list = soap_list          # 每项是一个 (N_i, soap_dim) 数组，对应各原子的SOAP描述符
        self.targets = data['Etot'].astype(np.float32)  # 能量值数组 (形状为[n_samples,])
        # 记录SOAP描述符的维度和数据集中原子序号的最大值，供模型构建使用
        self.soap_dim = self.soap_list[0].shape[1] if len(self.soap_list) > 0 else 0
        all_atoms = np.concatenate(self.atom_list, axis=0)  # 将所有原子序号汇总
        self.num_types = int(all_atoms.max()) + 1

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # 获取第 idx 个分子的原子序号、坐标、SOAP和能量
        atoms = torch.tensor(self.atom_list[idx], dtype=torch.long)
        coords = torch.tensor(self.coord_list[idx], dtype=torch.float32)
        soap = torch.tensor(self.soap_list[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return atoms, coords, soap, target

def collate_fn(batch):
    # 自定义collate，将batch内各分子的原子数据padding到统一长度
    # batch是一个列表，元素为__getitem__返回的 (atoms, coords, soap, target)
    batch_size = len(batch)
    # 找到本batch中原子数量的最大值
    max_n_atoms = max(sample[0].shape[0] for sample in batch)
    # 初始化张量：原子序号、坐标、SOAP描述符的padding矩阵，以及掩码mask
    atom_tensor = torch.zeros((batch_size, max_n_atoms), dtype=torch.long)
    coord_tensor = torch.zeros((batch_size, max_n_atoms, 3), dtype=torch.float32)
    soap_tensor = torch.zeros((batch_size, max_n_atoms, batch[0][2].shape[1]), dtype=torch.float32)
    mask_tensor = torch.zeros((batch_size, max_n_atoms), dtype=torch.bool)  # mask_tensor[i,j]=True表示第i个样本的第j个原子为真实原子
    target_list = []

    # 将每个分子的原子信息复制到相应张量中
    for i, (atoms, coords, soap, target) in enumerate(batch):
        n = atoms.shape[0]  # 该分子原子数
        atom_tensor[i, :n] = atoms  # 填充原子序号
        coord_tensor[i, :n, :] = coords
        soap_tensor[i, :n, :] = soap
        mask_tensor[i, :n] = True   # 前n个位置为真实原子
        target_list.append(target)
    # 将目标能量列表合并为tensor
    target_tensor = torch.stack(target_list, dim=0)
    return atom_tensor, coord_tensor, soap_tensor, mask_tensor, target_tensor

# 加载数据集并划分训练/验证集
dataset = MoleculeDataset('DFT_uniques.npz', "soap_features_cache.h5")
# 这里将数据按8:2拆分为训练集和验证集（可以根据需要调整比例或使用预定义划分）
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# 创建DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 模型组件定义 =====
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
    def forward(self, x, adj):
        h = self.linear(x)
        deg = adj.sum(-1)
        deg_inv_sqrt = torch.where(deg>0, deg.pow(-0.5), torch.zeros_like(deg))
        h_norm = h * deg_inv_sqrt.unsqueeze(-1)
        agg = torch.bmm(adj, h_norm) * deg_inv_sqrt.unsqueeze(-1)
        out = self.dropout(agg)
        res = self.res_proj(x) if self.res_proj else x
        out = F.relu(self.norm(out + res))
        return out

class TransformerEncoderLayerPreNorm(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout)
        )
    def forward(self, x, key_padding_mask=None):
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask)
        x = x + self.drop1(attn_out)
        y2 = self.norm2(x)
        x = x + self.ff(y2)
        return x

class GNNTransformer(nn.Module):
    def __init__(self, soap_dim, num_types, hidden_dim=128,
                 gcn_layers=2, tr_layers=2, n_heads=4, ff_dim=256, dropout=0.1, cutoff=5.0):
        super().__init__()
        self.embed = nn.Embedding(num_types, 16)
        self.gcn_layers = nn.ModuleList()
        in_d = soap_dim + 16
        for _ in range(gcn_layers):
            self.gcn_layers.append(GCNLayer(in_d, hidden_dim, dropout)); in_d = hidden_dim
        self.tr_layers = nn.ModuleList([
            TransformerEncoderLayerPreNorm(hidden_dim, n_heads, ff_dim, dropout)
            for _ in range(tr_layers)
        ])
        self.attn_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.Tanh(), nn.Linear(hidden_dim//2,1))
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,1))
        self.cutoff = cutoff
    def forward(self, atoms, coords, soap, mask):
        B,N = atoms.shape
        atom_e = self.embed(atoms)
        x = torch.cat([soap, atom_e], dim=-1)
        # 构建邻接矩阵
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)
        adj = (dist<=self.cutoff).float()
        mask_f = mask.float(); mat = mask_f.unsqueeze(2)*mask_f.unsqueeze(1)
        adj = adj*mat
        idx = torch.arange(N, device=adj.device)
        adj[:,idx,idx] = mask_f
        # GCN
        for g in self.gcn_layers: x = g(x, adj)
        # Transformer
        pad_mask = ~mask
        for tr in self.tr_layers: x = tr(x, key_padding_mask=pad_mask)
        # Attention Pooling
        scores = self.attn_pool(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        mol = torch.sum(weights.unsqueeze(-1)*x, dim=1)
        return self.mlp(mol).squeeze(-1)

import torch.optim as optim

# 初始化GNNTransformer
gnn_model = GNNTransformer(
    soap_dim=dataset.soap_dim,
    num_types=dataset.num_types,
    hidden_dim=128, gcn_layers=2, tr_layers=2, n_heads=4, ff_dim=256, dropout=0.1
)

def init_weights(m):
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
gnn_model.apply(init_weights)

# 线性回归子模型：原子计数 -> 线性输出
linear_model = nn.Linear(dataset.num_types, 1)
linear_model.apply(init_weights)

# 设备与多GPU
# 将模型组件移至device，不在此处使用DataParallel包装gnn_model和linear_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_model.to(device)
linear_model.to(device)

# ===== 1. 预训练线性模型 =====
opt_lin = optim.Adam(linear_model.parameters(), lr=1e-2)
crit = nn.MSELoss()
pretrain_epochs = 200
for ep in range(1, pretrain_epochs+1):
    linear_model.train(); total=0.0
    for atoms, coords, soap, mask, target in train_loader:
        atoms, mask = atoms.to(device), mask.to(device)
        target = target.to(device)
        # 计算原子计数
        one_hot = F.one_hot(atoms, num_classes=dataset.num_types).float().to(device)
        counts = (one_hot * mask.unsqueeze(-1).float()).sum(dim=1)
        # 直接使用原始目标值进行回归
        pred = linear_model(counts).squeeze(-1)
        loss = crit(pred, target)
        opt_lin.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(linear_model.parameters(),5.0); opt_lin.step()
        total += loss.item()*target.size(0)
    print(f"Linear Pretrain Epoch {ep}: Loss={total/len(train_dataset):.6f}")
# 冻结线性模型参数
for p in linear_model.parameters(): p.requires_grad=False
linear_model.eval()

# ===== 2. 组合模型 (GNN + 线性) =====
class CombinedModel(nn.Module):
    def __init__(self, gnn, linear, num_types):
        super().__init__()
        self.gnn = gnn
        self.linear = linear
        self.num_types = num_types
    def forward(self, atoms, coords, soap, mask):
        # GNN部分预测
        gnn_pred = self.gnn(atoms, coords, soap, mask)
        # 线性部分预测
        one_hot = F.one_hot(atoms, num_classes=self.num_types).float().to(atoms.device)
        counts = (one_hot * mask.unsqueeze(-1).float()).sum(dim=1)
        lin_pred = self.linear(counts).squeeze(-1)
        return gnn_pred + lin_pred

model = CombinedModel(gnn_model, linear_model, dataset.num_types).to(device)
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model)

# ===== 3. 训练组合模型 =====
opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
early_stop=10; best_val=float('inf'); bad=0
for epoch in range(1,1001):
    model.train(); tr_loss=0.0
    for atoms, coords, soap, mask, target in train_loader:
        atoms, coords, soap, mask = atoms.to(device), coords.to(device), soap.to(device), mask.to(device)
        target = target.to(device)
        # 直接使用原始目标值进行回归
        pred = model(atoms, coords, soap, mask)
        loss = crit(pred, target)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        tr_loss += loss.item()*target.size(0)
    tr_loss /= len(train_dataset)
    # 验证
    model.eval(); val_l=0.0
    with torch.no_grad():
        for atoms, coords, soap, mask, target in val_loader:
            atoms, coords, soap, mask = atoms.to(device), coords.to(device), soap.to(device), mask.to(device)
            target = target.to(device)
            pred = model(atoms, coords, soap, mask)
            val_l += crit(pred, target).item()*target.size(0)
    val_l /= len(val_dataset)
    print(f"Epoch {epoch}: Train={tr_loss:.6f}, Val={val_l:.6f}")
    # 早停与保存最佳
    if val_l < best_val:
        best_val = val_l; bad=0
        sav = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(sav.state_dict(), 'GCN_transformers_best_model4.pth')
    else:
        bad+=1
        if bad>=early_stop:
            print(f"Early stop at epoch {epoch}"); break
print(f"Best Val Loss: {best_val:.6f}")

# 加载最佳模型
gnn_model = GNNTransformer(
    soap_dim=dataset.soap_dim,
    num_types=dataset.num_types,
    hidden_dim=128, gcn_layers=2, tr_layers=2, n_heads=4, ff_dim=256, dropout=0.1
)

linear_model = nn.Linear(dataset.num_types, 1)

# ===== 2. 组合模型 (GNN + 线性) =====
class CombinedModel(nn.Module):
    def __init__(self, gnn, linear, num_types):
        super().__init__()
        self.gnn = gnn
        self.linear = linear
        self.num_types = num_types
    def forward(self, atoms, coords, soap, mask):
        # GNN部分预测
        gnn_pred = self.gnn(atoms, coords, soap, mask)
        # 线性部分预测
        one_hot = F.one_hot(atoms, num_classes=self.num_types).float().to(atoms.device)
        counts = (one_hot * mask.unsqueeze(-1).float()).sum(dim=1)
        lin_pred = self.linear(counts).squeeze(-1)
        return gnn_pred + lin_pred

best_model = CombinedModel(gnn_model, linear_model, dataset.num_types)

best_model.load_state_dict(torch.load('GCN_transformers_best_model4.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model.to(device)

# 画一下预测值与目标值的散点图
import matplotlib.pyplot as plt
def plot_predictions_vs_targets(predictions, targets):
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid()
    plt.axis('equal')
    plt.show()
    
# 在验证集上评估最佳模型预测值与目标值的MAE、RMSE和R2分数
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for atoms, coords, soap, mask, target in dataloader:
            atoms, coords, soap, mask = atoms.to(device), coords.to(device), soap.to(device), mask.to(device)
            target = target.to(device)
            pred = model(atoms, coords, soap, mask)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    print(f"MAE: {mae:.10f}, RMSE: {rmse:.10f}, R2: {r2:.10f}")
    plot_predictions_vs_targets(all_preds, all_targets)
# 在验证集上评估最佳模型
evaluate_model(best_model, val_loader)
