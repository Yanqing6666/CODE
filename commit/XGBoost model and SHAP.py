# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 11:00:27 2026

@author: YCC
"""

# -*- coding: utf-8 -*-
"""
GeoSHAP模型与空间解释分析 - 矢量格式输出版
适用于高质量地理学/生态学学术论文发表
作者: YCC
日期: 2026-02-06

主要特性:
1. 高分辨率矢量图输出（SVG + PDF + PNG）
2. 所有图表可在Adobe Illustrator/Inkscape中完美编辑
3. 优化字体大小和排版，符合顶级期刊要求
4. 添加LOWESS拟合和拐点检测
5. 删除所有子图标注，标题居中
6. 修复所有matplotlib布局冲突

输出格式说明:
- PNG: 300 DPI位图，用于预览和演示
- SVG: 可缩放矢量图形，可在AI/Inkscape中编辑所有元素
- PDF: 高质量矢量PDF，适合直接用于论文投稿

编辑建议:
- 使用Adobe Illustrator打开SVG/PDF文件
- 可自由调整字体、颜色、线条粗细
- 可添加或删除任何图形元素
- 可导出为任何分辨率的位图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import KMeans
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== 高质量绘图配置 ====================
# 符合Nature/Science/PNAS标准
# 【重要】字体进一步调大以便阅读

# 【关键】禁用constrained_layout以避免与SHAP的tight_layout冲突
import matplotlib
matplotlib.rcParams['figure.constrained_layout.use'] = False
matplotlib.rcParams['figure.autolayout'] = False

plt.rcParams.update({
    'font.family': ['Times New Roman', 'Arial', 'DejaVu Sans'],
    'font.size': 16,              # 从14调大到16
    'axes.labelsize': 18,         # 从16调大到18
    'axes.titlesize': 19,         # 从17调大到19
    'xtick.labelsize': 16,        # 从14调大到16
    'ytick.labelsize': 16,        # 从14调大到16
    'legend.fontsize': 15,        # 从13调大到15
    'axes.linewidth': 2.0,        # 从1.8调粗到2.0
    'xtick.major.width': 1.8,     # 从1.5调粗到1.8
    'ytick.major.width': 1.8,     # 从1.5调粗到1.8
    'xtick.major.size': 6,        # 从5调大到6
    'ytick.major.size': 6,        # 从5调大到6
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.unicode_minus': False,
    'figure.constrained_layout.use': False,  # 再次确认禁用
    'figure.autolayout': False
})

# 学术期刊配色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'positive': '#06A77D',
    'negative': '#D74E09',
    'neutral': '#6C757D',
    'lowess': '#FF6B35',
    'threshold': '#4ECDC4',
    'diverging': ['#D73027', '#FC8D59', '#FEE090', '#E0F3F8', '#91BFDB', '#4575B4']
}

# ==================== 拐点检测类 ====================
class ThresholdDetector:
    """基于LOWESS拟合的拐点检测"""
    
    def detect_with_lowess(self, x, y, frac=0.3):
        """使用LOWESS拟合并检测拐点"""
        # 数据清洗
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 10:
            return None, None, []
        
        # 排序
        sort_idx = np.argsort(x_clean)
        x_sorted, y_sorted = x_clean[sort_idx], y_clean[sort_idx]
        
        # LOWESS拟合
        try:
            lowess_fit = lowess(y_sorted, x_sorted, frac=frac)
            lowess_x, lowess_y = lowess_fit[:, 0], lowess_fit[:, 1]
        except:
            return None, None, []
        
        # 检测拐点
        thresholds = self._detect_thresholds(lowess_x, lowess_y)
        
        return lowess_x, lowess_y, thresholds
    
    def _detect_thresholds(self, x, y):
        """综合检测拐点"""
        thresholds = []
        
        # 方法1: 二阶导数法
        if len(y) >= 11:
            window_size = min(max(5, len(y)//10), 51)
            if window_size % 2 == 0:
                window_size += 1
            
            try:
                y_smooth = savgol_filter(y, window_size, 3)
                dx = np.mean(np.diff(x)) if len(x) > 1 else 1
                second_deriv = np.gradient(np.gradient(y_smooth, dx), dx)
                
                # 找零交叉点
                zero_cross = np.where(np.diff(np.sign(second_deriv)) != 0)[0]
                for idx in zero_cross:
                    if 0 < idx < len(x) - 1:
                        thresholds.append({
                            'x': x[idx],
                            'y': y[idx],
                            'score': abs(second_deriv[idx]),
                            'method': 'curvature'
                        })
            except:
                pass
        
        # 方法2: 斜率变化法
        if len(x) > 2:
            slopes = np.gradient(y, x)
            slope_changes = np.abs(np.gradient(slopes, x))
            
            if len(slope_changes) > 0:
                threshold_val = np.percentile(slope_changes, 90)
                significant = np.where(slope_changes > threshold_val)[0]
                
                for idx in significant:
                    if 0 < idx < len(x) - 1:
                        thresholds.append({
                            'x': x[idx],
                            'y': y[idx],
                            'score': slope_changes[idx],
                            'method': 'slope_change'
                        })
        
        # 去重并排序
        if thresholds:
            thresholds = sorted(thresholds, key=lambda t: t['score'], reverse=True)
            # 保留距离足够远的拐点
            filtered = [thresholds[0]] if thresholds else []
            for t in thresholds[1:]:
                if all(abs(t['x'] - ft['x']) > (x.max() - x.min()) * 0.1 for ft in filtered):
                    filtered.append(t)
                if len(filtered) >= 3:
                    break
            return filtered
        
        return []

# ==================== Moran's I 优化函数 ====================
def morans_i_optimized(values, coords, k=8, sample_size=5000):
    """内存优化的Moran's I计算"""
    from scipy.spatial import cKDTree
    from scipy.sparse import lil_matrix
    
    n = len(values)
    
    if n > sample_size:
        print(f"    数据量较大({n:,}), 使用{sample_size:,}个样本点进行估计")
        sample_idx = np.random.choice(n, sample_size, replace=False)
        values_sample = values.iloc[sample_idx] if hasattr(values, 'iloc') else values[sample_idx]
        coords_sample = coords.iloc[sample_idx]
        n = sample_size
    else:
        values_sample = values
        coords_sample = coords
    
    mean_val = np.mean(values_sample)
    tree = cKDTree(coords_sample)
    W = lil_matrix((n, n))
    
    for i in range(n):
        distances, indices = tree.query(coords_sample.iloc[i], k=min(k+1, n))
        for j in indices[1:]:
            W[i, j] = 1
    
    W = W.tocsr()
    W_sum = np.array(W.sum(axis=1)).flatten()
    W_sum[W_sum == 0] = 1
    W_normalized = W.multiply(1.0 / W_sum[:, np.newaxis])
    
    deviations = values_sample - mean_val
    numerator = 0
    for i in range(n):
        neighbors = W_normalized.getrow(i).toarray().flatten()
        numerator += deviations.iloc[i] * np.sum(neighbors * deviations.values)
    
    denominator = np.sum(deviations ** 2)
    I = (n / W.sum()) * (numerator / denominator)
    
    return I

# ==================== 配置区域 ====================
print("=" * 80)
print("GeoSHAP分析配置（矢量格式输出版）")
print("=" * 80)
print("\n【输出格式】")
print("  每张图将输出3种格式:")
print("  ✓ PNG - 300 DPI位图（预览）")
print("  ✓ SVG - 矢量图形（可在AI/Inkscape中编辑）")
print("  ✓ PDF - 矢量PDF（高质量印刷）")
print("=" * 80)

# 【重要】选择要分析的数据集
DATASET_TO_RUN = '2010'  # 可选: '变化', '2010', '2020'

# 路径配置
data_dir = r"F:\小论文\红黄壤退化\progress\新建文件夹\SHAP准备\表"
base_output_path = r"F:\小论文\红黄壤退化\progress\论文用图\Geo-SHAP调整1"

# 数据集配置
dataset_config = {
    '变化': {
        'file': '变化.csv',
        'target': 'ΔNPP',
        'description': 'NPP变化分析'
    },
    '2010': {
        'file': '2010.csv',
        'target': 'NPP_2010',
        'description': '2010年NPP分析'
    },
    '2020': {
        'file': '2020.csv',
        'target': 'NPP_2020',
        'description': '2020年NPP分析'
    }
}

# 空间坐标列名
COORD_COLS = ['经度', '纬度']

# 需要排除的列
EXCLUDE_COLS = ['FID','OBJECTID_1']

print(f"\n当前分析数据集: {DATASET_TO_RUN}")
print(f"目标变量: {dataset_config[DATASET_TO_RUN]['target']}")
print(f"数据文件: {dataset_config[DATASET_TO_RUN]['file']}")
print("=" * 80)

# ==================== 数据加载与预处理 ====================
print("\n" + "=" * 80)
print("步骤 1: 数据加载与预处理")
print("=" * 80)

config = dataset_config[DATASET_TO_RUN]
data_file = os.path.join(data_dir, config['file'])
target_col = config['target']
output_path = os.path.join(base_output_path, DATASET_TO_RUN)

# 创建输出文件夹
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'spatial_plots'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'feature_plots'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'statistics'), exist_ok=True)

# 读取数据
print(f"\n正在读取: {data_file}")
try:
    df = pd.read_csv(data_file, encoding='utf-8-sig')
    print(f"✓ 数据加载成功: {df.shape}")
    print(f"  行数: {df.shape[0]:,}")
    print(f"  列数: {df.shape[1]}")
except FileNotFoundError:
    print(f"✗ 错误: 找不到文件 {data_file}")
    print(f"\n请确保文件存在于: {data_dir}")
    exit()
except Exception as e:
    print(f"✗ 读取错误: {e}")
    exit()

# 检查目标变量
if target_col not in df.columns:
    print(f"\n✗ 错误: 未找到目标变量 '{target_col}'")
    npp_cols = [col for col in df.columns if 'NPP' in col.upper() or 'npp' in col]
    if npp_cols:
        print(f"\n可用的NPP相关列:")
        for col in npp_cols:
            print(f"  - {col}")
    exit()

print(f"✓ 目标变量确认: {target_col}")

# 检查空间坐标
spatial_cols = []
for possible_names in [COORD_COLS, ['lon', 'lat'], ['X', 'Y'], ['Longitude', 'Latitude']]:
    if all(col in df.columns for col in possible_names):
        spatial_cols = possible_names
        break

if not spatial_cols:
    print("\n⚠ 警告: 未找到空间坐标列，使用索引作为伪坐标")
    df['X'] = df.index % 1000
    df['Y'] = df.index // 1000
    spatial_cols = ['X', 'Y']

print(f"✓ 空间坐标列: {spatial_cols}")

# 处理FID
if 'FID' in df.columns:
    fid_col = df['FID'].copy()
    df = df.drop('FID', axis=1)
else:
    fid_col = pd.Series(range(len(df)), name='FID')

# 确定特征列
feature_cols = [col for col in df.columns 
                if col != target_col and col not in spatial_cols 
                and col not in EXCLUDE_COLS]

print(f"\n特征统计:")
print(f"  特征数量: {len(feature_cols)}")

# 数据清洗
print("\n数据清洗中...")
initial_size = len(df)

# 1. 删除缺失值
df_clean = df.dropna()
print(f"  1. 删除缺失值: 剩余 {len(df_clean):,} 行")

# 2. 删除PET异常值
if 'PET_2010' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[df_clean['PET_2010'] > -5000]
    print(f"  2. 删除PET异常值: 删除 {before - len(df_clean)} 行")

# 3. 删除MAT异常值
if 'MAT_2010' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[(df_clean['MAT_2010'] != 0) & 
                        (df_clean['MAT_2010'] > -10) & 
                        (df_clean['MAT_2010'] < 50)]
    print(f"  3. 删除MAT异常值: 删除 {before - len(df_clean)} 行")

# 4. 清洗目标变量（保留95%）
before_target = len(df_clean)
lower_bound = df_clean[target_col].quantile(0.025)
upper_bound = df_clean[target_col].quantile(0.975)
df_clean = df_clean[(df_clean[target_col] >= lower_bound) & 
                    (df_clean[target_col] <= upper_bound)]
print(f"  4. 清洗{target_col}: 删除 {before_target - len(df_clean)} 行")

print(f"\n清洗结果:")
print(f"  原始数据: {initial_size:,} 行")
print(f"  清洗后数据: {len(df_clean):,} 行")
print(f"  保留比例: {len(df_clean)/initial_size*100:.1f}%")

if len(df_clean) < 50:
    print("\n✗ 错误: 清洗后样本量过小")
    exit()

# 准备建模数据
fid_clean = fid_col.loc[df_clean.index].reset_index(drop=True)
spatial_coords = df_clean[spatial_cols].reset_index(drop=True)
spatial_coords.columns = ['Longitude', 'Latitude']

X = df_clean[feature_cols].reset_index(drop=True)
y = df_clean[target_col].reset_index(drop=True)

# 特征标准化
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# 数据集划分
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, X_scaled.index, test_size=0.2, random_state=42
)

print(f"\n数据集划分:")
print(f"  训练集: {len(X_train):,} 样本 ({len(X_train)/len(X_scaled)*100:.1f}%)")
print(f"  测试集: {len(X_test):,} 样本 ({len(X_test)/len(X_scaled)*100:.1f}%)")

# ==================== XGBoost模型训练 ====================
print("\n" + "=" * 80)
print("步骤 2: XGBoost模型训练与评估")
print("=" * 80)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

print("\n正在训练模型...")
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train, verbose=False)
print("✓ 模型训练完成")

# 模型评估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

metrics = {
    'Train_R2': r2_score(y_train, y_pred_train),
    'Test_R2': r2_score(y_test, y_pred_test),
    'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
    'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'Train_MAE': mean_absolute_error(y_train, y_pred_train),
    'Test_MAE': mean_absolute_error(y_test, y_pred_test)
}

print(f"\n模型性能:")
print(f"  测试集 R² = {metrics['Test_R2']:.4f}")
print(f"  测试集 RMSE = {metrics['Test_RMSE']:.4f}")

# 交叉验证
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1)
print(f"  交叉验证 R² = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ==================== SHAP值计算 ====================
print("\n" + "=" * 80)
print("步骤 3: 全样本SHAP值计算")
print("=" * 80)

print("\n计算SHAP值...")
explainer = shap.TreeExplainer(model)
shap_values_all = explainer.shap_values(X_scaled)
shap_values_test = shap_values_all[idx_test]
print(f"✓ SHAP值计算完成 (形状: {shap_values_all.shape})")

# 构建GeoSHAP数据框
geoshap_df = pd.DataFrame({
    'FID': fid_clean,
    'Longitude': spatial_coords['Longitude'],
    'Latitude': spatial_coords['Latitude'],
    target_col + '_Observed': y,
    target_col + '_Predicted': model.predict(X_scaled),
    'Residual': y - model.predict(X_scaled),
    'Absolute_Error': np.abs(y - model.predict(X_scaled))
})

# 添加SHAP值
shap_df = pd.DataFrame(shap_values_all, columns=[f'SHAP_{col}' for col in feature_cols])
geoshap_df = pd.concat([geoshap_df, shap_df], axis=1)

# SHAP统计量
geoshap_df['SHAP_Total_Positive'] = shap_df[shap_df > 0].sum(axis=1).fillna(0)
geoshap_df['SHAP_Total_Negative'] = shap_df[shap_df < 0].sum(axis=1).fillna(0)
geoshap_df['SHAP_Net_Effect'] = shap_df.sum(axis=1)
geoshap_df['SHAP_Magnitude'] = np.abs(shap_df).sum(axis=1)

# 保存完整数据
geoshap_output = os.path.join(output_path, 'geoshap_full_data.csv')
geoshap_df.to_csv(geoshap_output, index=False, encoding='utf-8-sig')
print(f"✓ GeoSHAP数据已保存: {geoshap_output}")

# ==================== 空间统计分析 ====================
print("\n" + "=" * 80)
print("步骤 4: 空间统计分析")
print("=" * 80)

print("\n空间自相关分析 (Moran's I):")
spatial_stats = {}

for col in [target_col + '_Observed', 'SHAP_Net_Effect', 'SHAP_Magnitude']:
    try:
        moran = morans_i_optimized(geoshap_df[col], spatial_coords, k=8, sample_size=5000)
        spatial_stats[col] = moran
        print(f"  {col:30s}: I = {moran:7.4f}")
    except Exception as e:
        print(f"  {col:30s}: 计算失败")
        spatial_stats[col] = np.nan

# 空间聚类
print("\n空间聚类分析:")
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
geoshap_df['Spatial_Cluster'] = kmeans.fit_predict(shap_df.values)

print(f"  聚类数: K = {optimal_k}")
for i in range(optimal_k):
    count = (geoshap_df['Spatial_Cluster'] == i).sum()
    pct = count / len(geoshap_df) * 100
    print(f"    Cluster {i}: {count:7,} 样本 ({pct:5.1f}%)")

# ==================== 可视化 ====================
print("\n" + "=" * 80)
print("步骤 5: 生成高质量图表")
print("=" * 80)

# 获取重要特征
shap_importance = np.abs(shap_values_all).mean(axis=0)
top_features_idx = np.argsort(shap_importance)[-6:][::-1]
top_features = [feature_cols[i] for i in top_features_idx]

# ==================== 图1: 空间分布热图（来自代码1）====================
print("\n生成图1: 空间分布热图（前4个子图）...")

# 完全清理matplotlib状态
plt.close('all')
plt.clf()

# 创建figure（2行2列，缩小行间距）
fig = plt.figure(figsize=(18, 15))  # 调整为更合适的尺寸
gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.28,  # 缩小hspace从0.30到0.18
                       left=0.06, right=0.95, top=0.95, bottom=0.05)

# 只保留前4个变量
variables_to_plot = [
    (target_col + '_Observed', f'Observed {target_col}', 'RdYlGn'),
    (target_col + '_Predicted', f'Predicted {target_col}', 'RdYlGn'),
    ('Residual', 'Prediction Residual', 'RdBu_r'),
    ('SHAP_Net_Effect', 'SHAP Net Effect', 'RdBu_r')
]

for idx, (var, title, cmap) in enumerate(variables_to_plot):
    row, col = idx // 2, idx % 2
    ax = fig.add_subplot(gs[row, col])
    
    scatter = ax.scatter(
        geoshap_df['Longitude'],
        geoshap_df['Latitude'],
        c=geoshap_df[var],
        cmap=cmap,
        s=20,
        alpha=0.7,
        edgecolors='none'
    )
    
    ax.set_xlabel('Longitude (°E)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=18, fontweight='bold')
    
    # 只有标题居中，删除标注
    ax.set_title(title, fontsize=19, fontweight='bold', pad=15, loc='center')
    
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=16, width=1.8, length=6)
    
    # 使用divider添加colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.12)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=15, width=1.5, length=5)

fig.suptitle(f'GeoSHAP Spatial Distribution - {DATASET_TO_RUN}', 
             fontsize=22, fontweight='bold', y=0.98)

plt.savefig(os.path.join(output_path, 'spatial_plots', '1_geoshap_spatial_distribution.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'spatial_plots', '1_geoshap_spatial_distribution.svg'),
            format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'spatial_plots', '1_geoshap_spatial_distribution.pdf'),
            format='pdf', bbox_inches='tight')
plt.close('all')
print("  ✓ 图1已保存（PNG + SVG + PDF格式）")

# ==================== 图2: 特征SHAP空间分布（来自代码2）====================
print("\n生成图2: 重要特征SHAP空间分布...")

# 清理状态
plt.close('all')

# 2.1 输出单个特征图
for idx, feat in enumerate(top_features):
    plt.close('all')  # 每张图前清理
    fig, ax = plt.subplots(figsize=(9, 7))  # 稍微调大图片尺寸
    shap_col = f'SHAP_{feat}'
    vmax = np.abs(geoshap_df[shap_col]).quantile(0.98)
    
    scatter = ax.scatter(
        geoshap_df['Longitude'], geoshap_df['Latitude'],
        c=geoshap_df[shap_col], cmap='RdBu_r', s=20, alpha=0.7,
        edgecolors='none', vmin=-vmax, vmax=vmax
    )
    
    ax.set_xlabel('Longitude (°E)', fontsize=17, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=17, fontweight='bold')
    ax.set_title(f'SHAP Spatial Distribution: {feat}', 
                 fontsize=18, fontweight='bold', pad=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    ax.tick_params(labelsize=15)  # 坐标刻度字体
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.10)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('SHAP Value', fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    plt.savefig(os.path.join(output_path, 'feature_plots', f'shap_spatial_{feat}.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'feature_plots', f'shap_spatial_{feat}.svg'),
                format='svg', bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'feature_plots', f'shap_spatial_{feat}.pdf'),
                format='pdf', bbox_inches='tight')
    plt.close('all')

print("  ✓ 单个特征图已保存（PNG + SVG + PDF格式）")

# 2.2 拼图版本
plt.close('all')  # 清理状态
fig = plt.figure(figsize=(22, 14))  # 调大整体尺寸
gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.28,  # 缩小hspace从0.30到0.18
                       left=0.05, right=0.96, top=0.95, bottom=0.05)

for idx, feat in enumerate(top_features):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    shap_col = f'SHAP_{feat}'
    vmax = np.abs(geoshap_df[shap_col]).quantile(0.98)
    
    scatter = ax.scatter(
        geoshap_df['Longitude'], geoshap_df['Latitude'],
        c=geoshap_df[shap_col], cmap='RdBu_r', s=18, alpha=0.7,
        edgecolors='none', vmin=-vmax, vmax=vmax
    )
    
    ax.set_xlabel('Longitude (°E)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=18, fontweight='bold')
    
    # 只有标题居中，删除标注
    ax.set_title(feat, fontsize=19, fontweight='bold', pad=15, loc='center')
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.tick_params(labelsize=16, width=1.8, length=6)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.12)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=14, width=1.5, length=5)

fig.suptitle(f'SHAP Spatial Distribution - Top 6 Features ({DATASET_TO_RUN})',
             fontsize=22, fontweight='bold', y=0.98)

plt.savefig(os.path.join(output_path, 'spatial_plots', '2_feature_shap_spatial_combined.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'spatial_plots', '2_feature_shap_spatial_combined.svg'),
            format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'spatial_plots', '2_feature_shap_spatial_combined.pdf'),
            format='pdf', bbox_inches='tight')
plt.close('all')
print("  ✓ 拼图版本已保存（PNG + SVG + PDF格式）")

# ==================== 图3: 空间聚类分析（来自代码1）====================
print("\n生成图3: 空间聚类分析...")

plt.close('all')  # 清理状态
fig = plt.figure(figsize=(22, 11))  # 调整高度，整体更合理
gs = fig.add_gridspec(1, 2, wspace=0.22,  # 调整间距
                       left=0.05, right=0.96, top=0.92, bottom=0.06)

# (a) 空间聚类分布
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(
    geoshap_df['Longitude'],
    geoshap_df['Latitude'],
    c=geoshap_df['Spatial_Cluster'],
    cmap='Set3',
    s=35,
    alpha=0.8,
    edgecolors='black',
    linewidth=0.4
)
ax1.set_xlabel('Longitude (°E)', fontsize=18, fontweight='bold')
ax1.set_ylabel('Latitude (°N)', fontsize=18, fontweight='bold')

# 只有标题居中，删除标注
ax1.set_title('Spatial Clustering Based on SHAP Values', 
              fontsize=19, fontweight='bold', pad=15, loc='center')

ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax1.set_aspect('equal', adjustable='box')
ax1.tick_params(labelsize=16, width=1.8, length=6)

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="4.5%", pad=0.12)
cbar = plt.colorbar(scatter, cax=cax, ticks=range(optimal_k))
cbar.set_label('Cluster ID', fontsize=17, fontweight='bold')
cbar.ax.tick_params(labelsize=15, width=1.5, length=5)

# (b) 雷达图
cluster_means = []
for i in range(optimal_k):
    mask = geoshap_df['Spatial_Cluster'] == i
    cluster_mean = shap_df[mask].mean().values
    cluster_means.append(cluster_mean)

angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
cluster_means_top = [[cluster_means[i][feature_cols.index(f)] for f in top_features] 
                     for i in range(optimal_k)]

ax2 = fig.add_subplot(gs[0, 1], projection='polar')
colors_cluster = plt.cm.Set3(np.linspace(0, 1, optimal_k))

for i, (cluster_val, color) in enumerate(zip(cluster_means_top, colors_cluster)):
    values = cluster_val + [cluster_val[0]]
    angles_plot = angles + [angles[0]]
    ax2.plot(angles_plot, values, 'o-', linewidth=3.5, 
             label=f'Cluster {i}', color=color, markersize=9)
    ax2.fill(angles_plot, values, alpha=0.15, color=color)

ax2.set_xticks(angles)
ax2.set_xticklabels(top_features, fontsize=15)

# 只有标题，删除标注
ax2.set_title('Mean SHAP by Cluster',
              fontsize=19, fontweight='bold', pad=25, loc='center')

# 修复：legend不接受linewidth参数，使用frameon和edgecolor
legend = ax2.legend(loc='upper right', bbox_to_anchor=(1.30, 1.15), 
                    fontsize=15, frameon=True, fancybox=False, 
                    edgecolor='black', shadow=False)
legend.get_frame().set_linewidth(1.8)  # 通过frame设置边框粗细
ax2.grid(True, alpha=0.3, linewidth=0.8)
ax2.tick_params(labelsize=15, width=1.8, length=6)

plt.savefig(os.path.join(output_path, 'spatial_plots', '3_spatial_clustering.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'spatial_plots', '3_spatial_clustering.svg'),
            format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'spatial_plots', '3_spatial_clustering.pdf'),
            format='pdf', bbox_inches='tight')
plt.close('all')
print("  ✓ 图3已保存（PNG + SVG + PDF格式）")

# ==================== 图4: 聚类箱线图（来自代码2）====================
print("\n生成图4: SHAP聚类分布箱线图...")

plt.close('all')  # 清理状态
fig = plt.figure(figsize=(20, 12))  # 调大尺寸
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25,
                       left=0.06, right=0.96, top=0.94, bottom=0.06)

for idx, feat in enumerate(top_features):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    shap_col = f'SHAP_{feat}'
    data_to_plot = [geoshap_df[geoshap_df['Spatial_Cluster'] == i][shap_col].values 
                    for i in range(optimal_k)]
    
    bp = ax.boxplot(data_to_plot, patch_artist=True, 
                    labels=[f'C{i}' for i in range(optimal_k)],
                    boxprops=dict(facecolor=COLORS['primary'], alpha=0.7, linewidth=1.8),
                    medianprops=dict(color='red', linewidth=3),
                    whiskerprops=dict(linewidth=1.8),
                    capprops=dict(linewidth=1.8),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.8)
    ax.set_xlabel('Cluster', fontsize=15, fontweight='bold')
    ax.set_ylabel('SHAP Value', fontsize=15, fontweight='bold')
    
    # 只有标题，删除标注
    ax.set_title(feat, fontsize=16, fontweight='bold', pad=12, loc='center')
    
    ax.grid(True, alpha=0.25, linestyle='--', axis='y', linewidth=0.6)
    ax.tick_params(labelsize=14)

fig.suptitle(f'SHAP Distribution by Cluster ({DATASET_TO_RUN})',
             fontsize=17, fontweight='bold', y=0.98)

plt.savefig(os.path.join(output_path, 'spatial_plots', '4_cluster_shap_distribution.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'spatial_plots', '4_cluster_shap_distribution.svg'),
            format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'spatial_plots', '4_cluster_shap_distribution.pdf'),
            format='pdf', bbox_inches='tight')
plt.close('all')
print("  ✓ 图4已保存（PNG + SVG + PDF格式）")

# ==================== 图5: SHAP摘要图（来自代码1）====================
print("\n生成图5: SHAP摘要图...")

# 彻底清理并重置
plt.close('all')
import gc
gc.collect()

# 使用try-except捕获并绕过布局错误
try:
    # 方法1: 标准方式
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_test, X_test, feature_names=feature_cols,
                      show=False, cmap='coolwarm', plot_size=(10, 6))
    plt.title(f'SHAP Summary Plot ({DATASET_TO_RUN})', 
              fontsize=15, fontweight='bold', pad=15)
except RuntimeError as e:
    if 'Colorbar layout' in str(e):
        # 方法2: 绕过布局错误
        print("    检测到布局冲突，使用替代方法...")
        plt.close('all')
        # 直接调用SHAP绘图，不添加额外元素
        shap.summary_plot(shap_values_test, X_test, feature_names=feature_cols,
                          show=False, cmap='coolwarm')
    else:
        raise

plt.savefig(os.path.join(output_path, '5_shap_summary_beeswarm.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(os.path.join(output_path, '5_shap_summary_beeswarm.svg'),
            format='svg', bbox_inches='tight', pad_inches=0.1)
plt.savefig(os.path.join(output_path, '5_shap_summary_beeswarm.pdf'),
            format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close('all')
gc.collect()
print("  ✓ 图5已保存（PNG + SVG + PDF格式）")

# ==================== 图6: SHAP重要性条形图（来自代码1）====================
print("\n生成图6: SHAP重要性条形图...")

# 彻底清理
plt.close('all')
gc.collect()

# 使用try-except捕获并绕过布局错误
try:
    # 方法1: 标准方式
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_test, X_test, feature_names=feature_cols,
                      plot_type='bar', show=False, color=COLORS['accent'])
    plt.title(f'SHAP Feature Importance ({DATASET_TO_RUN})', 
              fontsize=15, fontweight='bold', pad=15)
except RuntimeError as e:
    if 'Colorbar layout' in str(e):
        # 方法2: 绕过布局错误
        print("    检测到布局冲突，使用替代方法...")
        plt.close('all')
        shap.summary_plot(shap_values_test, X_test, feature_names=feature_cols,
                          plot_type='bar', show=False, color=COLORS['accent'])
    else:
        raise

plt.savefig(os.path.join(output_path, '6_shap_importance_bar.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(os.path.join(output_path, '6_shap_importance_bar.svg'),
            format='svg', bbox_inches='tight', pad_inches=0.1)
plt.savefig(os.path.join(output_path, '6_shap_importance_bar.pdf'),
            format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close('all')
gc.collect()
print("  ✓ 图6已保存（PNG + SVG + PDF格式）")

# ==================== 图7: SHAP依赖图（含拟合线和拐点）（来自代码2）====================
print("\n生成图7: SHAP依赖图（含LOWESS拟合和拐点）...")
detector = ThresholdDetector()

# 采样参数
MAX_POINTS_FOR_LOWESS = 2000
MAX_POINTS_FOR_PLOT = 200000

plt.close('all')  # 清理状态
fig = plt.figure(figsize=(26, 17))  # 进一步调大尺寸以容纳更大字体
gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.32,  # 调整间距
                       left=0.05, right=0.97, top=0.98, bottom=0.03)

for idx, feat in enumerate(top_features):
    print(f"  处理特征 {idx+1}/6: {feat}...", end=" ")
    
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    # 获取原始数据
    feature_values = X[feat].values
    shap_vals = shap_values_all[:, feature_cols.index(feat)]
    
    # 绘图采样
    if len(feature_values) > MAX_POINTS_FOR_PLOT:
        plot_idx = np.random.choice(len(feature_values), MAX_POINTS_FOR_PLOT, replace=False)
        plot_x, plot_y = feature_values[plot_idx], shap_vals[plot_idx]
    else:
        plot_x, plot_y = feature_values, shap_vals
    
    # 绘制散点
    scatter = ax.scatter(plot_x, plot_y, c=plot_y, 
                        cmap='coolwarm', s=20, alpha=0.5, edgecolors='none',
                        vmin=np.percentile(shap_vals, 5), 
                        vmax=np.percentile(shap_vals, 95))
    
    # LOWESS采样
    if len(feature_values) > MAX_POINTS_FOR_LOWESS:
        lowess_idx = np.random.choice(len(feature_values), MAX_POINTS_FOR_LOWESS, replace=False)
        lowess_x_in = feature_values[lowess_idx]
        lowess_y_in = shap_vals[lowess_idx]
    else:
        lowess_x_in = feature_values
        lowess_y_in = shap_vals
    
    # LOWESS拟合
    lowess_x, lowess_y, thresholds = detector.detect_with_lowess(lowess_x_in, lowess_y_in, frac=0.3)
    
    if lowess_x is not None:
        # 绘制拟合线
        ax.plot(lowess_x, lowess_y, color=COLORS['lowess'], linewidth=4.5, 
                label='LOWESS Fit', zorder=10)
        
        # 标注拐点
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        
        for i, t in enumerate(thresholds[:2]):
            # 拐点星号
            ax.scatter(t['x'], t['y'], s=350, marker='*', 
                      color=COLORS['threshold'], edgecolors='white', 
                      linewidths=4, zorder=15)
            
            # 垂直虚线
            ax.axvline(x=t['x'], color=COLORS['threshold'], 
                      linestyle='--', alpha=0.6, linewidth=3.5)
            
            # 文本标注（字体加大）
            text_y = ylim[1] - y_range * 0.1 if i == 0 else ylim[0] + y_range * 0.1
            ax.text(t['x'], text_y, f"{t['x']:.2f}", 
                   ha='center', va='center', fontsize=17, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                            edgecolor=COLORS['threshold'], alpha=0.95, linewidth=3.5))
    
    # 美化
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=2.5)
    
    # 坐标轴字体加大
    ax.set_xlabel(feat, fontsize=22, fontweight='bold')  # 从19调到22
    ax.set_ylabel('SHAP Value', fontsize=21, fontweight='bold')  # 从18调到21
    
    # 只有标题居中，删除标注
    ax.set_title(feat, fontsize=23, fontweight='bold', pad=20, loc='center')  # 从20调到23
    
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.tick_params(labelsize=19, width=2.2, length=8)  # 刻度字体从17调到19
    
    # 颜色条 - 字体加大
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label(feat, fontsize=18, fontweight='bold')  # 从15调到18
    cbar.ax.tick_params(labelsize=17, width=2, length=7)  # 从14调到17
    
    # 图例
    if lowess_x is not None:
        legend = ax.legend(loc='upper right', fontsize=16, framealpha=0.95,  # 从14调到16
                          edgecolor='black', fancybox=False)
        legend.get_frame().set_linewidth(2)  # 从1.8调到2
    
    print("✓")

# 注意：不添加总标题 (已删除 fig.suptitle)

plt.savefig(os.path.join(output_path, '7_shap_dependence_with_thresholds.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_path, '7_shap_dependence_with_thresholds.svg'),
            format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_path, '7_shap_dependence_with_thresholds.pdf'),
            format='pdf', bbox_inches='tight')
plt.close('all')
print("  ✓ 图7已保存（PNG + SVG + PDF格式）")

# 保存拐点数据
print("\n提取拐点数据...")
threshold_results = []
for feat in top_features:
    feature_values = X[feat].values
    shap_vals = shap_values_all[:, feature_cols.index(feat)]
    
    if len(feature_values) > MAX_POINTS_FOR_LOWESS:
        sample_idx = np.random.choice(len(feature_values), MAX_POINTS_FOR_LOWESS, replace=False)
        feature_values = feature_values[sample_idx]
        shap_vals = shap_vals[sample_idx]
    
    lowess_x, lowess_y, thresholds = detector.detect_with_lowess(feature_values, shap_vals)
    
    for i, t in enumerate(thresholds):
        threshold_results.append({
            'Feature': feat,
            'Threshold_X': round(t['x'], 4),
            'Threshold_SHAP': round(t['y'], 4),
            'Detection_Method': t['method'],
            'Confidence_Score': round(t['score'], 4)
        })

if threshold_results:
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(os.path.join(output_path, 'statistics', 'shap_thresholds.csv'),
                       index=False, encoding='utf-8-sig')
    print("  ✓ 拐点数据已保存")

# ==================== 保存统计结果 ====================
print("\n" + "=" * 80)
print("步骤 6: 保存统计结果")
print("=" * 80)

# 1. 模型性能
performance_df = pd.DataFrame({
    'Metric': ['R²', 'RMSE', 'MAE', 'CV_R²_Mean', 'CV_R²_Std'],
    'Training': [metrics['Train_R2'], metrics['Train_RMSE'], metrics['Train_MAE'], '-', '-'],
    'Testing': [metrics['Test_R2'], metrics['Test_RMSE'], metrics['Test_MAE'], 
                cv_scores.mean(), cv_scores.std()]
})
performance_df.to_csv(os.path.join(output_path, 'statistics', 'model_performance.csv'),
                     index=False, encoding='utf-8-sig')
print("  ✓ model_performance.csv")

# 2. 特征重要性
xgb_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'XGBoost_Importance': xgb_importance,
    'SHAP_Importance': shap_importance
}).sort_values('SHAP_Importance', ascending=False)
feature_importance_df.to_csv(os.path.join(output_path, 'statistics', 'feature_importance.csv'),
                            index=False, encoding='utf-8-sig')
print("  ✓ feature_importance.csv")

# 3. 空间自相关
if spatial_stats:
    spatial_stats_df = pd.DataFrame([spatial_stats])
    spatial_stats_df.to_csv(os.path.join(output_path, 'statistics', 'spatial_autocorrelation.csv'),
                           index=False, encoding='utf-8-sig')
    print("  ✓ spatial_autocorrelation.csv")

# ==================== 完成总结 ====================
print("\n" + "=" * 80)
print(f"✓✓✓ GeoSHAP分析完成 - {DATASET_TO_RUN} 数据集")
print("=" * 80)

print(f"\n输出路径: {output_path}")
print(f"\n生成文件清单（每张图均包含PNG、SVG、PDF三种格式）:")
print(f"  1. geoshap_full_data.csv ({len(geoshap_df):,} 行)")
print(f"\n  2. 空间图表 (4张 × 3格式 = 12个文件):")
print(f"     - 1_geoshap_spatial_distribution.[png|svg|pdf]")
print(f"     - 2_feature_shap_spatial_combined.[png|svg|pdf]")
print(f"     - 3_spatial_clustering.[png|svg|pdf]")
print(f"     - 4_cluster_shap_distribution.[png|svg|pdf]")
print(f"\n  3. SHAP分析图 (3张 × 3格式 = 9个文件):")
print(f"     - 5_shap_summary_beeswarm.[png|svg|pdf]")
print(f"     - 6_shap_importance_bar.[png|svg|pdf]")
print(f"     - 7_shap_dependence_with_thresholds.[png|svg|pdf]")
print(f"\n  4. 单个特征图 ({len(top_features)}张 × 3格式 = {len(top_features)*3}个文件):")
for feat in top_features:
    print(f"     - shap_spatial_{feat}.[png|svg|pdf]")
print(f"\n  5. 统计文件 (4个CSV):")
print(f"     - model_performance.csv")
print(f"     - feature_importance.csv")
print(f"     - spatial_autocorrelation.csv")
print(f"     - shap_thresholds.csv")

print(f"\n关键结果:")
print(f"  模型性能: R² = {metrics['Test_R2']:.4f}")
print(f"  最重要特征: {top_features[0]}")
print(f"  空间聚类数: {optimal_k} 类")

print("\n" + "=" * 80)
print("✓ 所有图表符合高质量论文发表标准")
print("✓ 输出格式：PNG（位图预览） + SVG（矢量可编辑） + PDF（高质量印刷）")
print("✓ SVG和PDF格式可在Adobe Illustrator/Inkscape中完美编辑")
print("✓ 字体、线条、颜色等所有元素均可自由调整")
print("✓ 适合投稿Nature/Science等顶级期刊")
print("=" * 80)
print("\n分析完成！祝论文顺利发表！")
print("=" * 80)