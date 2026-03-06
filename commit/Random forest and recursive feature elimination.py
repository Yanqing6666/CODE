# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:53:38 2025

@author: YCC
"""

# -*- coding: utf-8 -*-
"""
分块随机森林空间预测 - 修正版（鲁棒填充 / 裁剪 / scale0 保护 / 保存指标）
请确保 raster_files 的顺序与 Excel 中训练特征列（除目标列）顺序一致
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from osgeo import gdal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import rasterio
from rasterio.windows import Window
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# 参数（按需修改）
# --------------------------
excel_file = r'F:/小论文/红黄壤退化/progress/空间插值/2010点位环境输入_异常值处理_保留全列1.xlsx'
output_tif = r'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/空间预测结果/有机质3.tif'
metrics_path = output_tif.replace('.tif', '_metrics.txt')
# 栅格文件（顺序必须和下面 feature_names 一致）
raster_files = [
    'F:/小论文/红黄壤退化/progress/空间插值/环境变量已处理/DEM.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/环境变量已处理/CURVATURE.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/环境变量已处理/bd515.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/环境变量已处理/MAP_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/环境变量已处理/texcls515.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/环境变量已处理/thickness.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/距离公路距离2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/距河流距离2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/TRI.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/SAVI_max_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/EVI_seasonality_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/EVI_max_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/NDVI_seasonality_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/NDVI_max_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/AI_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/PET_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/PmaxMonth_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/PminMonth_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/Pseasonality_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/TmaxMonth_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/TminMonth_2010.tif',
    'F:/小论文/红黄壤退化/progress/空间插值/新增环境变量/裁剪/Tseasonality_2010.tif'
]

# 训练特征列（必须和上面 raster_files 对应顺序一致）
feature_names = ['DEM','CURVATURE','bd515','MAP_2010','texcls515','thickness',
                 '距河流距离2010','距离公路距离2010','TRI','SAVI_max_2010',
                 'EVI_seasonality_2010','EVI_max_2010','NDVI_seasonality_2010','NDVI_max_2010',
                 'AI_2010','PET_2010','PmaxMonth_2010',
                 'PminMonth_2010','Pseasonality_2010',
                 'TmaxMonth_2010','TminMonth_2010','Tseasonality_2010']
# target name: 'pH'

# --------------------------
# 1. 读取点样本（Excel）
# --------------------------
df = pd.read_excel(excel_file)
# 保证列存在
cols_required = feature_names + ['有机质']
for c in cols_required:
    if c not in df.columns:
        raise ValueError(f"Excel 缺少列: {c}")

# 按照 feature_names 顺序取特征
X_all = df[feature_names].copy()
y_all = df['有机质'].copy()

# --------------------------
# 2. 划分训练/测试并保存训练集特征统计量（用于填充/裁剪）
# --------------------------
data_train, data_test = train_test_split(pd.concat([X_all, y_all], axis=1), test_size=0.2, random_state=2)
print("训练样本数量：", len(data_train))
print("测试样本数量：", len(data_test))

X_train_raw = data_train[feature_names].copy()   # 原始（未标准化）
y_train_raw = data_train['有机质'].copy()

# 训练集统计量（每个特征）
feat_medians = X_train_raw.median().values.astype(np.float32)
feat_mins = X_train_raw.min().values.astype(np.float32)
feat_maxs = X_train_raw.max().values.astype(np.float32)

# --------------------------
# 3. 标准化（注意保留 standarder 对象用于预测）
# --------------------------
standarder = StandardScaler()
X_train = standarder.fit_transform(X_train_raw)
# 防止 scale_ 为 0（造成预测时除 0 产生 inf）
zero_scale_idx = np.isclose(standarder.scale_, 0.0)
if zero_scale_idx.any():
    standarder.scale_[zero_scale_idx] = 1.0

# 准备测试集（原始测试集按相同特征顺序）
X_test_raw = data_test[feature_names].copy()
X_test = standarder.transform(X_test_raw)

# 目标变量标准化
y_standarder = StandardScaler()
y_train = y_standarder.fit_transform(y_train_raw.values.reshape(-1,1)).squeeze()
# 保证 y scale 不为 0
if np.isclose(y_standarder.scale_, 0.0):
    y_standarder.scale_[y_standarder.scale_ == 0.0] = 1.0
y_test = y_standarder.transform(data_test['有机质'].values.reshape(-1,1)).squeeze()

# --------------------------
# 4. 随机森林调参（GridSearchCV）
# --------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print("最优参数组合：", grid_search.best_params_)

# --------------------------
# 5. 模型评估（以原始目标单位输出指标）
# --------------------------
y_test_pred_scaled = model.predict(X_test)
# 反标准化到原始单位
y_test_pred = y_standarder.inverse_transform(y_test_pred_scaled.reshape(-1,1)).squeeze()
y_test_orig = data_test['有机质'].values

train_score_scaled = model.score(X_train, y_train)
train_score = None
try:
    # 计算原始单位训练集 R2：需要对训练集预测并反标准化
    y_train_pred_scaled = model.predict(X_train)
    y_train_pred = y_standarder.inverse_transform(y_train_pred_scaled.reshape(-1,1)).squeeze()
    train_score = r2_score(y_train_raw.values, y_train_pred)
except Exception:
    train_score = train_score_scaled  # 退而求其次

rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
mae = mean_absolute_error(y_test_orig, y_test_pred)
r2 = r2_score(y_test_orig, y_test_pred)

print('train score (orig units):', train_score)
print('test rmse (orig units):', rmse)
print('test mae (orig units):', mae)
print('test r2 (orig units):', r2)

# 保存指标
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("最优参数组合: {}\n".format(grid_search.best_params_))
    f.write(f"训练集R² (原始单位,近似): {train_score:.4f}\n")
    f.write(f"测试集RMSE (原始单位): {rmse:.4f}\n")
    f.write(f"测试集MAE (原始单位): {mae:.4f}\n")
    f.write(f"测试集R² (原始单位): {r2:.4f}\n")
print("✅ 模型评估指标已保存到:", metrics_path)

# --------------------------
# 6. 分块预测整幅图（关键部分：稳健填充、裁剪、scale保护）
# --------------------------
# 检查 raster_files 数量是否和特征数量一致
if len(raster_files) != len(feature_names):
    raise ValueError("raster_files 数量和 feature_names 数量不一致，请检查顺序与列名！")

# 打开所有栅格（一次打开，重复使用）
datasets = [rasterio.open(f) for f in raster_files]
height, width = datasets[0].height, datasets[0].width
print("预测影像大小:", height, width)

# 使用第一个输入栅格作为参考模板创建输出
# 确保输出目录存在
os.makedirs(os.path.dirname(output_tif), exist_ok=True)
ref_file = raster_files[0]
ref = gdal.Open(ref_file)
if ref is None:
    raise FileNotFoundError(f"参考栅格 {ref_file} 无法打开，请检查路径！")
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(output_tif, width, height, 1, gdal.GDT_Float32)
out_ds.SetGeoTransform(ref.GetGeoTransform())
out_ds.SetProjection(ref.GetProjection())
out_ds.GetRasterBand(1).SetNoDataValue(-999)

# 分块大小（内存/速度折中，可按需调）
block_size = 512
total_blocks = int(np.ceil(height / block_size) * np.ceil(width / block_size))
block_counter = 0

# 主循环：按块读取、填充、裁剪、标准化、预测、写入
for row_off in range(0, height, block_size):
    for col_off in range(0, width, block_size):
        h = min(block_size, height - row_off)
        w = min(block_size, width - col_off)
        window = Window(col_off, row_off, w, h)

        block_stack = []
        # 逐波段读取并稳健填充/裁剪
        for idx, ds in enumerate(datasets):
            arr = ds.read(1, window=window).astype(np.float64)  # 使用 float64 避免中间精度问题

            # 把栅格自带的 nodata 替换为 np.nan（如果存在）
            nod = ds.nodata
            if nod is not None:
                arr[arr == nod] = np.nan

            # 把 inf 替换为 nan
            arr[~np.isfinite(arr)] = np.nan

            # 若整个块全为 nan，用训练集中位数填充；否则用该波段块内中位数填充 nan
            if np.all(np.isnan(arr)):
                fill_val = feat_medians[idx]
            else:
                # 优先使用窗口内部中位数（更本地化），若该值为 nan（罕见）则退回到训练中位数
                m = np.nanmedian(arr)
                fill_val = feat_medians[idx] if np.isnan(m) else float(m)
            arr[np.isnan(arr)] = fill_val

            # 裁剪到训练集 min/max（防止异常值）
            arr = np.clip(arr, feat_mins[idx], feat_maxs[idx])

            block_stack.append(arr.astype(np.float32))

        # 堆叠为 (h, w, bands)
        block_stack = np.stack(block_stack, axis=-1)
        n_bands = block_stack.shape[-1]
        # 转成形状(Npix, n_bands)
        block_2d = block_stack.reshape(-1, n_bands)

        # 标准化 - 先确保不会产生 inf（scale 已处理）
        # 如果某些像元仍然包含非有限值（理论上已填充），再一次替换
        block_2d = np.nan_to_num(block_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化（注意标准器期望 n_features 一致）
        try:
            block_2d_scaled = standarder.transform(block_2d)
        except Exception as e:
            # 出现异常时，做安全剪切并重试
            block_2d = np.clip(block_2d, feat_mins - 1e6, feat_maxs + 1e6)
            block_2d_scaled = standarder.transform(block_2d)

        # 防止极端值（float32 溢出），裁剪到合理范围
        block_2d_scaled = np.nan_to_num(block_2d_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        block_2d_scaled = np.clip(block_2d_scaled, -1e6, 1e6)

        # 预测（批量）
        # 若 block 大小为 M pixels，predict 接受 (M, n_features)
        preds_scaled = model.predict(block_2d_scaled)

        # 反标准化到原始有机质单位
        preds = y_standarder.inverse_transform(preds_scaled.reshape(-1,1)).squeeze()

        # 还原为块矩阵，并写入输出 tif（注意 GDAL 的 WriteArray 参数为 (array, xoff, yoff)）
        preds_block = preds.reshape(h, w).astype(np.float32)
        out_ds.GetRasterBand(1).WriteArray(preds_block, xoff=col_off, yoff=row_off)

        block_counter += 1
        if block_counter % 20 == 0:
            pct = block_counter / total_blocks * 100
            print(f"已处理 {block_counter}/{total_blocks} blocks ({pct:.1f}%)")

# 关闭并刷新
out_ds.FlushCache()
del out_ds
for ds in datasets:
    ds.close()

print("✅ 分块预测完成，结果保存到:", output_tif)
