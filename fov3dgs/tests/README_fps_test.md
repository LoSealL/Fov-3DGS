# render_compose_gazes_fps.py 配置参数

## Bicycle 数据集实际配置

根据 `D:\Works\third-party\Fov-3DGS\dataset\bicycle\4_12_0.01_1.0` 生成的产物分析：

### 分辨率
- **图像目录**: `images_4`
- **分辨率**: **1237 x 822**
- **相机数量**: 194

### 高斯点数量
- **PS1 模型** (`1_PS1_4_12/point_cloud/iteration_30002/point_cloud.ply`): **6,013,971** 点 (~600 万)
- **Composed 模型** (`composed_4_12/`): **6,013,971** 点

### 多层配置
- **层级数**: 4 层
- **各层 pooling_size**: 1, 4, 7, 12
- **SH 系数维度**: 3 (RGB)
- **不透明度**: 每层 1 个通道

### 文件结构
```
bicycle/4_12_0.01_1.0/
├── 1_PS1_4_12/
│   ├── chkpnt30002.pth
│   └── point_cloud/
│       └── iteration_30002/
│           └── point_cloud.ply          (6,013,971 点)
├── 3_1-1_4_surface/
│   └── point_cloud/
│       └── iteration_2/
│           └── point_cloud.ply
├── 7_1-1_4_surface/
│   └── point_cloud/
│       └── iteration_2/
│           └── point_cloud.ply
├── 12_1-1_4_surface/
│   └── point_cloud/
│       └── iteration_2/
│           └── point_cloud.ply
└── composed_4_12/
    ├── highest_levels.pt                (6013971, 1)
    ├── shs_dcs.pt                       (6013971, 4, 3)
    └── opacities.pt                     (6013971, 4)
```

### 训练配置
- **sh_degree**: 3
- **image_set**: images_4
- **eval**: True
- **pruning_metric**: max_comp_efficiency
- **relax_ratio**: 0.01

---

## 单元测试配置

测试文件：`tests/test_fov_rasterizer_fps.py`

### 默认配置（匹配 bicycle 数据集）
```python
# 分辨率
image_height = 822
image_width = 1237

# 高斯点数
num_points = 6_000_000  # 600 万
```

### 运行测试
```bash
# 使用 bicycle 配置（600 万点，1237x822）
uv run python tests/test_fov_rasterizer_fps.py
```

### 预期性能（RTX 4070 Ti）
- **分辨率**: 1237x822
- **高斯点数**: 6,000,000
- **平均 FPS**: ~10-15 FPS（估计）
- **帧时间**: ~70-100ms

> 注意：由于 600 万点数据量较大，完整 FPS 测试可能需要较长时间。测试时可适当减少点数。

---

## 与原始 render_compose_gazes_fps.py 的对比

| 参数 | render_compose_gazes_fps | 单元测试 |
|------|-------------------------|----------|
| 分辨率 | 1237x822 (images_4) | 1237x822 (默认) |
| 高斯点数 | ~600 万 | ~600 万 (默认) |
| 层级数 | 4 | 4 |
| Gaze 样本数 | 9 (3x3 网格) | 9 (3x3 网格) |
| 预热帧数 | 10 | 10 |
| 测试帧数 | 全部相机 | 100 帧 |
