# 强化学习改进Statistical狭窄检测 - 完整技术方案

## 🎯 核心思想

**让Agent学习如何为每张图像选择最优的检测参数，实现自适应检测。**

---

## 🏗️ RL框架设计

### 1. 环境定义（Environment）

#### 状态空间（State Space）
```python
State = {
    # 1. 图像特征（从原始图像提取）
    'image_features': [
        'mean_intensity',      # 平均亮度
        'std_intensity',       # 亮度标准差
        'contrast',            # 对比度
        'vessel_density',      # 血管密度
        'image_size'           # 图像尺寸
    ],
    
    # 2. Mask质量特征（关键！）
    'mask_quality': [
        'skeleton_connectivity',  # 骨架连通性（连通分量数）
        'skeleton_length',        # 骨架总长度
        'avg_vessel_width',       # 平均血管宽度
        'broken_segments',        # 断裂段数
        'mask_coverage'           # mask覆盖率
    ],
    
    # 3. Statistical检测中间结果
    'detection_features': [
        'num_segmentation_points',  # 分段点数量
        'avg_radius',               # 平均半径
        'radius_std',               # 半径标准差
        'path_success_rate'         # 路径查找成功率
    ],
    
    # 4. 当前检测状态
    'current_detections': {
        'num_candidates',      # 候选狭窄数
        'confidence_scores',   # 置信度分布
        'spatial_distribution' # 空间分布
    }
}

# 状态维度：约30-40维
```

#### 动作空间（Action Space）
```python
Action = {
    # 1. 核心检测参数（连续动作）
    'stenosis_threshold': [0.10, 0.40],  # 狭窄程度阈值
    'min_avg_radius': [1.0, 8.0],        # 最小平均半径
    
    # 2. 高级参数（连续）
    'segmentation_distance': [5.0, 15.0], # 分段点过滤距离
    'max_search_radius': [80, 150],       # 半径计算搜索范围
    
    # 3. 策略选择（离散）
    'skip_low_quality': {0, 1},           # 是否跳过低质量段
    'use_relaxed_path': {0, 1},           # 是否使用宽松路径查找
    
    # 4. 后处理参数
    'min_detection_distance': [5, 20],    # 检测点最小间距
    'confidence_threshold': [0.5, 0.95]   # 置信度阈值（如果加入）
}

# 动作维度：8维（5个连续 + 3个离散）
```

#### 奖励函数（Reward Function）
```python
def compute_reward(detections, ground_truth):
    """
    设计奖励函数：平衡Sensitivity和Precision
    """
    # 1. 匹配检测结果与GT
    tp, fp, fn = match_detections(detections, ground_truth, threshold=10mm)
    
    # 2. 计算基础指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # 3. 奖励设计（关键！）
    reward = 0
    
    # 3.1 F1-Score作为主要奖励（scaled to 0-10）
    reward += f1_score * 10
    
    # 3.2 True Positive奖励（鼓励检出）
    reward += tp * 1.0
    
    # 3.3 False Positive惩罚（抑制误报，但不要太重）
    reward -= fp * 0.3
    
    # 3.4 False Negative惩罚（漏检惩罚较重）
    reward -= fn * 0.5
    
    # 3.5 完美检测额外奖励
    if fp == 0 and fn == 0 and tp > 0:
        reward += 5.0  # Bonus
    
    # 3.6 无检测但有GT的惩罚
    if tp == 0 and len(ground_truth) > 0:
        reward -= 2.0
    
    return reward, {
        'tp': tp, 'fp': fp, 'fn': fn,
        'sensitivity': sensitivity,
        'precision': precision,
        'f1_score': f1_score
    }
```

---

## 💻 具体实现

### Environment实现

```python
import gym
from gym import spaces
import numpy as np
from stenosis_detection import StenosisDetector

class StenosisDetectionEnv(gym.Env):
    """
    狭窄检测的RL环境
    """
    def __init__(self, dataset_path, annotations_path):
        super().__init__()
        
        # 加载数据集
        self.dataset = self.load_dataset(dataset_path, annotations_path)
        self.current_idx = 0
        
        # 定义动作空间（连续）
        self.action_space = spaces.Box(
            low=np.array([0.10, 1.0, 5.0, 80, 0, 0, 5, 0.5]),
            high=np.array([0.40, 8.0, 15.0, 150, 1, 1, 20, 0.95]),
            dtype=np.float32
        )
        
        # 定义状态空间（约35维）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(35,),
            dtype=np.float32
        )
        
        self.max_steps = 1
        self.current_step = 0
    
    def reset(self):
        """重置环境，加载新图像"""
        # 随机选择一张图像
        self.current_idx = np.random.randint(0, len(self.dataset))
        sample = self.dataset[self.current_idx]
        
        self.image_path = sample['image_path']
        self.mask_path = sample['mask_path']
        self.ground_truth = sample['annotations']
        
        # 初始化检测器
        self.detector = StenosisDetector(
            self.image_path, 
            self.mask_path,
            resize_shape=(512, 512)
        )
        
        # 提取初始状态特征
        state = self._extract_state_features()
        
        self.current_step = 0
        return state
    
    def step(self, action):
        """执行动作"""
        # 解析动作
        stenosis_threshold = action[0]
        min_avg_radius = action[1]
        seg_distance = action[2]
        max_radius = action[3]
        skip_low_quality = bool(action[4] > 0.5)
        use_relaxed_path = bool(action[5] > 0.5)
        min_det_distance = action[6]
        confidence_threshold = action[7]
        
        # 执行检测（使用Agent选择的参数）
        try:
            self.detector.extract_skeleton()
            self.detector.calculate_vessel_radius(
                max_radius=int(max_radius), 
                use_gpu=False
            )
            self.detector.find_segmentation_points()
            self.detector.filter_segmentation_points(
                min_distance=int(seg_distance)
            )
            
            # 关键：使用RL选择的参数
            self.detector.detect_stenosis(
                stenosis_threshold=float(stenosis_threshold),
                min_avg_radius=float(min_avg_radius)
            )
            
            detections = self.detector.all_stenosis_points
            
        except Exception as e:
            # 检测失败，给予负奖励
            detections = []
            reward = -5.0
            done = True
            info = {'error': str(e)}
            return self._extract_state_features(), reward, done, info
        
        # 计算奖励
        reward, metrics = self._compute_reward(detections, self.ground_truth)
        
        # Episode结束
        done = True
        self.current_step += 1
        
        info = {
            'metrics': metrics,
            'action_params': {
                'stenosis_threshold': stenosis_threshold,
                'min_avg_radius': min_avg_radius
            }
        }
        
        return self._extract_state_features(), reward, done, info
    
    def _extract_state_features(self):
        """提取状态特征"""
        features = []
        
        # 1. 图像特征（5维）
        image = cv2.imread(self.image_path, 0)
        features.extend([
            np.mean(image) / 255.0,
            np.std(image) / 255.0,
            self._compute_contrast(image),
            np.sum(image > 50) / image.size,  # vessel density
            image.shape[0] / 1000.0
        ])
        
        # 2. Mask质量特征（5维）
        mask = cv2.imread(self.mask_path, 0)
        skeleton = skeletonize(mask > 0)
        
        # 连通分量分析
        num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))
        
        features.extend([
            num_labels / 10.0,  # 连通分量数（归一化）
            np.sum(skeleton) / 1000.0,  # 骨架长度
            np.mean(mask > 0) if np.sum(mask > 0) > 0 else 0,
            (num_labels - 1) / 5.0,  # 断裂估计
            np.sum(mask > 0) / mask.size  # mask覆盖率
        ])
        
        # 3. 检测中间特征（如果已执行）（25维）
        if hasattr(self.detector, 'point_data') and len(self.detector.point_data) > 0:
            radii = [p['radius'] for p in self.detector.point_data]
            features.extend([
                len(self.detector.point_data) / 1000.0,
                np.mean(radii) / 50.0,
                np.std(radii) / 20.0,
                len(self.detector.segmentation_points) / 20.0 if hasattr(self.detector, 'segmentation_points') else 0,
            ])
            # padding to 25 dims
            features.extend([0] * 21)
        else:
            features.extend([0] * 25)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_reward(self, detections, ground_truth):
        """计算奖励（与上面的reward函数一致）"""
        if len(detections) == 0 and len(ground_truth) == 0:
            return 5.0, {'tp': 0, 'fp': 0, 'fn': 0, 'f1_score': 1.0}
        
        if len(ground_truth) == 0:
            return -len(detections) * 0.3, {'tp': 0, 'fp': len(detections), 'fn': 0}
        
        # 匹配检测结果
        tp, fp, matches = self._match_detections(detections, ground_truth)
        fn = len(ground_truth) - tp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # 奖励计算
        reward = f1 * 10 + tp * 1.0 - fp * 0.3 - fn * 0.5
        
        if fp == 0 and fn == 0 and tp > 0:
            reward += 5.0
        
        return reward, {
            'tp': tp, 'fp': fp, 'fn': fn,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1_score': f1
        }
    
    def _match_detections(self, detections, ground_truth, threshold_mm=10.0):
        """匹配检测结果与GT"""
        pixel_spacing = 0.1953125  # mm/pixel
        threshold_pixels = threshold_mm / pixel_spacing
        
        if len(detections) == 0:
            return 0, 0, []
        
        gt_matched = [False] * len(ground_truth)
        tp = 0
        matches = []
        
        for det in detections:
            best_match = -1
            best_dist = float('inf')
            
            for i, gt in enumerate(ground_truth):
                if gt_matched[i]:
                    continue
                
                dist = np.linalg.norm(np.array(det) - np.array(gt))
                if dist < best_dist:
                    best_dist = dist
                    best_match = i
            
            if best_match != -1 and best_dist <= threshold_pixels:
                gt_matched[best_match] = True
                tp += 1
                matches.append((det, ground_truth[best_match]))
        
        fp = len(detections) - tp
        return tp, fp, matches
```

---

## 🎓 训练策略

### 使用PPO算法训练

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# 1. 创建环境
env = StenosisDetectionEnv(
    dataset_path='/mnt/sda1/luoyu/xzjc_data/dataset',
    annotations_path='/mnt/sda1/luoyu/xzjc_data/train_labels.csv'
)

# 2. 向量化环境（并行训练）
vec_env = make_vec_env(
    lambda: env,
    n_envs=4  # 4个并行环境
)

# 3. 配置PPO
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./rl_logs/"
)

# 4. 回调函数
eval_callback = EvalCallback(
    env,
    best_model_save_path='./rl_checkpoints/',
    log_path='./rl_logs/',
    eval_freq=1000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path='./rl_checkpoints/',
    name_prefix='rl_stenosis'
)

# 5. 训练
model.learn(
    total_timesteps=100000,
    callback=[eval_callback, checkpoint_callback]
)

# 6. 保存最终模型
model.save("rl_stenosis_final")
```

---

## 📊 评估和分析

### 对比实验

```python
def compare_rl_vs_fixed():
    """
    对比RL策略 vs 固定参数
    """
    test_dataset = load_test_dataset()
    
    # 1. 固定参数（原始Statistical）
    results_fixed = []
    for sample in test_dataset:
        detector = StenosisDetector(sample['image'], sample['mask'])
        detector.run_full_detection(
            stenosis_threshold=0.25,  # 固定
            min_avg_radius=4          # 固定
        )
        results_fixed.append(evaluate(detector.all_stenosis_points, sample['gt']))
    
    # 2. RL自适应策略
    model = PPO.load("rl_stenosis_final")
    results_rl = []
    
    for sample in test_dataset:
        env = StenosisDetectionEnv(...)
        state = env.reset()
        action, _ = model.predict(state, deterministic=True)
        
        # 执行RL选择的动作
        _, reward, _, info = env.step(action)
        results_rl.append(info['metrics'])
    
    # 3. 对比
    print("=" * 60)
    print("RL vs Fixed Parameters")
    print("=" * 60)
    print(f"Fixed Params:")
    print(f"  Sensitivity: {np.mean([r['sensitivity'] for r in results_fixed]):.2%}")
    print(f"  Precision: {np.mean([r['precision'] for r in results_fixed]):.2%}")
    print(f"  F1-Score: {np.mean([r['f1_score'] for r in results_fixed]):.3f}")
    print()
    print(f"RL Adaptive:")
    print(f"  Sensitivity: {np.mean([r['sensitivity'] for r in results_rl]):.2%}")
    print(f"  Precision: {np.mean([r['precision'] for r in results_rl]):.2%}")
    print(f"  F1-Score: {np.mean([r['f1_score'] for r in results_rl]):.3f}")
    print()
    print(f"Improvement:")
    print(f"  F1-Score: +{(np.mean([r['f1_score'] for r in results_rl]) - np.mean([r['f1_score'] for r in results_fixed])) * 100:.1f}%")
```

### 策略分析

```python
def analyze_learned_strategy(model, test_dataset):
    """
    分析Agent学到的策略
    """
    actions_by_quality = {
        'high_quality': [],
        'medium_quality': [],
        'low_quality': []
    }
    
    for sample in test_dataset:
        env = StenosisDetectionEnv(...)
        state = env.reset()
        action, _ = model.predict(state)
        
        # 根据mask质量分类
        quality = assess_mask_quality(sample['mask'])
        actions_by_quality[quality].append(action)
    
    # 可视化策略
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist([a[0] for a in actions_by_quality['high_quality']], label='High')
    plt.hist([a[0] for a in actions_by_quality['low_quality']], label='Low')
    plt.xlabel('Stenosis Threshold')
    plt.legend()
    plt.title('Threshold Selection by Mask Quality')
    
    plt.subplot(132)
    plt.hist([a[1] for a in actions_by_quality['high_quality']], label='High')
    plt.hist([a[1] for a in actions_by_quality['low_quality']], label='Low')
    plt.xlabel('Min Avg Radius')
    plt.legend()
    plt.title('Radius Selection by Mask Quality')
    
    plt.tight_layout()
    plt.savefig('rl_strategy_analysis.pdf')
```

---

## 📈 预期结果

### 定量提升

| 方法 | Sensitivity | Precision | F1-Score | 备注 |
|------|------------|-----------|----------|------|
| Statistical (固定0.25, 4) | 3.24% | 5.62% | 0.041 | Baseline |
| Statistical (RL优化后) | **20-30%** | **30-40%** | **0.25-0.35** | 自适应参数 |
| 提升倍数 | **6-9x** | **5-7x** | **6-8x** | 显著提升 |

### 定性发现

**Agent学到的策略**:
1. **高质量Mask**: 使用严格参数（threshold≈0.25-0.30）
2. **中等质量**: 适度放宽（threshold≈0.18-0.22）
3. **低质量Mask**: 更宽松或选择跳过（threshold≈0.12-0.15）

---

## 🎓 论文价值

### 创新点

1. **首次应用RL于Statistical狭窄检测**
   - 医学图像分析领域的新方向
   - 自适应参数选择的范式

2. **针对数据质量的鲁棒性**
   - 针对Mask质量问题的实用解决方案
   - 可解释的策略学习

3. **与深度学习互补**
   - 保留Statistical方法的可解释性
   - 提升性能接近深度学习

### 论文标题

"Reinforcement Learning-based Adaptive Parameter Selection for Statistical Stenosis Detection in Coronary Angiography"

### 投稿目标

- **IEEE TMI** (IF ~11, 顶刊)
- **Medical Image Analysis** (IF ~11, 顶刊)
- **MICCAI** 会议（顶会）

---

## ⏰ 实现时间表

### Month 1-2: 环境搭建
- Week 1-2: 实现Environment
- Week 3-4: 设计奖励函数和状态特征

### Month 3-4: RL训练
- Week 5-7: PPO训练和调优
- Week 8: 初步评估

### Month 5: 分析和实验
- Week 9-10: 策略分析和可视化
- Week 11: 对比实验（vs固定参数）
- Week 12: 消融实验

### Month 6: 论文撰写
- Week 13-14: 方法和实验
- Week 15: Introduction和Related Work
- Week 16: 润色和投稿

---

## 💡 技术难点和解决方案

### 难点1: 稀疏奖励
**问题**: 很多图像检测为0，难以学习

**解决**:
- 奖励shaping（中间奖励）
- Curriculum learning（从易到难）
- 使用预训练策略

### 难点2: 样本效率
**问题**: 每个episode需要完整检测，计算昂贵

**解决**:
- 并行环境（4-8个）
- 经验回放
- 使用较小数据集先训练

### 难点3: 动作空间大
**问题**: 8维连续+离散动作

**解决**:
- 先简化为核心2-3个参数
- 逐步增加动作维度
- 使用Hierarchical RL

---

## 🚀 下一步

我可以立即帮你：
1. ✅ 完整的Environment代码实现
2. ✅ PPO训练脚本
3. ✅ 评估和可视化代码
4. ✅ 与Statistical baseline对比

**要开始实现吗？** 🎯
