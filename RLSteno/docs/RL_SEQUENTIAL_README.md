# 🚀 序列决策RL狭窄检测系统

**基于深度强化学习(PPO-LSTM)的血管狭窄序列决策检测**

根据 `RL思路.md` 实现的完整系统，将狭窄检测建模为**序列决策问题**，Agent沿血管中心线逐点判断，使用LSTM记忆模块区分"渐变"和"突变"。

---

## 📦 项目文件

| 文件 | 说明 |
|------|------|
| `rl_sequential_stenosis_env.py` | 序列决策环境（MDP建模） |
| `rl_ppo_lstm_policy.py` | PPO-LSTM策略网络 |
| `train_sequential_rl_agent.py` | 训练脚本 |
| `evaluate_sequential_rl_agent.py` | 评估脚本（对比Statistical Baseline） |
| `run_rl_stenosis_detection.sh` | 快速启动脚本 ⭐ |
| `RL_SEQUENTIAL_USAGE_GUIDE.md` | 详细使用指南 📚 |

---

## ⚡ 快速开始（3步）

### Step 1: 准备数据

确保已运行 `batch_inference_xzjc.py` 生成血管分割mask：

```bash
# 检查数据结构
ls /mnt/sda1/luoyu/xzjc_data/dataset/*.bmp  # 原始图像
ls /mnt/sda1/luoyu/xzjc_data/dataset/*.xml  # 狭窄标注
ls /mnt/sda1/luoyu/xzjc_data/masks/*.png    # 分割mask
```

### Step 2: 训练RL Agent（后台运行推荐）

```bash
# 方式1: 后台训练（推荐）⭐
./run_rl_stenosis_detection.sh --mode train --timesteps 50000 --background

# 方式2: 前台训练
./run_rl_stenosis_detection.sh --mode train --timesteps 50000

# 方式3: 直接调用Python
python train_sequential_rl_agent.py --total_timesteps 50000
```

**预计时间**: 30-60分钟（取决于数据集大小）

**监控训练**: 
```bash
# 使用监控工具
./monitor_rl_training.sh

# 或查看日志
tail -f rl_sequential_logs/run_*/training.log
```

### Step 3: 评估性能

```bash
# 评估最佳模型（自动对比Statistical Baseline）
./run_rl_stenosis_detection.sh --mode evaluate \
    --model_path rl_sequential_logs/run_20231129_150230/best_model/best_model.zip
```

**输出**:
- `rl_evaluation_results/comparison_report.txt` - 详细对比报告
- `rl_evaluation_results/comparison_chart.png` - 可视化对比图

---

## 🎯 核心创新点

### 1. 序列决策建模

传统方法使用**滑动窗口**，只看局部信息。我们的方法：

```
传统: [窗口1] [窗口2] [窗口3] ... (独立判断)
       ↓       ↓       ↓
      是/否   是/否   是/否

RL: 起点 → [点1] → [点2] → [点3] → ... → 终点
           ↓       ↓       ↓
          LSTM记忆全局趋势，连续决策
```

### 2. LSTM记忆模块

**问题**: 如何区分"自然变细"和"病理狭窄"？

**解决**: LSTM记住血管走过的历史：
- 如果血管从10px慢慢变成5px（渐变） → LSTM认为正常
- 如果血管从10px突变成3px（突变） → LSTM识别出狭窄

### 3. 奖励塑造

医疗AI的特殊考虑：

| 情况 | 奖励 | 原因 |
|------|------|------|
| TP (正确检测) | +1.0 | 基础奖励 |
| FP (误报) | -2.0 | 严厉惩罚（避免过度检测） |
| FN (漏检) | -2.0 | **医疗AI大忌**（漏掉病变） |
| TN (正确放过) | +0.1 | 鼓励谨慎判断 |
| 连续性 | +0.2 | 狭窄通常是连续区域 |

---

## 📊 预期性能

基于RL思路.md的预期（相比Statistical Baseline）：

| 指标 | Statistical Baseline | **RL Agent (PPO-LSTM)** ⭐ |
|------|---------------------|---------------------------|
| Sensitivity | 75-85% | **84-88%** (+4-8%) |
| Precision | 70-80% | **88-92%** (+12-18%) |
| F1-Score | 72-82% | **86-90%** (+8-14%) |
| 误报率(FP) | 高 | **显著降低** (-30-50%) |

**关键优势**: 大幅降低误报率，同时保持高召回率

---

## 🔬 消融实验（Ablation Study）

验证LSTM模块的有效性：

```bash
# 实验A: PPO (无LSTM)
python train_sequential_rl_agent.py --lstm_hidden_size 0 --log_dir rl_logs_no_lstm

# 实验B: PPO-LSTM (完整版)
python train_sequential_rl_agent.py --lstm_hidden_size 128 --log_dir rl_logs_with_lstm

# 对比结果
python evaluate_sequential_rl_agent.py --model_path rl_logs_no_lstm/best_model/best_model.zip
python evaluate_sequential_rl_agent.py --model_path rl_logs_with_lstm/best_model/best_model.zip
```

**预期发现**: LSTM版本在处理渐变血管时误报率显著降低

---

## 🛠️ 自定义配置

### 调整窗口大小

```bash
# 小窗口（更敏感，适合细节）
python train_sequential_rl_agent.py --window_size 3

# 大窗口（更稳定，适合粗血管）
python train_sequential_rl_agent.py --window_size 7
```

### 调整奖励权重

编辑 `rl_sequential_stenosis_env.py` 的 `_compute_reward()` 方法：

```python
# 增加对漏检的惩罚
if is_near_gt and action == 0:
    reward = -3.0  # 原来是 -2.0

# 增加对误报的惩罚
if not is_near_gt and action == 1:
    reward = -3.0  # 原来是 -2.0
```

### 多GPU并行训练

```bash
# 使用4个并行环境加速训练
python train_sequential_rl_agent.py \
    --n_envs 4 \
    --total_timesteps 200000
```

---

## 📚 论文写作建议

### Abstract

```
We propose a novel deep reinforcement learning approach for vessel 
stenosis detection by formulating it as a sequential decision-making 
problem. Our PPO-LSTM agent navigates along the vessel centerline, 
leveraging LSTM memory to distinguish natural tapering from pathological 
stenosis. Experiments show X% improvement in F1-score and Y% reduction 
in false positives compared to statistical baselines.
```

### 核心贡献点

1. ✅ **首次将狭窄检测建模为序列决策问题**
2. ✅ **LSTM记忆模块区分渐变和突变**
3. ✅ **医疗特定的奖励塑造（重视FN惩罚）**
4. ✅ **显著降低误报率（提高临床可用性）**

### 必备实验

- [x] 对比实验: RL vs Statistical Baseline
- [x] 消融实验: PPO vs PPO-LSTM
- [ ] 泛化实验: 跨数据集测试（可选）
- [ ] 可视化: 展示Agent决策过程

---

## 🐛 故障排除

### 问题1: 训练不收敛（F1一直很低）

**检查**:
```bash
# 1. 验证数据集有Ground Truth
python -c "from rl_sequential_stenosis_env import SequentialStenosisEnv; \
           env = SequentialStenosisEnv(); \
           print(f'样本数: {len(env.samples)}'); \
           print(f'GT数量: {len(env.samples[0][\"ground_truth\"])}')"

# 2. 检查奖励是否合理
# 编辑 rl_sequential_stenosis_env.py 添加打印语句
```

### 问题2: GPU内存不足

```bash
# 使用CPU训练
export CUDA_VISIBLE_DEVICES=""
python train_sequential_rl_agent.py --n_envs 1
```

### 问题3: Mask文件缺失

```bash
# 生成Mask
cd /mnt/sda1/luoyu/SAM-VMNet
python batch_inference_xzjc.py
```

---

## 📖 详细文档

更多信息请查看：

- **[RL_SEQUENTIAL_USAGE_GUIDE.md](RL_SEQUENTIAL_USAGE_GUIDE.md)** - 完整使用指南
- **[RL思路.md](RL思路.md)** - 理论设计文档
- **[STATISTICAL_ALGORITHM_AND_SENSITIVITY.md](STATISTICAL_ALGORITHM_AND_SENSITIVITY.md)** - Baseline算法原理

---

## 🎓 理论参考

1. **PPO算法**: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
2. **LSTM网络**: [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)
3. **医疗RL**: [Gottesman et al., 2019](https://arxiv.org/abs/1805.12298)

---

## ✨ 快速命令参考

```bash
# 训练（快速验证）
./run_rl_stenosis_detection.sh --mode train --timesteps 50000

# 训练（完整）
./run_rl_stenosis_detection.sh --mode train --timesteps 200000 --n_envs 4

# 评估
./run_rl_stenosis_detection.sh --mode evaluate \
    --model_path rl_sequential_logs/run_xxx/best_model/best_model.zip

# 查看训练曲线
xdg-open rl_sequential_logs/run_xxx/training_curves.png

# TensorBoard
tensorboard --logdir rl_sequential_logs/run_xxx/tensorboard

# 查看评估结果
cat rl_evaluation_results/comparison_report.txt
xdg-open rl_evaluation_results/comparison_chart.png
```

---

## 📧 联系

如有问题，请检查：
1. 数据集格式是否正确（.bmp + .xml + mask）
2. 依赖是否安装完整（gymnasium, stable-baselines3, torch）
3. 查看详细日志文件

**Happy Researching! 🎯🚀**
