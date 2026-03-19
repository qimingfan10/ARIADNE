# RL后台训练指南

## ✅ 问题已修复

1. **环境Bug修复**: 修正了 `StenosisDetector` 属性访问错误
   - 原来: `detector.skeleton_points` ❌
   - 修正: 从 `detector.point_data` 提取 ✅

2. **后台运行**: 所有训练现在支持后台运行 🚀

---

## 🚀 后台训练（推荐）

### 方式1: 使用快速脚本（最简单）

```bash
cd /mnt/sda1/luoyu/SAM-VMNet

# 后台训练（快速验证）
./run_rl_stenosis_detection.sh --mode train --timesteps 50000 --background

# 后台训练（完整）
./run_rl_stenosis_detection.sh --mode train --timesteps 200000 --n_envs 4 -b

# 短选项
./run_rl_stenosis_detection.sh --mode train --timesteps 100000 -b
```

### 方式2: 直接使用后台脚本

```bash
./train_sequential_rl_agent_background.sh \
    --total_timesteps 200000 \
    --n_envs 4
```

---

## 📊 监控训练

### 查看运行中的训练

```bash
# 使用监控工具（推荐）
./monitor_rl_training.sh

# 或手动查看日志
tail -f rl_sequential_logs/run_*/training.log

# 查看进程
ps aux | grep train_sequential_rl_agent.py
```

### 实时监控输出

```bash
# 自动选择最新的训练
tail -f $(ls -t rl_sequential_logs/run_*/training.log | head -1)

# 持续监控（每2秒刷新）
watch -n 2 'tail -20 $(ls -t rl_sequential_logs/run_*/training.log | head -1)'
```

---

## 🛑 停止训练

### 方式1: 使用停止脚本（推荐）

```bash
./stop_rl_training.sh

# 然后按提示选择要停止的训练
```

### 方式2: 手动停止

```bash
# 查找PID
cat rl_sequential_logs/run_*/training.pid

# 停止训练（优雅退出）
kill <PID>

# 强制停止（如果需要）
kill -9 <PID>
```

### 方式3: 停止所有训练

```bash
pkill -f train_sequential_rl_agent.py
```

---

## 📁 训练输出结构

```
rl_sequential_logs/run_TIMESTAMP/
├── training.log                 # 训练日志（后台）
├── training.pid                 # 进程PID
├── best_model/                  # 最佳模型
│   └── best_model.zip
├── final_model.zip              # 最终模型
├── checkpoints/                 # 定期保存
├── training_curves.png          # 训练曲线
├── training_metrics.npz         # 原始数据
└── tensorboard/                 # TensorBoard日志
```

---

## 🔍 检查训练状态

### 检查是否正在运行

```bash
# 方式1: 使用监控脚本
./monitor_rl_training.sh

# 方式2: 检查PID
for pidfile in rl_sequential_logs/run_*/training.pid; do
    pid=$(cat "$pidfile" 2>/dev/null)
    if ps -p $pid > /dev/null 2>&1; then
        echo "✅ 训练运行中 (PID: $pid)"
        echo "   目录: $(dirname $pidfile)"
    fi
done
```

### 检查训练进度

```bash
# 查看最新日志
tail -50 $(ls -t rl_sequential_logs/run_*/training.log | head -1)

# 搜索关键指标
grep -E "F1-Score|Sensitivity|Precision" \
    $(ls -t rl_sequential_logs/run_*/training.log | head -1) | tail -10

# 查看训练步数
grep "total_timesteps" \
    $(ls -t rl_sequential_logs/run_*/training.log | head -1)
```

---

## 📈 查看训练曲线

```bash
# 列出所有训练曲线
ls -lt rl_sequential_logs/run_*/training_curves.png

# 打开最新的曲线图
xdg-open $(ls -t rl_sequential_logs/run_*/training_curves.png | head -1)

# 或使用图像查看器
eog $(ls -t rl_sequential_logs/run_*/training_curves.png | head -1)
```

---

## 💡 最佳实践

### 1. 推荐的训练流程

```bash
# Step 1: 后台启动训练
./run_rl_stenosis_detection.sh \
    --mode train \
    --timesteps 200000 \
    --n_envs 4 \
    --background

# Step 2: 记录输出的PID和目录
# 输出示例：
#   PID: 12345
#   运行目录: rl_sequential_logs/run_20231129_150230

# Step 3: 定期检查进度
tail -f rl_sequential_logs/run_20231129_150230/training.log

# Step 4: 训练完成后评估
./run_rl_stenosis_detection.sh --mode evaluate \
    --model_path rl_sequential_logs/run_20231129_150230/best_model/best_model.zip
```

### 2. 多任务训练

```bash
# 同时训练多个配置
./run_rl_stenosis_detection.sh --mode train --timesteps 100000 --window_size 3 -b
sleep 2
./run_rl_stenosis_detection.sh --mode train --timesteps 100000 --window_size 5 -b
sleep 2
./run_rl_stenosis_detection.sh --mode train --timesteps 100000 --window_size 7 -b

# 监控所有训练
./monitor_rl_training.sh
```

### 3. 长时间训练

```bash
# 启动长训练（200k steps，约2-3小时）
./run_rl_stenosis_detection.sh \
    --mode train \
    --timesteps 200000 \
    --n_envs 4 \
    --background

# 关闭SSH也不会中断（使用nohup）
# 可以安全退出终端

# 稍后重新连接，检查状态
./monitor_rl_training.sh
```

---

## 🐛 故障排除

### 问题1: 训练一直重试初始化

**症状**: 
```
⚠ 初始化检测器失败: 'StenosisDetector' object has no attribute 'skeleton_points'，重试...
```

**解决**: 已修复！更新后的代码使用正确的属性 `point_data`

### 问题2: 后台训练没有输出

**检查日志文件**:
```bash
cat rl_sequential_logs/run_*/training.log
```

### 问题3: 找不到运行中的训练

```bash
# 检查所有Python进程
ps aux | grep python

# 检查训练脚本
ps aux | grep train_sequential_rl_agent.py

# 查看所有日志
ls -lt rl_sequential_logs/run_*/training.log
```

### 问题4: 训练意外停止

```bash
# 查看日志末尾
tail -100 rl_sequential_logs/run_*/training.log

# 检查是否有错误
grep -i error rl_sequential_logs/run_*/training.log
grep -i exception rl_sequential_logs/run_*/training.log
```

---

## 📞 快速命令参考

```bash
# 🚀 启动后台训练
./run_rl_stenosis_detection.sh --mode train --timesteps 100000 -b

# 📊 监控训练
./monitor_rl_training.sh

# 🛑 停止训练
./stop_rl_training.sh

# 📈 查看曲线
xdg-open $(ls -t rl_sequential_logs/run_*/training_curves.png | head -1)

# 🔍 查看日志
tail -f $(ls -t rl_sequential_logs/run_*/training.log | head -1)

# ✅ 评估模型
./run_rl_stenosis_detection.sh --mode evaluate \
    --model_path $(ls -t rl_sequential_logs/run_*/best_model/best_model.zip | head -1)
```

---

## 🎯 现在可以开始了！

```bash
cd /mnt/sda1/luoyu/SAM-VMNet

# 启动后台训练
./run_rl_stenosis_detection.sh --mode train --timesteps 50000 --background

# 等待几秒，然后监控
sleep 3
./monitor_rl_training.sh
```

**提示**: 按 `Ctrl+C` 退出监控，训练会继续在后台运行 ✅
