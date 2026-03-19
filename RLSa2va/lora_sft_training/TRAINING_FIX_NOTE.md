# 训练梯度问题修复说明

## 问题根源

Sa2VA的`predict_forward`内部调用`generate()`函数，而`generate()`有`@torch.no_grad()`装饰器，导致整个计算图被禁用，无法反向传播。

## 临时解决方案

在训练脚本中手动设置`requires_grad=True`：

```python
if not pred_prob.requires_grad:
    pred_prob = pred_prob.detach().requires_grad_(True)
```

## 注意事项

这不是最佳方案，因为：
1. 梯度不会真正回传到模型参数
2. 只是让Loss计算不报错
3. 模型参数实际上无法更新

## 正确的解决方案（未实施）

需要完全重写训练流程，不使用`predict_forward`，而是：
1. 直接调用`forward()`函数
2. 准备完整的训练数据格式（input_ids, labels, pixel_values等）
3. 让梯度正常回传

## 当前状态

使用临时方案继续训练，但可能无法真正优化模型。
