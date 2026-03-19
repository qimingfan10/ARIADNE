#!/usr/bin/env python3
"""
Sa2VA V11 - 与predict_forward一致的训练路径

核心改进：
1. 使用与predict_forward完全一致的推理路径
2. 只训练SAM2 mask decoder和text_hidden_fcs
3. 7:3混合采样 + Dice/BCE Loss
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import random

sys.path.insert(0, '/home/ubuntu/Sa2VA')

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def compute_dice(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)


def dice_loss(pred, target):
    return 1 - compute_dice(pred, target)


class ConsistentTrainer:
    """与predict_forward一致的训练器"""
    
    def __init__(self):
        # 从V8 DPO模型开始（Dice ~0.82）
        self.model_path = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v8/step_100"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        # 训练参数
        self.lr = 1e-6  # 极低学习率
        self.epochs = 2
        self.grad_accum = 4
        self.max_samples = 200
        
        # 7:3混合采样
        self.hard_threshold = 0.75
        self.easy_threshold = 0.85
        self.hard_ratio = 0.7
        self.easy_ratio = 0.3
        
        print("=" * 60)
        print("🚀 V11 Consistent Training")
        print("   与predict_forward一致的推理路径")
        print("=" * 60)
        
        self._load_model()
        self._load_mixed_data()
        self._setup_training()
    
    def _load_model(self):
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map='auto',
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()  # 保持eval模式
        
        # 初始化模型（必须先调用）
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        
        # 缓存关键token id
        self.seg_token_id = self.model.seg_token_idx
        
        # 冻结整个模型
        print("\n🔧 Freezing all parameters...")
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 只解冻text_hidden_fcs和SAM2 mask decoder
        trainable = 0
        
        for param in self.model.text_hidden_fcs.parameters():
            param.requires_grad = True
            trainable += param.numel()
        
        for name, param in self.model.grounding_encoder.named_parameters():
            if 'mask_decoder' in name or 'output_upscaling' in name:
                param.requires_grad = True
                trainable += param.numel()
        
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
    def _load_mixed_data(self):
        """加载7:3混合数据"""
        print("\n📊 Loading mixed data...")
        
        ann_file = f"{self.data_root}/dpo_annotations.json"
        with open(ann_file) as f:
            annotations = json.load(f)
        
        hard_samples = []
        easy_samples = []
        
        for item in annotations:
            if 'chosen_mask' not in item or 'rejected_mask' not in item:
                continue
            
            chosen_path = os.path.join(self.data_root, item['chosen_mask'])
            rejected_path = os.path.join(self.data_root, item['rejected_mask'])
            
            if not os.path.exists(chosen_path) or not os.path.exists(rejected_path):
                continue
            
            chosen = np.array(Image.open(chosen_path).convert('L')) > 127
            rejected = np.array(Image.open(rejected_path).convert('L')) > 127
            
            inter = (chosen & rejected).sum()
            baseline_dice = (2 * inter) / (chosen.sum() + rejected.sum() + 1e-8)
            
            sample = {
                'image': os.path.join(self.data_root, item['image']),
                'gt_mask': chosen_path,
                'baseline_dice': baseline_dice,
            }
            
            if baseline_dice < self.hard_threshold:
                hard_samples.append(sample)
            elif baseline_dice > self.easy_threshold:
                easy_samples.append(sample)
        
        print(f"   Hard (Dice < {self.hard_threshold}): {len(hard_samples)}")
        print(f"   Easy (Dice > {self.easy_threshold}): {len(easy_samples)}")
        
        n_hard = int(self.max_samples * self.hard_ratio)
        n_easy = int(self.max_samples * self.easy_ratio)
        
        random.shuffle(hard_samples)
        random.shuffle(easy_samples)
        
        self.samples = hard_samples[:n_hard] + easy_samples[:n_easy]
        random.shuffle(self.samples)
        
        print(f"   Mixed: {len(self.samples)} samples")
    
    def _setup_training(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        
        total_steps = len(self.samples) * self.epochs // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-8
        )
    
    def _prepare_inputs(self, image):
        """准备输入（与predict_forward完全一致）"""
        # dynamic_preprocess定义在modeling_sa2va_chat.py中
        import importlib
        modeling_module = importlib.import_module(
            'transformers_modules.sa2va_vessel_hf.modeling_sa2va_chat'
        )
        dynamic_preprocess = modeling_module.dynamic_preprocess
        
        ori_size = image.size
        
        # 准备SAM2输入 (g_pixel_values)
        g_image = np.array(image)
        g_image = self.model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.model.torch_dtype)
        g_pixel_values = self.model.grounding_encoder.preprocess_image(g_pixel_values).unsqueeze(0)
        
        # 准备Vision输入 (pixel_values) - 使用dynamic_preprocess
        images = dynamic_preprocess(
            image, 
            self.model.min_dynamic_patch,
            self.model.max_dynamic_patch,
            self.model.image_size, 
            self.model.use_thumbnail
        )
        pixel_values = [self.model.transformer(img) for img in images]
        pixel_values = torch.stack(pixel_values).to(self.model.torch_dtype)
        num_image_tokens = pixel_values.shape[0] * self.model.patch_token
        
        return {
            'pixel_values': pixel_values,
            'g_pixel_values': g_pixel_values,
            'num_image_tokens': num_image_tokens,
            'ori_size': ori_size,
            'vp_overall_mask': None,
        }
    
    def _get_seg_embedding_from_generate(self, image, image_inputs):
        """通过generate获取[SEG] embedding"""
        text = '<image>Please segment the blood vessel.'
        
        # 构建输入（与predict_forward一致）
        num_image_tokens = image_inputs['num_image_tokens']
        image_token_str = (
            f'{self.model.IMG_START_TOKEN}'
            f'{self.model.IMG_CONTEXT_TOKEN * num_image_tokens}'
            f'{self.model.IMG_END_TOKEN}'
        )
        
        text = text.replace('<image>', image_token_str + '\n')
        input_text = self.model.template['INSTRUCTION'].format(
            input=text, round=1, bot_name=self.model.bot_name
        )
        
        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        attention_mask = torch.ones_like(ids, dtype=torch.bool)
        
        mm_inputs = {
            'pixel_values': image_inputs['pixel_values'],
            'input_ids': ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': None,
            'prompt_masks': None,
            'vp_overall_mask': image_inputs.get('vp_overall_mask'),
        }
        
        # Generate (LLM冻结，不需要梯度)
        with torch.no_grad():
            generate_output = self.model.generate(
                **mm_inputs,
                generation_config=self.model.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.model.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        # 获取[SEG] hidden states（使用官方实现）
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        
        output_ids = generate_output.sequences[0][:-1]
        seg_mask = output_ids == self.seg_token_id
        n_out = len(seg_mask)
        
        if seg_mask.sum() == 0:
            return None
        
        # 使用官方的索引方式：取最后n_out个hidden states
        seg_mask = seg_mask.to(last_hidden_states.device)
        seg_hidden = last_hidden_states[-n_out:][seg_mask]
        
        # text_hidden_fcs需要梯度
        seg_embedding = self.model.text_hidden_fcs(seg_hidden)
        
        return seg_embedding
    
    def _forward_sam2_consistent(self, g_pixel_values, seg_embedding):
        """SAM2 forward（与predict_forward一致）"""
        # 获取SAM2 embeddings
        sam_states = self.model.grounding_encoder.get_sam2_embeddings(g_pixel_values)
        
        # 单帧、单对象
        seg_embedding = seg_embedding.unsqueeze(0)  # [1, C]
        language_embd = [[seg_embedding]]  # [[tensor]]
        
        # 使用language_embd_training（保持梯度）
        pred_masks = self._language_embd_with_grad(sam_states, language_embd)
        
        return pred_masks
    
    def _language_embd_with_grad(self, inference_state, language_embd):
        """带梯度的language embedding推理"""
        sam2 = self.model.grounding_encoder.sam2_model
        
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame_idx in range(num_frame):
                for obj_idx in range(num_obj):
                    _language_embd = language_embd[frame_idx][obj_idx][None][None]
                    # add_language_embd内部会更新inference_state
                    _, _, out_mask_logits = sam2.add_language_embd(
                        inference_state,
                        frame_idx,
                        obj_idx + 100,
                        _language_embd,
                        inference=True,
                    )
            
            # propagate获取最终mask
            mask_out = []
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2.propagate_in_video(inference_state):
                mask_out.append(out_mask_logits)
            
            if mask_out:
                mask_out = torch.cat(mask_out, dim=0)
            else:
                mask_out = out_mask_logits
        
        return mask_out
    
    def train_step(self, sample):
        """单步训练"""
        try:
            image = Image.open(sample['image']).convert('RGB')
            gt_mask = np.array(Image.open(sample['gt_mask']).convert('L')) > 127
            gt_mask = torch.from_numpy(gt_mask.astype(np.float32))
            w, h = image.size
            
            # 准备输入
            image_inputs = self._prepare_inputs(image)
            
            # 获取seg embedding（通过generate，但text_hidden_fcs有梯度）
            seg_embedding = self._get_seg_embedding_from_generate(image, image_inputs)
            if seg_embedding is None:
                return None
            
            # SAM2 forward
            pred_masks = self._forward_sam2_consistent(
                image_inputs['g_pixel_values'], 
                seg_embedding[0]  # 取第一个seg
            )
            
            if pred_masks is None:
                return None
            
            # 调整尺寸
            pred_logits = F.interpolate(
                pred_masks.float(),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            pred_prob = torch.sigmoid(pred_logits)
            gt_mask = gt_mask.to(pred_prob.device)
            
            # Loss: Dice + BCE
            d_loss = dice_loss(pred_prob, gt_mask)
            bce_loss = F.binary_cross_entropy(pred_prob, gt_mask)
            loss = d_loss + 0.5 * bce_loss
            
            dice = compute_dice((pred_prob > 0.5).float(), gt_mask)
            
            return {'loss': loss, 'dice': dice.item()}
            
        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()
            return None
    
    def train(self):
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {self.epochs}, LR: {self.lr}")
        
        for epoch in range(self.epochs):
            print(f"\n📌 Epoch {epoch+1}/{self.epochs}")
            
            metrics = {'loss': 0, 'dice': 0, 'count': 0}
            random.shuffle(self.samples)
            
            pbar = tqdm(self.samples, desc=f"Epoch {epoch+1}")
            
            for idx, sample in enumerate(pbar):
                result = self.train_step(sample)
                
                if result is None:
                    continue
                
                loss = result['loss'] / self.grad_accum
                if loss.requires_grad:
                    loss.backward()
                
                metrics['loss'] += result['loss'].item()
                metrics['dice'] += result['dice']
                metrics['count'] += 1
                
                if (idx + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    if metrics['count'] > 0:
                        c = metrics['count']
                        pbar.set_postfix({
                            'loss': f"{metrics['loss']/c:.4f}",
                            'dice': f"{metrics['dice']/c:.4f}",
                        })
        
        # 评估
        self._evaluate()
        
        print("\n" + "=" * 60)
        print("🎉 Training completed!")
        print("=" * 60)
    
    def _evaluate(self):
        """使用predict_forward评估"""
        print("\n📊 Evaluating with predict_forward...")
        self.model.eval()
        
        eval_root = "/home/ubuntu/Sa2VA/data/merged_vessel_data"
        with open(f"{eval_root}/annotations.json") as f:
            anns = json.load(f)
        
        valid = [a for a in anns if 'mask' in a and len(a['mask']) > 0]
        random.seed(42)
        eval_samples = random.sample(valid, 10)
        
        dices = []
        for s in tqdm(eval_samples, desc="Eval"):
            try:
                img = Image.open(f"{eval_root}/images/{s['image']}").convert('RGB')
                w, h = img.size
                
                gt = np.zeros((h, w), dtype=np.uint8)
                for m in s['mask']:
                    if len(m) >= 6:
                        pts = np.array(m).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(gt, [pts], 255)
                
                # 使用predict_forward
                with torch.no_grad():
                    out = self.model.predict_forward(
                        image=img,
                        text='<image>Please segment the blood vessel.',
                        tokenizer=self.tokenizer,
                        processor=None,
                    )
                
                pred = out['prediction_masks'][0][0]
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                pred = pred.astype(np.float32)
                if pred.max() <= 1.0:
                    pred = pred * 255
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                
                pred_binary = (pred > 127).astype(np.uint8)
                gt_binary = (gt > 127).astype(np.uint8)
                
                inter = (pred_binary & gt_binary).sum()
                dice = (2 * inter) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
                dices.append(dice)
                
            except Exception as e:
                print(f"  Error: {e}")
        
        if dices:
            mean_dice = np.mean(dices)
            print(f"\n🎯 Results:")
            print(f"   Mean Dice: {mean_dice:.4f}")
            print(f"   Baseline: 0.8191")
            print(f"   Change: {mean_dice - 0.8191:+.4f}")


if __name__ == "__main__":
    trainer = ConsistentTrainer()
    trainer.train()
