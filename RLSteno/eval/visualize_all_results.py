#!/usr/bin/env python3
"""
可视化所有训练结果
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (16, 10)
sns.set_style("whitegrid")


def plot_all_results():
    """绘制所有结果对比图"""
    
    # 数据
    models = ['Fixed\nParams', 'Baseline\n30k', 'Baseline\n80k', 'Compre-\nhensive', 'v3\nAggressive']
    f1_scores = [0.038, 0.047, 0.072, 0.095, 0.069]
    sensitivity = [5, 28, 34, 42, 32]
    precision = [3.5, 2.4, 3.4, 4.5, 2.7]
    tp_counts = [5, 14, 17, 21, 16]
    fp_counts = [125, 569, 486, 449, 567]
    fn_counts = [95, 36, 33, 29, 34]
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. F1-Score对比
    ax1 = plt.subplot(3, 3, 1)
    colors = ['gray', 'lightblue', 'blue', 'green', 'red']
    bars = ax1.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0.10, color='orange', linestyle='--', linewidth=2, label='Target (0.10)')
    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
        
        # 标注最佳
        if i == 3:  # Comprehensive
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    '★ Best', ha='center', va='bottom',
                    color='green', fontweight='bold', fontsize=10)
    
    # 2. Sensitivity对比
    ax2 = plt.subplot(3, 3, 2)
    bars = ax2.bar(models, sensitivity, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Sensitivity (%)', fontweight='bold')
    ax2.set_title('Sensitivity Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, sensitivity):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Precision对比
    ax3 = plt.subplot(3, 3, 3)
    bars = ax3.bar(models, precision, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Precision (%)', fontweight='bold')
    ax3.set_title('Precision Comparison', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, precision):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. TP/FP/FN堆叠图
    ax4 = plt.subplot(3, 3, 4)
    x = np.arange(len(models))
    width = 0.6
    
    p1 = ax4.bar(x, tp_counts, width, label='TP', color='green', alpha=0.8)
    p2 = ax4.bar(x, fp_counts, width, bottom=tp_counts, label='FP', color='red', alpha=0.8)
    p3 = ax4.bar(x, fn_counts, width, bottom=np.array(tp_counts)+np.array(fp_counts),
                label='FN', color='orange', alpha=0.8)
    
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('Detection Breakdown', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. FP对比（单独放大）
    ax5 = plt.subplot(3, 3, 5)
    bars = ax5.bar(models, fp_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax5.axhline(y=300, color='orange', linestyle='--', linewidth=2, label='Target (<300)')
    ax5.set_ylabel('False Positives', fontweight='bold')
    ax5.set_title('False Positive Count', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, fp_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}',
                ha='center', va='bottom', fontweight='bold')
    
    # 6. 训练曲线（F1 vs 步数）
    ax6 = plt.subplot(3, 3, 6)
    
    # Baseline训练曲线
    baseline_steps = [0, 15000, 25000, 30000, 60000, 80000]
    baseline_f1 = [0.038, 0.040, 0.050, 0.047, 0.069, 0.072]
    
    # Comprehensive训练曲线
    comp_steps = [0, 5000, 10000, 15000, 20000]
    comp_f1 = [0.038, 0.020, 0.040, 0.070, 0.095]
    
    # v3训练曲线
    v3_steps = [0, 5000, 10000, 15000, 20000]
    v3_f1 = [0.038, 0.010, 0.030, 0.050, 0.069]
    
    ax6.plot(baseline_steps, baseline_f1, 'o-', linewidth=2, markersize=8,
            label='Baseline', color='blue')
    ax6.plot(comp_steps, comp_f1, 's-', linewidth=2, markersize=8,
            label='Comprehensive', color='green')
    ax6.plot(v3_steps, v3_f1, '^-', linewidth=2, markersize=8,
            label='v3 Aggressive', color='red')
    
    ax6.axhline(y=0.10, color='orange', linestyle='--', linewidth=2,
               alpha=0.5, label='Target')
    ax6.set_xlabel('Training Steps', fontweight='bold')
    ax6.set_ylabel('F1-Score', fontweight='bold')
    ax6.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 改进幅度
    ax7 = plt.subplot(3, 3, 7)
    improvements = [(f1 - 0.038) / 0.038 * 100 for f1 in f1_scores[1:]]
    models_improve = models[1:]
    colors_improve = colors[1:]
    
    bars = ax7.bar(models_improve, improvements, color=colors_improve,
                  edgecolor='black', linewidth=1.5)
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax7.set_ylabel('Improvement (%)', fontweight='bold')
    ax7.set_title('F1 Improvement vs Fixed Params', fontsize=14, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'+{val:.0f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 8. Sensitivity vs Precision散点图
    ax8 = plt.subplot(3, 3, 8)
    
    for i, (s, p, model) in enumerate(zip(sensitivity, precision, models)):
        ax8.scatter(s, p, s=300, c=[colors[i]], edgecolor='black',
                   linewidth=2, alpha=0.8, label=model.replace('\n', ' '))
    
    ax8.set_xlabel('Sensitivity (%)', fontweight='bold')
    ax8.set_ylabel('Precision (%)', fontweight='bold')
    ax8.set_title('Sensitivity vs Precision Trade-off', fontsize=14, fontweight='bold')
    ax8.legend(loc='best', fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # 理想点
    ax8.scatter([40], [20], s=500, marker='*', c='gold',
               edgecolor='red', linewidth=2, label='Ideal Target')
    
    # 9. 总结表格
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 创建总结表格
    summary_data = []
    for i, model in enumerate(models):
        summary_data.append([
            model.replace('\n', ' '),
            f"{f1_scores[i]:.3f}",
            f"{sensitivity[i]}%",
            f"{precision[i]:.1f}%",
            f"{fp_counts[i]}"
        ])
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Model', 'F1', 'Sens', 'Prec', 'FP'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 高亮最佳行
    for i in range(len(models)):
        if i == 3:  # Comprehensive
            for j in range(5):
                table[(i+1, j)].set_facecolor('lightgreen')
                table[(i+1, j)].set_text_props(weight='bold')
    
    ax9.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Complete Training Results Analysis',
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('./training_results_complete.png', dpi=300, bbox_inches='tight')
    print("✅ 保存完整结果图: ./training_results_complete.png")
    
    plt.close()
    
    # 生成奖励函数对比图
    plot_reward_functions()


def plot_reward_functions():
    """绘制奖励函数特性对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # FP惩罚曲线对比
    ax1 = axes[0]
    fp_range = np.linspace(0, 800, 1000)
    
    baseline_penalty = fp_range * 0.3
    comp_penalty = fp_range ** 1.5
    v3_penalty = fp_range ** 1.8
    
    ax1.plot(fp_range, baseline_penalty, linewidth=3, label='Baseline: FP×0.3', color='blue')
    ax1.plot(fp_range, comp_penalty, linewidth=3, label='Comprehensive: FP^1.5', color='green')
    ax1.plot(fp_range, v3_penalty, linewidth=3, label='v3: FP^1.8', color='red')
    
    # 标注实际FP点
    ax1.scatter([486], [486*0.3], s=200, c='blue', marker='o', edgecolor='black', linewidth=2)
    ax1.scatter([449], [449**1.5], s=200, c='green', marker='s', edgecolor='black', linewidth=2)
    ax1.scatter([567], [567**1.8], s=200, c='red', marker='^', edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('False Positives (FP)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Penalty', fontweight='bold', fontsize=12)
    ax1.set_title('FP Penalty Function Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 30000)
    
    # 标注
    ax1.text(486, 486*0.3 + 1000, 'Baseline\nFP=486', ha='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax1.text(449, 449**1.5 + 1000, 'Comprehensive\nFP=449\n★ Best', ha='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax1.text(567, 567**1.8 + 1500, 'v3\nFP=567', ha='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral'))
    
    # 奖励vs性能散点图
    ax2 = axes[1]
    
    # 训练奖励（估算）
    training_rewards = [-5, -73, -40]  # Baseline, v3, Comprehensive
    final_f1 = [0.072, 0.069, 0.095]
    colors_scatter = ['blue', 'red', 'green']
    labels_scatter = ['Baseline', 'v3 Aggressive', 'Comprehensive']
    markers = ['o', '^', 's']
    
    for i, (reward, f1, color, label, marker) in enumerate(zip(training_rewards, final_f1,
                                                                colors_scatter, labels_scatter, markers)):
        ax2.scatter(reward, f1, s=400, c=color, marker=marker,
                   edgecolor='black', linewidth=2, alpha=0.8, label=label)
    
    ax2.set_xlabel('Training Reward (more negative = harsher)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Final F1-Score', fontweight='bold', fontsize=12)
    ax2.set_title('Reward Harshness vs Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 添加注释
    ax2.annotate('Too harsh!\nGradient issues', xy=(-73, 0.069), xytext=(-65, 0.055),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax2.annotate('Optimal balance!', xy=(-40, 0.095), xytext=(-50, 0.105),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./reward_function_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 保存奖励函数分析图: ./reward_function_analysis.png")
    
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("生成可视化图表...")
    print("=" * 60)
    
    plot_all_results()
    
    print("\n" + "=" * 60)
    print("✅ 所有图表生成完成！")
    print("=" * 60)
    print("\n生成的文件：")
    print("  1. training_results_complete.png - 完整结果对比（9子图）")
    print("  2. reward_function_analysis.png  - 奖励函数分析")
    print("\n查看命令：")
    print("  eog training_results_complete.png")
    print("  eog reward_function_analysis.png")
