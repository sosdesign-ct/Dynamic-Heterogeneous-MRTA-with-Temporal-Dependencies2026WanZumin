from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class TrainConfig:
    seed: int = 42  # 训练随机种子
    total_steps: int = 1000000  # 总环境交互步数
    rollout_steps: int = 768  # 每次更新前采样步数
    num_envs: int = 12  # 并行环境数量（1 表示原始串行）
    update_epochs: int = 3  # 每轮 rollout 的优化轮数
    minibatch_size: int = 1536  # 迷你批大小（更大 batch 以降低新局部奖励带来的方差）
    max_episodes: int = 0  # 最大训练回合数，<=0 表示不限制

    lr: float = 7e-5  # 学习率
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE 参数
    clip_ratio: float = 0.12  # PPO 截断范围
    value_clip_ratio: float = 0.2  # Value function clipping (MAPPO/PPO2)
    entropy_coef: float = 0.005  # 熵正则系数
    entropy_coef_final: float = 0.0005 # 熵系数退火终点
    entropy_anneal_steps: int = 500000  # 熵系数线性退火步数，<=0 关闭退火
    value_coef: float = 0.25  # 价值损失系数（回报量纲较大时可适当减小）
    aux_loss_coef: float = 0.2  # 辅助可行性/依赖监督损失系数
    aux_task_start_coef: float = 1.0
    aux_task_ready_coef: float = 0.5
    aux_pair_feasible_coef: float = 1.0
    noop_feasible_loss_coef: float = 0.04  # 有可行动作时直接压低 NOOP 概率，补足奖励信号滞后
    noop_feasible_adv_bonus: float = 0.25  # 对“可行动作却 NOOP”的负优势额外放大，缓解停滞策略
    noop_feasible_loss_anneal_steps: int = 300000  # NOOP 抑制项线性退火步数，后期更多交给环境奖励
    noop_feasible_loss_floor_frac: float = 0.25  # 退火后保留的 NOOP 辅助比例，避免续训阶段完全失去约束
    normalize_value_loss: bool = True  # 是否按 rollout 回报标准差归一化 value loss
    value_norm_eps: float = 1e-6  # value loss 归一化稳定项
    use_huber_value_loss: bool = True  # critic 是否使用 Huber loss（比 MSE 更抗异常回报）
    huber_delta: float = 1.0  # Huber loss delta
    target_kl: float = 0.02  # PPO 近似 KL 早停阈值，<=0 表示关闭
    max_grad_norm: float = 0.45  # 梯度裁剪阈值

    # 收敛判定（按更新轮次）
    convergence_window: int = 20  # 滚动窗口大小
    convergence_patience: int = 30  # 无改进容忍轮数
    convergence_min_delta: float = 0.01  # 认为有改进的最小幅度
    convergence_return_std_max: float = 0.1  # 收敛时窗口回报方差上限
    convergence_min_completion: float = 0.8  # 收敛时最小完成率
    convergence_max_reject_rate: float = 0.25  # 收敛时最大拒绝率
    convergence_min_success_rate: float = 0.0  # 已弃用：收敛不再依赖 strict success 判据
    convergence_max_gen_makespan: float = 40.0  # 收敛时广义makespan上限（越小越好）
    convergence_max_violation_rate: float = 0.35  # 收敛时最大违例率

    # 基于 step 的收敛判定：比较前后两个窗口（各 step_convergence_window 步）
    enable_step_convergence: bool = False
    step_convergence_window: int = 10000
    step_convergence_min_delta: float = 0.01
    step_convergence_std_max: float = 0.1
    post_convergence_steps: int = 15000  # 判定收敛后继续训练的步数

    hidden_dim: int = 128  # 策略网络隐藏维度
    rgat_layers: int = 2  # RGAT 层数
    rgat_heads: int = 4  # 预留：RGAT 头数（当前实现未使用）
    use_graph_transformer: bool = True  # 默认使用关系感知 Graph Transformer 任务编码器
    graph_transformer_heads: int = 4  # Graph Transformer 注意力头数

    # 运行与日志
    log_every: int = 10
    tb: bool = True
    tb_logdir: str = "runs/MRTA_new"
    tb_flush_secs: int = 10
    verbose_episode_log: bool = False

    # checkpoint
    save_path: str = "model/save/mrta_checkpoint.pt"
    save_every_updates: int = 10
    resume_path: str = ""

    # 环境并行与自动调优
    env_backend: str = "subproc"  # auto|sync|subproc
    env_start_method: str = "fork"  # spawn|fork|forkserver
    auto_tune_runtime: bool = True
    auto_tune_seconds: float = 20.0
    auto_tune_rollout_steps: int = 48
    auto_tune_max_cases: int = 8
    torch_num_threads: int = 8
    torch_num_interop_threads: int = 1

    # 模块开关（消融）
    wo_rgat: bool = False
    wo_mask: bool = False
    wo_commitment: bool = False

    # 在完整难度域上均衡采样；首轮宽域实验关闭额外课程偏置。
    curriculum_steps: int = 0
    curriculum_start_difficulty_max: float = 0.80
    env_difficulty_bins: str = (
        "0.20:0.25,0.25:0.30,0.30:0.35,0.35:0.40,0.40:0.45,0.45:0.50,"
        "0.50:0.55,0.55:0.60,0.60:0.65,0.65:0.70,0.70:0.75,0.75:0.80"
    )

    # 高难度聚焦采样
    env_hard_focus_prob: float = 0.0
    env_hard_focus_min: float | None = None
    env_hard_focus_max: float | None = None

    # 固定测试集周期评估
    eval_fixed_every_steps: int = 0
    eval_fixed_set_dir: str = "MRTAFixedTestSet_fixed240_d020_080_balanced_v1"
    eval_fixed_max_scenes: int = 240
    eval_fixed_batch_size: int = 8  # 固定测试集评估的并行批大小；1 表示逐场景串行

    # 固定集最优模型保存
    save_best_fixed_eval: bool = False
    best_fixed_metric: str = "recognized_done_rate"  # return|completion|success|recognized_done_tasks|recognized_done_rate|neg_violation
    best_fixed_min_delta: float = 1e-6
    best_ckpt_path: str = ""


@dataclass
class EvalConfig:
    episodes: int = 240  # 每个 seed 评估回合数；固定集协议统一为 240 场景
    seeds: int = 5  # 评估随机种子数量
