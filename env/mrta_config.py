from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class MRTAConfig:
    area_size: Tuple[float, float] = (100.0, 100.0)  # 空间范围（x,y）
    dt_seconds: float = 2.0  # 环境一步对应的真实秒数
    max_steps: int = 800  # 单回合最大离散步数

    num_agents: int = 20  # 机器人数量
    cap_dim: int = 3  # 能力维度
    v_max_range: Tuple[float, float] = (1.5, 3.0)  # 机器人最大速度采样范围

    init_tasks_range: Tuple[int, int] = (20, 50)  # 初始任务数量范围
    total_tasks_floor: int = 20  # 单回合任务总数下限（初始+动态到达）
    total_tasks_cap: int = 50  # 单回合任务总数上限（初始+动态到达）
    dynamic_arrival_every: int = 16  # 每隔多少步触发动态到达（<=0 表示关闭）
    dynamic_arrival_range: Tuple[int, int] = (0, 1)  # 每次动态到达的新任务数范围
    max_tasks: int = 80  # 观测/动作中的任务槽位上限
    enforce_difficulty_range: bool = False  # 是否强制将随机场景难度限制在区间内
    difficulty_min: float = 0.2  # 训练/测试共同难度域下限
    difficulty_max: float = 0.8  # 训练/测试共同难度域上限
    difficulty_max_reset_attempts: int = 160  # 宽域分箱 reset 时重采样次数上限
    hard_focus_prob: float = 0.0  # 正式宽域实验采用均衡分箱，不额外偏向高难度
    hard_focus_min: Optional[float] = None
    hard_focus_max: Optional[float] = None

    relation_ratio: Dict[str, float] = field(default_factory=lambda: {
        "FS": 0.7,
        "SS": 0.2,
        "FF": 0.08,
        "SF": 0.02,
    })
    lag_range: Tuple[int, int] = (0, 5)  # 依赖关系 lag 采样范围
    relation_window_seconds: Dict[str, float] = field(default_factory=lambda: {
        "FS": 60.0,
        "SS": 60.0,
        "FF": 60.0,
        "SF": 60.0,
    })
    edge_density: float = 0.12  # 任务图连边概率
    use_structured_graph_generator: bool = True  # 是否使用结构化任务图生成器
    structured_parent_span: int = 12  # 结构化生成中候选前驱回看范围
    structured_min_parents: int = 0  # 结构化生成中最小前驱数量
    structured_max_parents: int = 2  # 结构化生成中最大前驱数量
    edge_feasibility_guard: bool = True  # 连边时启用轻量可行性守门（本地+增量+低频全局）
    edge_max_in_degree: int = 2  # 每个任务最大入度（防止依赖过密）
    edge_candidate_resamples: int = 8  # 单个候选父边失败后的重采样次数
    edge_global_check_interval: int = 2  # 每接受多少条边触发一次全局快速检查
    sf_edge_accept_prob: float = 0.05  # 额外抑制 SF 边，避免语义不现实
    window_slack_steps: int = 6  # 关系窗口可达性检查的额外安全步数

    agent_type_cap_range: Tuple[float, float] = (1.0, 3.0)  # 同类型能力采样范围
    task_demand_range: Tuple[float, float] = (0.6, 2.5)  # 任务对其类型能力的需求范围
    task_workload_range: Tuple[float, float] = (10.0, 30.0)  # 任务工作量范围
    task_k_range: Tuple[int, int] = (1, 1)  # 单智能体执行（无组队）

    # 任务类型划分：在 distance_task_types 内的类型按“距离型任务”处理，其余按“非距离型任务”处理。
    distance_task_types: Tuple[int, ...] = (0, 2)
    # 非距离型任务的类型参数（按 task_type 索引），用于：需求 / 团队能力和 * 参数。
    # 若长度与 cap_dim 不一致，则会在环境中回退到自动标定值。
    non_distance_type_scales: Tuple[float, ...] = (180.0, 180.0, 180.0)
    non_distance_time_bounds: Tuple[float, float] = (60.0, 600.0)  # 非距离型物理执行时间边界（秒）

    # ------------------------------------------------------------------
    # 奖励函数超参数（已降维：15 → 4 个语义参数）
    # 设计原则：4 个可调旋钮锚定不可约的设计权衡，其余折叠为推导常量。
    # w1: 主信号锚点（一次合规完成值多少）
    # w2: 前驱推进稠密 shaping（长依赖链稀疏奖励的唯一补偿）
    # w3: 累积完成持续梯度（"多完成" vs "早完成"的偏好）
    # w4: 违规相对代价（策略保守/激进的唯一控制旋钮）
    # λ = w_eff/w_vio 为推导常量，保证效率项永远不可能压过违规。
    # ------------------------------------------------------------------
    reward_recognized_immediate_weight: float = 36.0  # w1：首要主奖励
    reward_progress_weight: float = 0.12              # w2：次级密集引导
    reward_total_completion_weight: float = 3.0       # w3：持续强化"多完成才更优"
    reward_violation_weight: float = 1.2              # w4：违规惩罚，确保合规优先

    allow_pre_gathering: bool = True  # 承诺后允许预集结（先移动再等待）
    pre_gather_radius: float = 2.5  # 预集结到任务点附近的半径
    use_policy_commitment_logic: bool = True  # 主算法环境绑定启用任务吸引力/释放等承诺策略逻辑
    task_attractiveness_relative_margin: float = 2.2  # 动作掩码保留的相对分数带宽
    task_attractiveness_mask_threshold: float = -0.25  # 动作掩码保留的绝对阈值
    task_attractiveness_min_keep: int = 1  # 常规吸引力筛选后至少保留的候选数
    dependency_open_horizon_steps: int = 10  # 若依赖任务将在近期满足开始条件，允许其进入动作候选
    dependency_open_min_ready_ratio: float = 0.8  # 对存在前驱的任务，仅在大部分前驱已就绪时提前开放
    dependency_open_min_criticality: float = 1.6  # 仅对关键度足够高的依赖任务提前开放，避免泛化放松
    dependency_anchor_min_keep: int = 0  # 默认不额外扩张无前驱锚点任务，避免掩码回到宽松状态
    capacity_topk_enabled: bool = True  # 是否启用容量感知 Top-K 候选收缩
    capacity_topk_per_agent: int = 3  # 每个 agent 最多保留的高质量候选任务数
    capacity_topk_min_keep: int = 2  # 容量收缩后至少保留的候选任务数
    capacity_topk_remaining_slots_weight: float = 1.1  # 剩余槽位越少，候选分数抑制越强
    capacity_topk_fullness_weight: float = 1.4  # 越接近满员的任务，其拥挤惩罚越强
    capacity_topk_shared_candidate_weight: float = 0.7  # 同一步被更多 agent 同时视为候选的任务，额外降低其候选分数
    capacity_topk_dependency_protect_threshold: float = 1.8  # 关键依赖任务的保护阈值
    capacity_topk_ready_protect_threshold: float = 0.85  # 高前驱就绪率依赖任务的保护阈值
    capacity_topk_ready_bucket_keep: int = 1  # 分层 Top-K 中优先保留的已可开工/近可开工候选数
    capacity_topk_dependency_bucket_keep: int = 1  # 分层 Top-K 中优先保留的依赖关键候选数
    capacity_topk_imminent_horizon_steps: int = 8  # 任务被视为近可开工的预测时间窗
    dependency_criticality_bonus_weight: float = 1.2  # 对关键依赖任务增加吸引力，强化依赖链推进
    dependency_window_urgency_bonus_weight: float = 1.4  # 对窗口临近任务增加吸引力，减少错过关键时序
    dependency_ready_bonus_weight: float = 0.8  # 对高前驱准备度依赖任务增加吸引力
    pre_gather_start_horizon_steps: int = 12  # 允许预集结的近期开始视野
    pre_gather_max_wait_steps: int = 30  # 允许预集结任务的最长预测等待步数
    pre_gather_team_cap: int = 1  # 单个未就绪任务最多允许的预集结人数
    pre_gather_team_cap_when_ready_exists: int = 1  # 存在同类型立即可开工任务时的预集结上限
    pre_gather_team_cap_when_imminent_exists: int = 1  # 存在同类型近期可开工任务时的预集结上限
    far_pre_gather_team_cap: int = 0  # 远期锁定任务默认不允许预集结
    pre_gather_reserve_ratio: float = 0.10  # 预集结后保留的全局空闲比例
    pre_gather_reserve_min: int = 1  # 预集结后至少保留的空闲 agent 数
    pre_gather_type_reserve_min: int = 0  # 预集结后至少保留的同类型空闲 agent 数
    allow_stale_wait_reassignment: bool = True  # 允许等待过久的承诺被释放重分配
    stale_wait_steps: int = 14  # 多久未取得进展后触发 stale reassignment 检查
    stale_reassign_score_margin: float = 0.20  # 候选任务明显更优时释放旧承诺
    strict_start_in_mask: bool = False  # 动作掩码不硬卡开始时序；允许预集结，完成/激活阶段再判定时序
    capacity_aware_commitment_arbitration: bool = True  # 同步裁剪同一任务的过量承诺提案，避免把容量竞争误记为违规
    ordered_commitment: bool = True  # 是否按固定 id 顺序处理承诺

    # === 方案 C：依赖信息通路重构（信息瓶颈） ===
    # 开启后，节点/配对特征移除"依赖派生量"（前驱满足度、最早/最晚时间窗、
    # slack、temporal feasibility、criticality、窗口紧迫度、按依赖推算的开始/完成时间等），
    # 使这些依赖推理信息只能经由 RGAT 沿依赖边传播获得（RGAT 成为唯一通路）。
    # 主模型与所有消融变体共享同一瓶颈特征，对照唯一差别是有无图边传播（公平归因）。
    sparse_dependency_features: bool = True  # 是否启用依赖信息瓶颈（默认开启）
    # RGAT 沿边时间松弛传播：在消息中注入由依赖边推出的到达源节点的 earliest 时间，
    # 让网络显式执行沿依赖边的时间可行性传播（FS/SS 用源完成/开始时间 + lag）。
    rgat_temporal_propagation: bool = True  # 是否启用沿边时间传播（结构对齐）