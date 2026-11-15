from typing import Dict, Tuple
import math

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskGraphObsRequired(TypedDict):
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_lag: torch.Tensor
    node_mask: torch.Tensor


class TaskGraphObs(TaskGraphObsRequired, total=False):
    edge_mask: torch.Tensor
    node_t_start: torch.Tensor
    node_t_finish_phys: torch.Tensor


class MRTAObs(TypedDict):
    task_graph: TaskGraphObs
    agent_feats: torch.Tensor
    pair_feats: torch.Tensor
    action_mask: torch.Tensor
    global_time: torch.Tensor


class AuxPredictions(TypedDict, total=False):
    action_logits: torch.Tensor
    task_start_logits: torch.Tensor
    task_ready_logits: torch.Tensor
    pair_feasible_logits: torch.Tensor
    task_h: torch.Tensor
    task_z: torch.Tensor
    agent_actor_z: torch.Tensor
    graph_h: torch.Tensor


class RelationGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 4):
        # 面向任务依赖边的关系感知图注意力层。
        super().__init__()
        # 关系类型数量（例如 FS/SS/FF/SF）。
        self.num_relations = num_relations
        # 每种关系独立的 Q/K/V 投影。
        self.q_rel = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.k_rel = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.v_rel = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        # 每种关系的 lag 偏置投影。
        self.lag_bias_rel = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(num_relations)])
        # 聚合后再做一次统一投影。
        self.out_proj = nn.Linear(out_dim, out_dim)
        # 显式 self-loop 投影，保证无入边节点也有可学习更新。
        self.self_proj = nn.Linear(in_dim, out_dim, bias=False)

    def _message_pass(self, x, edge_index, edge_type, edge_lag, node_mask, edge_mask):
        """支持单样本和批量样本的关系注意力消息传递。"""
        if x.dim() == 2:
            x = x.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
            edge_type = edge_type.unsqueeze(0)
            edge_lag = edge_lag.unsqueeze(0)
            node_mask = node_mask.unsqueeze(0)
            if edge_mask is None:
                edge_mask = torch.ones_like(edge_type, dtype=torch.bool)
            else:
                edge_mask = edge_mask.unsqueeze(0)
            squeeze_back = True
        else:
            if edge_mask is None:
                edge_mask = torch.ones_like(edge_type, dtype=torch.bool)
            squeeze_back = False

        bsz, n_nodes, hid = x.shape
        x_flat = x.reshape(bsz * n_nodes, hid)
        out_flat = torch.zeros((bsz * n_nodes, hid), device=x.device, dtype=x.dtype)

        src_all = edge_index[:, 0, :].long()
        dst_all = edge_index[:, 1, :].long()
        offset = (torch.arange(bsz, device=x.device).unsqueeze(-1) * n_nodes).long()
        src_all = src_all + offset
        dst_all = dst_all + offset

        edge_type = edge_type.long()
        edge_lag = edge_lag.float()
        edge_mask = edge_mask.bool()

        total_nodes = bsz * n_nodes
        for rel in range(self.num_relations):
            rel_mask = (edge_type == rel) & edge_mask
            if not torch.any(rel_mask):
                continue

            src = src_all[rel_mask]
            dst = dst_all[rel_mask]
            q_dst = self.q_rel[rel](x_flat[dst])
            k_src = self.k_rel[rel](x_flat[src])
            v_src = self.v_rel[rel](x_flat[src])
            lag = edge_lag[rel_mask].unsqueeze(-1)
            lag_bias = self.lag_bias_rel[rel](lag).squeeze(-1)
            score = (q_dst * k_src).sum(dim=-1) / (k_src.shape[-1] ** 0.5)
            score = score + lag_bias

            max_per_dst = torch.full((total_nodes,), float("-inf"), device=x.device, dtype=score.dtype)
            max_per_dst.scatter_reduce_(0, dst, score, reduce="amax", include_self=True)
            exp_score = torch.exp(score - max_per_dst[dst])
            sum_per_dst = torch.zeros((total_nodes,), device=x.device, dtype=score.dtype)
            sum_per_dst.scatter_add_(0, dst, exp_score)
            alpha = exp_score / (sum_per_dst[dst] + 1e-9)

            out_flat.index_add_(0, dst, alpha.unsqueeze(-1) * v_src)

        out_flat = out_flat + self.self_proj(x_flat)

        out = out_flat.reshape(bsz, n_nodes, hid)
        out = self.out_proj(out)
        out = F.relu(out)
        out = out * node_mask.unsqueeze(-1)

        if squeeze_back:
            return out.squeeze(0)
        return out

    def forward(self, x, edge_index, edge_type, edge_lag, node_mask, edge_mask=None):
        """使用关系特定参数进行消息传递。

        参数：
            x: 任务节点嵌入，形状 [N, D]。
            edge_index: 有向边索引，形状 [2, E]。
            edge_type: 每条边的关系类型 id，形状 [E]。
            edge_lag: 每条边的时间滞后特征，形状 [E]。
            node_mask: 有效任务掩码，形状 [N]。

        返回：
            更新后的节点嵌入，形状 [N, D]。
        """
        return self._message_pass(x, edge_index, edge_type, edge_lag, node_mask, edge_mask)


class RGATEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, layers: int = 2):
        # 多层关系感知 GAT 堆叠，带残差连接与 LayerNorm。
        super().__init__()
        # 把原始任务节点特征映射到隐藏维度。
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        # 堆叠多层关系图注意力。
        self.layers = nn.ModuleList([RelationGATLayer(hidden_dim, hidden_dim) for _ in range(layers)])
        # 每层对应一个 LayerNorm，稳定训练。
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

    def forward(self, node_x, edge_index, edge_type, edge_lag, node_mask, edge_mask=None):
        """编码任务图，输出节点级表示与图级表示。"""
        # 输入先投影并激活，得到初始节点嵌入 h。
        h = F.relu(self.input_proj(node_x))
        # 在依赖图上重复执行消息传递。
        for layer, norm in zip(self.layers, self.norms):
            # 残差结构：h + GAT(h)，再做归一化。
            h = norm(h + layer(h, edge_index, edge_type, edge_lag, node_mask, edge_mask=edge_mask))
        # 仅在有效节点上做均值池化。
        # graph 是图级全局表示，供 agent 解码与 value 估计使用。
        if h.dim() == 3:
            graph = (h * node_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(node_mask.sum(dim=1, keepdim=True), min=1.0)
        else:
            graph = (h * node_mask.unsqueeze(-1)).sum(dim=0) / torch.clamp(node_mask.sum(), min=1.0)
        # 返回节点表示与图表示。
        return h, graph


class RelationGraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int = 4, num_heads: int = 4, dropout: float = 0.0,
                 temporal_propagation: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_heads = max(1, num_heads)
        if hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = hidden_dim // self.num_heads
        self.scale = math.sqrt(float(self.head_dim))
        self.temporal_propagation = bool(temporal_propagation)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_bias = nn.Embedding(num_relations, self.num_heads)
        self.rel_gate = nn.Embedding(num_relations, hidden_dim)
        self.lag_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_heads),
        )
        self.msg_lag_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        if self.temporal_propagation:
            # 沿边时间传播：把"源节点时间 + lag"编码为消息分量，使依赖时间下界沿边流动。
            # 输入 = [源开始时间, 源完成时间, lag, 源开始有效, 源完成有效]。
            self.temporal_mlp = nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def _prepare_batch(self, x, edge_index, edge_type, edge_lag, node_mask, edge_mask):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
            edge_type = edge_type.unsqueeze(0)
            edge_lag = edge_lag.unsqueeze(0)
            node_mask = node_mask.unsqueeze(0)
            if edge_mask is None:
                edge_mask = torch.ones_like(edge_type, dtype=torch.bool)
            else:
                edge_mask = edge_mask.unsqueeze(0)
            squeeze_back = True
        else:
            if edge_mask is None:
                edge_mask = torch.ones_like(edge_type, dtype=torch.bool)
            squeeze_back = False
        return x, edge_index, edge_type.long(), edge_lag.float(), node_mask.float(), edge_mask.bool(), squeeze_back

    def forward(self, x, edge_index, edge_type, edge_lag, node_mask, edge_mask=None,
                node_t_start=None, node_t_finish_phys=None):
        x, edge_index, edge_type, edge_lag, node_mask, edge_mask, squeeze_back = self._prepare_batch(
            x, edge_index, edge_type, edge_lag, node_mask, edge_mask
        )

        bsz, n_nodes, _ = x.shape
        total_nodes = bsz * n_nodes
        offset = (torch.arange(bsz, device=x.device).unsqueeze(-1) * n_nodes).long()
        src = edge_index[:, 0, :].long() + offset
        dst = edge_index[:, 1, :].long() + offset

        q = self.q_proj(x).reshape(bsz, n_nodes, self.num_heads, self.head_dim).reshape(total_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(bsz, n_nodes, self.num_heads, self.head_dim).reshape(total_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, n_nodes, self.num_heads, self.head_dim).reshape(total_nodes, self.num_heads, self.head_dim)

        if self.temporal_propagation and (node_t_start is None or node_t_finish_phys is None):
            raise ValueError("temporal_propagation requires node_t_start and node_t_finish_phys")
        if self.temporal_propagation:
            if node_t_start.dim() == 1:
                node_t_start = node_t_start.unsqueeze(0)
                node_t_finish_phys = node_t_finish_phys.unsqueeze(0)
            tstart_flat = node_t_start.reshape(total_nodes).float()
            tfin_flat = node_t_finish_phys.reshape(total_nodes).float()

        valid_edges = edge_mask
        flat_src = src[valid_edges]
        flat_dst = dst[valid_edges]
        flat_type = edge_type[valid_edges]
        flat_lag = edge_lag[valid_edges].unsqueeze(-1)

        agg = torch.zeros((total_nodes, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        if flat_src.numel() > 0:
            q_dst = q[flat_dst]
            k_src = k[flat_src]
            v_src = v[flat_src]
            rel_bias = self.rel_bias(flat_type)
            lag_bias = self.lag_mlp(flat_lag)
            score = (q_dst * k_src).sum(dim=-1) / self.scale
            score = score + rel_bias + lag_bias

            max_per_dst = torch.full((total_nodes, self.num_heads), float("-inf"), device=x.device, dtype=x.dtype)
            dst_expand = flat_dst.unsqueeze(-1).expand(-1, self.num_heads)
            max_per_dst.scatter_reduce_(0, dst_expand, score, reduce="amax", include_self=True)
            attn = torch.exp(score - max_per_dst[flat_dst])
            sum_per_dst = torch.zeros((total_nodes, self.num_heads), device=x.device, dtype=x.dtype)
            sum_per_dst.scatter_add_(0, dst_expand, attn)
            attn = attn / (sum_per_dst[flat_dst] + 1e-9)
            attn = self.attn_drop(attn)

            rel_gate = torch.sigmoid(self.rel_gate(flat_type)).reshape(-1, self.num_heads, self.head_dim)
            lag_msg = self.msg_lag_mlp(flat_lag).reshape(-1, self.num_heads, self.head_dim)
            msg = (v_src + lag_msg) * rel_gate
            if self.temporal_propagation:
                # 源节点沿边推出的时间下界：FS/FF 用完成时间，SS/SF 用开始时间，加 lag。
                src_tstart = tstart_flat[flat_src]
                src_tfin = tfin_flat[flat_src]
                lag_raw = flat_lag.squeeze(-1)
                # 关系类型: 0=FS 1=SS 2=FF 3=SF（与 env REL_TYPES 一致）
                is_finish_rel = (flat_type == 0) | (flat_type == 2)
                base_time = torch.where(is_finish_rel, src_tfin, src_tstart)
                base_valid = (base_time >= 0.0).float()
                propagated = base_time + lag_raw
                temporal_in = torch.stack([
                    src_tstart, src_tfin, lag_raw,
                    (src_tstart >= 0.0).float(), (src_tfin >= 0.0).float(),
                ], dim=-1)
                temporal_msg = self.temporal_mlp(temporal_in).reshape(-1, self.num_heads, self.head_dim)
                # 仅当源时间有效时才注入时间消息（无效源不贡献下界）。
                temporal_msg = temporal_msg * base_valid.reshape(-1, 1, 1)
                msg = msg + temporal_msg
            agg.index_add_(0, flat_dst, attn.unsqueeze(-1) * msg)

        out = agg.reshape(bsz, n_nodes, self.hidden_dim) + self.self_proj(x)
        out = self.norm1(x + self.out_proj(out))
        out = self.norm2(out + self.ffn(out))
        out = out * node_mask.unsqueeze(-1)
        if squeeze_back:
            return out.squeeze(0)
        return out


class GraphTransformerEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, layers: int = 2, num_relations: int = 4, num_heads: int = 4,
                 temporal_propagation: bool = False):
        super().__init__()
        self.temporal_propagation = bool(temporal_propagation)
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                RelationGraphTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_relations=num_relations,
                    num_heads=num_heads,
                    temporal_propagation=temporal_propagation,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, node_x, edge_index, edge_type, edge_lag, node_mask, edge_mask=None,
                node_t_start=None, node_t_finish_phys=None):
        h = F.relu(self.input_proj(node_x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_lag, node_mask, edge_mask=edge_mask,
                      node_t_start=node_t_start, node_t_finish_phys=node_t_finish_phys)
        if h.dim() == 3:
            graph = (h * node_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(node_mask.sum(dim=1, keepdim=True), min=1.0)
        else:
            graph = (h * node_mask.unsqueeze(-1)).sum(dim=0) / torch.clamp(node_mask.sum(), min=1.0)
        return h, graph


class MLPNodeEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        # 消融版编码器：忽略边信息，仅使用逐节点 MLP。
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, node_x, edge_index, edge_type, edge_lag, node_mask, edge_mask=None):
        """不使用图注意力进行节点编码（RGAT 关闭时使用）。"""
        h = self.net(node_x)
        if h.dim() == 3:
            graph = (h * node_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(node_mask.sum(dim=1, keepdim=True), min=1.0)
        else:
            graph = (h * node_mask.unsqueeze(-1)).sum(dim=0) / torch.clamp(node_mask.sum(), min=1.0)
        return h, graph


class MRTAPolicy(nn.Module):
    def __init__(
        self,
        node_dim: int,
        agent_dim: int,
        pair_dim: int,
        hidden_dim: int = 128,
        rgat_layers: int = 2,
        use_rgat: bool = True,
        use_graph_transformer: bool = True,
        graph_transformer_heads: int = 4,
        self_only_graph: bool = False,
        rgat_temporal_propagation: bool = False,
    ):
        # 多机器人任务分配的 MAPPO 风格 Actor-Critic 策略网络。
        # 注：独立双解码器分支已移除——消融显示其为负贡献（去掉反优），
        # 统一走浅通路（task_z=task_h + actor_agent_shallow）。
        super().__init__()
        self.self_only_graph = self_only_graph
        # 沿边时间传播（方案 C 结构对齐）。
        self.rgat_temporal_propagation = bool(rgat_temporal_propagation)
        # 任务编码器：可选 RGAT（默认）或 MLP（消融）。
        if use_rgat:
            if use_graph_transformer:
                self.task_encoder = GraphTransformerEncoder(
                    node_dim,
                    hidden_dim,
                    rgat_layers,
                    num_relations=4,
                    num_heads=graph_transformer_heads,
                    temporal_propagation=rgat_temporal_propagation,
                )
            else:
                self.task_encoder = RGATEncoder(node_dim, hidden_dim, rgat_layers)
        else:
            self.task_encoder = MLPNodeEncoder(node_dim, hidden_dim)

        # 将全局时间编码为隐藏向量，并用于 actor/critic 的时序上下文。
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # actor 的 agent 编码器仅使用局部自身状态 + time，不输入其他 agent 特征。
        # 浅通路：任务侧直接用 RGAT 输出 task_h，不再过独立 MLP 解码器。
        self.actor_agent_shallow = nn.Sequential(
            nn.Linear(agent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 全局任务上下文（来自 task graph + time），可广播给每个 agent。
        self.task_global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Actor 输入维度：local_agent + task_node + pair + global_task_ctx。
        actor_in = hidden_dim * 3 + pair_dim
        # Actor 头：输出每个 (agent, task) 组合的 logit。
        self.actor_head = nn.Sequential(
            nn.Linear(actor_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # NOOP 头：基于 local_agent + global_task_ctx 输出每个 agent 的 NOOP logit。
        self.noop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # critic 的全信息编码器：可访问所有 agent 与 pair 信息。
        self.critic_agent_encoder = nn.Sequential(
            nn.Linear(agent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.critic_pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.critic_mask_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Critic 头：输入全局任务上下文 + 全体 agent/pair 池化统计 + 时间上下文。
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.task_start_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.task_ready_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_feasible_head = nn.Sequential(
            nn.Linear(actor_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _apply_action_mask_with_fallback(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.masked_fill(mask < 0.5, -1e9)
        valid_any = (mask > 0.5).any(dim=-1)
        if bool(torch.any(~valid_any)):
            masked_logits = masked_logits.clone()
            # 数值安全兜底：若某个 agent 的动作全无效，则强制 NOOP 可选。
            masked_logits[~valid_any] = -1e9
            masked_logits[~valid_any, 0] = 0.0
        return masked_logits

    def _masked_task_max_single(self, task_h: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        valid = node_mask.unsqueeze(-1) > 0.5
        neg_inf = torch.full_like(task_h, float("-inf"))
        masked = torch.where(valid, task_h, neg_inf)
        mx = masked.amax(dim=0)
        return torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

    def _masked_task_max_batch(self, task_h: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        valid = node_mask.unsqueeze(-1) > 0.5
        neg_inf = torch.full_like(task_h, float("-inf"))
        masked = torch.where(valid, task_h, neg_inf)
        mx = masked.amax(dim=1)
        return torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

    def _encode_time(self, global_time: torch.Tensor, is_batch: bool, dtype: torch.dtype) -> torch.Tensor:
        if is_batch:
            time_in = global_time.reshape(-1, 1)
        else:
            time_in = global_time.reshape(1, 1)
        # tanh 压缩可避免训练后期大时间步导致数值尺度偏移。
        time_in = torch.tanh(time_in.to(dtype=dtype) / 100.0)
        time_h = self.time_encoder(time_in)
        if is_batch:
            return time_h
        return time_h.squeeze(0)

    def _masked_pair_stats_single(self, pair_z: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pair_z: [A, T, H], node_mask: [T]
        w = node_mask.unsqueeze(0).unsqueeze(-1)
        w = w.expand(pair_z.shape[0], pair_z.shape[1], 1).to(dtype=pair_z.dtype)
        denom = torch.clamp(w.sum(dim=(0, 1)), min=1.0)
        pair_mean = (pair_z * w).sum(dim=(0, 1)) / denom

        neg_inf = torch.full_like(pair_z, float("-inf"))
        valid = w > 0.5
        pair_for_max = torch.where(valid, pair_z, neg_inf)
        pair_max = pair_for_max.amax(dim=(0, 1))
        pair_max = torch.where(torch.isfinite(pair_max), pair_max, torch.zeros_like(pair_max))
        return pair_mean, pair_max

    def _masked_pair_stats_batch(self, pair_z: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pair_z: [B, A, T, H], node_mask: [B, T]
        w = node_mask.unsqueeze(1).unsqueeze(-1)
        w = w.expand(pair_z.shape[0], pair_z.shape[1], pair_z.shape[2], 1).to(dtype=pair_z.dtype)
        denom = torch.clamp(w.sum(dim=(1, 2)), min=1.0)
        pair_mean = (pair_z * w).sum(dim=(1, 2)) / denom

        neg_inf = torch.full_like(pair_z, float("-inf"))
        valid = w > 0.5
        pair_for_max = torch.where(valid, pair_z, neg_inf)
        pair_max = pair_for_max.amax(dim=(1, 2))
        pair_max = torch.where(torch.isfinite(pair_max), pair_max, torch.zeros_like(pair_max))
        return pair_mean, pair_max

    def forward(self, obs: MRTAObs, use_mask: bool = True, return_aux: bool = False):
        """为每个机器人生成动作 logits，并输出共享状态价值估计。

        每个机器人的动作空间为 [NOOP, task_0, ..., task_(max_tasks-1)]。
        """
        # 解包观测张量。
        # 任务节点特征。
        node_x = obs["task_graph"]["node_features"]
        # 图边索引（2,E）。
        edge_index = obs["task_graph"]["edge_index"].long()
        # 边类型 id（E）。
        edge_type = obs["task_graph"]["edge_type"].long()
        # 边 lag 特征（E）。
        edge_lag = obs["task_graph"]["edge_lag"].float()
        # 任务有效位掩码（max_tasks）。
        node_mask = obs["task_graph"]["node_mask"].float()
        edge_mask = obs["task_graph"].get("edge_mask", None)
        if edge_mask is not None:
            edge_mask = edge_mask.float()
        if self.self_only_graph:
            if edge_mask is None:
                edge_mask = torch.zeros_like(edge_type, dtype=node_x.dtype)
            else:
                edge_mask = torch.zeros_like(edge_mask)
        # 机器人特征（num_agents, agent_dim）。
        agent_x = obs["agent_feats"]
        # 机器人-任务配对特征（num_agents, max_tasks, pair_dim）。
        pair_x = obs["pair_feats"]
        global_time = obs["global_time"]
        # 方案 C：沿边时间传播所需的环境真值时间（可选）。
        node_t_start = obs["task_graph"].get("node_t_start", None)
        node_t_finish_phys = obs["task_graph"].get("node_t_finish_phys", None)

        # 编码任务节点与全局图上下文。
        # task_h: 节点表示；graph_h: 图级表示。
        encoder_kwargs = {}
        if isinstance(self.task_encoder, GraphTransformerEncoder) and self.task_encoder.temporal_propagation:
            encoder_kwargs["node_t_start"] = node_t_start
            encoder_kwargs["node_t_finish_phys"] = node_t_finish_phys
        task_h, graph_h = self.task_encoder(node_x, edge_index, edge_type, edge_lag, node_mask, edge_mask=edge_mask, **encoder_kwargs)
        if node_x.dim() == 2:
            time_h = self._encode_time(global_time, is_batch=False, dtype=node_x.dtype)
            global_task_ctx = self.task_global_proj(torch.cat([graph_h, time_h], dim=-1))

            task_z = task_h
            agent_actor_z = self.actor_agent_shallow(
                torch.cat([agent_x, time_h.unsqueeze(0).expand(agent_x.shape[0], -1)], dim=-1)
            )

            num_agents, num_tasks = pair_x.shape[0], pair_x.shape[1]
            agent_expand = agent_actor_z.unsqueeze(1).expand(num_agents, num_tasks, -1)
            task_expand = task_z.unsqueeze(0).expand(num_agents, num_tasks, -1)
            global_expand = global_task_ctx.unsqueeze(0).unsqueeze(0).expand(num_agents, num_tasks, -1)
            task_valid = node_mask.unsqueeze(0).unsqueeze(-1)

            pair_input = torch.cat([agent_expand, task_expand, pair_x, global_expand], dim=-1)
            logits_task = self.actor_head(pair_input).squeeze(-1)
            logits_task = logits_task.masked_fill(task_valid.squeeze(-1) < 0.5, -1e9)

            noop_in = torch.cat([agent_actor_z, global_task_ctx.unsqueeze(0).expand(num_agents, -1)], dim=-1)
            noop_logit = self.noop_head(noop_in).squeeze(-1).unsqueeze(-1)
            logits = torch.cat([noop_logit, logits_task], dim=-1)

            if use_mask and "action_mask" in obs:
                mask = obs["action_mask"]
                logits = self._apply_action_mask_with_fallback(logits, mask)

            critic_agent_z = self.critic_agent_encoder(
                torch.cat([agent_x, time_h.unsqueeze(0).expand(agent_x.shape[0], -1)], dim=-1)
            )
            pooled_mean = critic_agent_z.mean(dim=0)
            pooled_max = critic_agent_z.max(dim=0).values

            pair_z = self.critic_pair_encoder(pair_x)
            pair_mean, pair_max = self._masked_pair_stats_single(pair_z, node_mask)
            task_max = self._masked_task_max_single(task_h, node_mask)

            action_mask = obs.get("action_mask", None)
            if action_mask is not None:
                valid_task_ratio = node_mask.mean()
                valid_action_ratio = action_mask[:, 1:].mean()
                mask_feat = torch.stack([valid_task_ratio, valid_action_ratio], dim=0).unsqueeze(0)
                mask_ctx = self.critic_mask_encoder(mask_feat).squeeze(0)
            else:
                mask_ctx = torch.zeros_like(time_h)

            value = self.critic(
                torch.cat([global_task_ctx, pooled_mean, pooled_max, pair_mean, pair_max, task_max, mask_ctx, time_h], dim=-1)
            ).squeeze(-1)
            if not return_aux:
                return logits, value
            aux: AuxPredictions = {
                "action_logits": logits,
                "task_start_logits": self.task_start_head(task_h).squeeze(-1),
                "task_ready_logits": self.task_ready_head(task_h).squeeze(-1),
                "pair_feasible_logits": self.pair_feasible_head(pair_input).squeeze(-1),
                "task_h": task_h,
                "task_z": task_z,
                "agent_actor_z": agent_actor_z,
                "graph_h": graph_h,
            }
            return logits, value, aux

        # 批量分支：obs 形状为 [B, ...]。
        time_h = self._encode_time(global_time, is_batch=True, dtype=node_x.dtype)
        global_task_ctx = self.task_global_proj(torch.cat([graph_h, time_h], dim=-1))

        task_z = task_h
        agent_actor_z = self.actor_agent_shallow(
            torch.cat([agent_x, time_h.unsqueeze(1).expand(-1, agent_x.shape[1], -1)], dim=-1)
        )

        # 扩展机器人与任务表示，为每个 (agent, task) 组合打分。
        # 机器人数量与任务槽位数量。
        bsz, num_agents, num_tasks = pair_x.shape[0], pair_x.shape[1], pair_x.shape[2]
        agent_expand = agent_actor_z.unsqueeze(2).expand(bsz, num_agents, num_tasks, -1)
        task_expand = task_z.unsqueeze(1).expand(bsz, num_agents, num_tasks, -1)
        global_expand = global_task_ctx.unsqueeze(1).unsqueeze(2).expand(bsz, num_agents, num_tasks, -1)
        task_valid = node_mask.unsqueeze(1).unsqueeze(-1)

        # 任务动作 logits，并屏蔽不存在的任务槽位。
        # 拼接 (agent, task, pair) 特征并输出每个任务动作分数。
        pair_input = torch.cat([agent_expand, task_expand, pair_x, global_expand], dim=-1)
        logits_task = self.actor_head(pair_input).squeeze(-1)
        # 对无效任务位写入极小值，softmax 后概率近似为 0。
        logits_task = logits_task.masked_fill(task_valid.squeeze(-1) < 0.5, -1e9)

        # NOOP 动作单独打分。
        # 每个机器人得到一个 NOOP 分数。
        noop_in = torch.cat([agent_actor_z, global_task_ctx.unsqueeze(1).expand(-1, num_agents, -1)], dim=-1)
        noop_logit = self.noop_head(noop_in).squeeze(-1).unsqueeze(-1)
        # 拼接后动作维度为 [NOOP + 全任务槽位]。
        logits = torch.cat([noop_logit, logits_task], dim=-1)

        # 启用时应用环境提供的可行性掩码。
        if use_mask and "action_mask" in obs:
            # action_mask 由环境根据约束生成（不可行动作为 0）。
            mask = obs["action_mask"]
            # 把不可行动作对应 logit 设为极小值。
            logits = self._apply_action_mask_with_fallback(logits, mask)

        # Critic 使用全体 agent + pair 的全信息池化统计。
        critic_agent_z = self.critic_agent_encoder(
            torch.cat([agent_x, time_h.unsqueeze(1).expand(-1, num_agents, -1)], dim=-1)
        )
        pooled_mean = critic_agent_z.mean(dim=1)
        pooled_max = critic_agent_z.max(dim=1).values

        pair_z = self.critic_pair_encoder(pair_x)
        pair_mean, pair_max = self._masked_pair_stats_batch(pair_z, node_mask)
        task_max = self._masked_task_max_batch(task_h, node_mask)

        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            valid_task_ratio = node_mask.mean(dim=1, keepdim=True)
            valid_action_ratio = action_mask[:, :, 1:].mean(dim=(1, 2)).unsqueeze(-1)
            mask_feat = torch.cat([valid_task_ratio, valid_action_ratio], dim=-1)
            mask_ctx = self.critic_mask_encoder(mask_feat)
        else:
            mask_ctx = torch.zeros_like(time_h)

        value = self.critic(
            torch.cat([global_task_ctx, pooled_mean, pooled_max, pair_mean, pair_max, task_max, mask_ctx, time_h], dim=-1)
        ).squeeze(-1)
        # 返回 actor logits 与 critic value。
        if not return_aux:
            return logits, value
        aux: AuxPredictions = {
            "action_logits": logits,
            "task_start_logits": self.task_start_head(task_h).squeeze(-1),
            "task_ready_logits": self.task_ready_head(task_h).squeeze(-1),
            "pair_feasible_logits": self.pair_feasible_head(pair_input).squeeze(-1),
            "task_h": task_h,
            "task_z": task_z,
            "agent_actor_z": agent_actor_z,
            "graph_h": graph_h,
        }
        return logits, value, aux

    def act(self, obs: MRTAObs, deterministic: bool = False, use_mask: bool = True):
        """为每个机器人采样或贪心选择动作，并返回 PPO 所需统计量。"""
        logits, value = self.forward(obs, use_mask=use_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            # 评估阶段使用贪心解码。
            actions = torch.argmax(logits, dim=-1)
        else:
            # 训练阶段使用随机采样以保持探索。
            actions = dist.sample()
        logp = dist.log_prob(actions)
        # 返回逐智能体熵，便于训练时结合 active 掩码做一致聚合。
        entropy = dist.entropy()
        return actions, logp, value, entropy

    def evaluate_actions(self, obs: MRTAObs, actions: torch.Tensor, use_mask: bool = True):
        """对已选动作重算 log-prob/value/entropy（用于 PPO 更新）。"""
        logits, value, aux = self.forward(obs, use_mask=use_mask, return_aux=True)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        # 返回逐智能体熵，和 logp/active 的形状保持一致。
        entropy = dist.entropy()
        return logp, entropy, value, aux
