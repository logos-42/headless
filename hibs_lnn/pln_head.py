"""PLN 头 (lm1 V31_PLN 的可移植移植版) + OML 双循环原语。

来源: tests/run_v31_meta_learning.py:114 V31_PLN / head_forward / per_feature_step
      tests/run_lm3_bpe.py:193  oml2 双循环协议

lm1 原设计(见 run_lm1_production.py 文件头):
  SSM (RLN) + OML 双层 (内循环 PLN 适应 + 外循环合并) + 价值提议器

本模块只搬 RLN 之外的两件事:
  1. PLN: 小 MLP 头 + **per-feature 步长** (Meta-SGD / IDBD), 内循环只应用不修改 β
  2. 双循环协议: 内循环克隆头适应 support → 外循环用 query loss 更新全模型

与 lm4 的结合点: lm4 原本是 `head = nn.Linear(d_model, n_classes)` 的**单循环**训练。
把 head 换成 PLN 并改用双循环, 即 lm1 的架构。
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PLNHead(nn.Module):
    """预测学习网络 (Prediction Learning Network).

    结构与 V31_PLN 一致: Linear(in, d) → Tanh → Linear(d, out)
    差别: 输入是连续特征聚合 (lm4) 而非离散 hidden, 故 input_dim 可独立指定。

    step_beta 是 **meta-parameter**: 外循环学它, 内循环只用它做
    `W ← W - exp(beta) * grad`. 这样内循环有 per-feature 自适应步长,
    而不是所有参数共享一个标量 lr。
    """

    def __init__(self, input_dim, d_model, n_classes, inner_lr=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_classes = n_classes
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, n_classes),
        )
        # log 空间初始化 → exp(beta) = inner_lr
        self.step_beta = nn.Parameter(
            torch.full((self._n_params(),), math.log(inner_lr)))

    # ---------- 参数工具 ----------
    def _n_params(self) -> int:
        return sum(p.numel() for p in self.predictor.parameters())

    def clone_params(self) -> List[torch.Tensor]:
        """克隆可适应参数 (保留到共享参数的图, 供外循环反传)。"""
        lin0 = self.predictor[0]
        lin2 = self.predictor[2]
        return [lin0.weight.clone(), lin0.bias.clone(),
                lin2.weight.clone(), lin2.bias.clone()]

    def per_feature_step(self, W: List[torch.Tensor],
                         grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """Meta-SGD 内循环更新: 逐参数自适应步长 (与 V31_PLN.per_feature_step 同式)。"""
        beta = self.step_beta
        new_W, off = [], 0
        for w, g in zip(W, grads):
            n = w.numel()
            b = beta[off:off + n].view_as(w)
            alpha = b.exp().clamp(max=1.0)
            new_W.append(w - alpha * g)
            off += n
        return new_W

    def load_cloned(self, W: List[torch.Tensor]) -> None:
        """把克隆参数写回本体 (仅评估/合并用)。"""
        with torch.no_grad():
            self.predictor[0].weight.copy_(W[0])
            self.predictor[0].bias.copy_(W[1])
            self.predictor[2].weight.copy_(W[2])
            self.predictor[2].bias.copy_(W[3])

    # ---------- 前向 ----------
    @staticmethod
    def fwd_with(W: List[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """用给定的参数集做前向 (functional, 支持内循环克隆参数)。"""
        w1, b1, w2, b2 = W
        x = torch.tanh(h @ w1.t() + b1)
        return x @ w2.t() + b2

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.predictor(h)


def inner_adapt(head: PLNHead, h_sup: torch.Tensor, y_sup: torch.Tensor,
                K: int = 2, per_feature: bool = True):
    """内循环: 在 support 上适应**克隆的**头参数, 返回适应后的 W。

    per_feature=True  → 用 head.step_beta (lm1 的 Meta-SGD 步长)
    per_feature=False → 固定标量 lr (lm3 oml2 的做法, lr=0.1)

    注意: 整个过程对 head.predictor 的参数**保留梯度图** (没有 detach),
    这样外循环的 query loss 反传时, 梯度会经过内循环更新流回 meta-parameters。
    """
    W = head.clone_params()
    for _ in range(K):
        logits = PLNHead.fwd_with(W, h_sup)
        loss = F.cross_entropy(logits, y_sup)
        grads = torch.autograd.grad(loss, W, create_graph=True)
        if per_feature:
            W = head.per_feature_step(W, grads)
        else:
            W = [w - 0.1 * g for w, g in zip(W, grads)]
    return W


def oml_step(model, head: PLNHead, enc, h_sup, y_sup, h_qry, y_qry,
             opt_outer, K: int = 2, per_feature: bool = True,
             consolidate: bool = True, reptile_lr: float = 0.0):
    """一次 OML 双循环步。

      内循环: 克隆头在 support 上适应 K 步 (慢参数冻结)
      外循环: fast head 在 query 上的 loss → 反传更新 RLN + PLN meta 参数
      合并:   把适应后的 fast head 写回本体 (OML 权重整合)
      Reptile: 再把本体 init 朝适应后权重拉一步 (lm1 meta_step 的 init 平均)
               init ← init + reptile_lr * (W_adapted - init)

    返回 (query_loss, support_loss) 标量。
    """
    # --- 内循环 (RLN 冻结, 只适应头; 梯度图保留供外循环用) ---
    W = inner_adapt(head, h_sup, y_sup, K=K, per_feature=per_feature)

    # --- 外循环: query loss 反传全模型 ---
    opt_outer.zero_grad()
    logits_q = PLNHead.fwd_with(W, h_qry)
    loss_q = F.cross_entropy(logits_q, y_qry)
    loss_q.backward()
    # 二阶 (create_graph) 梯度必须裁剪 —— lm1 meta_step 的 clip_grad_norm_(1.0)。
    # 不裁剪时外循环会把编码器打爆, 表现为"当前域刚学会就整体归零"。
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt_outer.step()

    # --- OML 权重整合: 适应后的头合并回本体 ---
    if consolidate:
        with torch.no_grad():
            head.load_cloned([w.detach() for w in W])
            # 合并后这些是叶子参数, 外循环下一步能继续更新
            head.predictor[0].weight.requires_grad_(True)
            head.predictor[0].bias.requires_grad_(True)
            head.predictor[2].weight.requires_grad_(True)
            head.predictor[2].bias.requires_grad_(True)

    # --- Reptile init 平均 (lm1 V31.2 meta_step 的做法) ---
    if reptile_lr > 0.0:
        with torch.no_grad():
            for p, w in zip(head.predictor.parameters(), W):
                p.add_(reptile_lr * (w.detach() - p))

    with torch.no_grad():
        loss_s = F.cross_entropy(PLNHead.fwd_with(
            [w.detach() for w in W], h_sup), y_sup).item()
    return loss_q.item(), loss_s
