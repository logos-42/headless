"""
V30.3: 内部纠缠 SSM 层 (减法版)
===================================
核心改动: 在 SSM 递推的每一步, 用当前隐藏状态的跨维度投影
调制下一步的 A 矩阵相位。5 行代码, 无外部引擎。

减法:
  V30.1: +EntanglementEngine (后处理)
  V30.2: +EntangleDecohereEngine (纠缠+退相干+熵正则)
  V30.3: 只改 SSM 内部, 不加任何外部模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSM_Layer_V30_3(nn.Module):
    """
    V30.3 SSM 层: V29 基线 + 内部纠缠。

    在串行扫描的每一步(h_mean 计算后):
      if ent_mode != 'none':
        # 调 h 的相位: 纯相位调制, 不改变振幅
        phase_shift = ent_mod_net(h_norm)  # 跨 N 维度投影
        h = h * exp(i · phase_shift)        # 纯相位旋转

    纠缠特性:
      - 发生在 SSM 递推内部 (每个 token 步)
      - 跨 N 维度耦合 (ent_k=3 时 top-3)
      - 纯相位 (不破坏已学习的振幅结构)
      - 因果: 只影响未来状态, 不影响过去
    """

    def __init__(self, d_model, d_state=8, fiber=None, layer_idx=0,
                 ent_mode='none', ent_k=3, ent_strength=0.05):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ent_mode = ent_mode
        self.ent_k = ent_k
        self.ent_strength = ent_strength

        # 沿用 V29 架构
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3)
        self.out_proj = nn.Linear(d_model, d_model)

        # SSM 参数
        self.ssm_logsigma = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.ssm_theta = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.ssm_D = nn.Parameter(torch.ones(d_model))

        # V33 路由进递推 (INTACT 意图条件化): cond 调制 dt/B (读写门意图感知)
        # 零初始化 → 加载 Stage1 缓存后 cond 任意值 ≡ 无条件 (无需重训 Stage1)
        self.cond_dt = nn.Linear(d_model, d_model)
        self.cond_b = nn.Linear(d_model, d_state)
        with torch.no_grad():
            self.cond_dt.weight.zero_(); self.cond_dt.bias.zero_()
            self.cond_b.weight.zero_(); self.cond_b.bias.zero_()

        # KL 相关 (从 V29 继承)
        self.logvar_net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_state),
        )
        self.base_prior_logvar = nn.Parameter(torch.zeros(d_state))
        self.fiber_prior_strength = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('kl_scale', torch.tensor(0.0))

        # ── V30.3 内部纠缠: 极小调制网络 ──
        # 输入: |h| 投影到 N 维 (跨维度耦合)
        # 输出: 相位调制量 (纯相位, 不改变振幅)
        if ent_mode != 'none':
            self.ent_mod = nn.Linear(d_state, d_state, bias=False)
            self.ent_amp = nn.Linear(d_state, 1, bias=False)  # 振幅门控

        self._entropy_trace = []

    def compute_entropy(self, h):
        B, D, N = h.shape
        amp = h.abs()
        rho = amp.view(B, D, N, 1) * amp.view(B, D, 1, N)
        rho = rho / (rho.sum(dim=(-1, -2), keepdim=True) + 1e-10)
        try:
            e = torch.linalg.eigvalsh(rho).clamp(min=1e-10)
            return -(e * e.log()).sum(dim=-1).mean().item()
        except:
            return 0.0

    def forward(self, x, cond=None):
        r = x
        x = self.norm(x)
        a, b = self.in_proj(x).chunk(2, dim=-1)
        a = F.silu(self.conv(a.transpose(-1, -2))[:, :, :x.shape[1]].transpose(-1, -2))

        B_, L_, D_ = a.shape
        S_ = self.ssm_logsigma.shape[-1]
        sigma = -self.ssm_logsigma.exp()
        theta0 = self.ssm_theta
        A_eff = sigma.unsqueeze(0).unsqueeze(0) + 1j * theta0.unsqueeze(0).unsqueeze(0)

        if cond is not None:
            # V33 路由: 意图 cond 调制 dt 与 B (读写门意图感知), 广播到每步
            c_dt = self.cond_dt(cond)              # (..., d)
            c_b = self.cond_b(cond)                # (..., d_state)
            if c_dt.dim() == 1:
                c_dt = c_dt.unsqueeze(0)
                c_b = c_b.unsqueeze(0)
            dt = F.softplus(self.dt_proj(a) + c_dt.unsqueeze(1)) + 1e-4
            Bk = self.B_proj(a) + c_b.unsqueeze(1)
        else:
            dt = F.softplus(self.dt_proj(a)) + 1e-4
            Bk = self.B_proj(a)
        Ck = self.C_proj(a)
        Ab = torch.exp(dt.unsqueeze(-1) * A_eff)
        bk = (Ab - 1.0) / (A_eff + 1e-8) * Bk.unsqueeze(2) * a.unsqueeze(-1)

        h = torch.zeros(B_, D_, S_, dtype=torch.complex64, device=Ab.device)
        ys = torch.zeros(B_, L_, D_, device=a.device)

        for k in range(L_):
            h_mean = Ab[:, k] * h + bk[:, k]

            # ── V30.3: 内部纠缠 ──────────────
            if self.ent_mode != 'none' and k > 0:
                # 从 |h| 导出纯相位调制 (跨 N 维度耦合)
                h_norm = h_mean / (h_mean.abs() + 1e-8)  # 归一化复数
                h_amp = h_mean.abs()                     # 振幅
                phase_input = h_norm.real * h_amp        # (B, D, N) 振幅加权相位信号

                # 跨 N 维度的线性耦合
                delta_raw = self.ent_mod(phase_input)  # (B, D, N)

                if self.ent_k > 0 and self.ent_k < S_:
                    # top-k 稀疏化: 只纠缠振幅最大的维度
                    _, top_idx = h_amp.topk(self.ent_k, dim=-1)
                    mask = torch.zeros_like(delta_raw)
                    mask.scatter_(-1, top_idx, 1.0)
                    delta_raw = delta_raw * mask

                # 振幅门控: 强信号有更强的纠缠, 但不改变本身振幅
                amp_gate = torch.sigmoid(self.ent_amp(h_amp))  # (B, D, 1)
                phase_shift = torch.tanh(delta_raw) * self.ent_strength * amp_gate

                # 纯相位旋转: 不改变振幅
                h = h_mean * torch.exp(1j * phase_shift)
            else:
                h = h_mean

            y = (Ck[:, k].unsqueeze(1) * h).sum(dim=-1).real + self.ssm_D * a[:, k]
            ys[:, k] = y

            if self.ent_mode != 'none':
                self._entropy_trace.append(self.compute_entropy(h))

        out = self.out_proj(ys * F.silu(b)) + r

        # 兼容 V30_CodeLoop: 保存最后一个激活
        self._last_a = a[:, -1, :]

        return out, 0.0  # (y, kl=0 兼容 V30_CodeLoop)
