"""
V31.0/V31.2: MetaRuleWorld — 元学习任务分布世界
============================================
元学习设定: 每个"任务" = 一个规则变体 (remap 参数化)。

任务 T_i 的规则 = 一个寄存器重映射:
  remap=None              → 标准执行 (规则 A)
  remap=(0,1)             → R0↔R1 互换 (规则 B)   [兼容旧接口: pair]
  remap=(1,3,2,0)         → 任意排列 perm[reg]    [V31.2: 全 24 种排列]

V31.2 长流扩展 (S4 对称群, 24 条规则):
  S4 = 1 identity + 6 对换 + 8 三轮换 + 6 四轮换 + 3 双对换
  训练: identity + 6 对换 + 5 三轮换 = 12 条
  未见: 3 三轮换 + 6 四轮换 + 3 双对换 = 12 条 (结构上不同 → "没学过的学习方法")

目标 (Javed & White NeurIPS 2019 精神):
  不学"预测所有规则"的单一 head (V30.4 失败模式),
  而是学一个表示 φ, 使任意规则可以快速适应 (内循环 head 更新)。

数据格式 (离线监督式):
  context = 最近 window 个 READ 观察 (5 tokens each) → 预测下一个 READ (5 tokens)
  模型从观察历史推断当前规则 → 预测 EXEC 后的寄存器状态。
"""

import torch
import random
from itertools import permutations
from .code_world import (
    RegisterMachine, CodeSandboxWorld,
    N_REGISTERS, INSTR_LEN, OPCODE_TOKEN_OFFSET,
    OPCODES, OPCODES_INV,
)


def s4_groups():
    """S4 按结构分组: (identity, swaps, 3cycles, 4cycles, double_swaps)。"""
    allp = list(permutations(range(4)))
    identity = [(0, 1, 2, 3)]
    swaps, c3, c4, dbl = [], [], [], []
    for p in allp:
        if p == (0, 1, 2, 3):
            continue
        fixed = sum(1 for i in range(4) if p[i] == i)
        if fixed == 2:
            swaps.append(p)          # 单对换
        elif fixed == 1:
            c3.append(p)             # 三轮换
        elif fixed == 0:
            # 四轮换 (阶 4) vs 双对换 (阶 2)
            if all(p[p[i]] == i for i in range(4)):
                dbl.append(p)        # 双对换 (Klein)
            else:
                c4.append(p)         # 四轮换
    return identity, swaps, c3, c4, dbl


# V31.2 规则划分: 训练 = identity + 6 对换 + 5 三轮换 (12)
#                 未见 = 3 三轮换 + 6 四轮换 + 3 双对换 (12, 结构不同)
def build_rule_splits(seed=42):
    identity, swaps, c3, c4, dbl = s4_groups()
    rng = random.Random(seed)
    rng.shuffle(c3); rng.shuffle(c4); rng.shuffle(dbl)
    train = identity + swaps + c3[:5]
    test = c3[5:] + c4 + dbl
    return train, test


# 各 opcode 的寄存器操作数位置 (立即数位置不重映射):
#   SET dst,val   → a1 寄存器, a2 立即数 (0-99)
#   MOV src,dst   → a1,a2 寄存器
#   ADD/SUB a,b,d → a1,a2,a3 寄存器
#   INC/DEC reg   → a1 寄存器
#   SWP a,b       → a1,a2 寄存器
_REG_OPPOS = {0: (), 1: (0,), 2: (0, 1), 3: (0, 1, 2), 4: (0, 1, 2),
              5: (0,), 6: (0,), 7: (0, 1)}


class MetaRuleMachine(RegisterMachine):
    """RegisterMachine + 参数化寄存器重映射 (任意排列或 pair)。

    V31.2 修复 (原 V31.0 bug): 只重映射**寄存器操作数**位置,
    立即数 (SET 的 a2, 0-99) 不受 remap 影响。
    原实现 `_remap` 作用于所有操作数 → 立即数被 mod 4 重映射,
    "SET R0 42 → R1=42" 变成 "SET R1 2" — 规则语义被污染
    (V31.0 世界自测的 remap 项其实一直失败, 计数器有 bug 显示 7/7)。
    """

    def __init__(self, V, remap_pair=None, remap=None):
        super().__init__(V)
        if remap is None:
            if remap_pair is None:
                remap = (0, 1, 2, 3)
            else:
                i, j = remap_pair
                remap = tuple(j if k == i else (i if k == j else k)
                              for k in range(4))
        self.remap_pair = remap_pair  # 兼容旧接口 (V31.0)
        self.remap = remap            # V31.2: 完整排列 (identity = (0,1,2,3))

    def _remap(self, reg: int) -> int:
        return self.remap[reg % N_REGISTERS]

    def step_execute(self) -> bool:
        if self.halted or self.pc >= len(self.program):
            self.halted = True
            return False
        if self.step_count >= self.max_steps:
            self.halted = True
            return False

        opcode, a1, a2, a3 = self.program[self.pc]
        name = OPCODES_INV[opcode]

        if self.remap != (0, 1, 2, 3):
            args = [a1, a2, a3]
            for pos in _REG_OPPOS[opcode]:
                args[pos] = self._remap(args[pos])
            a1, a2, a3 = args

        try:
            if name == 'NOP':
                pass
            elif name == 'SET':
                self.regs[a1 % N_REGISTERS] = a2 % self.V
            elif name == 'MOV':
                self.regs[a2 % N_REGISTERS] = self.regs[a1 % N_REGISTERS]
            elif name == 'ADD':
                self.regs[a3 % N_REGISTERS] = (
                    self.regs[a1 % N_REGISTERS] + self.regs[a2 % N_REGISTERS]
                ) % self.V
            elif name == 'SUB':
                self.regs[a3 % N_REGISTERS] = max(
                    0, self.regs[a1 % N_REGISTERS] - self.regs[a2 % N_REGISTERS]
                )
            elif name == 'INC':
                self.regs[a1 % N_REGISTERS] = (
                    self.regs[a1 % N_REGISTERS] + 1
                ) % self.V
            elif name == 'DEC':
                self.regs[a1 % N_REGISTERS] = max(
                    0, self.regs[a1 % N_REGISTERS] - 1
                )
            elif name == 'SWP':
                ri, rj = a1 % N_REGISTERS, a2 % N_REGISTERS
                self.regs[ri], self.regs[rj] = self.regs[rj], self.regs[ri]
        except Exception:
            return False

        self.pc += 1
        self.step_count += 1
        return True


class MetaRuleWorld(CodeSandboxWorld):
    """代码执行世界, 规则由 remap_pair 参数化。观察格式与 CodeSandboxWorld 一致。"""

    K = 5
    READ_WINDOW = N_REGISTERS + 1  # 4 regs + 1 opcode
    ACTIONS = ['READ', 'EXEC', 'RESET', 'NEWREG', 'TOME']

    def __init__(self, V: int, remap_pair=None, remap=None,
                 max_program_len: int = 8):
        self.V = V
        self.TOME = V + 1
        self.world_vocab_size = V + 2
        self.max_program_len = max_program_len
        self.remap_pair = remap_pair
        self.remap = remap
        self._new_episode()

    def _new_episode(self):
        self.vm = MetaRuleMachine(self.V, remap_pair=self.remap_pair,
                                  remap=self.remap)
        self.exec_count = 0
        self.current_step = 0
        self.n_erases = 0

        prog_len = random.randint(2, self.max_program_len)
        self.vm.generate_random_program(prog_len)

        for i in range(N_REGISTERS):
            self.vm.regs[i] = random.randint(0, min(self.V - 1, 50))

        pre_steps = random.randint(0, min(2, len(self.vm.program) - 1))
        for _ in range(pre_steps):
            self.vm.step_execute()
            self.exec_count += 1

    def reset(self):
        self._new_episode()


# ============================================================
# 任务数据生成 (离线监督式)
# ============================================================
# V31 观察格式 (8 tokens): [R0,R1,R2,R3, opcode+100, a1, a2, a3]
#   → 完整指令可见, "预测 EXEC 后状态"是确定性问题 (信息完备)
#   → 规则差异 (remap) 真正可学习: SET R0 42 → A: R0=42 / B: R1=42
# 注意: 这与 V30 系列不同 (V30 只有 opcode 无操作数 → 信息不足 → pred 2-8%)

def _instr_tokens(vm):
    """当前 PC 指令的 4 个 token: [opcode+100, a1, a2, a3], 程序结束用 TOME。"""
    if vm.halted or vm.pc >= len(vm.program):
        return [vm.V + 1] * 4  # TOME
    opcode, a1, a2, a3 = vm.program[vm.pc]
    return [opcode + 100, min(a1, 99), min(a2, 99), min(a3, 99)]


def _world_obs(world):
    """8-token 观察: [R0,R1,R2,R3, instr...]"""
    vm = world.vm
    state = [min(r, world.V - 1) for r in vm.regs]
    return state + _instr_tokens(vm)


def gen_episode_pairs(world, V, max_len=8, window=3, stop_at_halt=False):
    """
    从一个 episode 生成 (context, target) 监督对。

    context = 最近 window 个观察拼接 (window*8 tokens)
    target  = EXEC 后的观察 (8 tokens)

    模型从观察历史推断当前规则 → 预测 EXEC 后的完整状态。
    由于观察含完整指令, 给定规则下 target 是确定性的。

    stop_at_halt (V33 世界修复): 程序结束 (halted) 后立即停止,
    不生成含 TOME(201) 填充的 target → 每个 target 都有信息量
    (旧世界 ~77.5% target 后 4 token 是 TOME, 虚高 full-8 acc)。
    """
    pairs = []
    obs = torch.tensor(_world_obs(world), dtype=torch.long)
    history = [obs]
    for _ in range(max_len):
        world.query(1)  # EXEC
        obs = torch.tensor(_world_obs(world), dtype=torch.long)
        if stop_at_halt and world.vm.halted:
            break  # 程序结束: 不再追加 TOME 观察
        history.append(obs)
    for t in range(window, len(history)):
        ctx = torch.cat(history[t - window:t]).long()   # window*8
        tgt = history[t].long()                          # 8
        pairs.append((ctx, tgt))
    return pairs


def gen_task_data(remap_pair, V, n_episodes=48, max_len=8, window=3,
                  seed=42, max_program_len=8, remap=None,
                  stop_at_halt=False):
    """
    生成一个任务 (规则) 的数据集。
    返回: list of (context_ids, target_ids), 均 clamp 到 [0, V+1]。
    """
    rng_state = random.getstate()
    random.seed(seed)
    pairs = []
    for _ in range(n_episodes):
        w = MetaRuleWorld(V, remap_pair=remap_pair, remap=remap,
                          max_program_len=max_program_len)
        pairs.extend(gen_episode_pairs(w, V, max_len=max_len, window=window,
                                       stop_at_halt=stop_at_halt))
    random.setstate(rng_state)
    # clamp
    pairs = [(c.clamp(0, V + 1), t.clamp(0, V + 1)) for c, t in pairs]
    return pairs


def gen_task_data_split(remap_pair, V, n_support=48, n_query=24,
                        max_len=8, window=3, seed=42, max_program_len=8,
                        remap=None, stop_at_halt=False):
    """生成任务的 support/query 划分 (不同 seed 保证不重叠)。"""
    support = gen_task_data(remap_pair, V, n_episodes=n_support,
                            max_len=max_len, window=window,
                            seed=seed, max_program_len=max_program_len,
                            remap=remap, stop_at_halt=stop_at_halt)
    query = gen_task_data(remap_pair, V, n_episodes=n_query,
                          max_len=max_len, window=window,
                          seed=seed + 1000, max_program_len=max_program_len,
                          remap=remap, stop_at_halt=stop_at_halt)
    return support, query


def _self_test():
    ok = 0
    total = 0
    def check(cond, msg):
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
            print(f"  OK {msg}")
        else:
            print(f"  FAIL {msg}")

    random.seed(42)
    V = 100

    # 1. 规则 A (无 remap): SET R0 42 → R0=42
    vm_a = MetaRuleMachine(V=100, remap_pair=None)
    vm_a.load_instruction([101, 0, 42, 0])
    vm_a.step_execute()
    check(vm_a.regs[0] == 42 and vm_a.regs[1] == 0,
          f"A: SET R0 42 → R0={vm_a.regs[0]}, R1={vm_a.regs[1]}")

    # 2. 规则 B (0,1): SET R0 42 → R1=42
    vm_b = MetaRuleMachine(V=100, remap_pair=(0, 1))
    vm_b.load_instruction([101, 0, 42, 0])
    vm_b.step_execute()
    check(vm_b.regs[1] == 42 and vm_b.regs[0] == 0,
          f"B: SET R0 42 → R1={vm_b.regs[1]}, R0={vm_b.regs[0]}")

    # 3. 规则 C (0,2): SET R0 42 → R2=42
    vm_c = MetaRuleMachine(V=100, remap_pair=(0, 2))
    vm_c.load_instruction([101, 0, 42, 0])
    vm_c.step_execute()
    check(vm_c.regs[2] == 42 and vm_c.regs[0] == 0,
          f"C: SET R0 42 → R2={vm_c.regs[2]}, R0={vm_c.regs[0]}")

    # 4. 规则 D (1,2): SET R1 42 → R2=42
    vm_d = MetaRuleMachine(V=100, remap_pair=(1, 2))
    vm_d.load_instruction([101, 1, 42, 0])
    vm_d.step_execute()
    check(vm_d.regs[2] == 42 and vm_d.regs[1] == 0,
          f"D: SET R1 42 → R2={vm_d.regs[2]}, R1={vm_d.regs[1]}")

    # 5. 数据生成: context/target shapes (8-token 观察)
    w = MetaRuleWorld(V=100, remap_pair=(0, 1))
    pairs = gen_episode_pairs(w, V, max_len=8, window=3)
    check(len(pairs) > 0, f"episode pairs = {len(pairs)}")
    ctx, tgt = pairs[0]
    check(ctx.shape[0] == 24 and tgt.shape[0] == 8,
          f"ctx={tuple(ctx.shape)} tgt={tuple(tgt.shape)} (24/8)")

    # 5b. 信息完备性: 给定规则, target 确定 (SET R0 42 在规则 A/B 下不同)
    random.seed(3)
    wA = MetaRuleWorld(V=100, remap_pair=None)
    wA.vm.program = [(1, 0, 42, 0)]; wA.vm.pc = 0; wA.vm.regs = [0, 0, 0, 0]
    wA.vm.halted = False
    obsA = _world_obs(wA)
    wA.query(1)  # EXEC
    afterA = _world_obs(wA)
    wB = MetaRuleWorld(V=100, remap_pair=(0, 1))
    wB.vm.program = [(1, 0, 42, 0)]; wB.vm.pc = 0; wB.vm.regs = [0, 0, 0, 0]
    wB.vm.halted = False
    obsB = _world_obs(wB)
    wB.query(1)  # EXEC
    afterB = _world_obs(wB)
    check(obsA[:4] == obsB[:4], "same state before EXEC")
    check(afterA[:4] == [42, 0, 0, 0], f"A after SET R0 42: {afterA[:4]}")
    check(afterB[:4] == [0, 42, 0, 0], f"B after SET R0 42: {afterB[:4]}")
    check(afterA[:4] != afterB[:4], "rule difference visible in targets")

    # 6. support/query 划分
    sup, qry = gen_task_data_split((0, 1), V, n_support=8, n_query=4, seed=42)
    check(len(sup) > 0 and len(qry) > 0, f"support={len(sup)} query={len(qry)}")

    # 7. 不同规则产生不同数据分布 (相同 seed, 相同程序 → 不同状态)
    random.seed(7)
    w1 = MetaRuleWorld(V=100, remap_pair=None)
    w2 = MetaRuleWorld(V=100, remap_pair=(0, 1))
    # 手动构造相同程序
    prog = [(1, 0, 42, 0), (5, 0, 0, 0)]  # SET R0 42; INC R0
    w1.vm.program = prog[:]; w1.vm.pc = 0; w1.vm.regs = [0, 0, 0, 0]
    w2.vm.program = prog[:]; w2.vm.pc = 0; w2.vm.regs = [0, 0, 0, 0]
    w1.vm.step_execute(); w2.vm.step_execute()
    s1 = w1.vm.regs[:]; s2 = w2.vm.regs[:]
    check(s1[0] == 42 and s2[1] == 42,
          f"same prog diverges: A={s1[:2]}, B={s2[:2]}")

    # 8. V31.2 排列规则: 只重映射寄存器, 立即数不动
    vm_p = MetaRuleMachine(V=100, remap=(1, 3, 2, 0))
    vm_p.load_instruction([101, 0, 42, 0])   # SET R0 42
    vm_p.step_execute()
    check(vm_p.regs[1] == 42,
          f"perm(1,3,2,0): SET R0 42 → R1={vm_p.regs[1]} (立即数 42 不受影响)")
    vm_p2 = MetaRuleMachine(V=100, remap=(1, 3, 2, 0))
    vm_p2.load_instruction([104, 0, 1, 2])   # SUB R0 R1 → R2
    vm_p2.step_execute()
    # SUB R0(→1) R1(→3) → R2(→2): regs[2] = regs[1] - regs[3] = 0
    check(vm_p2.regs[2] == 0,
          f"perm SUB: R0,R1→R2 remap 后 regs[2]=regs[1]-regs[3]")

    print(f"\n  {ok}/{total} passed")
    return ok == total


if __name__ == "__main__":
    _self_test()
