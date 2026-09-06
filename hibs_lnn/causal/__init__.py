"""headless · Causal AI 技术栈

把《Causal AI》一书的因果技术落地到 headless 的 S4 元学习世界：

  dag.py            DAG 数据结构 + d-separation（第 3-4 章）
  scm.py            结构因果模型 SCM + do() 干预 + 反事实（第 6-9 章）
  identification.py 后门调整 / 因果效应识别（第 10-11 章）
  discovery.py      因果发现（条件独立检验 + PC 式骨架，第 4/发现章）
  causal_proposer.py 因果驱动的自主数据提议器（改进 S4ValueProposer）
  causal_rl.py      Causal RL：状态/动作/奖励 + 因果 credit assignment（第 12 章）

实现原则：S4 世界是有限离散代数结构，核心算法用 numpy 自实现（无重型外部依赖），
对应书上 pgmpy / DoWhy / causal-learn 的轻量可复现等价物。
"""
from .dag import Graph, d_separated, is_dag, ancestors, descendants
from .scm import SCM, S4WorldSCM
from .identification import (
    backdoor_adjustment_set, causal_effect, g_formula)
from .discovery import (
    ci_test, pc_skeleton, CausalDiscovery)
from .causal_proposer import CausalProposer
from .causal_rl import CausalRLAgent

__all__ = [
    "Graph", "d_separated", "is_dag", "ancestors", "descendants",
    "SCM", "S4WorldSCM",
    "backdoor_adjustment_set", "causal_effect", "g_formula",
    "ci_test", "pc_skeleton", "CausalDiscovery",
    "CausalProposer", "CausalRLAgent",
]