# spark_seed_sosa.py
# 火种源自组织算法（Spark Seed Self-Organizing Algorithm, SOSA）
# 稀疏 Markov + Binary-Twin + 时间窗口 + 组合数编码
# By: 430 + GPT-5.1 Thinking

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Iterable, Optional, Set
import math
import time


# =========================
# 0. 基本数据结构
# =========================

@dataclass
class Event:
    """
    一条原始事件：
    - obs: 观测（日志、错误信息、环境信号等）
    - action: 系统采取的行为（可以是字符串/枚举/整数）
    - timestamp: 时间戳（float, 秒）
    """
    obs: Any
    action: Any
    timestamp: float


@dataclass
class BinaryTwin:
    """
    Binary-Twin 双态数块（窗口级经验单元）：
    - x_cont: 连续部分（如平均局部势能、行为多样性、窗口规模等）
    - b_bits: 离散二进制部分（窗口级结构/模式标志）
    """
    x_cont: List[float] = field(default_factory=list)
    b_bits: List[int] = field(default_factory=list)


class SparseMarkov:
    """
    稀疏 Markov 链：
    - N: 状态数
    - edges: from_state -> list of (to_state, prob)
    """

    def __init__(self, N: int):
        self.N: int = N
        self.edges: Dict[int, List[Tuple[int, float]]] = {}

    def add_edge(self, i: int, j: int, prob: float) -> None:
        """添加转移 i -> j 概率 prob（不自动归一化）。"""
        if i < 0 or i >= self.N or j < 0 or j >= self.N:
            raise ValueError("state index out of range")
        self.edges.setdefault(i, []).append((j, prob))

    def normalize_outgoing(self) -> None:
        """
        按每个 from_state 的出边归一化概率，使其和为 1。
        如果某个状态出边和为 0，则保持原样。
        """
        for i, lst in self.edges.items():
            s = sum(p for _, p in lst)
            if s > 0:
                self.edges[i] = [(j, p / s) for j, p in lst]

    def step(self, pi: List[float]) -> List[float]:
        """
        稀疏一步推进：
        pi_next[j] = sum_i pi[i] * P_ij
        """
        if len(pi) != self.N:
            raise ValueError("pi length must equal N")

        pi_next = [0.0] * self.N
        for i, p_i in enumerate(pi):
            if p_i == 0.0:
                continue
            if i not in self.edges:
                continue
            for j, p_ij in self.edges[i]:
                pi_next[j] += p_i * p_ij
        return pi_next


# =========================
# 1. 组合数工具（组合数编码/解码）
# =========================

def comb(n: int, k: int) -> int:
    """安全封装 math.comb，处理边界情况。"""
    if n < 0 or k < 0 or k > n:
        return 0
    return math.comb(n, k)


def comb_encode_subset(sorted_ids: List[int], M: int) -> Tuple[int, float]:
    """
    用组合数系统为子集赋一个 index，并粗略归一化到 [0,1]：

    输入：
      - sorted_ids: 升序排列的子集元素列表（例如 [0,2,5]）
      - M: 总元素数（行为组数）

    输出：
      - idx: 组合数索引（整数）
      - c_r: 归一化指标 ∈ [0,1]，可视为“组合占用度”
    """
    k = len(sorted_ids)
    idx = 0
    for j in range(1, k + 1):
        i_j = sorted_ids[j - 1]
        idx += comb(i_j, j)

    if M <= 0:
        return 0, 0.0

    # 使用 C(M, floor(M/2)) 作为上界的一个近似，用于归一化
    C_max = comb(M, M // 2)
    if C_max <= 0:
        c_r = 0.0
    else:
        c_r = max(0.0, min(1.0, idx / C_max))

    return idx, c_r


def comb_decode_subset(idx: int, k: int) -> List[int]:
    """
    组合数系统的解码：给定 idx 与子集大小 k，恢复 sorted_ids。
    需要知道 k，实际中多用于离线分析/行为链重放。
    """
    if k <= 0:
        return []

    ids: List[int] = []
    remaining = idx

    for j in reversed(range(1, k + 1)):  # j = k,k-1,...,1
        i_j = j
        while comb(i_j + 1, j) <= remaining:
            i_j += 1
        ids.insert(0, i_j)
        remaining -= comb(i_j, j)

    return ids


# =========================
# 2. 火种源自组织算法主体
# =========================

@dataclass
class SparkSeedState:
    markov: SparseMarkov
    pi: List[float]                  # 当前 Markov 状态分布
    window_buf: List[Event]          # 当前时间窗口内的事件
    last_flush: float                # 上次窗口刷新的时间戳
    dt_window: float                 # 窗口长度（秒）
    M_groups: int                    # 行为模式组数
    BT_history: List[BinaryTwin] = field(default_factory=list)


class SparkSeedSOSA:
    """
    火种源自组织算法（Spark Seed Self-Organizing Algorithm, SOSA）

    结构：
    - 稀疏 Markov 链：描述高层“模式状态”的转移（全局动态）
    - Binary-Twin：按时间窗口聚合经验块 (x_cont, b_bits)
    - 行为分组器：将具体行为映射到有限行为组
    - 组合数编码：对行为组集合编码为 (idx, c_r)，1 - c_r 代表未占用自由度
    - 更新规则：用 Binary-Twin + c_r 调节 Markov 的“探索 vs 固化”
    """

    def __init__(
        self,
        N_states: int,
        M_groups: int,
        dt_window: float = 5.0,
    ):
        """
        N_states: Markov 状态数
        M_groups: 行为模式组数
        dt_window: 时间窗口长度（秒）
        """
        markov = SparseMarkov(N_states)
        pi0 = [1.0 / N_states for _ in range(N_states)]
        now = time.time()
        self.state = SparkSeedState(
            markov=markov,
            pi=pi0,
            window_buf=[],
            last_flush=now,
            dt_window=dt_window,
            M_groups=M_groups,
        )

    # ------------- 对外接口 -------------

    def process_event(self, obs: Any, action: Any, timestamp: Optional[float] = None) -> None:
        """
        输入一条事件（适合流式调用）。
        当与上次刷新时间间隔超过 dt_window 时，会自动触发窗口刷新。
        """
        if timestamp is None:
            timestamp = time.time()
        e = Event(obs=obs, action=action, timestamp=timestamp)
        self.state.window_buf.append(e)

        if timestamp - self.state.last_flush >= self.state.dt_window:
            self._flush_window()
            self.state.last_flush = timestamp

    def force_flush(self) -> None:
        """手动触发一次窗口刷新（例如程序结束或阶段性检查时调用）。"""
        self._flush_window()
        self.state.last_flush = time.time()

    def get_state_distribution(self) -> List[float]:
        """当前 Markov 状态分布 pi。"""
        return list(self.state.pi)

    def get_markov(self) -> SparseMarkov:
        """访问稀疏 Markov 对象（方便外部设置/学习转移概率）。"""
        return self.state.markov

    def get_history(self) -> List[BinaryTwin]:
        """返回历史 Binary-Twin 块（可用于离线分析/可视化）。"""
        return list(self.state.BT_history)

    # ------------- 内部主流程 -------------

    def _flush_window(self) -> None:
        """刷新一个时间窗口，是算法的核心周期。"""
        buf = self.state.window_buf
        if not buf:
            return

        # 1) 生成窗口级 Binary-Twin
        BT_r = self._compute_binary_twin(buf)

        # 2) 产生行为分组集合 G_r
        G_r = self._group_actions(buf, self.state.M_groups)

        # 3) 组合数编码: G_r -> (idx_r, c_r)
        sorted_ids = sorted(list(G_r))
        idx_r, c_r = comb_encode_subset(sorted_ids, self.state.M_groups)

        # 4) 用 Binary-Twin + c_r 更新 Markov 状态分布
        self._update_markov_state(BT_r, c_r)

        # 5) 记录 Binary-Twin
        self.state.BT_history.append(BT_r)

        # 6) 清空窗口缓冲
        self.state.window_buf.clear()

    # ------------- Binary-Twin 计算 -------------

    def _compute_binary_twin(self, events: List[Event]) -> BinaryTwin:
        """
        窗口级 Binary-Twin 定义：

        连续部分 x_cont:
        - avg_energy: 平均局部势能（0~1）
        - diversity: 行为多样性（不同行为数 / 事件数）
        - size_norm: 窗口规模归一化（窗口越大越接近 1）

        离散部分 b_bits:
        - bit0: 是否存在高能行为（局部势能 > 0.8）
        - bit1: 行为模式是否 >= 3 种
        - bit2: 窗口事件数是否 >= 10
        """
        BT = BinaryTwin()

        scores: List[float] = []
        action_types: Set[Any] = set()

        for e in events:
            score_e = self._estimate_action_score(e)
            scores.append(score_e)
            action_types.add(e.action)

        avg_energy = sum(scores) / len(scores) if scores else 0.0
        diversity = len(action_types) / max(1, len(events))
        size_norm = math.tanh(len(events) / 10.0)

        BT.x_cont = [avg_energy, diversity, size_norm]

        # 离散标志位
        high_energy = any(s > 0.8 for s in scores)
        many_types = len(action_types) >= 3
        big_window = len(events) >= 10

        BT.b_bits = [
            1 if high_energy else 0,
            1 if many_types else 0,
            1 if big_window else 0,
        ]

        return BT

    def _estimate_action_score(self, e: Event) -> float:
        """
        估计单条行为的“局部势能”，完全内生于火种算法。

        思路：
        - obs 的长度代表信息量/复杂度；
        - action 决定一个基权重（不同类型行为对系统的冲击不同）；
        然后压缩到 [0,1] 区间。
        """
        obs_len = len(str(e.obs))
        obs_term = math.tanh(obs_len / 100.0)  # 越长信息量越大

        a = str(e.action).lower()
        if "observe" in a or "watch" in a:
            base = 0.3
        elif "patch_small" in a or "minor" in a:
            base = 0.6
        elif "patch_big" in a or "major" in a:
            base = 0.8
        elif "rollback" in a or "revert" in a:
            base = 0.5
        else:
            base = 0.4

        score = base * (0.5 + 0.5 * obs_term)
        return max(0.0, min(1.0, score))

    # ------------- 行为分组 -------------

    def _group_actions(self, events: List[Event], M_groups: int) -> Set[int]:
        """
        将具体行为映射到行为组 ID 的集合 G_r。

        默认实现：使用 hash(action) % M 的简单分组，
        后续可以替换为基于规则或聚类的更精细分组。
        """
        G_r: Set[int] = set()
        if M_groups <= 0:
            return G_r

        for e in events:
            gid = self._map_action_to_group(e.action, M_groups)
            G_r.add(gid)

        return G_r

    def _map_action_to_group(self, action: Any, M_groups: int) -> int:
        """默认行为映射：行为的 hash 模 M_groups。"""
        return abs(hash(action)) % M_groups

    # ------------- Markov 更新规则 -------------

    def _update_markov_state(self, BT_r: BinaryTwin, c_r: float) -> None:
        """
        使用 Binary-Twin + 行为组合占用度 c_r 更新 Markov 状态分布 pi。

        降维后的火种规则：

        1) pi_next = pi * P       # 稀疏 Markov 一步推进，全局模式更新
        2) 从窗口经验中提取：
           - avg_energy: 当前时间段整体表现（越高越稳定）
           - diversity: 行为多样性（越高说明在试更多策略）
           - size_norm: 窗口大小（越大说明这一段观测更粗）
           - c_r: 行为组合占用度（越大说明已经占用更多组合空间）

        3) 计算探索度 explore_factor ∈ [0,1]：
           - explore 高：更信任 pi_next，允许模式多走几步（偏探索）
           - explore 低：更多拉回吸引子分布（偏固化）
        """
        pi = self.state.pi
        P = self.state.markov

        # 1) 稀疏 Markov 一步推进
        pi_next = P.step(pi)

        # 2) 提取经验特征
        avg_energy = BT_r.x_cont[0] if len(BT_r.x_cont) > 0 else 0.0
        diversity = BT_r.x_cont[1] if len(BT_r.x_cont) > 1 else 0.0
        size_norm = BT_r.x_cont[2] if len(BT_r.x_cont) > 2 else 0.0

        # 3) 探索度：
        #    - c_r 越大：组合越满，探索度越低
        #    - diversity 越高：已经在试很多策略，可适当收敛
        #    - size_norm 越大：窗口信息更粗略，探索度适度降低
        base_explore = 1.0 - c_r
        explore_factor = (
            base_explore *
            (1.0 - 0.5 * diversity) *
            (0.5 + 0.5 * (1.0 - size_norm))
        )
        explore_factor = max(0.0, min(1.0, explore_factor))

        # 4) 吸引子分布：根据 avg_energy、diversity 设定低熵收缩中心
        attractor = self._prior_state_distribution(avg_energy, diversity)

        # 5) 混合 + 归一化
        N = P.N
        pi_mixed = [0.0] * N
        for s in range(N):
            pi_mixed[s] = explore_factor * pi_next[s] + (1.0 - explore_factor) * attractor[s]

        Z = sum(pi_mixed)
        if Z > 0:
            pi_mixed = [x / Z for x in pi_mixed]
        else:
            pi_mixed = [1.0 / N for _ in range(N)]

        self.state.pi = pi_mixed

    def _prior_state_distribution(self, avg_energy: float, diversity: float) -> List[float]:
        """
        吸引子状态分布：火种的“低熵收缩中心”。

        规则示例：
        - avg_energy 高：偏向某些“高质量模式”状态；
        - diversity 低：更集中于单个模式；
        - diversity 高：在多个模式之间平均铺开。
        """
        N = self.state.markov.N
        prior = [1.0 / N for _ in range(N)]

        # 根据 avg_energy 选一个“高质量模式”索引
        idx = int(max(0, min(N - 1, avg_energy * N)))
        # 多样性越低，越偏向单一模式
        bias_strength = 0.5 * (1.0 - diversity)

        prior = [(1.0 - bias_strength) * p for p in prior]
        prior[idx] += bias_strength

        Z = sum(prior)
        if Z > 0:
            prior = [x / Z for x in prior]
        else:
            prior = [1.0 / N for _ in range(N)]

        return prior


# =========================
# 3. 简单示例（可选 demo）
# =========================

if __name__ == "__main__":
    """
    极简 demo：
    - 构造 4 状态 Markov 链（环 + 自环）
    - 构造一个火种算法实例
    - 喂入一些假的事件，查看 pi 的演化情况
    """
    sosa = SparkSeedSOSA(N_states=4, M_groups=3, dt_window=2.0)

    # 设置一个简单的 Markov 链（0→1→2→3→0）
    mk = sosa.get_markov()
    mk.add_edge(0, 0, 0.2)
    mk.add_edge(0, 1, 0.8)
    mk.add_edge(1, 2, 1.0)
    mk.add_edge(2, 3, 1.0)
    mk.add_edge(3, 0, 1.0)
    mk.normalize_outgoing()

    # 模拟一串事件流
    now = time.time()
    actions = ["patch_small", "patch_big", "rollback", "observe"]

    for i in range(20):
        obs = f"log-{i}: something happened"
        action = actions[i % len(actions)]
        t = now + i * 0.5  # 每 0.5s 一个事件
        sosa.process_event(obs=obs, action=action, timestamp=t)

    # 强制刷新最后窗口
    sosa.force_flush()

    print("Final state distribution pi:", sosa.get_state_distribution())
    print("Binary-Twin blocks:", len(sosa.get_history()))