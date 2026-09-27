"""核心：地址空间、两级页表、带 ASID 的 LRU TLB、失效票据协议。

公开模型见 :mod:`vmshoot` 包文档字符串。本模块不做任何真实 OS 内存映射。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum

from .errors import (
    AsidBusy,
    AddressSpaceLimit,
    CrossPageAccess,
    InvalidAddressSpace,
    InvalidationsPending,
    MappingConflict,
    MappingExists,
    NotMapped,
    TicketError,
)
from .memory import PhysicalMemory
from .primitives import (
    HUGE_PAGE_MASK,
    HUGE_PAGE_SIZE,
    L1_SIZE,
    L2_SIZE,
    MAX_ADDRESS_SPACES,
    NUM_CORES,
    PAGE_BITS,
    PAGE_MASK,
    PAGE_SIZE,
    PERM_EXEC,
    PERM_READ,
    PERM_WRITE,
    TLB_SIZE,
    VADDR_MAX,
    perm_to_str,
    split_vaddr,
    validate_int,
    validate_perms,
    validate_size,
    validate_vaddr,
)


class AccessType(IntEnum):
    """translate 的访问类型，位值与权限位一一对应。"""

    READ = PERM_READ
    WRITE = PERM_WRITE
    EXECUTE = PERM_EXEC


# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class PTE:
    """页表项。``huge=True`` 时它是一级叶子，``ppn`` 为 16 对齐的起始帧号。"""

    ppn: int
    perms: int
    huge: bool = False


@dataclass
class Ticket:
    """失效票据。

    * ``kind``：``map`` / ``unmap`` / ``protect``；
    * ``keys``：需要失效的 TLB key（小页粒度的 ``(asid, vpn)``）；
    * ``targets``：实际收到失效请求、必须 ACK 的核。目标集在发票时按
      “确实可能缓存过该地址空间”的参与范围快照得出（判断依据不足时保守扩大），
      不再恒为全部核；迟到的旧 ACK 表明此前漏判时可在后续票据上动态扩展；
    * ``acks``：已 ACK 的核集合（含迟到 ACK）；
    * 票据在 :meth:`MMU.finish` 成功后关闭并从活动票据表移除。
    """

    id: int
    asid: int
    kind: str
    vaddr: int
    keys: tuple[tuple[int, int], ...]
    targets: list[int]
    acks: set[int] = field(default_factory=set)
    closed: bool = False
    # unmap 摘下的帧；在 finish（所有目标核 ACK）后才允许回收到帧池，
    # 防止“旧物理页被提前复用”。None 表示该票据不涉及帧回收。
    deferred_frames: tuple[int, ...] | None = None

    @property
    def pending(self) -> tuple[int, ...]:
        return tuple(c for c in self.targets if c not in self.acks)

    @property
    def done(self) -> bool:
        return not self.pending

    @property
    def notified_cores(self) -> tuple[int, ...]:
        """实际通知（必须等待 ACK）的核，按核号升序。"""
        return tuple(sorted(self.targets))


@dataclass
class TranslationResult:
    """translate 的结果（永不因 not-mapped / 权限不足抛异常）。

    * 成功：``ok=True``，``fault=None``，``phys_addr`` 为物理地址；
    * 失败：``ok=False``，``fault`` 取值
      ``"not_mapped" | "read" | "write" | "execute"``；
    * ``via``：``"tlb"``（命中缓存，可能是尚未 ACK 的旧项）或 ``"walk"``（遍历页表）。
    """

    ok: bool
    phys_addr: int | None
    fault: str | None
    access: str
    via: str
    asid: int
    vaddr: int
    perms: int | None = None

    def __repr__(self) -> str:  # 紧凑可读
        if self.ok:
            return (
                f"TranslationResult(ok, asid={self.asid}, vaddr=0x{self.vaddr:x}, "
                f"phys=0x{self.phys_addr:x}, perms={perm_to_str(self.perms or 0)}, via={self.via})"
            )
        return (
            f"TranslationResult(fault={self.fault}, asid={self.asid}, "
            f"vaddr=0x{self.vaddr:x}, via={self.via})"
        )


@dataclass
class WalkResult:
    """绕过 TLB 的纯页表遍历结果（“无 TLB 遍历器”）。"""

    ppn: int
    perms: int
    huge: bool
    vaddr: int

    @property
    def phys_addr(self) -> int:
        off_mask = HUGE_PAGE_MASK if self.huge else PAGE_MASK
        return self.ppn * PAGE_SIZE + (self.vaddr & off_mask)


# ===========================================================================
# 地址空间（页表结构操作；不含 TLB / 票据逻辑）
# ===========================================================================
class AddressSpace:
    def __init__(self, asid: int, mem: PhysicalMemory):
        self.asid = asid
        self.mem = mem
        self.l1: list[None | PTE | list] = [None] * L1_SIZE

    # ---- 遍历 ---------------------------------------------------------------
    def walk(self, vaddr: int) -> WalkResult | None:
        l1i, l2i, _ = split_vaddr(vaddr)
        e = self.l1[l1i]
        if e is None:
            return None
        if isinstance(e, PTE):
            if not e.huge:  # 内部不变量：L1 上的 PTE 必为大页叶子
                raise AssertionError("内部错误：一级页表中出现非叶子 PTE")
            return WalkResult(e.ppn, e.perms, True, vaddr)
        pte = e[l2i]
        if pte is None:
            return None
        return WalkResult(pte.ppn, pte.perms, False, vaddr)

    # ---- map ----------------------------------------------------------------
    def map_small(self, vaddr: int, perms: int) -> tuple[PTE, int]:
        """建立小页映射。返回 (新 PTE, 新帧号)；已被覆盖则 MappingExists。"""
        if vaddr & PAGE_MASK:
            raise ValueError(f"小页 map 要求 256 对齐，得到 vaddr=0x{vaddr:x}")
        l1i, l2i, _ = split_vaddr(vaddr)
        l1e = self.l1[l1i]
        if isinstance(l1e, PTE):
            raise MappingExists(
                f"vaddr=0x{vaddr:x} 已被起始于 0x{(l1i << 12):x} 的大页覆盖"
            )
        if l1e is not None and l1e[l2i] is not None:
            raise MappingExists(f"vaddr=0x{vaddr:x} 已存在映射")
        ppn = self.mem.alloc_frame()  # 先分配：若 FrameExhausted，页表完全不变
        if l1e is None:
            l1e = [None] * L2_SIZE
            self.l1[l1i] = l1e
        pte = PTE(ppn=ppn, perms=perms, huge=False)
        l1e[l2i] = pte
        return pte, ppn

    def map_huge(self, vaddr: int, perms: int) -> tuple[PTE, int]:
        if vaddr & HUGE_PAGE_MASK:
            raise ValueError(f"大页 map 要求 4096 对齐，得到 vaddr=0x{vaddr:x}")
        l1i, _, _ = split_vaddr(vaddr)
        if self.l1[l1i] is not None:
            cur = self.l1[l1i]
            if isinstance(cur, PTE):
                raise MappingExists(f"vaddr=0x{vaddr:x} 已存在大页映射")
            raise MappingExists(f"vaddr=0x{vaddr:x} 的一级项已含小页表，不能直接大页映射")
        ppn = self.mem.alloc_huge()
        pte = PTE(ppn=ppn, perms=perms, huge=True)
        self.l1[l1i] = pte
        return pte, ppn

    # ---- unmap --------------------------------------------------------------
    def unmap(self, vaddr: int) -> tuple[PTE, bool]:
        """移除映射。

        ``vaddr`` 必须是映射区起始：小页为 256 对齐地址；大页必须给其 4096
        对齐的起始地址——unmap 不做隐式大页拆分（大页内部地址抛
        :class:`MappingConflict`，避免静默曲解调用方意图）。
        """
        if vaddr & PAGE_MASK:
            raise ValueError(f"unmap 要求 256 对齐，得到 vaddr=0x{vaddr:x}")
        l1i, l2i, _ = split_vaddr(vaddr)
        l1e = self.l1[l1i]
        if isinstance(l1e, PTE):
            if l2i != 0:
                raise MappingConflict(
                    f"vaddr=0x{vaddr:x} 位于大页 0x{(l1i << 12):x} 内部；"
                    "unmap 不隐式拆分大页，请对大页起始地址 unmap，"
                    "或先 protect 拆分后再逐页 unmap"
                )
            pte = l1e
            self.l1[l1i] = None       # 只摘表；帧的回收延迟到票据 finish
            frames = tuple(pte.ppn + j for j in range(L2_SIZE))
            return pte, True, frames
        if l1e is None or l1e[l2i] is None:
            raise NotMapped(f"vaddr=0x{vaddr:x} 没有映射")
        pte = l1e[l2i]
        l1e[l2i] = None              # 只摘表；帧的回收延迟到票据 finish
        if all(x is None for x in l1e):
            self.l1[l1i] = None  # 回收空的二级表
        return pte, False, (pte.ppn,)

    # ---- protect ------------------------------------------------------------
    def protect_small(self, vaddr: int, perms: int) -> PTE:
        l1i, l2i, _ = split_vaddr(vaddr)
        l1e = self.l1[l1i]
        if not isinstance(l1e, list) or l1e[l2i] is None:
            raise NotMapped(f"vaddr=0x{vaddr:x} 没有映射，不能 protect")
        pte = l1e[l2i]
        pte.perms = perms
        return pte

    def split_huge_for_protect(self, vaddr: int, perms: int) -> tuple[list, int]:
        """就地把大页拆成 16 个小页：帧完全不变，目标小页用 ``perms``，其余 15
        页保留原权限。

        大页本来就占用 16 个连续、16 对齐的帧；拆分只是在同一批帧上安装一张
        二级表，因此物理地址不变、不分配也不释放帧、操作不可能中途失败，
        也就不存在部分变更。返回 (新二级表, 目标 l2 下标)。
        """
        l1i, l2i, _ = split_vaddr(vaddr)
        l1e = self.l1[l1i]
        if not isinstance(l1e, PTE) or not l1e.huge:
            raise NotMapped(f"vaddr=0x{vaddr:x} 处不是大页映射，不能按大页拆分保护")
        base_ppn = l1e.ppn
        old_perms = l1e.perms
        table = [
            PTE(ppn=base_ppn + j, perms=old_perms, huge=False)
            for j in range(L2_SIZE)
        ]
        table[l2i].perms = perms
        self.l1[l1i] = table
        return table, l2i

    # ---- 销毁 ---------------------------------------------------------------
    def destroy(self) -> None:
        """释放全部帧与页表结构。"""
        for l1e in self.l1:
            if isinstance(l1e, PTE):
                self.mem.free_huge(l1e.ppn)
            elif isinstance(l1e, list):
                for pte in l1e:
                    if pte is not None:
                        self.mem.free_frame(pte.ppn)
        self.l1 = [None] * L1_SIZE

    def mappings(self) -> list[tuple[int, int, int, bool]]:
        """枚举 ``(起始 vaddr, 字节数, 权限, 是否大页)``，按地址升序。"""
        out: list[tuple[int, int, int, bool]] = []
        for l1i, l1e in enumerate(self.l1):
            base = l1i * HUGE_PAGE_SIZE
            if isinstance(l1e, PTE):
                out.append((base, HUGE_PAGE_SIZE, l1e.perms, True))
            elif isinstance(l1e, list):
                for l2i, pte in enumerate(l1e):
                    if pte is not None:
                        out.append((base + l2i * PAGE_SIZE, PAGE_SIZE, pte.perms, False))
        return out


# ===========================================================================
# MMU：TLB + 票据协议门面
# ===========================================================================
class MMU:
    """两级页表 + 每核 8 项 LRU TLB（key 含 ASID）+ 失效票据的模拟机。"""

    def __init__(self, mem: PhysicalMemory | None = None):
        if mem is not None and not isinstance(mem, PhysicalMemory):
            raise TypeError("mem 必须是 PhysicalMemory 或 None")
        self.mem = mem if mem is not None else PhysicalMemory()
        self._spaces: dict[int, AddressSpace] = {}
        # 每核一个 OrderedDict：key=(asid, vpn) -> (ppn, perms, huge)，头部为 LRU
        self._tlb: list[OrderedDict] = [OrderedDict() for _ in range(NUM_CORES)]
        self._tickets: dict[int, Ticket] = {}
        self._next_ticket = 1
        # 每个地址空间一个“参与核”集合：其中的核确实（或无法排除地）可能在
        # 该 ASID 上缓存过 TLB 项。票据目标核 = 参与核 ∪ 该 ASID 上所有未完成
        # 票据的目标核（后者保证旧票据迟到 ACK 前，后续票据不漏等任何核）。
        self._participants: dict[int, set[int]] = {}
        # 每核当前运行的地址空间；None 表示该核未运行在任何地址空间。
        self._current_asid: list[int | None] = [None] * NUM_CORES

    # ---- 基础校验 -----------------------------------------------------------
    @staticmethod
    def _check_core(core_id) -> int:
        c = validate_int(core_id, "core_id")
        if not 0 <= c < NUM_CORES:
            raise ValueError(f"core_id={c} 越界（只有核 0..{NUM_CORES - 1}）")
        return c

    def _space(self, asid) -> AddressSpace:
        a = validate_int(asid, "asid")
        sp = self._spaces.get(a)
        if sp is None:
            raise InvalidAddressSpace(f"ASID {a} 不存在或已销毁")
        return sp

    def _pending_for_asid(self, asid: int) -> list[int]:
        return [t.id for t in self._tickets.values() if t.asid == asid and not t.closed]

    # ---- ASID 生命周期 ------------------------------------------------------
    def create_address_space(self) -> int:
        """分配一个新地址空间，返回 ASID（0..3）。

        * 4 个槽位全活动 -> :class:`AddressSpaceLimit`；
        * 空闲槽位若仍挂着未 finish 的票据 -> 该号不能复用；
          全部空闲槽位都被未完成票据占据时抛 :class:`AsidBusy`。
        """
        active = set(self._spaces)
        if len(active) >= MAX_ADDRESS_SPACES:
            raise AddressSpaceLimit(f"活动地址空间已达上限 {MAX_ADDRESS_SPACES}")
        busy: list[int] = []
        for cand in range(MAX_ADDRESS_SPACES):
            if cand in active:
                continue
            pending = self._pending_for_asid(cand)
            if pending:
                busy.append(cand)
                continue
            sp = AddressSpace(cand, self.mem)
            self._spaces[cand] = sp
            self._participants[cand] = set()  # 清掉销毁后迟到 ACK 留下的残留集合
            return cand
        raise AsidBusy(busy[0], self._pending_for_asid(busy[0]))

    def destroy_address_space(self, asid) -> None:
        """销毁地址空间。

        销毁时同步冲刷所有核上该 ASID 的全部 TLB 项并释放页表与物理帧；
        此后该 ASID 上的翻译一律失败（:class:`InvalidAddressSpace`）。

        但**尚未 finish 的票据会作为“僵尸票据”保留**：各核仍须 ACK、调用方仍须
        finish；在此之前该 ASID 编号不能被新地址空间复用（:class:`AsidBusy`），
        这正是题面“新 ASID 不能复用未完成票据”的约束。全部票据 finish 后编号立即可用。
        """
        sp = self._space(asid)
        for tlb in self._tlb:
            dead = [k for k in tlb if k[0] == sp.asid]
            for k in dead:
                del tlb[k]
        sp.destroy()
        del self._spaces[sp.asid]
        # 该空间已不存在：参与核集合随之撤销，每核的“当前空间”若指向它则置空。
        self._participants.pop(sp.asid, None)
        for core in range(NUM_CORES):
            if self._current_asid[core] == sp.asid:
                self._current_asid[core] = None

    # ---- 票据 ---------------------------------------------------------------
    def _core_holds_asid_keys(self, core: int, asid: int) -> bool:
        """该核 TLB 中当前是否还有属于 ``asid`` 的 key。"""
        return any(k[0] == asid for k in self._tlb[core].keys())

    def _shrink_participants(self, asid: int) -> None:
        """从参与范围中移除确定不可能再持有该 ASID 旧项的核。

        移除判据（两者同时满足，依据不足则保守保留）：

        1. 该核 TLB 中已没有任何该 ASID 的 key（陈旧项已被 ACK/淘汰/冲刷清空）；
        2. 该核不是该 ASID 上任何未完成票据的目标核
           （它可能正持有对更早票据的“迟到 ACK”，在 ACK 真正到达前不能漏等）。
        """
        group = self._participants.get(asid)
        if not group:
            return
        stale_targets: set[int] = set()
        for t in self._tickets.values():
            if t.asid == asid and not t.closed:
                stale_targets.update(t.targets)
        for core in range(NUM_CORES):
            if core not in group or core in stale_targets:
                continue
            # 移出需要同时具备：核已显式切走（current 不是该空间）、TLB 中该
            # 空间 key 已清空。仍正运行在该空间、或 TLB 仍有 key（含切出后靠
            # ASID 保留的缓存）都必须保留。从未显式 switch_to 过的核其 current
            # 为 None，但缺少“切出”这一明确证据，也保守保留，不漏通知。
            if self._current_asid[core] is not None \
                    and self._current_asid[core] != asid \
                    and not self._core_holds_asid_keys(core, asid):
                group.discard(core)

    def _snapshot_targets(self, asid: int) -> list[int]:
        """计算新票据的目标核：参与核 ∪ 该 ASID 未完成票据的全部目标核。

        参与核覆盖“核心切换地址空间、第一次进入地址空间、翻译填充过 TLB”的
        所有来源；未完成票据目标核的并集保证：更早票据的 ACK 迟到时，新票据
        不会漏掉任何可能仍持有旧项、需要被通知的核。
        """
        self._shrink_participants(asid)
        targets = set(self._participants.get(asid, ()))
        for t in self._tickets.values():
            if t.asid == asid and not t.closed:
                targets.update(t.targets)
        return sorted(targets)

    def _release_frames(self, ppns: tuple[int, ...]) -> None:
        """finish 后回收 unmap 延迟释放的帧。

        小页票据可能只有 1 帧；大页 16 帧且 16 对齐。统一逐帧释放：
        这些帧在摘表时已从所有活动页表移除，此刻所有目标核也已 ACK，
        旧物理页才允许被首位适配重新分配。
        """
        if len(ppns) == L2_SIZE and ppns[0] % L2_SIZE == 0 and \
                ppns == tuple(range(ppns[0], ppns[0] + L2_SIZE)):
            self.mem.free_huge(ppns[0])
        else:
            for ppn in ppns:
                self.mem.free_frame(ppn)

    def _issue(self, asid: int, kind: str, vaddr: int, keys: tuple,
               deferred_frames: tuple[int, ...] | None = None) -> Ticket:
        targets = self._snapshot_targets(asid)
        t = Ticket(
            id=self._next_ticket,
            asid=asid,
            kind=kind,
            vaddr=vaddr,
            keys=tuple(keys),
            targets=targets,
            deferred_frames=deferred_frames,
        )
        self._next_ticket += 1
        # 无任何核可能缓存过该空间（TLB 可证为空且无未完成票据目标）：
        # 不发任何失效请求；unmap 的旧帧此刻可安全立即回收，票据直接关闭，
        # 不进入活动票据表（finish/ack 未知票据的既有语义随之适用）。
        if not targets:
            if deferred_frames is not None:
                self._release_frames(deferred_frames)
                t.deferred_frames = None
            t.closed = True
            return t
        self._tickets[t.id] = t
        return t

    @staticmethod
    def _keys_for(asid: int, vaddr: int, huge: bool) -> tuple[tuple[int, int], ...]:
        if huge:
            base_vpn = (vaddr & ~HUGE_PAGE_MASK) >> PAGE_BITS
            # 大页标记项位于 16 对齐 vpn；同时列出 16 个小页 key，
            # 以失效“表已更新但 ACK 未到”期间其他核新填入的小页项。
            return tuple((asid, base_vpn + j) for j in range(L2_SIZE))
        return ((asid, vaddr >> PAGE_BITS),)

    def ack(self, core_id, ticket_id) -> Ticket:
        """核 ``core_id`` 确认票据：立刻从该核 TLB 移除票据所列 key。

        若该核**不在**发票时快照的目标核里，这是一张“迟到的旧确认”：它证明
        此前的参与范围判断依据不足、该核确实缓存过该地址空间。此时不报错、
        也不丢确认，而是：

        1. 照样清掉该核上票据所列 key；
        2. 把该核保守补入该 ASID 的参与范围；
        3. 把该核扩展为该 ASID 上所有仍未 finish 的后续票据的目标核
           （当前票据也立即把它记为已响应，不重复等待）。

        这样参与范围只可能因证据而扩大、绝不会漏通知。
        """
        c = self._check_core(core_id)
        tid = validate_int(ticket_id, "ticket_id")
        t = self._tickets.get(tid)
        if t is None:
            raise TicketError(f"票据 {tid} 不存在或已关闭")
        if c in t.acks:
            raise TicketError(f"核 {c} 已 ACK 过票据 {tid}，不能重复 ACK")
        tlb = self._tlb[c]
        for k in t.keys:
            tlb.pop(k, None)
        t.acks.add(c)
        if c not in t.targets:
            # 迟到的旧确认：扩大参与范围并补进所有未完成的同空间票据。
            group = self._participants.setdefault(t.asid, set())
            group.add(c)
            t.targets.append(c)
            for other in self._tickets.values():
                if other.asid == t.asid and not other.closed and c not in other.targets:
                    other.targets.append(c)
                    # 只有当后续票据所列 key 都已被本次 ACK 清空（key 集合被
                    # 包含，例如同一页的连续 protect），才顺带视为已响应；
                    # 否则该核仍需对自己那份失效请求单独 ACK。
                    if set(other.keys).issubset(set(t.keys)):
                        other.acks.add(c)
        self._shrink_participants(t.asid)
        return t

    def finish(self, ticket_id) -> Ticket:
        """票据收尾：所有目标核都已 ACK 才成功，否则抛 InvalidationsPending。"""
        tid = validate_int(ticket_id, "ticket_id")
        t = self._tickets.get(tid)
        if t is None:
            raise TicketError(f"票据 {tid} 不存在或已 finish")
        if t.pending:
            raise InvalidationsPending(tid, t.pending)
        t.closed = True
        if t.deferred_frames is not None:
            # 所有实际持有旧缓存可能的核都已 ACK：旧物理页此刻才允许复用。
            self._release_frames(t.deferred_frames)
            t.deferred_frames = None
        del self._tickets[tid]
        self._shrink_participants(t.asid)
        return t

    def pending_tickets(self, asid=None) -> list[int]:
        if asid is None:
            return [t.id for t in self._tickets.values()]
        a = validate_int(asid, "asid")
        return self._pending_for_asid(a)

    # ---- 核心地址空间切换（参与范围维护） ----------------------------------
    def switch_to(self, core_id, asid) -> None:
        """让核 ``core_id`` 切换（或第一次进入）地址空间 ``asid``。

        这是参与范围的显式来源：

        * **第一次进入某个地址空间 / 切入**：该核成为该 ASID 的参与核——它
          随后随时可能填入该空间的翻译，后续失效请求必须通知它（保守）。
        * **切出旧地址空间**：仅当该核 TLB 中已没有旧空间的任何 key、且它也不
          在旧空间未完成票据的目标列表里时，才把它移出旧空间的参与范围；
          依据不足时保守保留，绝不在无法证伪缓存时漏通知。

        本模拟不冲刷 TLB（key 含 ASID，硬件切 ASID 通常保留缓存），因此切出
        后仍持有旧 key 的核继续留在参与范围内，直到对应 ACK/淘汰清空。
        """
        core = self._check_core(core_id)
        sp = self._space(asid)
        old = self._current_asid[core]
        if old is not None and old != sp.asid:
            self._shrink_participants(old)
        self._participants.setdefault(sp.asid, set()).add(core)
        self._current_asid[core] = sp.asid

    def current_address_space(self, core_id) -> int | None:
        """诊断用：核当前运行的 ASID（从未 :meth:`switch_to` 过则为 None）。"""
        core = self._check_core(core_id)
        return self._current_asid[core]

    def participants(self, asid) -> tuple[int, ...]:
        """诊断用：该地址空间当前“可能缓存过它”的参与核（升序）。"""
        sp = self._space(asid)
        return tuple(sorted(self._participants.get(sp.asid, ())))

    # ---- map / unmap / protect（先改表，后发票） ----------------------------
    def map(self, asid, vaddr, perms, *, huge: bool = False) -> Ticket:
        if not isinstance(huge, bool):
            raise TypeError("huge 必须是 bool")
        sp = self._space(asid)
        va = validate_vaddr(vaddr, "vaddr")
        p = validate_perms(perms, "perms")
        if huge:
            sp.map_huge(va, p)
        else:
            sp.map_small(va, p)
        return self._issue(sp.asid, "map", va, self._keys_for(sp.asid, va, huge))

    def unmap(self, asid, vaddr) -> Ticket:
        sp = self._space(asid)
        va = validate_vaddr(vaddr, "vaddr")
        _, huge, frames = sp.unmap(va)
        return self._issue(sp.asid, "unmap", va,
                           self._keys_for(sp.asid, va, huge),
                           deferred_frames=frames)

    def protect(self, asid, vaddr, perms) -> Ticket:
        """改权限。``vaddr`` 必须 256 对齐（不做静默向下取整）。

        落在大页中间时拆分为 16 个小页：目标页新权限，其余 15 页保持原权限
        （保留其他映射，帧内容在本模拟中无实际字节，只迁移权限与帧占用）。
        """
        sp = self._space(asid)
        va = validate_vaddr(vaddr, "vaddr")
        p = validate_perms(perms, "perms")
        if va & PAGE_MASK:
            raise ValueError(f"protect 要求 256 对齐，得到 vaddr=0x{va:x}")
        l1i, _, _ = split_vaddr(va)
        l1e = sp.l1[l1i]
        if isinstance(l1e, PTE):
            sp.split_huge_for_protect(va, p)
            return self._issue(
                sp.asid, "protect", va,
                self._keys_for(sp.asid, va & ~HUGE_PAGE_MASK, True),
            )
        sp.protect_small(va, p)
        return self._issue(sp.asid, "protect", va, self._keys_for(sp.asid, va, False))

    # ---- TLB ----------------------------------------------------------------
    def tlb_keys(self, core_id) -> list[tuple[int, int]]:
        """诊断用：按 LRU 顺序（旧→新）返回核上 TLB 的 key 列表。"""
        c = self._check_core(core_id)
        return list(self._tlb[c].keys())

    def tlb_len(self, core_id) -> int:
        c = self._check_core(core_id)
        return len(self._tlb[c])

    def _lookup_tlb(self, core: int, asid: int, vpn: int):
        tlb = self._tlb[core]
        k = (asid, vpn)
        if k in tlb:
            tlb.move_to_end(k)
            return tlb[k], k
        huge_k = (asid, vpn & ~(L2_SIZE - 1))
        if huge_k in tlb:
            tlb.move_to_end(huge_k)
            entry = tlb[huge_k]
            if entry[2]:  # huge
                return entry, huge_k
        return None, None

    # ---- 遍历与翻译 ---------------------------------------------------------
    def walk_table(self, asid, vaddr) -> WalkResult | None:
        """无 TLB 遍历器：直接查页表，返回 :class:`WalkResult` 或 None。"""
        sp = self._space(asid)
        va = validate_vaddr(vaddr, "vaddr")
        return sp.walk(va)

    def translate(self, asid, core_id, vaddr, access, length: int = 1) -> TranslationResult:
        """翻译一次访问。

        * ``access``：:class:`AccessType`；
        * ``length``：1..4096；访问越过所在页（小页 256 / 大页 4096）边界时
          抛 :class:`CrossPageAccess`（题面要求的“跨页访问拒绝”）；
        * 未映射 / 权限不足以 fault 结果返回，不抛异常；
        * ACK 之前允许命中旧 TLB 项（陈旧翻译是可观察的预期行为）。
        """
        sp = self._space(asid)
        core = self._check_core(core_id)
        va = validate_vaddr(vaddr, "vaddr")
        if not isinstance(access, AccessType):
            raise TypeError(f"access 必须是 AccessType，得到 {type(access).__name__}")
        length = validate_size(length, "length")
        if va + length - 1 > VADDR_MAX:
            raise ValueError(f"访问区间末端 0x{va + length - 1:x} 超出虚拟地址空间")

        vpn = va >> PAGE_BITS
        entry, hit_key = self._lookup_tlb(core, sp.asid, vpn)
        via = "tlb"
        if entry is None:
            wr = sp.walk(va)
            if wr is None:
                return TranslationResult(False, None, "not_mapped", access.name.lower(),
                                         "walk", sp.asid, va)
            entry = (wr.ppn, wr.perms, wr.huge)
            via = "walk"
            # 填入 TLB：大页用 16 对齐的标记 key
            key = (sp.asid, vpn if not wr.huge else (vpn & ~(L2_SIZE - 1)))
            tlb = self._tlb[core]
            if key in tlb:
                tlb.move_to_end(key)
            else:
                tlb[key] = entry
                if len(tlb) > TLB_SIZE:
                    tlb.popitem(last=False)
            # 一次成功的遍历填充证明该核确实缓存了该空间：登记为参与核。
            self._participants.setdefault(sp.asid, set()).add(core)
            self._current_asid[core] = sp.asid

        ppn, perms, huge = entry
        region = HUGE_PAGE_SIZE if huge else PAGE_SIZE
        if (va & (region - 1)) + length > region:
            raise CrossPageAccess(
                f"vaddr=0x{va:x} length={length} 跨越了{'大页' if huge else '小页'}边界"
                f"（{'4096' if huge else '256'} 字节）"
            )
        need = int(access)
        if not (perms & need):
            fault = {PERM_READ: "read", PERM_WRITE: "write", PERM_EXEC: "execute"}[need]
            return TranslationResult(False, None, fault, access.name.lower(), via, sp.asid, va, perms)
        off = va & (HUGE_PAGE_MASK if huge else PAGE_MASK)
        return TranslationResult(True, ppn * PAGE_SIZE + off, None, access.name.lower(),
                                 via, sp.asid, va, perms)

    # ---- 诊断 ---------------------------------------------------------------
    def mappings(self, asid) -> list[tuple[int, int, int, bool]]:
        sp = self._space(asid)
        return sp.mappings()
