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
    HUGE_BITS,
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
    VADDR_SPACE,
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

    * ``kind``：``map`` / ``unmap`` / ``protect`` / ``remap``；
    * ``keys``：需要失效的 TLB key（小页粒度的 ``(asid, vpn)``）；
    * ``targets``：必须 ACK 的核（本实现中恒为全部核）；
    * ``acks``：已 ACK 的核集合；
    * ``remap`` 票据额外携带 ``dst``（目标起始虚址）与 ``length``（字节数），
      其 keys 同时覆盖源与目标区间（含被触动源大页的全部 16 个小页 key）；
    * 票据在 :meth:`MMU.finish` 成功后关闭并从活动票据表移除。
    """

    id: int
    asid: int
    kind: str
    vaddr: int
    keys: tuple[tuple[int, int], ...]
    targets: tuple[int, ...]
    acks: set[int] = field(default_factory=set)
    closed: bool = False
    dst: int | None = None
    length: int | None = None

    @property
    def pending(self) -> tuple[int, ...]:
        return tuple(c for c in self.targets if c not in self.acks)

    @property
    def done(self) -> bool:
        return not self.pending


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
            self.l1[l1i] = None       # 先摘表，再释放帧
            self.mem.free_huge(pte.ppn)
            return pte, True
        if l1e is None or l1e[l2i] is None:
            raise NotMapped(f"vaddr=0x{vaddr:x} 没有映射")
        pte = l1e[l2i]
        l1e[l2i] = None              # 先摘表，再释放帧
        self.mem.free_frame(pte.ppn)
        if all(x is None for x in l1e):
            self.l1[l1i] = None  # 回收空的二级表
        return pte, False

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

    # ---- remap_range：先规划（纯只读），后安装（纯指针、零帧操作） ----------
    def plan_remap(self, src: int, dst: int,
                   npages: int) -> tuple[tuple, ...]:
        """构造“把 src 起 npages 个小页搬到 dst”的搬迁计划（纯只读）。

        返回有序动作元组序列：

        * ``("huge_move", src_l1i, dst_l1i, pte)``：整张大页平移——
          源 L1 槽置空，原大页叶子 PTE 原样装进目标 L1 槽（帧与权限不变）；
        * ``("huge_split", src_l1i, table, moves)``：源大页只搬一部分——
          源 L1 槽就地换为 16 个小页 PTE（引用同一批帧，不分配/释放帧），
          搬走的槽置 None；``moves`` 为 ``(dst_l1i, dst_l2i, pte)`` 序列；
        * ``("small_move", src_l1i, src_l2i, dst_l1i, dst_l2i, pte)``：
          单个小页 PTE 从源槽移到目标槽。

        计划成立条件：源区间每页都有映射（不能有洞，否则 :class:`NotMapped`）；
        目标区间每页都空闲（不能碰现有映射，否则 :class:`MappingExists`）。
        对齐/越界由 MMU 层校验。检查全部先于任何修改。
        """
        actions: list[tuple] = []

        # 第一步：源完整性检查（任何一个源小页缺失即拒绝，不做修改）。
        for i in range(npages):
            if self.walk(src + i * PAGE_SIZE) is None:
                raise NotMapped(
                    f"remap_range 源区间 0x{src:x}..0x{src + npages * PAGE_SIZE:x} "
                    f"在 0x{src + i * PAGE_SIZE:x} 处有洞（源必须是完整映射）"
                )

        # 第二步：目标空闲检查（源、目标重叠也会在此被发现：重叠处源已有映射）。
        for i in range(npages):
            if self.walk(dst + i * PAGE_SIZE) is not None:
                raise MappingExists(
                    f"remap_range 目标区间 0x{dst:x}..0x{dst + npages * PAGE_SIZE:x} "
                    f"在 0x{dst + i * PAGE_SIZE:x} 处与现有映射冲突"
                )

        # 第三步：按源 L1 槽分组生成动作。源区间在小页粒度连续，同一 L1 槽内
        # 命中的小页下标必连续，逐组扫描即可。
        j = 0
        while j < npages:
            sva = src + j * PAGE_SIZE
            src_l1i, src_l2, _ = split_vaddr(sva)
            l1e = self.l1[src_l1i]
            # 本组在该 L1 槽内连续覆盖多少页（至多到槽末或区间末）。
            cnt = 1
            while (j + cnt < npages
                   and split_vaddr(src + (j + cnt) * PAGE_SIZE)[0] == src_l1i):
                cnt += 1
            dva = dst + j * PAGE_SIZE
            dst_l1i, _, _ = split_vaddr(dva)

            if isinstance(l1e, PTE):
                if cnt == L2_SIZE and (dva & HUGE_PAGE_MASK) == 0:
                    # 整张大页且目标 4096 对齐：保持大页平移。
                    actions.append(("huge_move", src_l1i, dst_l1i, l1e))
                else:
                    # 只搬一部分：就地拆分，搬走的槽置 None，其余小页保留。
                    table = [
                        PTE(ppn=l1e.ppn + q, perms=l1e.perms, huge=False)
                        for q in range(L2_SIZE)
                    ]
                    moves: list[tuple] = []
                    for q in range(cnt):
                        table[src_l2 + q] = None
                        tl1i, tl2i, _ = split_vaddr(dst + (j + q) * PAGE_SIZE)
                        moves.append((
                            tl1i, tl2i,
                            PTE(ppn=l1e.ppn + src_l2 + q,
                                perms=l1e.perms, huge=False),
                        ))
                    actions.append(("huge_split", src_l1i, table, tuple(moves)))
            else:
                # 小页表（源检查保证 l1e 是列表且相应槽有 PTE）。
                for q in range(cnt):
                    tl1i, tl2i, _ = split_vaddr(dst + (j + q) * PAGE_SIZE)
                    actions.append(("small_move", src_l1i, src_l2 + q,
                                    tl1i, tl2i, l1e[src_l2 + q]))
            j += cnt

        return tuple(actions)

    def execute_remap(self, actions: tuple[tuple, ...]) -> None:
        """按 :meth:`plan_remap` 的动作表安装。

        全部动作只是指针改写：移动原有 PTE，或安装引用同一批帧的新 PTE。
        **不分配也不释放任何物理帧**，没有资源失败窗口，因此不存在部分变更；
        调用前源/目标检查必须已全部通过。
        """
        for a in actions:
            kind = a[0]
            if kind == "huge_move":
                _, src_l1i, dst_l1i, pte = a
                self.l1[src_l1i] = None
                self.l1[dst_l1i] = pte
            elif kind == "huge_split":
                _, src_l1i, table, moves = a
                self.l1[src_l1i] = table
                for tl1i, tl2i, pte in moves:
                    slot = self.l1[tl1i]
                    if slot is None:
                        slot = [None] * L2_SIZE
                        self.l1[tl1i] = slot
                    slot[tl2i] = pte
            else:  # small_move
                _, src_l1i, src_l2i, dst_l1i, dst_l2i, pte = a
                self.l1[src_l1i][src_l2i] = None
                slot = self.l1[dst_l1i]
                if slot is None:
                    slot = [None] * L2_SIZE
                    self.l1[dst_l1i] = slot
                slot[dst_l2i] = pte

        # 收尾：回收已全空的二级表（纯结构回收；帧仍被目标 PTE 引用，
        # 此过程从不调用 mem 的分配/释放）。
        for l1i, l1e in enumerate(self.l1):
            if isinstance(l1e, list) and all(x is None for x in l1e):
                self.l1[l1i] = None

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

    # ---- 票据 ---------------------------------------------------------------
    def _issue(self, asid: int, kind: str, vaddr: int, keys: tuple,
               dst: int | None = None, length: int | None = None) -> Ticket:
        t = Ticket(
            id=self._next_ticket,
            asid=asid,
            kind=kind,
            vaddr=vaddr,
            keys=tuple(keys),
            targets=tuple(range(NUM_CORES)),
            dst=dst,
            length=length,
        )
        self._next_ticket += 1
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
        """核 ``core_id`` 确认票据：立刻从该核 TLB 移除票据所列 key。"""
        c = self._check_core(core_id)
        tid = validate_int(ticket_id, "ticket_id")
        t = self._tickets.get(tid)
        if t is None:
            raise TicketError(f"票据 {tid} 不存在或已关闭")
        if c not in t.targets:
            raise TicketError(f"核 {c} 不是票据 {tid} 的目标核")
        if c in t.acks:
            raise TicketError(f"核 {c} 已 ACK 过票据 {tid}，不能重复 ACK")
        tlb = self._tlb[c]
        for k in t.keys:
            tlb.pop(k, None)
        t.acks.add(c)
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
        del self._tickets[tid]
        return t

    def pending_tickets(self, asid=None) -> list[int]:
        if asid is None:
            return [t.id for t in self._tickets.values()]
        a = validate_int(asid, "asid")
        return self._pending_for_asid(a)

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
        _, huge = sp.unmap(va)
        return self._issue(sp.asid, "unmap", va, self._keys_for(sp.asid, va, huge))

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

    def remap_range(self, asid, src, dst, length) -> Ticket:
        """把同一 ASID 内 ``[src, src+length)`` 的完整映射平移到 ``[dst, dst+length)``。

        契约：

        * ``src``/``dst`` 必须 256 对齐，``length`` 必须是 256 的正整数倍，
          且整个区间不越出 16 位虚拟地址空间（否则 :class:`ValueError`）；
        * 源区间每个小页都必须已映射（不能有洞，否则 :class:`NotMapped`）；
        * 目标区间每个小页都必须空闲（不能与现有映射重叠，包括与源区间
          自身重叠，否则 :class:`MappingExists`）；
        * 物理帧与权限逐页原样搬运：**不分配、不释放任何物理帧**，也不复制
          任何字节（本模拟不持有页内数据）；只改页表指针；
        * 源大页整段（16 页）搬走且目标 4096 对齐时，目标保持大页（原叶子
          PTE 平移）；源大页只搬一部分时，源处就地拆成 16 个小页（同一批帧），
          搬走部分与目标一律装小页，不做重新合并；
        * 先整体检查再动页表：任何一项检查失败，页表与帧位图完全不变；
        * 返回 ``kind="remap"`` 的失效票据，keys 同时覆盖源与目标区间
          （含被触动源大页的全部 16 个小页 key）。ACK 前旧 TLB 项按原协议
          允许存在；finish 后源地址访问必须 not_mapped，目标得到原物理地址。
        """
        sp = self._space(asid)
        s = validate_vaddr(src, "src")
        d = validate_vaddr(dst, "dst")
        ln = validate_int(length, "length")
        if ln == 0:
            raise ValueError("length 必须为正整数，得到 0")
        if ln & PAGE_MASK:
            raise ValueError(f"length={ln} 必须是 {PAGE_SIZE} 的整数倍（小页粒度）")
        if s & PAGE_MASK:
            raise ValueError(f"src=0x{s:x} 未按 {PAGE_SIZE} 字节对齐")
        if d & PAGE_MASK:
            raise ValueError(f"dst=0x{d:x} 未按 {PAGE_SIZE} 字节对齐")
        if s + ln > VADDR_SPACE:
            raise ValueError(
                f"源区间末端 0x{s + ln:x} 超出虚拟地址空间（最大 0x{VADDR_SPACE:x}）"
            )
        if d + ln > VADDR_SPACE:
            raise ValueError(
                f"目标区间末端 0x{d + ln:x} 超出虚拟地址空间（最大 0x{VADDR_SPACE:x}）"
            )

        npages = ln >> PAGE_BITS
        # 先整体检查（纯只读），全部通过后才动页表。
        actions = sp.plan_remap(s, d, npages)
        sp.execute_remap(actions)

        # 失效 keys：源、目标区间涉及的每个 4096 块都列出全部 16 个小页 key。
        # 这样即使某核上残留该块**旧大页**的 16 对齐标记 TLB 项（块里现在
        # 已是小页表或空表、且此前票据尚未 ACK），也会被一并摘掉；逐页区间
        # 本身的源/目标 key 自然包含在内。
        blocks: set[int] = set()
        for base, span in ((s, ln), (d, ln)):
            first = base >> HUGE_BITS
            last = (base + span - 1) >> HUGE_BITS
            blocks.update(range(first, last + 1))
        vpns: set[int] = set()
        for l1i in blocks:
            vpns.update(l1i * L2_SIZE + j for j in range(L2_SIZE))
        keys = tuple((sp.asid, v) for v in sorted(vpns))
        return self._issue(sp.asid, "remap", s, keys, dst=d, length=ln)

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
