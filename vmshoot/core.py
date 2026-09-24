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

    * ``kind``：``map`` / ``unmap`` / ``protect`` / ``protect_range``；
    * ``keys``：需要失效的 TLB key（小页粒度的 ``(asid, vpn)``）；
    * ``targets``：必须 ACK 的核（本实现中恒为全部核）；
    * ``acks``：已 ACK 的核集合；
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

    # ---- protect_range ------------------------------------------------------
    def protect_range_pages(self, start: int, end: int, perms: int) -> tuple[int, ...]:
        """把半开区间 ``[start, end)``（均已 256 对齐）内所有小页的权限改为 ``perms``。

        规则：

        * 先对区间覆盖的每个 4096 区域做完整预检（空洞 -> :class:`NotMapped`），
          全部通过后才修改任何页表项——本方法不分配/释放帧，预检之后不可能失败，
          因此失败时页表、帧占用均无任何部分变更；
        * 被区间**完整覆盖**的大页：只改权限、保留大页形式（不隐式拆分）；
        * 被区间**部分覆盖**的大页：就地拆成 16 个小页（帧不变），只把区间内
          小页改为新权限，区间外小页保留原权限；
        * 小页表区域：区间内已映射小页改权限。

        物理帧号一律不变；同一物理帧在其他虚拟页上的别名 PTE 不随之改权限。
        返回“修改后仍为大页形式”的 L1 区域下标元组（供票据失效 key 计算）。
        """
        l1_lo = start >> HUGE_BITS
        l1_hi = (end - 1) >> HUGE_BITS
        # ---- 预检：任何修改之前 -------------------------------------------
        for l1i in range(l1_lo, l1_hi + 1):
            region_lo = max(start, l1i << HUGE_BITS)
            region_hi = min(end, (l1i + 1) << HUGE_BITS)
            l1e = self.l1[l1i]
            if l1e is None:
                raise NotMapped(
                    f"区间 [0x{start:x}, 0x{end:x}) 在 0x{region_lo:x} 处遇到未映射空洞"
                )
            if isinstance(l1e, list):
                lo_l2 = (region_lo >> PAGE_BITS) & (L2_SIZE - 1)
                hi_l2 = ((region_hi - 1) >> PAGE_BITS) & (L2_SIZE - 1)
                for l2i in range(lo_l2, hi_l2 + 1):
                    if l1e[l2i] is None:
                        raise NotMapped(
                            f"区间 [0x{start:x}, 0x{end:x}) 在 0x{(l1i << HUGE_BITS) + (l2i << PAGE_BITS):x} 处遇到未映射空洞"
                        )
        # ---- 应用：预检全部通过，以下不会失败 --------------------------------
        kept_huge: list[int] = []
        for l1i in range(l1_lo, l1_hi + 1):
            region_lo = max(start, l1i << HUGE_BITS)
            region_hi = min(end, (l1i + 1) << HUGE_BITS)
            lo_l2 = (region_lo >> PAGE_BITS) & (L2_SIZE - 1)
            hi_l2 = ((region_hi - 1) >> PAGE_BITS) & (L2_SIZE - 1)
            l1e = self.l1[l1i]
            if isinstance(l1e, PTE):
                if lo_l2 == 0 and hi_l2 == L2_SIZE - 1:
                    l1e.perms = perms          # 整段大页统一改权限：保留大页形式
                    kept_huge.append(l1i)
                else:
                    self.split_huge_region(l1i, lo_l2, hi_l2, perms)
            else:
                for l2i in range(lo_l2, hi_l2 + 1):
                    l1e[l2i].perms = perms
        return tuple(kept_huge)

    def split_huge_region(self, l1i: int, lo_l2: int, hi_l2: int, perms: int) -> None:
        """把 L1 区域 ``l1i`` 上的大页就地拆成 16 个小页，只把 ``[lo_l2, hi_l2]``
        下标内的小页改为 ``perms``，其余保留原权限。帧不变、无资源分配。"""
        l1e = self.l1[l1i]
        assert isinstance(l1e, PTE) and l1e.huge
        table = [
            PTE(ppn=l1e.ppn + j, perms=l1e.perms, huge=False)
            for j in range(L2_SIZE)
        ]
        for l2i in range(lo_l2, hi_l2 + 1):
            table[l2i].perms = perms
        self.l1[l1i] = table

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
    def _issue(self, asid: int, kind: str, vaddr: int, keys: tuple) -> Ticket:
        t = Ticket(
            id=self._next_ticket,
            asid=asid,
            kind=kind,
            vaddr=vaddr,
            keys=tuple(keys),
            targets=tuple(range(NUM_CORES)),
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

    def protect_range(self, asid, start, end, perms) -> Ticket:
        """把半开虚拟地址区间 ``[start, end)`` 内所有已映射页的权限改为 ``perms``。

        * ``start``/``end`` 均须 256 对齐，``0 <= start < end <= 0x10000``；
          区间内每个 256 小页都必须已有映射，否则抛 :class:`NotMapped`；
        * **原子性**：所有参数与“区间无空洞”的预检都先于任何修改完成；失败时
          页表、TLB、票据均不发生任何变化（也不会产生新票据）；
        * 被区间完整覆盖的大页保留大页形式、只改权限；部分覆盖的大页就地拆成
          16 个小页，只替换区间内权限；物理地址不变，同一物理页在别处的别名
          映射不随之改权限；
        * 失效走既有票据协议：返回**一张**票据（``kind="protect_range"``），
          其 keys 覆盖区间内全部小页 key（凡本次触及的 4096 区域一律列出该区域
          全部 16 个小页 key，以精确失效拆分前可能缓存的大页标记项）；调用方用
          既有的 :meth:`ack` / :meth:`finish` 完成确认，没有第二套确认机制。
        """
        sp = self._space(asid)
        s = validate_vaddr(start, "start")
        e = validate_int(end, "end", max_value=VADDR_SPACE)
        p = validate_perms(perms, "perms")
        if s & PAGE_MASK:
            raise ValueError(f"protect_range 要求 start 256 对齐，得到 start=0x{s:x}")
        if e & PAGE_MASK:
            raise ValueError(f"protect_range 要求 end 256 对齐，得到 end=0x{e:x}")
        if e <= s:
            raise ValueError(
                f"protect_range 要求 start < end（半开区间非空），得到 "
                f"start=0x{s:x} end=0x{e:x}"
            )
        sp.protect_range_pages(s, e, p)
        keys = self._keys_for_range(sp.asid, s, e)
        return self._issue(sp.asid, "protect_range", s, keys)

    @staticmethod
    def _keys_for_range(asid: int, start: int, end: int) -> tuple[tuple[int, int], ...]:
        """范围失效 key：每个被触及的 4096 区域一律列出 16 个小页 key。

        保留大页形式的区域同样需要 16 个 key：ACK 前其他核可能已缓存该区域
        任意小页位置上的大页标记项（key 为 16 对齐 vpn），只列区间内 key 会漏掉
        区间外小页上的陈旧大页项。
        """
        keys: list[tuple[int, int]] = []
        for l1i in range(start >> HUGE_BITS, ((end - 1) >> HUGE_BITS) + 1):
            base_vpn = l1i * L2_SIZE
            keys.extend((asid, base_vpn + j) for j in range(L2_SIZE))
        return tuple(keys)

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
