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
    NotCopyOnWrite,
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
    """页表项。``huge=True`` 时它是一级叶子，``ppn`` 为 16 对齐的起始帧号。

    ``cow=True`` 表示写时复制页：该 PTE 的帧可能与另一个地址空间共享，
    写访问必须先经 :meth:`MMU.handle_write_fault` 分裂。COW 页一律是小页
    （克隆时大页就地展开为 16 个小页），故 ``cow`` 不会与 ``huge`` 同时为真。
    """

    ppn: int
    perms: int
    huge: bool = False
    cow: bool = False


@dataclass
class Ticket:
    """失效票据。

    * ``kind``：``map`` / ``unmap`` / ``protect`` / ``clone`` / ``cow``；
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
    * ``via``：``"tlb"``（命中缓存，可能是尚未 ACK 的旧项）或 ``"walk"``（遍历页表）；
    * ``cow``：写访问命中 COW 页时为 ``True``，此时 ``fault="write"``——
      这不是权限不足，而是要求调用方先串行处理写故障
      （:meth:`MMU.handle_write_fault`）再重试写。
    """

    ok: bool
    phys_addr: int | None
    fault: str | None
    access: str
    via: str
    asid: int
    vaddr: int
    perms: int | None = None
    cow: bool = False

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
    cow: bool = False

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
            return WalkResult(e.ppn, e.perms, True, vaddr, cow=False)
        pte = e[l2i]
        if pte is None:
            return None
        return WalkResult(pte.ppn, pte.perms, False, vaddr, cow=pte.cow)

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
            self.l1[l1i] = None       # 先摘表，再释放帧引用
            for j in range(L2_SIZE):
                self.mem.release_frame(pte.ppn + j)
            return pte, True
        if l1e is None or l1e[l2i] is None:
            raise NotMapped(f"vaddr=0x{vaddr:x} 没有映射")
        pte = l1e[l2i]
        l1e[l2i] = None              # 先摘表，再释放帧引用
        self.mem.release_frame(pte.ppn)
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
        # 维持不变量“可写 PTE 若指向共享帧（引用计数 >1）必须处于 COW”：
        #  * 改成只读：无人能写，cow 清掉（对其写退化为普通写权限故障）；
        #  * 改成可写：按帧当前引用计数决定——仍被共享则重新置 COW，
        #    私有帧（refs==1）则可直接写。
        if perms & PERM_WRITE:
            pte.cow = self.mem.frame_refs(pte.ppn) > 1
        else:
            pte.cow = False
        return pte

    def cow_split(self, vaddr: int, perms: int) -> tuple[PTE, int, int]:
        """COW 写故障分裂：为该页分配一个私有新帧并安装私有 PTE。

        严格顺序（失败不留半更新）：

        1. 先分配新帧——若 :class:`FrameExhausted`，旧的共享 PTE 完全不变；
        2. 切换 PTE 指向新帧、置 ``cow=False``、写新权限；
        3. 最后释放旧帧的一个引用（最后引用时旧帧才真正回收）。

        本模拟不存放页内字节，“拷贝内容”是按契约隐含完成的；可观察的是
        物理帧号与引用计数。返回 ``(新 PTE, 新 ppn, 旧 ppn)``。
        """
        l1i, l2i, _ = split_vaddr(vaddr)
        l1e = self.l1[l1i]
        if not isinstance(l1e, list) or l1e[l2i] is None:
            raise NotMapped(f"vaddr=0x{vaddr:x} 没有映射，不能处理写故障")
        pte = l1e[l2i]
        if pte.huge:
            raise AssertionError("内部错误：大页不应处于 COW 状态")
        old_ppn = pte.ppn
        new_ppn = self.mem.alloc_frame()   # 先分配：失败则 PTE 原封不动
        pte.ppn = new_ppn
        pte.perms = perms
        pte.cow = False
        self.mem.release_frame(old_ppn)
        return pte, new_ppn, old_ppn

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
        """释放本空间对全部物理帧的引用（共享帧仅在最后引用时真正回收）。"""
        for l1e in self.l1:
            if isinstance(l1e, PTE):
                for j in range(L2_SIZE):
                    self.mem.release_frame(l1e.ppn + j)
            elif isinstance(l1e, list):
                for pte in l1e:
                    if pte is not None:
                        self.mem.release_frame(pte.ppn)
        self.l1 = [None] * L1_SIZE

    # ---- 克隆（写时复制） ----------------------------------------------------
    def clone_snapshot(self) -> list[tuple[int, str, object]]:
        """枚举本空间的全部映射，供克隆使用。

        返回项：``(l1 索引, "huge", (基 ppn, perms))`` 或
        ``(l1 索引, "small", [(l2 索引, ppn, perms), ...])``。
        """
        out: list[tuple[int, str, object]] = []
        for l1i, l1e in enumerate(self.l1):
            if isinstance(l1e, PTE):
                out.append((l1i, "huge", (l1e.ppn, l1e.perms)))
            elif isinstance(l1e, list):
                ptes = [
                    (l2i, pte.ppn, pte.perms)
                    for l2i, pte in enumerate(l1e)
                    if pte is not None
                ]
                if ptes:
                    out.append((l1i, "small", ptes))
        return out

    def install_clone_huge(self, l1i: int, base_ppn: int, perms: int) -> None:
        """把一个大页在本空间就地展开为 16 个共享小页（帧不变）。

        仅当页含写权限时才置 ``cow``；只读页父子直接共享，对其写仍是普通写故障。
        """
        if self.l1[l1i] is not None:
            raise AssertionError("内部错误：克隆目标的一级项必须为空")
        cow = bool(perms & PERM_WRITE)
        self.l1[l1i] = [
            PTE(ppn=base_ppn + j, perms=perms, huge=False, cow=cow)
            for j in range(L2_SIZE)
        ]

    def convert_huge_to_cow(self, l1i: int) -> None:
        """父空间侧：把大页叶子就地展开为 16 个共享小页（帧不变）。"""
        l1e = self.l1[l1i]
        if not isinstance(l1e, PTE) or not l1e.huge:
            raise AssertionError("内部错误：父空间克隆源不是大页")
        base_ppn = l1e.ppn
        perms = l1e.perms
        cow = bool(perms & PERM_WRITE)
        self.l1[l1i] = [
            PTE(ppn=base_ppn + j, perms=perms, huge=False, cow=cow)
            for j in range(L2_SIZE)
        ]

    def install_clone_small(self, l1i: int, ptes: list[tuple[int, int, int]]) -> None:
        """在本空间安装一组共享小页 PTE（共享帧已由 MMU 层 retain）。

        仅含写权限的页置 ``cow``；只读页直接共享。
        """
        table = self.l1[l1i]
        if table is None:
            table = [None] * L2_SIZE
            self.l1[l1i] = table
        for l2i, ppn, perms in ptes:
            if table[l2i] is not None:
                raise AssertionError("内部错误：克隆目标的二级项必须为空")
            table[l2i] = PTE(ppn=ppn, perms=perms, huge=False,
                             cow=bool(perms & PERM_WRITE))

    def mark_small_cow(self) -> None:
        """克隆后把本空间（父空间）现有的可写普通小页 PTE 标记为 COW。

        只读页不标记：它们父子直接共享，对其写应继续是普通写权限故障。
        """
        for l1e in self.l1:
            if isinstance(l1e, list):
                for pte in l1e:
                    if pte is not None and (pte.perms & PERM_WRITE):
                        pte.cow = True

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
        # 每核一个 OrderedDict：key=(asid, vpn) -> (ppn, perms, huge, cow)，头部为 LRU
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

    def _alloc_slot(self) -> tuple[int, AddressSpace]:
        """分配一个空闲 ASID 槽位并注册空地址空间（create 与 clone 共用）。"""
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
            return cand, sp
        raise AsidBusy(busy[0], self._pending_for_asid(busy[0]))

    # ---- ASID 生命周期 ------------------------------------------------------
    def create_address_space(self) -> int:
        """分配一个新地址空间，返回 ASID（0..3）。

        * 4 个槽位全活动 -> :class:`AddressSpaceLimit`；
        * 空闲槽位若仍挂着未 finish 的票据 -> 该号不能复用；
          全部空闲槽位都被未完成票据占据时抛 :class:`AsidBusy`。
        """
        asid, _ = self._alloc_slot()
        return asid

    def clone_address_space(self, parent_asid) -> tuple[int, Ticket]:
        """克隆地址空间（写时复制），返回 ``(子 ASID, 失效票据)``。

        语义：

        * **普通页共享**：父子两个空间的 PTE 指向**同一批物理帧**，帧引用计数
          各加 1，克隆本身**不分配任何新物理帧**（``used_frames`` 不变）；
        * 双方的共享页都置 ``cow`` 标志；父空间若含大页，父子两侧都**就地展开**
          为 16 个小页（物理帧不变，与大页拆分同一机制），因为写时复制以
          256B 小页为粒度；
        * 任一方之后写该页 -> ``translate`` 返回 ``fault="write", cow=True``，
          须显式串行调用 :meth:`handle_write_fault` 分裂后重试；
        * 克隆会让父空间核上旧的 TLB 翻译（不带 cow 标记）过期，故发出一张
          ``kind="clone"`` 的失效票据（目标核仍为全部核），沿用既有 ACK/finish
          协议；子空间是全新的，没有 TLB 项。

        失败不留半更新：槽位不足（:class:`AddressSpaceLimit`/:class:`AsidBusy`）
        或父 ASID 非法时，在任何帧引用变更之前就抛错；克隆不分配新帧，之后的
        retain 与 PTE 安装都是纯内存操作（只会在内部不变量被破坏时报错，与大页
        拆分同一性质），因此不存在 :class:`FrameExhausted` 或部分更新窗口。
        """
        parent = self._space(parent_asid)
        child_asid, child = self._alloc_slot()
        snapshot = parent.clone_snapshot()
        # 阶段 1：为全部共享帧增加引用（retain 只在内部不变量被破坏时抛错）。
        for l1i, kind, payload in snapshot:
            if kind == "huge":
                base_ppn, _ = payload
                for j in range(L2_SIZE):
                    self.mem.retain_frame(base_ppn + j)
            else:
                for _, ppn, _ in payload:
                    self.mem.retain_frame(ppn)
        # 阶段 2：装入子空间。
        for l1i, kind, payload in snapshot:
            if kind == "huge":
                base_ppn, perms = payload
                child.install_clone_huge(l1i, base_ppn, perms)
            else:
                child.install_clone_small(l1i, payload)
        # 阶段 3：父空间大页就地展开为共享小页，其余可写小页打 cow 标记。
        for l1i, kind, payload in snapshot:
            if kind == "huge":
                parent.convert_huge_to_cow(l1i)
        parent.mark_small_cow()
        keys = self._clone_keys(parent.asid, snapshot)
        ticket = self._issue(parent.asid, "clone", 0, keys)
        return child_asid, ticket

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

    @staticmethod
    def _clone_keys(asid: int, snapshot) -> tuple[tuple[int, int], ...]:
        """克隆票据要失效的父空间 key：大页展开影响 16 个小页，小页表影响各页。"""
        keys: list[tuple[int, int]] = []
        for l1i, kind, payload in snapshot:
            base_vpn = l1i * L2_SIZE
            if kind == "huge":
                keys.extend((asid, base_vpn + j) for j in range(L2_SIZE))
            else:
                for l2i, _ppn, _perms in payload:
                    keys.append((asid, base_vpn + l2i))
        return tuple(keys)

    def handle_write_fault(self, asid, vaddr) -> Ticket:
        """显式串行处理一次写时复制写故障（“写故障事件”）。

        前置条件：``vaddr`` 256 对齐且该页当前确实是 COW 页（普通未映射页 ->
        :class:`NotMapped`；已映射但非 COW -> :class:`NotCopyOnWrite`；
        大页内部不会出现 COW，但非对齐地址仍按 :class:`ValueError` 拒绝）。

        处理顺序（原子化，不留半更新）：

        1. **先分配私有新帧**；若帧池耗尽抛 :class:`FrameExhausted`，旧共享映射
           原封不动（引用计数、PTE、TLB 全不变），调用方可稍后重试；
        2. 切换该 PTE 指向新帧、恢复其写权限（``cow=False``），并释放旧帧的一个
           引用（最后引用时旧帧才真正回收——这是“最后引用释放”的核对点）；
        3. 发出 ``kind="cow"`` 的失效票据：ACK 之前其他核上带旧 cow 标记的 TLB
           项允许存在，ACK 后删除，下一次写重走页表命中私有新帧。

        返回该票据；调用方仍需双核 ACK + finish。
        """
        sp = self._space(asid)
        va = validate_vaddr(vaddr, "vaddr")
        if va & PAGE_MASK:
            raise ValueError(f"handle_write_fault 要求 256 对齐，得到 vaddr=0x{va:x}")
        l1i, l2i, _ = split_vaddr(va)
        l1e = sp.l1[l1i]
        if isinstance(l1e, PTE):
            # 大页叶子不是 COW 状态（克隆时大页一律展开）
            raise NotCopyOnWrite(f"vaddr=0x{va:x} 是大页映射，不处于写时复制状态")
        if not isinstance(l1e, list) or l1e[l2i] is None:
            raise NotMapped(f"vaddr=0x{va:x} 没有映射，不能处理写故障")
        pte = l1e[l2i]
        if not pte.cow:
            raise NotCopyOnWrite(
                f"vaddr=0x{va:x} 的页不是写时复制页（cow=False），无需处理写故障"
            )
        # 恢复该页的写权限：保留原 r/x 位并置上 w。
        new_perms = pte.perms | PERM_WRITE
        _pte, _new_ppn, _old_ppn = sp.cow_split(va, new_perms)
        return self._issue(sp.asid, "cow", va, self._keys_for(sp.asid, va, False))

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
            entry = (wr.ppn, wr.perms, wr.huge, wr.cow)
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

        ppn, perms, huge, cow = entry
        region = HUGE_PAGE_SIZE if huge else PAGE_SIZE
        if (va & (region - 1)) + length > region:
            raise CrossPageAccess(
                f"vaddr=0x{va:x} length={length} 跨越了{'大页' if huge else '小页'}边界"
                f"（{'4096' if huge else '256'} 字节）"
            )
        need = int(access)
        # 写命中 COW 页：这不是普通权限不足，而是必须先串行处理写故障、分裂出私有帧。
        # 注意 COW 页的 perms 仍保留写位（分裂时据此恢复权限），故此判定先于权限位检查。
        if access is AccessType.WRITE and cow:
            return TranslationResult(False, None, "write", access.name.lower(), via,
                                     sp.asid, va, perms, cow=True)
        if not (perms & need):
            fault = {PERM_READ: "read", PERM_WRITE: "write", PERM_EXEC: "execute"}[need]
            return TranslationResult(False, None, fault, access.name.lower(), via, sp.asid, va, perms)
        off = va & (HUGE_PAGE_MASK if huge else PAGE_MASK)
        return TranslationResult(True, ppn * PAGE_SIZE + off, None, access.name.lower(),
                                 via, sp.asid, va, perms, cow=False)

    # ---- 诊断 ---------------------------------------------------------------
    def mappings(self, asid) -> list[tuple[int, int, int, bool]]:
        sp = self._space(asid)
        return sp.mappings()
