"""核心实现：ASID、两级页表、每核 LRU TLB 与失效票据（shootdown ticket）。

本模块是纯模拟：
* 物理地址只是整数，物理帧由本模块内部分配器簿记，**从不**调用 mmap/mmap64
  等任何真实 OS 映射原语；
* 页表为内存中的 Python 对象；
* TLB 为每核一个 OrderedDict，按 LRU 淘汰。

地址划分（虚拟地址 16 位，空间 64 KiB）：

    | 15..12 一级索引 (4 bit, 16 项) | 11..8 二级索引 (4 bit, 16 项) | 7..0 页内偏移 |

一级表项有两种形态：
* 二级页表（:class:`L2Table`，16 个 :class:`PageEntry`）；
* 大页叶（``huge=True`` 的 :class:`PageEntry`），一次映射 4096 字节
  （= 16 个 256 字节小页），物理基址 4096 对齐。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

# ---------------------------------------------------------------------------
# 固定常量（本题契约，不是可配置项）
# ---------------------------------------------------------------------------

PAGE_SIZE = 256
L1_ENTRIES = 16
L2_ENTRIES = 16
HUGE_SIZE = PAGE_SIZE * L2_ENTRIES          # 4096
VIRT_SIZE = PAGE_SIZE * L2_ENTRIES * L1_ENTRIES  # 65536
PHYS_SIZE = 1024 * 1024                     # 模拟物理内存 1 MiB

MAX_ASIDS = 4
CORE_COUNT = 2
TLB_SIZE = 8

PAGE_MASK = PAGE_SIZE - 1
HUGE_MASK = HUGE_SIZE - 1
OFFSET_BITS = 8
L2_BITS = 4


# ---------------------------------------------------------------------------
# 异常层次（所有异常都可定位：消息带参数名与取值）
# ---------------------------------------------------------------------------

class VmError(Exception):
    """本库所有异常的公共基类。"""


class VmTypeError(VmError, TypeError):
    """参数类型非法（例如把 bool 当整数、把浮点当地址）。"""


class VmValueError(VmError, ValueError):
    """参数取值非法（越界、NaN/Infinity、未知标识等）。"""


class AddressRangeError(VmValueError):
    """地址区间越界或跨越不允许的边界。"""


class AlignmentError(VmValueError):
    """地址或长度未按契约对齐。"""


class UnknownAsidError(VmValueError):
    """引用了未分配或已释放的 ASID。"""


class AsidExhaustedError(VmValueError):
    """4 个 ASID 均不可用（在用，或被未完成票据阻塞复用）。"""


class MappingExistsError(VmValueError):
    """map 的目标虚拟页已有映射。"""


class NotMappedError(VmValueError):
    """unmap/protect 触及了不存在的映射。"""


class UnknownTicketError(VmValueError):
    """票据不存在或已 finish 并归档。"""


class TicketStateError(VmValueError):
    """票据状态机非法转移的基类。"""


class AlreadyAckedError(TicketStateError):
    """同一核对同一票据重复 ack。"""


class NotTicketTargetError(TicketStateError):
    """ack 的核不在该票据的目标核集合内。"""


class PendingFinishError(TicketStateError):
    """finish 时仍有目标核未 ack；或在有未完成票据时释放地址空间。"""


class TranslationFault(VmError):
    """翻译失败的基类，携带 core/asid/va/access 便于定位。"""

    def __init__(self, message: str, *, core: int, asid: int, va: int,
                 access: "Access") -> None:
        super().__init__(message)
        self.core = core
        self.asid = asid
        self.va = va
        self.access = access


class NotPresentFault(TranslationFault):
    """遍历页表未找到有效映射（read/write/execute 均如此）。"""


class CrossPageFault(TranslationFault):
    """一次访问跨越 256 字节页边界（本题契约直接拒绝，见 README）。"""


class PermissionFault(TranslationFault):
    """映射存在但权限不足的基类。"""


class ReadFault(PermissionFault):
    """不可读。"""


class WriteFault(PermissionFault):
    """不可写。"""


class ExecuteFault(PermissionFault):
    """不可执行。"""


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------

class Access(Enum):
    """访问类型。"""

    READ = "r"
    WRITE = "w"
    EXECUTE = "x"


@dataclass(frozen=True)
class Permissions:
    """小页/大页权限位。只接受真正的布尔位（不做静默真值纠正）。"""

    readable: bool = True
    writable: bool = True
    executable: bool = True

    @staticmethod
    def coerce(value: object, *, name: str = "perms") -> "Permissions":
        if isinstance(value, Permissions):
            return Permissions(bool(value.readable), bool(value.writable),
                               bool(value.executable))
        if isinstance(value, (tuple, list)) and len(value) == 3:
            bits = []
            for i, bit in enumerate(value):
                if isinstance(bit, bool):
                    bits.append(bit)
                elif isinstance(bit, int) and not isinstance(bit, bool) \
                        and bit in (0, 1):
                    bits.append(bool(bit))
                else:
                    raise VmTypeError(
                        f"{name}[{i}] 必须是 bool 或 0/1，得到 {bit!r}")
            return Permissions(bits[0], bits[1], bits[2])
        raise VmTypeError(
            f"{name} 必须是 Permissions 或长度为 3 的 (r, w, x) 序列，"
            f"得到 {value!r}")

    def as_rwx(self) -> str:
        return ("r" if self.readable else "-") + \
               ("w" if self.writable else "-") + \
               ("x" if self.executable else "-")


@dataclass
class PageEntry:
    """一个映射项（二级表中的小页，或一级表中的大页叶）。"""

    phys: int
    perms: Permissions
    huge: bool = False
    # frame_base/frame_size 标识底层分配器拥有的物理帧。
    # 大页拆成小页后，16 个小页共享同一个 4096 字节帧；
    # 只有最后一个共享者消失时才把整帧归还分配器。
    frame_base: int = 0
    frame_size: int = PAGE_SIZE

    def clone(self) -> "PageEntry":
        return PageEntry(self.phys, Permissions(self.perms.readable,
                                               self.perms.writable,
                                               self.perms.executable),
                         self.huge, self.frame_base, self.frame_size)


class L2Table(list):
    """二级页表：固定 16 项，元素为 PageEntry 或 None。"""

    def __init__(self) -> None:
        super().__init__([None] * L2_ENTRIES)


@dataclass(frozen=True)
class WalkResult:
    """一次翻译（TLB 项或页表遍历）的快照结果。"""

    phys_base: int
    perms: Permissions
    huge: bool


@dataclass
class Ticket:
    """失效票据。

    生命周期::

        产生（map/unmap/protect/free_asid）
          -> 每个目标核 ack（ack 时该核本地陈旧项被清除）
          -> finish（必须所有目标核都 ack；目标集为空时可立即 finish）

    ack 之前，目标核内命中的旧 TLB 缓存允许继续存在并被使用（模拟真实
    shootdown 的延迟窗口）；finish 成功后票据归档，ASID 复用阻塞解除。
    """

    id: int
    asid: int
    kind: str
    ranges: tuple[tuple[int, int], ...]
    target_cores: frozenset[int]
    creator: str
    acked: set[int] = field(default_factory=set)
    finished: bool = False

    def pending_cores(self) -> list[int]:
        return sorted(self.target_cores - self.acked)

    def is_complete(self) -> bool:
        return self.target_cores <= self.acked


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def _strict_int(name: str, value: object) -> int:
    """整数参数：拒绝 bool（即使 bool 是 int 子类），拒绝浮点与 NaN/Infinity。"""
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float):
            reason = "NaN/Infinity" if value != value or value in (
                float("inf"), float("-inf")) else "非整数值"
            raise VmTypeError(f"{name} 必须是 int，得到浮点 {value!r}（{reason}）")
        raise VmTypeError(f"{name} 必须是 int，得到 {type(value).__name__}")
    return value


def _validate_va(name: str, va: object) -> int:
    va = _strict_int(name, va)
    if not 0 <= va < VIRT_SIZE:
        raise AddressRangeError(
            f"{name}=0x{va:x} 越界，合法区间 [0, 0x{VIRT_SIZE:x})")
    return va


def _validate_core(core: object) -> int:
    core = _strict_int("core", core)
    if not 0 <= core < CORE_COUNT:
        raise AddressRangeError(
            f"core={core} 越界，合法核编号 [0, {CORE_COUNT})")
    return core


def l1_index(va: int) -> int:
    return (va >> (OFFSET_BITS + L2_BITS)) & (L1_ENTRIES - 1)


def l2_index(va: int) -> int:
    return (va >> OFFSET_BITS) & (L2_ENTRIES - 1)


# ---------------------------------------------------------------------------
# 物理帧分配器（仅簿记整数区间，不触碰真实内存）
# ---------------------------------------------------------------------------

class _FrameAllocator:
    """按对齐分配/归还 [0, PHYS_SIZE) 内的连续帧，区间合并，首块适配。"""

    def __init__(self) -> None:
        self._used: list[tuple[int, int]] = []  # 已占用，按 start 排序

    def allocate(self, size: int, align: int) -> int:
        if size <= 0 or size > PHYS_SIZE:
            raise VmValueError(f"非法帧大小 {size}")
        cursor = 0
        for start, end in self._used:
            aligned = (cursor + align - 1) & ~(align - 1)
            if aligned + size <= start:
                result = (aligned, aligned + size)
                self._used.append(result)
                self._used.sort()
                return aligned
            cursor = max(cursor, end)
        aligned = (cursor + align - 1) & ~(align - 1)
        if aligned + size > PHYS_SIZE:
            raise VmValueError(
                f"模拟物理内存不足：无法分配 {size} 字节（对齐 {align}）")
        self._used.append((aligned, aligned + size))
        self._used.sort()
        return aligned

    def free(self, base: int, size: int) -> None:
        seg = (base, base + size)
        if seg not in self._used:
            raise VmError(
                f"内部错误：试图释放未登记的帧 [0x{base:x}, 0x{base + size:x})")
        self._used.remove(seg)

    def used_bytes(self) -> int:
        return sum(e - s for s, e in self._used)


# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------

class MemorySystem:
    """ASID + 两级页表 + 每核 LRU TLB + 失效票据的模拟系统。

    典型用法::

        sys = MemorySystem()
        asid = sys.alloc_asid()
        ticket, pa = sys.map(asid, 0x0000)
        sys.finish(ticket)                       # 无陈旧项时可立即完成
        pa2 = sys.translate(0, asid, 0x0010, Access.READ)
    """

    # 故障类型按访问方式选择
    _FAULT_FOR_ACCESS = {
        Access.READ: ReadFault,
        Access.WRITE: WriteFault,
        Access.EXECUTE: ExecuteFault,
    }

    def __init__(self) -> None:
        # asid -> 一级表（list[16]，元素 None | L2Table | PageEntry(huge)）
        self.root_tables: dict[int, list] = {}
        # asid -> True 表示已释放、等待票据结束后才能被重新分配
        self._released: dict[int, bool] = {}
        # asid -> 未完成票据 id 集合
        self._outstanding: dict[int, set[int]] = {}
        # 每核 TLB：(asid, 小页号 va & ~0xFF) -> WalkResult
        self.tlbs: list[OrderedDict[tuple[int, int], WalkResult]] = [
            OrderedDict() for _ in range(CORE_COUNT)]
        self._frames = _FrameAllocator()
        self._next_ticket = 1
        self._tickets_store: dict[int, Ticket] = {}

    # -- ASID 管理 ---------------------------------------------------------

    def alloc_asid(self) -> int:
        """分配一个 ASID（取最小可用编号）。

        已释放但仍挂有未完成票据的编号**不能**被复用；4 个编号都不可用时抛
        :class:`AsidExhaustedError`。
        """
        for asid in range(MAX_ASIDS):
            if asid in self.root_tables:
                continue
            if self._outstanding.get(asid):
                continue  # 未完成票据阻塞复用
            self.root_tables[asid] = [None] * L1_ENTRIES
            self._released.pop(asid, None)
            self._outstanding.setdefault(asid, set())
            return asid
        raise AsidExhaustedError(
            f"最多 {MAX_ASIDS} 个地址空间，且被释放编号仍有未完成票据")

    def free_asid(self, asid: int) -> Ticket:
        """销毁整个地址空间：归还全部物理帧并产生全目标失效票据。

        该 ASID 有任何未完成票据时拒绝（:class:`PendingFinishError`），
        保证失败不产生部分变更。票据 finish 之前编号不可复用。
        """
        asid = self._require_active_asid(asid)
        if self._outstanding.get(asid):
            raise PendingFinishError(
                f"asid={asid} 有未完成票据 {sorted(self._outstanding[asid])}，"
                "不能销毁地址空间")

        root = self.root_tables[asid]
        ranges: list[tuple[int, int]] = []
        frees: list[tuple[int, int]] = []
        for i, slot in enumerate(root):
            region = i * HUGE_SIZE
            if slot is None:
                continue
            if isinstance(slot, PageEntry):  # 大页叶
                ranges.append((region, region + HUGE_SIZE))
                frees.append((slot.frame_base, slot.frame_size))
            else:  # L2Table
                for j, entry in enumerate(slot):
                    if entry is not None:
                        va = region + j * PAGE_SIZE
                        ranges.append((va, va + PAGE_SIZE))
                # 收集共享大帧：同一 frame_base 只归还一次
                seen: dict[int, int] = {}
                for entry in slot:
                    if entry is not None:
                        seen[entry.frame_base] = entry.frame_size
                frees.extend(seen.items())

        stale = self._snapshot_stale(asid, ranges)
        for base, size in frees:
            self._frames.free(base, size)
        self._released[asid] = True
        # 移除活动根表（编号保留在 _released/_outstanding 中，
        # 在票据 finish 前阻塞 alloc_asid 复用，finish 后编号可重新分配）。
        del self.root_tables[asid]
        return self._issue("destroy", asid, ranges, stale)

    def _require_active_asid(self, asid: object) -> int:
        asid = _strict_int("asid", asid)
        if asid not in self.root_tables or self._released.get(asid):
            raise UnknownAsidError(f"asid={asid} 未分配或已释放")
        return asid

    # -- map ---------------------------------------------------------------

    def map(self, asid: int, va: int,
            perms: Union[Permissions, tuple[bool, bool, bool], None] = None,
            *, huge: bool = False,
            phys: Optional[int] = None) -> tuple[Ticket, int]:
        """建立映射，返回 ``(票据, 物理基址)``。

        * ``huge=False``：va 必须 256 对齐，映射 256 字节；
        * ``huge=True``：va 必须 4096 对齐，映射 4096 字节大页，
          要求一级表项为空（不能覆盖已有二级表/大页）；
        * ``phys`` 省略时由内部分配器给出；显式给出时校验对齐、越界与
          与已分配帧重叠；
        * 目标位置已映射时抛 :class:`MappingExistsError`，不产生任何变更。
        """
        asid = self._require_active_asid(asid)
        va = _validate_va("va", va)
        permissions = self._default_perms(perms)
        if not isinstance(huge, bool):
            raise VmTypeError(f"huge 必须是 bool，得到 {type(huge).__name__}")
        size = HUGE_SIZE if huge else PAGE_SIZE
        align = HUGE_SIZE if huge else PAGE_SIZE
        if va % align != 0:
            raise AlignmentError(
                f"va=0x{va:x} 未按 {align} 对齐（{'大页' if huge else '小页'}）")

        root = self.root_tables[asid]
        i1 = l1_index(va)
        slot = root[i1]
        new_table: Optional[L2Table] = None
        if huge:
            if slot is not None:
                raise MappingExistsError(
                    f"大页区域 0x{va:x} 已被占用（一级表项 {i1} 非空）")
        else:
            if isinstance(slot, PageEntry):
                raise MappingExistsError(
                    f"小页 0x{va:x} 落在已映射大页 [0x{i1 * HUGE_SIZE:x}, "
                    f"0x{(i1 + 1) * HUGE_SIZE:x}) 内")
            if slot is None:
                new_table = L2Table()  # 先不挂入根表
                slot = new_table
            j2 = l2_index(va)
            if slot[j2] is not None:
                raise MappingExistsError(f"小页 0x{va:x} 已映射")

        # 全部结构校验通过后才分配物理帧（失败不会留下空表等部分变更）
        if phys is None:
            pa = self._frames.allocate(size, align)
        else:
            pa = self._check_external_frame(phys, size, align)

        if huge:
            root[i1] = PageEntry(pa, permissions, huge=True,
                                 frame_base=pa, frame_size=HUGE_SIZE)
        else:
            if new_table is not None:
                root[i1] = new_table
            slot[j2] = PageEntry(pa, permissions,
                                 frame_base=pa, frame_size=PAGE_SIZE)
        ticket = self._issue("map", asid, [(va, va + size)],
                             self._snapshot_stale(asid, [(va, va + size)]))
        return ticket, pa

    @staticmethod
    def _default_perms(perms: object) -> Permissions:
        if perms is None:
            return Permissions(True, True, True)
        return Permissions.coerce(perms)

    def _check_external_frame(self, phys: object, size: int,
                              align: int) -> int:
        phys = _strict_int("phys", phys)
        if phys < 0 or phys + size > PHYS_SIZE:
            raise AddressRangeError(
                f"phys=0x{phys:x} 区间越出模拟物理内存 [0, 0x{PHYS_SIZE:x})")
        if phys % align != 0:
            raise AlignmentError(f"phys=0x{phys:x} 未按 {align} 对齐")
        end = phys + size
        for start, used_end in self._frames._used:
            if phys < used_end and start < end:
                raise MappingExistsError(
                    f"物理帧 [0x{phys:x}, 0x{end:x}) 与已分配帧 "
                    f"[0x{start:x}, 0x{used_end:x}) 重叠")
        self._frames._used.append((phys, end))
        self._frames._used.sort()
        return phys

    # -- unmap / protect（通用区间计划，先整体校验，再整体提交）-------------

    def unmap(self, asid: int, va: int, length: int = PAGE_SIZE) -> Ticket:
        """解除 ``[va, va+length)`` 的映射并产生失效票据。

        length 必须是 256 的正整数倍，va 256 对齐，区间在一个 4096 一级
        区域内；区间内任一文件前不存在映射都会整体失败
        （:class:`NotMappedError`）。大页被部分解除时先拆成小页，**其余小页
        映射保留**；共享的 4096 物理帧在最后一个小页解除时才整体归还。
        """
        return self._range_op("unmap", asid, va, length, None)

    def protect(self, asid: int, va: int,
                perms: Union[Permissions, tuple[bool, bool, bool]],
                length: int = PAGE_SIZE) -> Ticket:
        """修改 ``[va, va+length)`` 内映射的权限并产生失效票据。

        对齐/区间规则同 :meth:`unmap`。对大页的局部（length < 4096 或未
        完整覆盖大页区域）保护会先把大页拆成 16 个小页：被覆盖小页采用新
        权限，其余小页原样保留（物理地址、权限均不变）。
        """
        permissions = Permissions.coerce(perms)
        return self._range_op("protect", asid, va, length, permissions)

    def _validate_range(self, asid: object, va: object,
                        length: object) -> tuple[int, int, int]:
        asid = self._require_active_asid(asid)
        va = _validate_va("va", va)
        length = _strict_int("length", length)
        if length <= 0:
            raise VmValueError(f"length 必须为正整数，得到 {length}")
        if va % PAGE_SIZE != 0:
            raise AlignmentError(f"va=0x{va:x} 未按 {PAGE_SIZE} 对齐")
        if length % PAGE_SIZE != 0:
            raise AlignmentError(f"length={length} 不是 {PAGE_SIZE} 的倍数")
        end = va + length
        if end > VIRT_SIZE:
            raise AddressRangeError(
                f"区间 [0x{va:x}, 0x{end:x}) 越出虚拟空间 0x{VIRT_SIZE:x}")
        if l1_index(va) != l1_index(end - 1):
            raise AddressRangeError(
                "区间跨越一级区域（4096）边界；请拆成多次调用")
        return asid, va, length

    def _range_op(self, kind: str, asid: int, va: int, length: int,
                  new_perms: Optional[Permissions]) -> Ticket:
        asid, va, length = self._validate_range(asid, va, length)
        root = self.root_tables[asid]
        end = va + length

        # 计划阶段：纯校验，必要时在内存中先造好拆分表但不挂入根表；
        # 任一文件前校验失败都不会改动现有状态。
        # 动作形如：
        #   ("link", i1, L2Table)     大页拆分后挂入新二级表
        #   ("huge_remove", i1, leaf)
        #   ("huge_perm", i1, new_leaf)
        #   ("small_remove", table, j2, entry)
        #   ("small_perm", table, j2, new_entry)
        actions: list[tuple] = []
        i1 = l1_index(va)
        slot = root[i1]
        region = i1 * HUGE_SIZE
        if slot is None:
            raise NotMappedError(
                f"0x{va:x} 无映射（一级表项 {i1} 为空）")
        if isinstance(slot, PageEntry):
            covers_whole = (va == region and end - region >= HUGE_SIZE)
            if covers_whole:
                if kind == "unmap":
                    actions.append(("huge_remove", i1, slot))
                else:
                    actions.append(("huge_perm", i1, PageEntry(
                        slot.phys, new_perms, huge=True,
                        frame_base=slot.frame_base, frame_size=HUGE_SIZE)))
            else:
                table = self._split_leaf(slot)  # 计划期产物，提交时挂入
                actions.append(("link", i1, table))
                self._plan_small_pages(kind, actions, table, i1, va, end,
                                       new_perms)
        else:
            self._plan_small_pages(kind, actions, slot, i1, va, end,
                                   new_perms)

        # 提交阶段（计划已保证不会再失败；帧释放在最后随动作发生）
        stale = self._snapshot_stale(asid, [(va, end)])
        for action in actions:
            op = action[0]
            if op == "link":
                root[action[1]] = action[2]
            elif op == "huge_remove":
                leaf = root[action[1]]
                root[action[1]] = None
                self._frames.free(leaf.frame_base, leaf.frame_size)
            elif op == "huge_perm":
                root[action[1]] = action[2]
            elif op == "small_remove":
                _, table, j2, entry = action
                table[j2] = None
                self._reclaim_shared_frame(table, entry)
            elif op == "small_perm":
                _, table, j2, new_entry = action
                table[j2] = new_entry
            else:  # pragma: no cover - 防御
                raise VmError(f"内部错误：未知动作 {op}")
        # 二级表 16 项全空时回收表对象（帧已按共享规则归还）
        for i1, slot in enumerate(root):
            if isinstance(slot, L2Table) and all(e is None for e in slot):
                root[i1] = None
        return self._issue(kind, asid, [(va, end)], stale)

    def _plan_small_pages(self, kind: str, actions: list[tuple],
                          table: L2Table, i1: int, va: int, end: int,
                          new_perms: Optional[Permissions]) -> None:
        page = va
        region_end = min(end, i1 * HUGE_SIZE + HUGE_SIZE)
        while page < region_end:
            j2 = l2_index(page)
            entry = table[j2]
            if entry is None:
                raise NotMappedError(
                    f"0x{page:x} 无映射（二级表项 {i1}.{j2} 为空）")
            if kind == "unmap":
                actions.append(("small_remove", table, j2, entry))
            else:
                actions.append(("small_perm", table, j2, PageEntry(
                    entry.phys, new_perms,
                    frame_base=entry.frame_base,
                    frame_size=entry.frame_size)))
            page += PAGE_SIZE

    @staticmethod
    def _split_leaf(leaf: PageEntry) -> L2Table:
        """把大页叶展开为 16 个共享底层 4096 帧的小页项。"""
        if not leaf.huge:
            raise VmError("内部错误：拆分目标不是大页叶")
        table = L2Table()
        for j in range(L2_ENTRIES):
            table[j] = PageEntry(
                leaf.phys + j * PAGE_SIZE,
                Permissions(leaf.perms.readable, leaf.perms.writable,
                            leaf.perms.executable),
                huge=False,
                frame_base=leaf.frame_base, frame_size=HUGE_SIZE)
        return table

    def _reclaim_shared_frame(self, table: L2Table, removed: PageEntry) -> None:
        """小页解除后，若共享底层帧已无引用则整帧归还。"""
        for entry in table:
            if entry is not None and entry.frame_base == removed.frame_base:
                return
        self._frames.free(removed.frame_base, removed.frame_size)

    # -- 翻译 ---------------------------------------------------------------

    def translate(self, core: int, asid: int, va: int,
                  access: Access) -> int:
        """把虚拟地址翻译为物理地址。

        命中 TLB 直接用缓存快照；未命中则遍历页表并填充 TLB（LRU，8 项，
        key 为 ``(asid, va & ~0xFF)``）。

        返回物理地址；未映射抛 :class:`NotPresentFault`，权限不足抛
        :class:`ReadFault` / :class:`WriteFault` / :class:`ExecuteFault`。
        成功不返回缓存命中与否；需要观测请直接检查 :attr:`tlbs`。
        """
        result, offset = self._resolve(core, asid, va, access, fill_tlb=True)
        return result.phys_base + offset

    def check_access(self, core: int, asid: int, va: int, length: int,
                     access: Access) -> int:
        """校验一次 ``length`` 字节访问并返回起始字节物理地址。

        跨 256 字节页边界（即使落在同一个大页内）一律抛
        :class:`CrossPageFault`——本题的访问模型以小页为粒度，调用方需自行
        拆分跨页访问。起始地址的翻译与权限检查同 :meth:`translate`。
        """
        core = _validate_core(core)
        asid_chk = self._require_active_asid(asid)
        va = _validate_va("va", va)
        length = _strict_int("length", length)
        if length <= 0:
            raise VmValueError(f"length 必须为正整数，得到 {length}")
        if va + length > VIRT_SIZE:
            raise AddressRangeError(
                f"访问 [0x{va:x}, 0x{va + length:x}) 越出虚拟空间")
        if (va & PAGE_MASK) + length > PAGE_SIZE:
            raise CrossPageFault(
                f"访问 0x{va:x} 长度 {length} 跨越 256 字节页边界",
                core=core, asid=asid_chk, va=va, access=self._access(access))
        return self.translate(core, asid_chk, va, self._access(access))

    def _access(self, access: object) -> Access:
        if not isinstance(access, Access):
            raise VmTypeError(
                f"access 必须是 Access 枚举，得到 {type(access).__name__}")
        return access

    def _resolve(self, core: int, asid: int, va: int, access: object,
                 *, fill_tlb: bool) -> tuple[WalkResult, int]:
        core = _validate_core(core)
        asid = self._require_active_asid(asid)
        va = _validate_va("va", va)
        access = self._access(access)

        key = (asid, va & ~PAGE_MASK)
        tlb = self.tlbs[core]
        cached = tlb.get(key)
        if cached is not None:
            tlb.move_to_end(key)
            self._check_perms(cached, core, asid, va, access)
            return cached, va & (HUGE_MASK if cached.huge else PAGE_MASK)

        result = self._walk(asid, va, core, access)
        self._check_perms(result, core, asid, va, access)
        if fill_tlb:
            if key in tlb:
                tlb.move_to_end(key)
            tlb[key] = result
            while len(tlb) > TLB_SIZE:
                tlb.popitem(last=False)
        return result, va & (HUGE_MASK if result.huge else PAGE_MASK)

    def _walk(self, asid: int, va: int, core: int,
              access: Access) -> WalkResult:
        """无 TLB 的纯页表遍历（独立遍历器可对照此数学过程）。"""
        root = self.root_tables[asid]
        slot = root[l1_index(va)]
        if slot is None:
            raise NotPresentFault("一级表项为空", core=core, asid=asid, va=va,
                                  access=access)
        if isinstance(slot, PageEntry):
            return WalkResult(slot.phys, Permissions(
                slot.perms.readable, slot.perms.writable,
                slot.perms.executable), huge=True)
        entry = slot[l2_index(va)]
        if entry is None:
            raise NotPresentFault("二级表项为空", core=core, asid=asid, va=va,
                                  access=access)
        return WalkResult(entry.phys, Permissions(
            entry.perms.readable, entry.perms.writable,
            entry.perms.executable), huge=False)

    def direct_walk(self, asid: int, va: int) -> WalkResult:
        """公开的无 TLB 遍历：只读页表、不碰任何缓存，供验收核对。"""
        asid = self._require_active_asid(asid)
        va = _validate_va("va", va)
        return self._walk(asid, va, -1, Access.READ)

    @staticmethod
    def _check_perms(result: WalkResult, core: int, asid: int, va: int,
                     access: Access) -> None:
        p = result.perms
        allowed = (access is Access.READ and p.readable) or \
                  (access is Access.WRITE and p.writable) or \
                  (access is Access.EXECUTE and p.executable)
        if not allowed:
            fault_cls = MemorySystem._FAULT_FOR_ACCESS[access]
            label = {Access.READ: "读", Access.WRITE: "写",
                     Access.EXECUTE: "执行"}[access]
            raise fault_cls(
                f"0x{va:x} 权限 {p.as_rwx()} 拒绝{label}访问",
                core=core, asid=asid, va=va, access=access)

    # -- 失效票据 ----------------------------------------------------------

    def _snapshot_stale(self, asid: int,
                        ranges: list[tuple[int, int]]
                        ) -> dict[int, list[tuple[int, int]]]:
        """记录各核当前命中受影响区间的 TLB key（票据 ack 前允许其陈旧）。"""
        stale: dict[int, list[tuple[int, int]]] = {}
        for core in range(CORE_COUNT):
            keys = [k for k in self.tlbs[core]
                    if k[0] == asid and self._in_ranges(k[1], ranges)]
            if keys:
                stale[core] = keys
        return stale

    @staticmethod
    def _in_ranges(page_va: int,
                   ranges: list[tuple[int, int]]) -> bool:
        for start, end in ranges:
            if start <= page_va < end:
                return True
        return False

    def _issue(self, kind: str, asid: int,
               ranges: list[tuple[int, int]],
               stale: dict[int, list[tuple[int, int]]]) -> Ticket:
        ticket_id = self._next_ticket
        self._next_ticket += 1
        ticket = Ticket(
            id=ticket_id, asid=asid, kind=kind,
            ranges=tuple(ranges),
            target_cores=frozenset(stale),
            creator=f"asid{asid}:{kind}",
            acked=set())
        # 票据对象本身保存 ack 前应清除的陈旧 key 快照（对外不公开字段）
        ticket._stale_keys = stale  # type: ignore[attr-defined]
        self._outstanding.setdefault(asid, set()).add(ticket_id)
        self._tickets_store[ticket_id] = ticket
        return ticket

    def ack(self, core: int, ticket: Union[Ticket, int]) -> Ticket:
        """目标核确认失效：清除该核在票据快照内的本地陈旧 TLB 项。"""
        core = _validate_core(core)
        t = self._ticket_obj(ticket)
        if core not in t.target_cores:
            raise NotTicketTargetError(
                f"核 {core} 不是票据 #{t.id} 的目标（目标={sorted(t.target_cores)}）")
        if core in t.acked:
            raise AlreadyAckedError(f"核 {core} 已 ack 票据 #{t.id}")
        for key in t._stale_keys.get(core, []):  # type: ignore[attr-defined]
            self.tlbs[core].pop(key, None)
        t.acked.add(core)
        return t

    def finish(self, ticket: Union[Ticket, int]) -> Ticket:
        """结束票据；仍有目标核未 ack 时抛 :class:`PendingFinishError`。"""
        t = self._ticket_obj(ticket)
        if not t.is_complete():
            raise PendingFinishError(
                f"票据 #{t.id} 等待核 {t.pending_cores()} ack")
        t.finished = True
        self._outstanding[t.asid].discard(t.id)
        self._tickets_store.pop(t.id, None)
        return t

    def _ticket_obj(self, ticket: Union[Ticket, int]) -> Ticket:
        if isinstance(ticket, Ticket):
            tid = ticket.id
        else:
            tid = _strict_int("ticket", ticket)
        store = getattr(self, "_tickets_store", {})
        if tid not in store:
            raise UnknownTicketError(f"票据 #{tid} 不存在或已 finish")
        return store[tid]

    # -- 观测（演示/测试用，不改变状态）------------------------------------

    def tlb_keys(self, core: int) -> list[tuple[int, int]]:
        """返回某核 TLB 当前缓存的 ``(asid, 小页虚拟基址)`` 列表（LRU 序）。"""
        core = _validate_core(core)
        return list(self.tlbs[core].keys())

    def used_phys_bytes(self) -> int:
        return self._frames.used_bytes()

    def outstanding_tickets(self, asid: int) -> list[int]:
        asid = self._require_active_asid(asid)
        return sorted(self._outstanding.get(asid, ()))
