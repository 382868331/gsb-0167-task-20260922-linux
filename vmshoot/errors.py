"""异常层次：所有库抛出的异常都继承自 :class:`VMShootError`。

错误语义约定：

* 类型错误（含把 ``bool`` 当整数、把 ``float`` 当整数参数）抛 :class:`TypeError`；
* ``NaN`` / ``Infinity`` 浮点输入抛 :class:`ValueError`；
* 其余“值合法但违反本题契约”的情况抛本模块定义的异常，消息中带参数名与关键值，
  便于定位（不静默纠正）。
"""

from __future__ import annotations


class VMShootError(Exception):
    """本库所有自定义异常的根。"""


class InvalidAddressSpace(VMShootError):
    """ASID 不存在，或地址空间已被销毁。"""


class AddressSpaceLimit(VMShootError):
    """活动地址空间数量已达 ``MAX_ADDRESS_SPACES``（4）。"""


class AsidBusy(VMShootError):
    """想复用的 ASID 编号上还有未 finish 的失效票据。"""

    def __init__(self, asid: int, ticket_ids: list[int]):
        self.asid = asid
        self.ticket_ids = list(ticket_ids)
        super().__init__(
            f"ASID {asid} 仍有未完成的失效票据 {self.ticket_ids}，不能被新地址空间复用"
        )


class MappingExists(VMShootError):
    """目标虚拟页已经存在映射；改映射前必须先 unmap。"""


class NotMapped(VMShootError):
    """目标虚拟页没有映射（unmap/protect/写故障处理要求已映射）。"""


class NotCopyOnWrite(VMShootError):
    """对一个不是写时复制（COW）状态的页调用 :meth:`MMU.handle_write_fault`。"""


class MappingConflict(VMShootError):
    """结构性冲突：小页操作落在大页叶子上（或反向），且该操作不允许隐式拆分。"""


class FrameExhausted(VMShootError):
    """物理帧耗尽。分配器做的是确定性全扫描，因此这是“证明无解”，不是预算耗尽。"""


class InvalidationsPending(VMShootError):
    """:meth:`MMU.finish` 时仍有目标核没有 ACK。"""

    def __init__(self, ticket_id: int, outstanding: tuple[int, ...]):
        self.ticket_id = ticket_id
        self.outstanding = tuple(outstanding)
        super().__init__(
            f"票据 {ticket_id} 尚未被所有目标核 ACK，未响应核: {list(outstanding)}"
        )


class TicketError(VMShootError):
    """票据不存在、已关闭、重复 ACK，或被非目标核 ACK。"""


class CrossPageAccess(VMShootError):
    """一次访问跨越了其所在映射页（小页 256B / 大页 4096B）的边界，按契约拒绝。"""
