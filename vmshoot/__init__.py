"""vmshoot —— ASID 下的两级页表与 TLB 失效模拟库（纯标准库，离线）。

体系结构（题面固定子集，非行业标准兼容实现）：

* 虚拟地址 16 位：L1 索引 4 位 + L2 索引 4 位 + 页内偏移 8 位；
  两级页表每级 16 项，小页 256 字节；一级叶子可表示 4096 字节大页。
* 2 个核，每核 8 项 LRU TLB；TLB 的 key 为 ``(asid, vpn)``，故不同地址空间
  的缓存天然隔离，大页项以 16 对齐的 vpn 作为 key。
* 最多 4 个活动地址空间（ASID 0..3）。物理内存只是 Python 内的位图模拟，
  不调用任何 OS 映射（无 mmap）；每帧带引用计数以支持地址空间克隆共享。

地址空间克隆与写时复制（COW）：

* :meth:`MMU.clone_address_space` 让父子空间共享同一批普通物理页（引用计数 +1、
  不分配新帧），可写页双方打 ``cow`` 标记，大页在父子两侧就地展开为 16 小页；
* 对 COW 页的写访问令 :meth:`MMU.translate` 返回 ``fault="write", cow=True``，
  须显式串行调用 :meth:`MMU.handle_write_fault` 分配私有新帧、切换 PTE、释放旧帧
  引用，并发出 ``kind="cow"`` 的失效票据；最后一个引用释放时帧才回收。

失效协议（map/unmap/protect）：

1. 先更新页表；
2. 产生一张 :class:`Ticket`，列出需要失效的 TLB key 与全部目标核；
3. ACK 之前，其他核上的旧 TLB 项允许继续存在并被观察到（陈旧翻译）；
4. 每个目标核调用 :meth:`MMU.ack` 后该项才从该核 TLB 移除；
5. :meth:`MMU.finish` 必须等待所有目标核 ACK，否则抛
   :class:`InvalidationsPending`。
6. 被销毁/重建的 ASID 号在仍有未 finish 票据时不能复用（:class:`AsidBusy`）。

翻译：:meth:`MMU.translate` 返回 :class:`TranslationResult`，成功给物理地址，
失败给 ``not_mapped/read/write/execute`` fault；一次访问跨越所在页边界抛
:class:`CrossPageAccess`。另有 :meth:`MMU.walk_table` 作为“无 TLB 遍历器”
供验收在全部 ACK 后核对。
"""

from __future__ import annotations

from .core import (
    MMU,
    AddressSpace,
    AccessType,
    PTE,
    Ticket,
    TranslationResult,
    WalkResult,
)
from .errors import (
    VMShootError,
    AddressSpaceLimit,
    AsidBusy,
    CrossPageAccess,
    FrameExhausted,
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
    HUGE_PAGE_SIZE,
    L1_SIZE,
    L2_SIZE,
    MAX_ADDRESS_SPACES,
    NUM_CORES,
    PAGE_SIZE,
    PERM_EXEC,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PERM_WRITE,
    TLB_SIZE,
    VADDR_MAX,
    VADDR_SPACE,
    perm_to_str,
)

__all__ = [
    # 主门面
    "MMU",
    "AddressSpace",
    "PhysicalMemory",
    "AccessType",
    "PTE",
    "Ticket",
    "TranslationResult",
    "WalkResult",
    # 常量
    "PAGE_SIZE",
    "HUGE_PAGE_SIZE",
    "L1_SIZE",
    "L2_SIZE",
    "TLB_SIZE",
    "NUM_CORES",
    "MAX_ADDRESS_SPACES",
    "VADDR_MAX",
    "VADDR_SPACE",
    "PERM_READ",
    "PERM_WRITE",
    "PERM_EXEC",
    "PERM_RWX",
    "PERM_RW",
    "PERM_RX",
    "perm_to_str",
    # 异常
    "VMShootError",
    "AddressSpaceLimit",
    "AsidBusy",
    "CrossPageAccess",
    "FrameExhausted",
    "InvalidAddressSpace",
    "InvalidationsPending",
    "MappingConflict",
    "MappingExists",
    "NotCopyOnWrite",
    "NotMapped",
    "TicketError",
]

__version__ = "1.0.0"
