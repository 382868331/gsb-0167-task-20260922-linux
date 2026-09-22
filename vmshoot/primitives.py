"""固定常量、权限位、地址拆分与输入校验（纯函数，无状态）。

地址布局（虚拟地址 32 位、无符号解释）::

    | L1 idx (4b) | L2 idx (4b) | offset (8b) |
      bit 11..8     bit 7..4       bit 3..0? —— 实际见下

页大小 256 字节 => offset 占 8 位（bit 7..0）；
两级索引各 16 项 => 各占 4 位，L2 在 bit 11..8，L1 在 bit 15..12。
虚拟地址空间因此为 2^16 = 65536 字节；高位 bit 31..16 必须为 0。

大页 = 16 个连续小页 = 4096 字节，对齐 4096；大页内偏移占 12 位（bit 11..0）。
"""

from __future__ import annotations

import math

# ---- 体系结构常量（题面固定，不可配置） -----------------------------------
PAGE_BITS = 8
PAGE_SIZE = 1 << PAGE_BITS          # 256 字节
PAGE_MASK = PAGE_SIZE - 1
L2_BITS = 4
L2_SIZE = 1 << L2_BITS              # 每级 16 项
L1_BITS = 4
L1_SIZE = 1 << L1_BITS
# 小页模型下的虚拟地址宽度：4 + 4 + 8 = 16
VA_BITS = L1_BITS + L2_BITS + PAGE_BITS   # 16
VADDR_SPACE = 1 << VA_BITS          # 65536
VADDR_MAX = VADDR_SPACE - 1
HUGE_BITS = L2_BITS + PAGE_BITS     # 12
HUGE_PAGE_SIZE = 1 << HUGE_BITS     # 4096
HUGE_PAGE_MASK = HUGE_PAGE_SIZE - 1

# ---- 工程限制（题面固定） -------------------------------------------------
MAX_ADDRESS_SPACES = 4
NUM_CORES = 2
TLB_SIZE = 8  # 每核

# ---- 权限位 --------------------------------------------------------------
PERM_READ = 1 << 0
PERM_WRITE = 1 << 1
PERM_EXEC = 1 << 2
PERM_RWX = PERM_READ | PERM_WRITE | PERM_EXEC
PERM_RW = PERM_READ | PERM_WRITE
PERM_RX = PERM_READ | PERM_EXEC

PERM_NAMES = {
    PERM_READ: "r",
    PERM_WRITE: "w",
    PERM_EXEC: "x",
}


def perm_to_str(perms: int) -> str:
    """把权限位向量渲染成 ``rwx`` 风格字符串（缺失位显示 ``-``）。"""
    return (
        ("r" if perms & PERM_READ else "-")
        + ("w" if perms & PERM_WRITE else "-")
        + ("x" if perms & PERM_EXEC else "-")
    )


def validate_int(value, name: str, *, nonnegative: bool = True, max_value: int | None = None) -> int:
    """整数参数校验入口。

    * 拒绝 ``bool``（``True``/``False`` 不是合法整数参数）；
    * 拒绝 ``float``（含 ``nan``/``inf``，也含 1.0 之类——整数参数必须传 int）；
      其他非 int 类型拒绝；
    * 可要求非负与上界。错误消息带 ``name`` 便于定位。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float):
            # 明确点出 NaN / Infinity，便于定位
            if math.isnan(value):
                raise ValueError(f"{name} 不能是 NaN")
            if math.isinf(value):
                raise ValueError(f"{name} 不能是 Infinity")
            raise TypeError(f"{name} 必须是 int，得到 float: {value!r}")
        raise TypeError(f"{name} 必须是 int，得到 {type(value).__name__}: {value!r}")
    if nonnegative and value < 0:
        raise ValueError(f"{name} 不能为负，得到 {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name}={value} 超过上界 {max_value}")
    return value


def validate_vaddr(vaddr, name: str = "vaddr") -> int:
    """校验并规范化虚拟地址：非负 int，且 bit 31..16 为零。"""
    v = validate_int(vaddr, name, max_value=0xFFFFFFFF)
    if v > VADDR_MAX:
        raise ValueError(f"{name}=0x{v:x} 超出 16 位虚拟地址空间（最大 0x{VADDR_MAX:x}）")
    return v


def validate_size(size, name: str = "size") -> int:
    size = validate_int(size, name)
    if size == 0:
        raise ValueError(f"{name} 必须为正整数，得到 0")
    if size > HUGE_PAGE_SIZE:
        raise ValueError(
            f"{name}={size} 超过单次访问最大长度 {HUGE_PAGE_SIZE}（大页 4096 字节）"
        )
    return size


def validate_perms(perms, name: str = "perms") -> int:
    """校验权限位向量：int、非负、只含 r/w/x 三个已知位、且非零。"""
    p = validate_int(perms, name, max_value=0xFFFFFFFF)
    if p & ~PERM_RWX:
        raise ValueError(f"{name}=0x{p:x} 含未知权限位（仅允许 r/w/x = 0..7）")
    if p == 0:
        raise ValueError(f"{name} 不能为 0（至少需要一种权限）")
    return p


def split_vaddr(vaddr: int) -> tuple[int, int, int]:
    """虚拟地址 -> ``(L1 索引, L2 索引, 页内偏移)``。"""
    l1 = (vaddr >> (L2_BITS + PAGE_BITS)) & (L1_SIZE - 1)
    l2 = (vaddr >> PAGE_BITS) & (L2_SIZE - 1)
    off = vaddr & PAGE_MASK
    return l1, l2, off


def page_base(vaddr: int) -> int:
    return vaddr & ~PAGE_MASK


def huge_base(vaddr: int) -> int:
    return vaddr & ~HUGE_PAGE_MASK


def align_down(v: int, align: int) -> int:
    return v - (v % align)
