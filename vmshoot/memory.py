"""模拟物理内存：256 字节一帧的线性帧分配器。

* 不做任何真实 OS 内存映射（无 mmap/mmap 等价物），只用 Python 位图模拟占用；
* 大页需要 16 帧且首帧 16 对齐，分配采用确定性“首位适配”全扫描——
  找不到就是在当前占用状态下证明无解，抛 :class:`FrameExhausted`，
  本题不存在“搜索预算耗尽”这类情况。
"""

from __future__ import annotations

from .errors import FrameExhausted
from .primitives import L2_SIZE, PAGE_SIZE


class PhysicalMemory:
    def __init__(self, num_frames: int = L2_SIZE * L2_SIZE * 4 + 64):
        # 默认帧池：4 个地址空间各 256 帧 + 64 帧余量
        if isinstance(num_frames, bool) or not isinstance(num_frames, int):
            raise TypeError("num_frames 必须是 int")
        if num_frames <= 0:
            raise ValueError("num_frames 必须为正整数")
        self._used = bytearray(num_frames)
        self.num_frames = num_frames

    # ---- 内部 ---------------------------------------------------------------
    def _check_ppn(self, ppn: int) -> None:
        if not 0 <= ppn < self.num_frames:
            raise ValueError(f"ppn={ppn} 越界（帧池共 {self.num_frames} 帧）")
        if self._used[ppn]:
            raise AssertionError(f"内部错误：帧 {ppn} 重复分配")

    # ---- 公开接口 -----------------------------------------------------------
    def alloc_frame(self) -> int:
        """分配一个小页帧，返回物理页号 (ppn)。"""
        table = self._used
        for ppn in range(self.num_frames):
            if not table[ppn]:
                table[ppn] = 1
                return ppn
        raise FrameExhausted(f"物理帧已耗尽（共 {self.num_frames} 帧）")

    def alloc_huge(self) -> int:
        """分配 16 个连续、且起点 16 对齐的帧，返回起始 ppn。"""
        table = self._used
        n = self.num_frames
        for start in range(0, n - L2_SIZE + 1, L2_SIZE):
            if all(table[start + j] == 0 for j in range(L2_SIZE)):
                for j in range(L2_SIZE):
                    table[start + j] = 1
                return start
        raise FrameExhausted(f"无法分配 16 连续对齐帧（帧池共 {n} 帧）")

    def free_frame(self, ppn: int) -> None:
        if not 0 <= ppn < self.num_frames:
            raise ValueError(f"ppn={ppn} 越界（帧池共 {self.num_frames} 帧）")
        if not self._used[ppn]:
            raise ValueError(f"帧 {ppn} 未处于占用状态，不能释放")
        self._used[ppn] = 0

    def free_huge(self, ppn_base: int) -> None:
        if ppn_base % L2_SIZE != 0:
            raise ValueError(f"大页起始 ppn 必须 {L2_SIZE} 对齐，得到 {ppn_base}")
        for j in range(L2_SIZE):
            self.free_frame(ppn_base + j)

    def phys_addr(self, ppn: int, offset: int) -> int:
        """``ppn`` + 页内偏移 -> 物理地址。"""
        if not 0 <= offset < PAGE_SIZE:
            raise ValueError(f"offset={offset} 超出 0..{PAGE_SIZE - 1}")
        return ppn * PAGE_SIZE + offset

    def used_frames(self) -> int:
        return sum(self._used)
