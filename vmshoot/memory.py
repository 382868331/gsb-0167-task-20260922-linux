"""模拟物理内存：256 字节一帧的线性帧分配器，带引用计数。

* 不做任何真实 OS 内存映射（无 mmap/mmap 等价物），只用 Python 位图模拟占用；
* 大页需要 16 帧且首帧 16 对齐，分配采用确定性“首位适配”全扫描——
  找不到就是在当前占用状态下证明无解，抛 :class:`FrameExhausted`，
  本题不存在“搜索预算耗尽”这类情况；
* 引用计数：地址空间克隆（写时复制）让多个 PTE 可以共享同一物理帧。
  :meth:`retain_frame` 增加引用，:meth:`release_frame` 减少引用，**仅当最后一个
  引用释放时帧才真正回到空闲池**。``_used`` 位图表示“帧是否被物理占用”，
  ``_refs`` 是占用帧上的引用计数（空闲帧恒为 0）。
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
        self._refs = bytearray(num_frames)
        self.num_frames = num_frames

    # ---- 内部 ---------------------------------------------------------------
    def _check_ppn(self, ppn: int) -> None:
        if not 0 <= ppn < self.num_frames:
            raise ValueError(f"ppn={ppn} 越界（帧池共 {self.num_frames} 帧）")
        if self._used[ppn]:
            raise AssertionError(f"内部错误：帧 {ppn} 重复分配")

    # ---- 公开接口 -----------------------------------------------------------
    def alloc_frame(self) -> int:
        """分配一个小页帧，返回物理页号 (ppn)；新帧引用计数为 1。"""
        table = self._used
        for ppn in range(self.num_frames):
            if not table[ppn]:
                table[ppn] = 1
                self._refs[ppn] = 1
                return ppn
        raise FrameExhausted(f"物理帧已耗尽（共 {self.num_frames} 帧）")

    def alloc_huge(self) -> int:
        """分配 16 个连续、且起点 16 对齐的帧，返回起始 ppn；每帧引用计数为 1。"""
        table = self._used
        n = self.num_frames
        for start in range(0, n - L2_SIZE + 1, L2_SIZE):
            if all(table[start + j] == 0 for j in range(L2_SIZE)):
                for j in range(L2_SIZE):
                    table[start + j] = 1
                    self._refs[start + j] = 1
                return start
        raise FrameExhausted(f"无法分配 16 连续对齐帧（帧池共 {n} 帧）")

    def retain_frame(self, ppn: int) -> None:
        """为已占用帧增加一个共享引用（克隆地址空间时使用）。"""
        if not 0 <= ppn < self.num_frames:
            raise ValueError(f"ppn={ppn} 越界（帧池共 {self.num_frames} 帧）")
        if not self._used[ppn]:
            raise AssertionError(f"内部错误：不能 retain 未占用帧 {ppn}")
        self._refs[ppn] += 1

    def release_frame(self, ppn: int) -> bool:
        """释放一个帧引用。

        引用计数减 1；**仅当减到 0（最后一个引用）时帧才真正回收**，返回
        ``True``；否则帧仍被其他 PTE 共享占用，返回 ``False``。
        """
        if not 0 <= ppn < self.num_frames:
            raise ValueError(f"ppn={ppn} 越界（帧池共 {self.num_frames} 帧）")
        if not self._used[ppn] or self._refs[ppn] == 0:
            raise ValueError(f"帧 {ppn} 未处于占用状态，不能释放")
        self._refs[ppn] -= 1
        if self._refs[ppn] == 0:
            self._used[ppn] = 0
            return True
        return False

    def frame_refs(self, ppn: int) -> int:
        """诊断用：返回帧当前引用计数（0 表示空闲）。"""
        if not 0 <= ppn < self.num_frames:
            raise ValueError(f"ppn={ppn} 越界（帧池共 {self.num_frames} 帧）")
        return self._refs[ppn]

    def free_frame(self, ppn: int) -> None:
        """遗留的单引用释放接口（要求帧占用且引用计数为 1）。

        新代码（unmap / destroy / COW）一律使用 :meth:`release_frame`，
        以正确处理共享帧；本方法保留给直接操作帧池的既有调用方。
        """
        if not 0 <= ppn < self.num_frames:
            raise ValueError(f"ppn={ppn} 越界（帧池共 {self.num_frames} 帧）")
        if not self._used[ppn]:
            raise ValueError(f"帧 {ppn} 未处于占用状态，不能释放")
        if self._refs[ppn] != 1:
            raise ValueError(f"帧 {ppn} 引用计数为 {self._refs[ppn]}，不能用 free_frame 直接释放")
        self._refs[ppn] = 0
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
        """物理上被占用的帧数（共享帧只计一次）。"""
        return sum(self._used)
