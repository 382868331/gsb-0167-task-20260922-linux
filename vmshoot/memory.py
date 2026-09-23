"""模拟物理内存：256 字节一帧的线性帧分配器 + 帧引用计数 + 页内容后备。

* 不做任何真实 OS 内存映射（无 mmap/mmap 等价物），只用 Python 位图模拟占用；
* 大页需要 16 帧且首帧 16 对齐，分配采用确定性“首位适配”全扫描——
  找不到就是在当前占用状态下证明无解，抛 :class:`FrameExhausted`，
  本题不存在“搜索预算耗尽”这类情况。
* 每个物理帧带一个整数引用计数：地址空间克隆时父子共享同一帧（计数 +1），
  任一方解除映射只把计数减 1，计数归零时帧才真正回到帧池；写时复制分裂时
  分配新帧并把源帧 256 字节内容逐字节复制过去。
* 帧内容用惰性创建的 ``bytearray(256)`` 后备（不存放任何真实内存数据）。
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
        self.num_frames = num_frames
        # 引用计数：0 = 空闲；>=1 = 被多少个 PTE 共享
        self._refs = [0] * num_frames
        # 帧内容后备：惰性创建，帧最终释放时清空（重新分配即为全零帧）
        self._data: list[bytearray | None] = [None] * num_frames

    # ---- 内部 ---------------------------------------------------------------
    def _check_ppn(self, ppn: int) -> None:
        if not 0 <= ppn < self.num_frames:
            raise ValueError(f"ppn={ppn} 越界（帧池共 {self.num_frames} 帧）")

    def _check_live(self, ppn: int) -> None:
        self._check_ppn(ppn)
        if self._refs[ppn] == 0:
            raise AssertionError(f"内部错误：帧 {ppn} 未被占用")

    # ---- 公开接口 -----------------------------------------------------------
    def alloc_frame(self) -> int:
        """分配一个小页帧（引用计数置 1），返回物理页号 (ppn)。"""
        for ppn in range(self.num_frames):
            if self._refs[ppn] == 0:
                self._refs[ppn] = 1
                return ppn
        raise FrameExhausted(f"物理帧已耗尽（共 {self.num_frames} 帧）")

    def alloc_huge(self) -> int:
        """分配 16 个连续、且起点 16 对齐的帧（各置引用 1），返回起始 ppn。"""
        n = self.num_frames
        for start in range(0, n - L2_SIZE + 1, L2_SIZE):
            if all(self._refs[start + j] == 0 for j in range(L2_SIZE)):
                for j in range(L2_SIZE):
                    self._refs[start + j] = 1
                return start
        raise FrameExhausted(f"无法分配 16 连续对齐帧（帧池共 {n} 帧）")

    def retain_frame(self, ppn: int) -> None:
        """把已有帧的引用计数加 1（地址空间克隆共享帧时使用）。"""
        self._check_live(ppn)
        self._refs[ppn] += 1

    def release_frame(self, ppn: int) -> bool:
        """引用计数减 1；返回 True 表示这是最后一个引用、帧已回收。

        计数为 0（未占用）时抛 ``ValueError``，与旧版“释放未占帧”错误一致。
        """
        self._check_ppn(ppn)
        if self._refs[ppn] == 0:
            raise ValueError(f"帧 {ppn} 未处于占用状态，不能释放")
        self._refs[ppn] -= 1
        if self._refs[ppn] == 0:
            self._data[ppn] = None  # 回收即清零内容，重新分配得到全零帧
            return True
        return False

    # 旧名兼容：free_* 现在按引用计数语义释放
    def free_frame(self, ppn: int) -> None:
        self.release_frame(ppn)

    def free_huge(self, ppn_base: int) -> None:
        if ppn_base % L2_SIZE != 0:
            raise ValueError(f"大页起始 ppn 必须 {L2_SIZE} 对齐，得到 {ppn_base}")
        for j in range(L2_SIZE):
            self.release_frame(ppn_base + j)

    def retain_huge(self, ppn_base: int) -> None:
        """大页 16 帧全部引用计数加 1（克隆共享大页时使用）。"""
        if ppn_base % L2_SIZE != 0:
            raise ValueError(f"大页起始 ppn 必须 {L2_SIZE} 对齐，得到 {ppn_base}")
        for j in range(L2_SIZE):
            self.retain_frame(ppn_base + j)

    def alloc_cow_copy(self, src_ppn: int) -> int:
        """为写时复制分裂分配一个新帧，并把 ``src_ppn`` 的 256 字节复制过去。

        先做全扫描分配：若帧池耗尽抛 :class:`FrameExhausted`，此时不改动任何
        引用计数（调用方据此保证“失败时原映射保持”）。
        """
        self._check_live(src_ppn)
        dst_ppn = self.alloc_frame()  # 置 dst 引用 1
        src = self._data[src_ppn]
        dst = bytearray(PAGE_SIZE)
        if src is not None:
            dst[:] = src
        self._data[dst_ppn] = dst
        return dst_ppn

    def frame_refs(self, ppn: int) -> int:
        """诊断：帧当前引用计数。"""
        self._check_ppn(ppn)
        return self._refs[ppn]

    def buffer(self, ppn: int) -> bytearray:
        """取得帧的 256 字节内容后备（惰性创建，首访为全零）。"""
        self._check_live(ppn)
        buf = self._data[ppn]
        if buf is None:
            buf = bytearray(PAGE_SIZE)
            self._data[ppn] = buf
        return buf

    def phys_addr(self, ppn: int, offset: int) -> int:
        """``ppn`` + 页内偏移 -> 物理地址。"""
        if not 0 <= offset < PAGE_SIZE:
            raise ValueError(f"offset={offset} 超出 0..{PAGE_SIZE - 1}")
        return ppn * PAGE_SIZE + offset

    # ---- 物理地址级内容读写（逐帧，帧必须处于占用状态） ----------------------
    def read_phys(self, phys_addr, length) -> bytes:
        """从物理地址读 ``length`` 字节（可跨帧；途经帧必须都被占用）。"""
        from .primitives import validate_int

        pa = validate_int(phys_addr, "phys_addr")
        n = validate_int(length, "length")
        if n == 0:
            raise ValueError("length 必须为正整数，得到 0")
        if pa + n > self.num_frames * PAGE_SIZE:
            raise ValueError(
                f"物理访问区间 0x{pa:x}..0x{pa + n - 1:x} 超出帧池"
            )
        out = bytearray()
        while len(out) < n:
            cur = pa + len(out)
            frame, off = divmod(cur, PAGE_SIZE)
            take = min(n - len(out), PAGE_SIZE - off)
            out += self.buffer(frame)[off:off + take]
        return bytes(out)

    def write_phys(self, phys_addr, data) -> None:
        """向物理地址写入 bytes-like 内容（可跨帧；途经帧必须都被占用）。"""
        from .primitives import validate_int

        pa = validate_int(phys_addr, "phys_addr")
        if isinstance(data, (bytes, bytearray, memoryview)):
            payload = bytes(data)
        else:
            raise TypeError(f"data 必须是 bytes-like，得到 {type(data).__name__}")
        if not payload:
            raise ValueError("data 不能为空")
        if pa + len(payload) > self.num_frames * PAGE_SIZE:
            raise ValueError(
                f"物理访问区间 0x{pa:x}..0x{pa + len(payload) - 1:x} 超出帧池"
            )
        done = 0
        while done < len(payload):
            cur = pa + done
            frame, off = divmod(cur, PAGE_SIZE)
            take = min(len(payload) - done, PAGE_SIZE - off)
            self.buffer(frame)[off:off + take] = payload[done:done + take]
            done += take

    def used_frames(self) -> int:
        """占用帧数（引用计数 > 0 的物理帧数；共享只计一帧）。"""
        return sum(1 for r in self._refs if r > 0)
