"""地址空间克隆与写时复制（COW）测试。

期望值来源（不调用被测实现生成）：

* 帧号：按题面“线性帧池首位适配”手算——小页取最小空帧，大页取第一个全空的
  16 对齐块；本文件每个用例在注释里给出推导过程。
* 引用计数：共享 +1、解除映射/分裂 -1、归零回收，按定义手算。
* 页内容：克隆共享、分裂复制、解除映射不影响他方，按字节字面量核对。
* 差分部分：:class:`RefCOW` 是按契约独立重写的参考机（自己的帧池/引用计数/
  内容字典/页表/TLB），与被测实现不共享算法代码。
"""

from __future__ import annotations

import random
import unittest
from collections import OrderedDict

from vmshoot import (
    MMU,
    AccessType,
    AddressSpaceLimit,
    CrossPageAccess,
    FrameExhausted,
    InvalidAddressSpace,
    InvalidationsPending,
    InvalidWriteFault,
    MemoryAccessFault,
    NotMapped,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_WRITE,
    PhysicalMemory,
)


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        for c in (0, 1):
            mmu.ack(c, t.id)
        mmu.finish(t.id)


def settle_all(mmu: MMU) -> None:
    for tid in list(mmu.pending_tickets()):
        for c in (0, 1):
            mmu.ack(c, tid)
        mmu.finish(tid)


class TestCloneSharesSmallPages(unittest.TestCase):
    """手算场景 A：克隆共享小页 -> 父子双写各自分裂 -> 引用计数与回收。"""

    def setUp(self):
        self.mmu = MMU()
        self.a = self.mmu.create_address_space()          # asid 0
        t1 = self.mmu.map(self.a, 0x1000, PERM_RW)        # 帧 0
        t2 = self.mmu.map(self.a, 0x2000, PERM_RW)        # 帧 1
        settle(self.mmu, t1, t2)
        self.mmu.write_bytes(self.a, 0x1000, b"parent-data")  # 11 字节

    def test_clone_shares_frames_and_sets_cow(self):
        res = self.mmu.clone_address_space(self.a)
        self.assertEqual(res.child_asid, 1)
        # 手算：克隆不分配新帧；帧 0/1 引用计数 1 -> 2
        self.assertEqual(self.mmu.mem.used_frames(), 2)
        self.assertEqual(self.mmu.mem.frame_refs(0), 2)
        self.assertEqual(self.mmu.mem.frame_refs(1), 2)
        # 父子 PTE 同帧、都带 COW 标记
        wp = self.mmu.walk_table(self.a, 0x1000)
        wc = self.mmu.walk_table(res.child_asid, 0x1000)
        self.assertEqual((wp.ppn, wp.cow), (0, True))
        self.assertEqual((wc.ppn, wc.cow), (0, True))
        # 子空间经共享帧读到父空间克隆前写入的内容
        self.assertEqual(self.mmu.read_bytes(res.child_asid, 0x1000, 11), b"parent-data")
        # 克隆票据：kind=clone，覆盖父空间两个小页 key
        self.assertEqual(res.ticket.kind, "clone")
        self.assertEqual(res.ticket.keys, ((0, 0x10), (0, 0x20)))
        settle(self.mmu, res.ticket)

    def test_double_write_diverges_and_last_ref_reclaims(self):
        mmu, a = self.mmu, self.a
        child = mmu.clone_address_space(a).child_asid
        settle_all(mmu)
        # 克隆票据 ACK 后：父子写共享页都报 cow fault（走页表）
        for asid in (a, child):
            r = mmu.translate(asid, 0, 0x1000, AccessType.WRITE)
            self.assertFalse(r.ok)
            self.assertEqual(r.fault, "cow")
            self.assertEqual(r.via, "walk")
        # 父先分裂：手算 -> 新帧为最小空帧 2；帧 0 引用 2->1
        tp = mmu.handle_write_fault(a, 0x1000)
        self.assertEqual(tp.kind, "cow")
        self.assertEqual(tp.keys, ((a, 0x10),))
        settle(mmu, tp)
        self.assertEqual(mmu.walk_table(a, 0x1000).ppn, 2)
        self.assertFalse(mmu.walk_table(a, 0x1000).cow)
        self.assertEqual(mmu.mem.frame_refs(0), 1)
        self.assertEqual(mmu.mem.frame_refs(2), 1)
        # 父写自己的私有副本，子空间内容不变（手算字节核对）
        mmu.write_bytes(a, 0x1000, b"parent-v2!!")
        self.assertEqual(mmu.read_bytes(a, 0x1000, 11), b"parent-v2!!")
        self.assertEqual(mmu.read_bytes(child, 0x1000, 11), b"parent-data")
        # 子再分裂：帧 0/1/2 都占用 -> 新帧 3；帧 0 引用 1->0 回收
        tc = mmu.handle_write_fault(child, 0x1000)
        settle(mmu, tc)
        self.assertEqual(mmu.walk_table(child, 0x1000).ppn, 3)
        self.assertEqual(mmu.mem.frame_refs(0), 0)   # 最后引用释放，帧回收
        self.assertEqual(mmu.mem.used_frames(), 3)   # 帧 1,2,3
        # 子写自己的副本，父不受影响：双向分叉完成
        mmu.write_bytes(child, 0x1000, b"child-v1!!!")
        self.assertEqual(mmu.read_bytes(child, 0x1000, 11), b"child-v1!!!")
        self.assertEqual(mmu.read_bytes(a, 0x1000, 11), b"parent-v2!!")
        # 解除映射只减引用：子 unmap 0x2000，帧 1 引用 2->1，帧不回收
        t = mmu.unmap(child, 0x2000)
        settle(mmu, t)
        self.assertIsNone(mmu.walk_table(child, 0x2000))
        self.assertEqual(mmu.mem.frame_refs(1), 1)
        self.assertEqual(mmu.mem.used_frames(), 3)
        self.assertEqual(mmu.walk_table(a, 0x2000).ppn, 1)  # 父侧映射保持
        # 父也 unmap：最后引用释放，帧 1 回收
        t = mmu.unmap(a, 0x2000)
        settle(mmu, t)
        self.assertEqual(mmu.mem.frame_refs(1), 0)
        self.assertEqual(mmu.mem.used_frames(), 2)
        # 销毁两个空间后全部帧回收
        mmu.destroy_address_space(a)
        mmu.destroy_address_space(child)
        self.assertEqual(mmu.mem.used_frames(), 0)


class TestCloneStaleWindow(unittest.TestCase):
    """手算场景 B：克隆票据 ACK 前，父空间陈旧可写 TLB 项仍允许命中。"""

    def test_stale_writable_entry_until_clone_ticket_ack(self):
        mmu = MMU()
        a = mmu.create_address_space()
        t = mmu.map(a, 0x1000, PERM_RW)
        settle(mmu, t)
        mmu.translate(a, 0, 0x1000, AccessType.READ)   # 核 0 预热 TLB
        res = mmu.clone_address_space(a)
        # ACK 前：核 0 旧项仍命中（陈旧窗口，与既有失效语义一致）
        stale = mmu.translate(a, 0, 0x1000, AccessType.WRITE)
        self.assertTrue(stale.ok)
        self.assertEqual(stale.via, "tlb")
        # 核 1 无旧项：走页表 -> cow fault
        r1 = mmu.translate(a, 1, 0x1000, AccessType.WRITE)
        self.assertEqual((r1.ok, r1.fault, r1.via), (False, "cow", "walk"))
        # 内容写入路径不经过 TLB：即使陈旧窗口内，write_bytes 仍拒绝写共享页
        with self.assertRaises(MemoryAccessFault) as cm:
            mmu.write_bytes(a, 0x1000, b"x")
        self.assertEqual(cm.exception.fault, "cow")
        # finish 必须等双核 ACK
        with self.assertRaises(InvalidationsPending):
            mmu.finish(res.ticket.id)
        mmu.ack(0, res.ticket.id)
        mmu.ack(1, res.ticket.id)
        mmu.finish(res.ticket.id)
        self.assertNotIn((a, 0x10), mmu.tlb_keys(0))
        # ACK 后父空间写 -> cow fault
        r0 = mmu.translate(a, 0, 0x1000, AccessType.WRITE)
        self.assertEqual((r0.ok, r0.fault), (False, "cow"))


class TestCloneHugeDeepCopy(unittest.TestCase):
    """手算场景 C：大页不参与 COW 共享，克隆时深拷贝 16 帧内容。"""

    def test_huge_pages_are_deep_copied(self):
        mmu = MMU()
        a = mmu.create_address_space()
        t = mmu.map(a, 0x4000, PERM_RWX, huge=True)   # 帧 0..15
        settle(mmu, t)
        mmu.write_bytes(a, 0x4000, b"huge!")
        res = mmu.clone_address_space(a)
        child = res.child_asid
        settle(mmu, res.ticket)
        # 手算：子大页取第一个全空 16 对齐块 = 帧 16..31；父帧引用不变
        self.assertEqual(mmu.mem.used_frames(), 32)
        self.assertEqual(mmu.mem.frame_refs(0), 1)
        self.assertEqual(mmu.mem.frame_refs(16), 1)
        wp = mmu.walk_table(a, 0x4000)
        wc = mmu.walk_table(child, 0x4000)
        self.assertEqual((wp.ppn, wp.huge, wp.cow), (0, True, False))
        self.assertEqual((wc.ppn, wc.huge, wc.cow), (16, True, False))
        # 内容已复制；之后父写不影响子（深拷贝，非共享）
        self.assertEqual(mmu.read_bytes(child, 0x4000, 5), b"huge!")
        mmu.write_bytes(a, 0x4000, b"PARNT")
        mmu.write_bytes(a, 0x4F00, b"end!")           # 父帧 15
        self.assertEqual(mmu.read_bytes(a, 0x4000, 5), b"PARNT")
        self.assertEqual(mmu.read_bytes(child, 0x4000, 5), b"huge!")
        self.assertEqual(mmu.read_bytes(child, 0x4F00, 4), b"\x00\x00\x00\x00")
        # 父 unmap 大页释放 16 帧；子空间大页完好
        t = mmu.unmap(a, 0x4000)
        settle(mmu, t)
        self.assertEqual(mmu.mem.used_frames(), 16)
        self.assertEqual(mmu.read_bytes(child, 0x4000, 5), b"huge!")


class TestWriteFaultFailureKeepsMapping(unittest.TestCase):
    """手算场景 D：帧池耗尽时写故障失败，原映射完全保持。"""

    def test_frame_exhausted_keeps_original_mapping(self):
        mmu = MMU(PhysicalMemory(num_frames=3))
        a = mmu.create_address_space()
        t1 = mmu.map(a, 0x1000, PERM_RW)   # 帧 0
        t2 = mmu.map(a, 0x2000, PERM_RW)   # 帧 1
        settle(mmu, t1, t2)
        mmu.write_bytes(a, 0x1000, b"DATA")
        res = mmu.clone_address_space(a)   # 不分配帧；帧 0/1 引用=2
        child = res.child_asid
        settle(mmu, res.ticket)
        t3 = mmu.map(child, 0x3000, PERM_RW)  # 帧 2 -> 池满
        settle(mmu, t3)
        self.assertEqual(mmu.mem.used_frames(), 3)
        # 写故障失败：FrameExhausted，且原映射保持
        with self.assertRaises(FrameExhausted):
            mmu.handle_write_fault(a, 0x1000)
        w = mmu.walk_table(a, 0x1000)
        self.assertEqual((w.ppn, w.cow), (0, True))       # PTE 未换、COW 未清
        self.assertEqual(mmu.mem.frame_refs(0), 2)        # 引用计数未动
        self.assertEqual(mmu.mem.used_frames(), 3)
        self.assertEqual(mmu.read_bytes(a, 0x1000, 4), b"DATA")  # 内容未动
        self.assertEqual(mmu.walk_table(child, 0x1000).ppn, 0)   # 子侧未动
        # 腾出一帧后重试成功：手算 -> 新帧即刚释放的帧 2，内容复制
        t = mmu.unmap(child, 0x3000)
        settle(mmu, t)
        tp = mmu.handle_write_fault(a, 0x1000)
        settle(mmu, tp)
        self.assertEqual(mmu.walk_table(a, 0x1000).ppn, 2)
        self.assertEqual(mmu.mem.frame_refs(0), 1)
        self.assertEqual(mmu.read_bytes(a, 0x1000, 4), b"DATA")


class TestCloneFailureRollsBack(unittest.TestCase):
    """手算场景 E：克隆中途帧耗尽 -> 完整回滚，父空间零变化、不留子空间。"""

    def test_clone_frame_exhausted_rolls_back(self):
        # 64 帧：小页 f0；大页1 f16..31；大页2 f32..47（占用 33 帧）。
        # 克隆时子大页1 拷贝到 f48..63 成功，子大页2 无块可分 -> FrameExhausted。
        mmu = MMU(PhysicalMemory(num_frames=64))
        a = mmu.create_address_space()
        t1 = mmu.map(a, 0x0000, PERM_RW)
        t2 = mmu.map(a, 0x1000, PERM_RW, huge=True)
        t3 = mmu.map(a, 0x2000, PERM_RW, huge=True)
        settle(mmu, t1, t2, t3)
        self.assertEqual(mmu.mem.used_frames(), 33)
        with self.assertRaises(FrameExhausted):
            mmu.clone_address_space(a)
        # 回滚核对：占用、引用计数、父空间 COW 标记、票据、子空间
        self.assertEqual(mmu.mem.used_frames(), 33)
        self.assertEqual(mmu.mem.frame_refs(0), 1)
        self.assertEqual(mmu.mem.frame_refs(48), 0)   # 已分配的子大页1被退回
        self.assertFalse(mmu.walk_table(a, 0x0000).cow)
        self.assertEqual(mmu.pending_tickets(), [])
        with self.assertRaises(InvalidAddressSpace):
            mmu.walk_table(1, 0x0000)                 # 子空间未安装
        self.assertEqual(len(mmu.mappings(a)), 3)     # 父映射保持
        # 父空间仍可正常分裂写故障之外的克隆前置条件：腾出帧后克隆可成功
        t = mmu.unmap(a, 0x2000)
        settle(mmu, t)
        res = mmu.clone_address_space(a)              # 现在只需 16 帧拷贝
        self.assertIn(res.child_asid, (1, 2, 3))


class TestCloneChain(unittest.TestCase):
    """手算场景 F：链式克隆（祖->父->子）的引用计数。"""

    def test_grandchild_chain_refcounts(self):
        mmu = MMU()
        a = mmu.create_address_space()
        t = mmu.map(a, 0x1000, PERM_RW)   # 帧 0
        settle(mmu, t)
        b = mmu.clone_address_space(a).child_asid     # refs f0 = 2
        c = mmu.clone_address_space(b).child_asid     # refs f0 = 3
        settle_all(mmu)
        self.assertEqual(mmu.mem.frame_refs(0), 3)
        # 父分裂：b 得新帧 1；f0 引用 3->2
        t = mmu.handle_write_fault(b, 0x1000)
        settle(mmu, t)
        self.assertEqual(mmu.walk_table(b, 0x1000).ppn, 1)
        self.assertEqual(mmu.mem.frame_refs(0), 2)
        # 祖 unmap：f0 引用 2->1（只剩 c）
        t = mmu.unmap(a, 0x1000)
        settle(mmu, t)
        self.assertEqual(mmu.mem.frame_refs(0), 1)
        self.assertEqual(mmu.mem.used_frames(), 2)
        # 销毁 c：最后引用释放
        mmu.destroy_address_space(c)
        self.assertEqual(mmu.mem.frame_refs(0), 0)
        self.assertEqual(mmu.mem.used_frames(), 1)
        mmu.destroy_address_space(b)
        self.assertEqual(mmu.mem.used_frames(), 0)


class TestCowErrorCases(unittest.TestCase):
    """非法输入与非法目标的明确错误。"""

    def setUp(self):
        self.mmu = MMU()
        self.a = self.mmu.create_address_space()

    def test_write_fault_targets(self):
        mmu, a = self.mmu, self.a
        with self.assertRaises(NotMapped):
            mmu.handle_write_fault(a, 0x5000)          # 未映射
        t = mmu.map(a, 0x6000, PERM_RW)
        settle(mmu, t)
        with self.assertRaises(InvalidWriteFault):
            mmu.handle_write_fault(a, 0x6000)          # 私有页，非 COW
        th = mmu.map(a, 0x8000, PERM_RWX, huge=True)
        settle(mmu, th)
        with self.assertRaises(InvalidWriteFault):
            mmu.handle_write_fault(a, 0x8000)          # 大页不参与 COW
        with self.assertRaises(ValueError):
            mmu.handle_write_fault(a, 0x6001)          # 未对齐
        with self.assertRaises(TypeError):
            mmu.handle_write_fault(True, 0x6000)       # bool asid
        with self.assertRaises(InvalidAddressSpace):
            mmu.handle_write_fault(3, 0x6000)          # 不存在的空间

    def test_clone_errors(self):
        mmu = self.mmu
        with self.assertRaises(InvalidAddressSpace):
            mmu.clone_address_space(9)
        with self.assertRaises(TypeError):
            mmu.clone_address_space(True)
        for _ in range(3):
            mmu.create_address_space()                 # 4 个槽位占满
        with self.assertRaises(AddressSpaceLimit):
            mmu.clone_address_space(self.a)

    def test_byte_access_errors(self):
        mmu, a = self.mmu, self.a
        t = mmu.map(a, 0x1000, PERM_RW)
        tr = mmu.map(a, 0x2000, PERM_READ)
        settle(mmu, t, tr)
        with self.assertRaises(MemoryAccessFault) as cm:
            mmu.write_bytes(a, 0x3000, b"x")           # 未映射
        self.assertEqual(cm.exception.fault, "not_mapped")
        with self.assertRaises(MemoryAccessFault) as cm:
            mmu.write_bytes(a, 0x2000, b"x")           # 只读页
        self.assertEqual(cm.exception.fault, "write")
        with self.assertRaises(MemoryAccessFault) as cm:
            mmu.read_bytes(a, 0x3000, 1)
        self.assertEqual(cm.exception.fault, "not_mapped")
        with self.assertRaises(CrossPageAccess):
            mmu.read_bytes(a, 0x10FA, 16)              # 跨小页边界
        with self.assertRaises(CrossPageAccess):
            mmu.write_bytes(a, 0x10FA, b"0123456789abcdef")
        with self.assertRaises(TypeError):
            mmu.write_bytes(a, 0x1000, "not-bytes")
        with self.assertRaises(ValueError):
            mmu.write_bytes(a, 0x1000, b"")
        with self.assertRaises(TypeError):
            mmu.read_bytes(a, 0x1000, True)

    def test_cow_page_readable_not_writable(self):
        mmu, a = self.mmu, self.a
        t = mmu.map(a, 0x1000, PERM_RW)
        settle(mmu, t)
        mmu.write_bytes(a, 0x1000, b"AB")
        res = mmu.clone_address_space(a)
        settle(mmu, res.ticket)
        self.assertEqual(mmu.read_bytes(a, 0x1000, 2), b"AB")   # COW 页可读
        with self.assertRaises(MemoryAccessFault) as cm:
            mmu.write_bytes(a, 0x1000, b"CD")
        self.assertEqual(cm.exception.fault, "cow")
        with self.assertRaises(MemoryAccessFault) as cm:
            mmu.write_bytes(res.child_asid, 0x1000, b"CD")
        self.assertEqual(cm.exception.fault, "cow")


class TestCowProtectUnmapInterplay(unittest.TestCase):
    """protect / unmap 与 COW 标记的交互。"""

    def test_protect_keeps_cow_then_split(self):
        mmu = MMU()
        a = mmu.create_address_space()
        t = mmu.map(a, 0x1000, PERM_RW)   # 帧 0
        settle(mmu, t)
        res = mmu.clone_address_space(a)
        child = res.child_asid
        settle(mmu, res.ticket)
        tp = mmu.protect(a, 0x1000, PERM_READ)   # 改权限不清 COW 标记
        settle(mmu, tp)
        w = mmu.walk_table(a, 0x1000)
        self.assertEqual((w.perms, w.cow), (PERM_READ, True))
        # 权限不足先于 COW 判定：报 write 而非 cow
        r = mmu.translate(a, 0, 0x1000, AccessType.WRITE)
        self.assertEqual(r.fault, "write")
        # 仍可显式分裂（COW 标记仍在）：手算 -> 新帧 1，帧 0 引用 2->1
        tf = mmu.handle_write_fault(a, 0x1000)
        settle(mmu, tf)
        w = mmu.walk_table(a, 0x1000)
        self.assertEqual((w.ppn, w.perms, w.cow), (1, PERM_READ, False))
        self.assertEqual(mmu.mem.frame_refs(0), 1)
        # 子 unmap 释放最后引用
        t = mmu.unmap(child, 0x1000)
        settle(mmu, t)
        self.assertEqual(mmu.mem.frame_refs(0), 0)


# ===========================================================================
# 差分测试：独立参考机 RefCOW vs 被测 MMU
# ===========================================================================
NUM_FRAMES = 256 * 4 + 64
PERMS_CHOICES = (PERM_READ, PERM_WRITE, PERM_RW, PERM_RWX)
ACCESSES = (AccessType.READ, AccessType.WRITE, AccessType.EXECUTE)
NEED_OF = {AccessType.READ: PERM_READ, AccessType.WRITE: PERM_WRITE,
           AccessType.EXECUTE: 4}


class RefCOW:
    """按契约独立重写的参考机：引用计数帧池 + 内容字典 + 页表 + LRU TLB。

    不调用 vmshoot 的任何核心算法；仅复用题面常量与异常类型。
    页表表示：l1 项 = None / ("H", base, perms) / [16 项 None|(ppn, perms, cow)]。
    """

    def __init__(self):
        self.refs = [0] * NUM_FRAMES
        self.content: list[bytearray | None] = [None] * NUM_FRAMES
        self.spaces: dict[int, list] = {}
        self.tlb: list[OrderedDict] = [OrderedDict() for _ in range(2)]
        self.tickets: dict[int, dict] = {}
        self.next_id = 1

    # ---- 帧 ----------------------------------------------------------------
    def _alloc(self) -> int:
        for i in range(NUM_FRAMES):
            if self.refs[i] == 0:
                self.refs[i] = 1
                return i
        raise FrameExhausted("ref")

    def _alloc_huge(self) -> int:
        for s in range(0, NUM_FRAMES - 15, 16):
            if all(self.refs[s + j] == 0 for j in range(16)):
                for j in range(16):
                    self.refs[s + j] = 1
                return s
        raise FrameExhausted("ref huge")

    def _release(self, p: int) -> None:
        assert self.refs[p] > 0
        self.refs[p] -= 1
        if self.refs[p] == 0:
            self.content[p] = None

    def _buf(self, p: int) -> bytearray:
        b = self.content[p]
        if b is None:
            b = bytearray(256)
            self.content[p] = b
        return b

    # ---- ASID ---------------------------------------------------------------
    def _slot(self) -> int:
        if len(self.spaces) >= 4:
            raise AddressSpaceLimit("ref")
        for c in range(4):
            if c in self.spaces:
                continue
            if any(t["asid"] == c for t in self.tickets.values()):
                continue
            return c
        from vmshoot import AsidBusy
        raise AsidBusy(0, [])

    def create(self) -> int:
        c = self._slot()
        self.spaces[c] = [None] * 16
        return c

    def destroy(self, asid) -> None:
        for e in self.spaces[asid]:
            if isinstance(e, tuple):
                for j in range(16):
                    self._release(e[1] + j)
            elif isinstance(e, list):
                for pte in e:
                    if pte is not None:
                        self._release(pte[0])
        del self.spaces[asid]
        for tlb in self.tlb:
            for k in [k for k in tlb if k[0] == asid]:
                del tlb[k]

    def clone(self, asid) -> tuple[int, int]:
        l1 = self.spaces[asid]
        child = self._slot()
        new_l1 = [None] * 16
        allocated, retained = [], []
        try:
            for i, e in enumerate(l1):
                if isinstance(e, tuple):
                    nb = self._alloc_huge()
                    allocated.append(nb)
                    for j in range(16):
                        self._buf(nb + j)[:] = self._buf(e[1] + j)
                    new_l1[i] = ("H", nb, e[2])
                elif isinstance(e, list):
                    t = [None] * 16
                    for j, pte in enumerate(e):
                        if pte is not None:
                            self.refs[pte[0]] += 1
                            retained.append(pte[0])
                            t[j] = (pte[0], pte[1], True)
                    new_l1[i] = t
        except FrameExhausted:
            for b in allocated:
                for j in range(16):
                    self._release(b + j)
            for p in retained:
                self._release(p)
            raise
        keys = []
        for i, e in enumerate(l1):
            if isinstance(e, list):
                for j, pte in enumerate(e):
                    if pte is not None:
                        e[j] = (pte[0], pte[1], True)
                        keys.append((asid, i * 16 + j))
        self.spaces[child] = new_l1
        return child, self._issue(asid, "clone", 0, tuple(keys))

    # ---- 遍历 ---------------------------------------------------------------
    def walk(self, asid, va):
        e = self.spaces[asid][(va >> 12) & 0xF]
        if e is None:
            return None
        if isinstance(e, tuple):
            return (e[1], e[2], True, False)
        pte = e[(va >> 8) & 0xF]
        return None if pte is None else (pte[0], pte[1], False, pte[2])

    # ---- 结构操作 -----------------------------------------------------------
    def _issue(self, asid, kind, va, keys) -> int:
        tid = self.next_id
        self.next_id += 1
        self.tickets[tid] = {"asid": asid, "kind": kind, "keys": tuple(keys),
                             "acks": set()}
        return tid

    @staticmethod
    def _keys(asid, va, huge):
        if huge:
            return tuple((asid, ((va & ~0xFFF) >> 8) + j) for j in range(16))
        return ((asid, va >> 8),)

    def map(self, asid, va, perms, huge) -> int:
        l1i, l2i = (va >> 12) & 0xF, (va >> 8) & 0xF
        l1 = self.spaces[asid]
        if huge:
            if va & 0xFFF:
                raise ValueError("align")
            if l1[l1i] is not None:
                from vmshoot import MappingExists
                raise MappingExists("ref")
            base = self._alloc_huge()
            l1[l1i] = ("H", base, perms)
            return self._issue(asid, "map", va, self._keys(asid, va, True))
        if va & 0xFF:
            raise ValueError("align")
        e = l1[l1i]
        if isinstance(e, tuple):
            from vmshoot import MappingExists
            raise MappingExists("ref")
        if e is not None and e[l2i] is not None:
            from vmshoot import MappingExists
            raise MappingExists("ref")
        ppn = self._alloc()
        if e is None:
            e = [None] * 16
            l1[l1i] = e
        e[l2i] = (ppn, perms, False)
        return self._issue(asid, "map", va, self._keys(asid, va, False))

    def unmap(self, asid, va) -> int:
        if va & 0xFF:
            raise ValueError("align")
        l1i, l2i = (va >> 12) & 0xF, (va >> 8) & 0xF
        e = self.spaces[asid][l1i]
        if isinstance(e, tuple):
            if l2i != 0:
                from vmshoot import MappingConflict
                raise MappingConflict("ref")
            for j in range(16):
                self._release(e[1] + j)
            self.spaces[asid][l1i] = None
            return self._issue(asid, "unmap", va, self._keys(asid, va, True))
        if e is None or e[l2i] is None:
            raise NotMapped("ref")
        self._release(e[l2i][0])
        e[l2i] = None
        if all(x is None for x in e):
            self.spaces[asid][l1i] = None
        return self._issue(asid, "unmap", va, self._keys(asid, va, False))

    def protect(self, asid, va, perms) -> int:
        if va & 0xFF:
            raise ValueError("align")
        l1i, l2i = (va >> 12) & 0xF, (va >> 8) & 0xF
        e = self.spaces[asid][l1i]
        if isinstance(e, tuple):
            table = [(e[1] + j, e[2], False) for j in range(16)]
            table[l2i] = (e[1] + l2i, perms, False)
            self.spaces[asid][l1i] = table
            return self._issue(asid, "protect", va, self._keys(asid, va & ~0xFFF, True))
        if e is None or e[l2i] is None:
            raise NotMapped("ref")
        e[l2i] = (e[l2i][0], perms, e[l2i][2])
        return self._issue(asid, "protect", va, self._keys(asid, va, False))

    def write_fault(self, asid, va) -> int:
        if va & 0xFF:
            raise ValueError("align")
        l1i, l2i = (va >> 12) & 0xF, (va >> 8) & 0xF
        e = self.spaces[asid][l1i]
        if isinstance(e, tuple):
            raise InvalidWriteFault("ref huge")
        if e is None or e[l2i] is None:
            raise NotMapped("ref")
        ppn, perms, cow = e[l2i]
        if not cow:
            raise InvalidWriteFault("ref not cow")
        np = self._alloc()                     # 失败则上面状态不变
        self._buf(np)[:] = self._buf(ppn)
        self._release(ppn)
        e[l2i] = (np, perms, False)
        return self._issue(asid, "cow", va, ((asid, va >> 8),))

    # ---- 票据 ---------------------------------------------------------------
    def ack(self, core, tid) -> None:
        t = self.tickets.get(tid)
        if t is None:
            from vmshoot import TicketError
            raise TicketError("ref")
        if core in t["acks"]:
            from vmshoot import TicketError
            raise TicketError("ref dup")
        for k in t["keys"]:
            self.tlb[core].pop(k, None)
        t["acks"].add(core)

    def finish(self, tid) -> None:
        t = self.tickets.get(tid)
        if t is None:
            from vmshoot import TicketError
            raise TicketError("ref")
        if t["acks"] != {0, 1}:
            raise InvalidationsPending(
                tid, tuple(c for c in (0, 1) if c not in t["acks"]))
        del self.tickets[tid]

    # ---- 翻译与字节访问 -------------------------------------------------------
    def translate(self, asid, core, va, need_bit, length):
        if length <= 0 or length > 4096:
            raise ValueError("ref length")
        if va + length - 1 > 0xFFFF:
            raise ValueError("ref range")
        vpn = va >> 8
        tlb = self.tlb[core]
        entry, via, cow = None, "walk", False
        k = (asid, vpn)
        if k in tlb:
            tlb.move_to_end(k)
            entry, via = tlb[k], "tlb"
        else:
            hk = (asid, vpn & ~15)
            if hk in tlb and tlb[hk][2]:
                tlb.move_to_end(hk)
                entry, via = tlb[hk], "tlb"
        if entry is None:
            w = self.walk(asid, va)
            if w is None:
                return (False, None, "not_mapped", "walk")
            entry = (w[0], w[1], w[2])
            cow = w[3]
            if not cow:
                key = (asid, vpn if not w[2] else vpn & ~15)
                if key in tlb:
                    tlb.move_to_end(key)
                else:
                    tlb[key] = entry
                    if len(tlb) > 8:
                        tlb.popitem(last=False)
        ppn, perms, huge = entry
        region = 4096 if huge else 256
        if (va & (region - 1)) + length > region:
            raise CrossPageAccess("ref cross")
        if not (perms & need_bit):
            fault = {1: "read", 2: "write", 4: "execute"}[need_bit]
            return (False, None, fault, via)
        if cow and need_bit == 2:
            return (False, None, "cow", via)
        off = va & (0xFFF if huge else 0xFF)
        return (True, ppn * 256 + off, None, via)

    def _resolve(self, asid, va, n, need_bit):
        w = self.walk(asid, va)
        if w is None:
            raise MemoryAccessFault("not_mapped", asid, va)
        region = 4096 if w[2] else 256
        if (va & (region - 1)) + n > region:
            raise CrossPageAccess("ref cross")
        if not (w[1] & need_bit):
            raise MemoryAccessFault("read" if need_bit == 1 else "write", asid, va)
        return w

    def read_bytes(self, asid, va, n):
        w = self._resolve(asid, va, n, 1)
        base = w[0] * 256 + (va & (0xFFF if w[2] else 0xFF))
        out = bytearray()
        while len(out) < n:
            cur = base + len(out)
            f, off = divmod(cur, 256)
            take = min(n - len(out), 256 - off)
            out += self._buf(f)[off:off + take]
        return bytes(out)

    def write_bytes(self, asid, va, data):
        w = self._resolve(asid, va, len(data), 2)
        if w[3]:
            raise MemoryAccessFault("cow", asid, va)
        base = w[0] * 256 + (va & (0xFFF if w[2] else 0xFF))
        done = 0
        while done < len(data):
            cur = base + done
            f, off = divmod(cur, 256)
            take = min(len(data) - done, 256 - off)
            self._buf(f)[off:off + take] = data[done:done + take]
            done += take
        return len(data)


class TestCowDifferential(unittest.TestCase):
    """固定种子随机操作序列：被测 MMU 与 RefCOW 逐步核对。"""

    def test_seeded_sequences(self):
        for seed in (20260924, 7, 31337):
            with self.subTest(seed=seed):
                self._run_one(seed, 300)

    # ---- 比较器 --------------------------------------------------------------
    def _cmp_refs(self, sut: MMU, ref: RefCOW):
        for i in range(NUM_FRAMES):
            self.assertEqual(sut.mem.frame_refs(i), ref.refs[i], f"frame {i}")
        self.assertEqual(sut.mem.used_frames(), sum(1 for r in ref.refs if r))

    def _cmp_walks(self, sut: MMU, ref: RefCOW):
        self.assertEqual(sorted(sut._spaces), sorted(ref.spaces))
        for asid in sorted(ref.spaces):
            for vpn in range(256):
                va = vpn * 256 + 7
                rw = ref.walk(asid, va)
                sw = sut.walk_table(asid, va)
                if rw is None:
                    self.assertIsNone(sw, f"asid={asid} va=0x{va:x}")
                else:
                    self.assertEqual((sw.ppn, sw.perms, sw.huge, sw.cow), rw,
                                     f"asid={asid} va=0x{va:x}")

    def _cmp_content(self, sut: MMU, ref: RefCOW):
        for asid in sorted(ref.spaces):
            for vpn in range(256):
                va = vpn * 256
                if ref.walk(asid, va) is None:
                    continue
                try:
                    rb = ref.read_bytes(asid, va, 8)
                except MemoryAccessFault as e:
                    with self.assertRaises(MemoryAccessFault) as cm:
                        sut.read_bytes(asid, va, 8)
                    self.assertEqual(cm.exception.fault, e.fault)
                    continue
                self.assertEqual(sut.read_bytes(asid, va, 8), rb,
                                 f"content asid={asid} va=0x{va:x}")

    def _cmp_ticket(self, fn, sut, ref):
        try:
            rid = fn(ref)
        except Exception as exc:
            with self.assertRaises(type(exc)):
                fn(sut)
            return
        st = fn(sut)
        rt = ref.tickets[rid]
        self.assertEqual(st.id, rid)
        self.assertEqual(st.keys, rt["keys"])
        self.assertEqual(st.kind, rt["kind"])

    def _cmp_clone(self, sut, ref, asid):
        try:
            rchild, rid = ref.clone(asid)
        except Exception as exc:
            with self.assertRaises(type(exc)):
                sut.clone_address_space(asid)
            return
        res = sut.clone_address_space(asid)
        self.assertEqual(res.child_asid, rchild)
        self.assertEqual(res.ticket.id, rid)
        self.assertEqual(res.ticket.keys, ref.tickets[rid]["keys"])
        self.assertEqual(res.ticket.kind, "clone")

    def _cmp_translate(self, sut, ref, asid, core, va, acc, length):
        try:
            rv = ref.translate(asid, core, va, NEED_OF[acc], length)
        except Exception as exc:
            with self.assertRaises(type(exc)):
                sut.translate(asid, core, va, acc, length)
            return
        sr = sut.translate(asid, core, va, acc, length)
        self.assertEqual((sr.ok, sr.phys_addr, sr.fault, sr.via), rv,
                         f"asid={asid} core={core} va=0x{va:x} {acc.name} len={length}")

    def _cmp_read(self, sut, ref, asid, va, n):
        try:
            rb = ref.read_bytes(asid, va, n)
        except MemoryAccessFault as exc:
            with self.assertRaises(MemoryAccessFault) as cm:
                sut.read_bytes(asid, va, n)
            self.assertEqual(cm.exception.fault, exc.fault)
            return
        except Exception as exc:
            with self.assertRaises(type(exc)):
                sut.read_bytes(asid, va, n)
            return
        self.assertEqual(sut.read_bytes(asid, va, n), rb)

    def _cmp_write(self, sut, ref, asid, va, data):
        try:
            rn = ref.write_bytes(asid, va, data)
        except MemoryAccessFault as exc:
            with self.assertRaises(MemoryAccessFault) as cm:
                sut.write_bytes(asid, va, data)
            self.assertEqual(cm.exception.fault, exc.fault)
            return
        except Exception as exc:
            with self.assertRaises(type(exc)):
                sut.write_bytes(asid, va, data)
            return
        self.assertEqual(sut.write_bytes(asid, va, data), rn)

    # ---- 驱动 -----------------------------------------------------------------
    def _run_one(self, seed: int, steps: int):
        rng = random.Random(seed)
        sut = MMU(PhysicalMemory(NUM_FRAMES))
        ref = RefCOW()
        for step in range(steps):
            r = rng.random()
            active = sorted(ref.spaces)
            with self.subTest(step=step):
                if r < 0.05:
                    try:
                        ra = ref.create()
                    except Exception as exc:
                        with self.assertRaises(type(exc)):
                            sut.create_address_space()
                    else:
                        self.assertEqual(sut.create_address_space(), ra)
                elif r < 0.10 and active:
                    a = rng.choice(active)
                    ref.destroy(a)
                    sut.destroy_address_space(a)
                elif r < 0.17 and active:
                    self._cmp_clone(sut, ref, rng.choice(active))
                elif r < 0.40 and active:
                    a = rng.choice(active)
                    huge = rng.random() < 0.3
                    va = (rng.randrange(16) * 4096 if huge
                          else rng.randrange(256) * 256)
                    p = rng.choice(PERMS_CHOICES)
                    self._cmp_ticket(
                        lambda m, a=a, va=va, p=p, h=huge:
                            m.map(a, va, p, h) if isinstance(m, RefCOW)
                            else m.map(a, va, p, huge=h), sut, ref)
                elif r < 0.50 and active:
                    a = rng.choice(active)
                    va = rng.randrange(256) * 256
                    self._cmp_ticket(lambda m, a=a, va=va: m.unmap(a, va), sut, ref)
                elif r < 0.58 and active:
                    a = rng.choice(active)
                    va = rng.randrange(256) * 256
                    p = rng.choice(PERMS_CHOICES)
                    self._cmp_ticket(
                        lambda m, a=a, va=va, p=p: m.protect(a, va, p), sut, ref)
                elif r < 0.66 and active:
                    a = rng.choice(active)
                    va = rng.randrange(256) * 256
                    self._cmp_ticket(
                        lambda m, a=a, va=va:
                            m.write_fault(a, va) if isinstance(m, RefCOW)
                            else m.handle_write_fault(a, va), sut, ref)
                elif r < 0.74 and active:
                    a = rng.choice(active)
                    va = rng.randrange(0x10000)
                    n = rng.randrange(1, 9)
                    data = bytes(rng.randrange(256) for _ in range(n))
                    self._cmp_write(sut, ref, a, va, data)
                elif r < 0.80 and active:
                    a = rng.choice(active)
                    va = rng.randrange(0x10000)
                    self._cmp_read(sut, ref, a, va, rng.randrange(1, 9))
                elif r < 0.92 and active:
                    a = rng.choice(active)
                    self._cmp_translate(sut, ref, a, rng.randrange(2),
                                        rng.randrange(0x10000),
                                        rng.choice(ACCESSES),
                                        rng.choice([1, 1, 1, 4, 32]))
                elif ref.tickets:
                    tid = rng.choice(sorted(ref.tickets))
                    if rng.random() < 0.6:
                        c = rng.randrange(2)
                        try:
                            ref.ack(c, tid)
                        except Exception as exc:
                            with self.assertRaises(type(exc)):
                                sut.ack(c, tid)
                        else:
                            sut.ack(c, tid)
                    else:
                        try:
                            ref.finish(tid)
                        except Exception as exc:
                            with self.assertRaises(type(exc)):
                                sut.finish(tid)
                        else:
                            sut.finish(tid)
                else:
                    self._cmp_walks(sut, ref)
                self._cmp_refs(sut, ref)
                if step % 10 == 9:
                    self._cmp_walks(sut, ref)
                    self._cmp_content(sut, ref)
        # 收尾：结清全部票据后做最终全量核对
        for tid in sorted(ref.tickets):
            for c in (0, 1):
                if c not in ref.tickets[tid]["acks"]:
                    ref.ack(c, tid)
                    sut.ack(c, tid)
            ref.finish(tid)
            sut.finish(tid)
        self.assertEqual(sut.pending_tickets(), [])
        self._cmp_walks(sut, ref)
        self._cmp_content(sut, ref)
        self._cmp_refs(sut, ref)
        for core in range(2):
            self.assertEqual(list(sut._tlb[core].keys()), list(ref.tlb[core].keys()))
            for k in ref.tlb[core]:
                self.assertEqual(sut._tlb[core][k], ref.tlb[core][k])


if __name__ == "__main__":
    unittest.main()
