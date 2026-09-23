"""地址空间克隆与写时复制（COW）测试。

所有物理帧号、引用计数均为手算/解析期望（空池首位适配：第 k 个新帧取最小空闲
号），不读取被测实现内部算法来生成期望值；只通过公开接口与 walk_table /
frame_refs 诊断核对。覆盖：

* 克隆共享普通页、不增帧、引用计数 +1；
* 父子双写各自分裂、互不影响，最后一个引用释放时帧才回收；
* 只读页不进入 COW；大页克隆在父子两侧就地展开为 16 小页；
* 写故障 / 解除映射的显式串行票据（ACK 前陈旧、ACK 后生效）；
* 失败原子性：帧耗尽时 handle_write_fault 抛 FrameExhausted 且原映射保持；
* clone / handle_write_fault 的非法输入与不留半更新。
"""

from __future__ import annotations

import unittest

from vmshoot import (
    MMU,
    AccessType,
    AddressSpaceLimit,
    FrameExhausted,
    InvalidAddressSpace,
    NotCopyOnWrite,
    NotMapped,
    PAGE_SIZE,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PhysicalMemory,
)


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        mmu.ack(0, t.id)
        mmu.ack(1, t.id)
        mmu.finish(t.id)


class TestCloneSharesFrames(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.parent = self.mmu.create_address_space()

    def test_clone_shares_normal_page_no_new_frame(self):
        # 手算：空池首个小页 -> ppn 0
        t = self.mmu.map(self.parent, 0x1000, PERM_RW)
        settle(self.mmu, t)
        used_before = self.mmu.mem.used_frames()
        self.assertEqual(used_before, 1)

        child, tc = self.mmu.clone_address_space(self.parent)
        # 克隆不分配新物理帧
        self.assertEqual(self.mmu.mem.used_frames(), used_before)
        # 共享帧引用计数 2（父 + 子）
        self.assertEqual(self.mmu.mem.frame_refs(0), 2)
        # 父子解析到同一物理帧，且双方都标记为 COW
        wp = self.mmu.walk_table(self.parent, 0x1000)
        wc = self.mmu.walk_table(child, 0x1000)
        self.assertEqual((wp.ppn, wp.cow), (0, True))
        self.assertEqual((wc.ppn, wc.cow), (0, True))
        self.assertFalse(wp.huge or wc.huge)
        settle(self.mmu, tc)

    def test_clone_multiple_distinct_pages_refcounts(self):
        # 三个独立小页 -> 手算 ppn 0,1,2
        for va in (0x0000, 0x0100, 0x0500):
            settle(self.mmu, self.mmu.map(self.parent, va, PERM_RW))
        child, tc = self.mmu.clone_address_space(self.parent)
        settle(self.mmu, tc)
        for ppn in (0, 1, 2):
            self.assertEqual(self.mmu.mem.frame_refs(ppn), 2)
        self.assertEqual(self.mmu.mem.used_frames(), 3)  # 仍是 3 个物理帧
        # 子空间映射布局与父一致（同虚址、同权限）
        self.assertEqual(self.mmu.mappings(child), self.mmu.mappings(self.parent))

    def test_clone_empty_space_emits_empty_ticket(self):
        child, tc = self.mmu.clone_address_space(self.parent)
        self.assertEqual(tc.keys, ())
        self.assertEqual(self.mmu.mem.used_frames(), 0)
        settle(self.mmu, tc)
        self.assertEqual(self.mmu.mappings(child), [])

    def test_clone_readonly_page_is_not_cow(self):
        # 只读页父子直接共享：没有任何一方能写，无需 COW
        t = self.mmu.map(self.parent, 0x2000, PERM_READ)
        settle(self.mmu, t)
        child, tc = self.mmu.clone_address_space(self.parent)
        settle(self.mmu, tc)
        self.assertEqual(self.mmu.mem.frame_refs(0), 2)   # 帧仍共享
        self.assertFalse(self.mmu.walk_table(self.parent, 0x2000).cow)
        self.assertFalse(self.mmu.walk_table(child, 0x2000).cow)
        # 对只读共享页写：普通写权限故障，而不是 COW 写故障
        rp = self.mmu.translate(self.parent, 0, 0x2000, AccessType.WRITE)
        rc = self.mmu.translate(child, 1, 0x2000, AccessType.WRITE)
        self.assertEqual((rp.fault, rp.cow), ("write", False))
        self.assertEqual((rc.fault, rc.cow), ("write", False))
        # 显式写故障处理也应拒绝（它本就不是 COW 页）
        with self.assertRaises(NotCopyOnWrite):
            self.mmu.handle_write_fault(child, 0x2000)


class TestParentChildDoubleWrite(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.parent = self.mmu.create_address_space()
        settle(self.mmu, self.mmu.map(self.parent, 0x3000, PERM_RW))  # ppn 0
        self.child, self.tc = self.mmu.clone_address_space(self.parent)
        settle(self.mmu, self.tc)

    def test_write_faults_before_split_for_both(self):
        for space, core in ((self.parent, 0), (self.child, 1)):
            r = self.mmu.translate(space, core, 0x3000, AccessType.WRITE)
            self.assertFalse(r.ok)
            self.assertEqual((r.fault, r.cow), ("write", True))
        # 读不受影响（共享内容）
        self.assertTrue(self.mmu.translate(self.parent, 0, 0x3000, AccessType.READ).ok)
        self.assertTrue(self.mmu.translate(self.child, 1, 0x3000, AccessType.READ).ok)

    def test_parent_split_then_child_split_independent(self):
        # 父分裂：帧 0 占用中，首位适配新帧 -> ppn 1
        ts1 = self.mmu.handle_write_fault(self.parent, 0x3000)
        settle(self.mmu, ts1)
        wp = self.mmu.walk_table(self.parent, 0x3000)
        wc = self.mmu.walk_table(self.child, 0x3000)
        self.assertEqual((wp.ppn, wp.cow), (1, False))
        self.assertEqual((wc.ppn, wc.cow), (0, True))   # 子仍共享旧帧
        # 手算引用计数：旧帧只剩子引用=1，新帧=1
        self.assertEqual(self.mmu.mem.frame_refs(0), 1)
        self.assertEqual(self.mmu.mem.frame_refs(1), 1)
        # 父写成功（私有帧），子仍须分裂
        self.assertTrue(self.mmu.translate(self.parent, 0, 0x3000, AccessType.WRITE).ok)
        self.assertTrue(self.mmu.translate(self.child, 1, 0x3000, AccessType.WRITE).cow)

        # 子分裂：帧 0 仍占（refs=1），帧 1 占，新帧 -> ppn 2
        ts2 = self.mmu.handle_write_fault(self.child, 0x3000)
        settle(self.mmu, ts2)
        wc2 = self.mmu.walk_table(self.child, 0x3000)
        self.assertEqual((wc2.ppn, wc2.cow), (2, False))
        # 旧帧 0 的最后一个引用释放 -> 真正回收
        self.assertEqual(self.mmu.mem.frame_refs(0), 0)
        self.assertEqual(self.mmu.mem.used_frames(), 2)  # 只剩帧 1、2
        # 回收帧可被首位适配重新取回
        settle(self.mmu, self.mmu.map(self.parent, 0x3100, PERM_RW))
        self.assertEqual(self.mmu.walk_table(self.parent, 0x3100).ppn, 0)

    def test_sibling_pages_independent_only_written_page_splits(self):
        # 单次克隆下，父有两页 0x3000(ppn0)、0x3200(ppn1)，只分裂其中一页
        mmu = MMU()
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x3000, PERM_RW))   # ppn 0
        settle(mmu, mmu.map(p, 0x3200, PERM_RW))   # ppn 1
        c, tck = mmu.clone_address_space(p)
        settle(mmu, tck)
        settle(mmu, mmu.handle_write_fault(p, 0x3000))  # 仅父的 0x3000 分裂 -> ppn 2
        # 0x3000：父私有 ppn2；子仍共享 ppn0
        self.assertEqual(mmu.walk_table(p, 0x3000).ppn, 2)
        self.assertEqual(mmu.walk_table(c, 0x3000).ppn, 0)
        # 0x3200：父子仍共享 ppn1（引用计数 2，未分裂）
        self.assertEqual(mmu.walk_table(p, 0x3200).ppn, 1)
        self.assertEqual(mmu.walk_table(c, 0x3200).ppn, 1)
        self.assertEqual(mmu.mem.frame_refs(1), 2)
        self.assertEqual(mmu.mem.frame_refs(0), 1)  # 只因子还引用
        self.assertTrue(mmu.walk_table(c, 0x3200).cow)


class TestLastReferenceRelease(unittest.TestCase):
    def test_unmap_then_destroy_reclaims_only_at_last_ref(self):
        mmu = MMU()
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x0000, PERM_RW))  # ppn 0
        c, tc = mmu.clone_address_space(p)
        settle(mmu, tc)
        self.assertEqual(mmu.mem.frame_refs(0), 2)

        # 父解除映射：引用 2 -> 1，帧不回收，子仍可翻译
        settle(mmu, mmu.unmap(p, 0x0000))
        self.assertEqual(mmu.mem.frame_refs(0), 1)
        self.assertIsNone(mmu.walk_table(p, 0x0000))
        self.assertEqual(mmu.walk_table(c, 0x0000).ppn, 0)

        # 子再解除映射：最后引用释放，帧回收
        settle(mmu, mmu.unmap(c, 0x0000))
        self.assertEqual(mmu.mem.frame_refs(0), 0)
        self.assertEqual(mmu.mem.used_frames(), 0)

    def test_destroy_parent_keeps_child_mapping_intact(self):
        mmu = MMU()
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x4000, PERM_RW))  # ppn 0
        c, tc = mmu.clone_address_space(p)
        settle(mmu, tc)
        mmu.destroy_address_space(p)
        # 父释放其引用：每共享帧 refs 2 -> 1，子映射完好
        self.assertEqual(mmu.mem.frame_refs(0), 1)
        self.assertEqual(mmu.walk_table(c, 0x4000).ppn, 0)
        self.assertTrue(mmu.walk_table(c, 0x4000).cow)
        # 子分裂仍可正常拿到私有新帧
        settle(mmu, mmu.handle_write_fault(c, 0x4000))
        self.assertEqual(mmu.walk_table(c, 0x4000).ppn, 1)
        self.assertEqual(mmu.mem.frame_refs(0), 0)  # 旧帧最后引用释放

    def test_three_way_clone_refcount_three(self):
        mmu = MMU()
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x1000, PERM_RW))  # ppn 0
        c1, t1 = mmu.clone_address_space(p)
        settle(mmu, t1)
        c2, t2 = mmu.clone_address_space(p)       # 父再克隆一个兄弟
        settle(mmu, t2)
        # 三个空间共享 ppn0 -> refs 3
        self.assertEqual(mmu.mem.frame_refs(0), 3)
        mmu.destroy_address_space(p)
        self.assertEqual(mmu.mem.frame_refs(0), 2)
        settle(mmu, mmu.handle_write_fault(c1, 0x1000))  # c1 分裂 -> ppn1
        self.assertEqual(mmu.mem.frame_refs(0), 1)       # 只剩 c2
        self.assertEqual(mmu.walk_table(c2, 0x1000).ppn, 0)
        mmu.destroy_address_space(c2)
        self.assertEqual(mmu.mem.frame_refs(0), 0)       # 最后引用释放


class TestCloneHugePageExpansion(unittest.TestCase):
    def test_huge_expanded_to_sixteen_shared_small_pages(self):
        mmu = MMU()
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x4000, PERM_RWX, huge=True))  # 空池首个 16 对齐块 ppn 0..15
        self.assertEqual(mmu.mem.used_frames(), 16)
        c, tc = mmu.clone_address_space(p)
        settle(mmu, tc)
        # 不增帧，16 帧全部引用计数 2
        self.assertEqual(mmu.mem.used_frames(), 16)
        for ppn in range(0, 16):
            self.assertEqual(mmu.mem.frame_refs(ppn), 2)
        # 手算：0x4300 = L1 区 4 内第 3 小页 -> ppn 0+3 = 3，父子都展开为小页 COW
        for space in (p, c):
            w = mmu.walk_table(space, 0x4300)
            self.assertFalse(w.huge)
            self.assertTrue(w.cow)
            self.assertEqual(w.ppn, 3)
            self.assertEqual(w.phys_addr, 3 * PAGE_SIZE)
        # 父只写第 3 页 -> 仅该页分裂；新私有帧首位适配取 ppn16，其余 15 页仍共享
        ts = mmu.handle_write_fault(p, 0x4300)
        settle(mmu, ts)
        self.assertEqual(mmu.walk_table(p, 0x4300).ppn, 16)  # 新私有帧
        self.assertEqual(mmu.walk_table(c, 0x4300).ppn, 3)   # 子仍在旧帧
        self.assertEqual(mmu.mem.frame_refs(3), 1)
        self.assertEqual(mmu.mem.frame_refs(4), 2)           # 邻页(0x4400)仍共享
        self.assertTrue(mmu.walk_table(p, 0x4400).cow)       # 邻页仍是 COW


class TestCowTicketSerialStaleness(unittest.TestCase):
    def test_clone_ticket_stale_parent_tlb_before_ack(self):
        mmu = MMU()
        p = mmu.create_address_space()
        t = mmu.map(p, 0x6000, PERM_RW)
        settle(mmu, t)
        # 父核 0 预热：TLB 里是“非 COW”的旧项
        self.assertTrue(mmu.translate(p, 0, 0x6000, AccessType.READ).ok)
        c, tc = mmu.clone_address_space(p)
        # ACK 前：父核 0 命中旧 TLB 项（无 cow 标记），写仍成功——协议允许的陈旧窗口
        stale = mmu.translate(p, 0, 0x6000, AccessType.WRITE)
        self.assertTrue(stale.ok and stale.via == "tlb")
        # 全部 ACK + finish 后旧项移除，重走页表 -> COW 写故障
        settle(mmu, tc)
        fresh = mmu.translate(p, 0, 0x6000, AccessType.WRITE)
        self.assertEqual((fresh.fault, fresh.cow, fresh.via), ("write", True, "walk"))
        # 子是全新空间，从未有过 TLB 项
        self.assertEqual(mmu.tlb_keys(1), [])

    def test_cow_split_ticket_other_core_stale_until_ack(self):
        mmu = MMU()
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x7000, PERM_RW))  # ppn0
        c, tc = mmu.clone_address_space(p)
        settle(mmu, tc)
        # 核 1 在子空间触发一次 COW 写故障 -> TLB 填入带 cow 的项
        r = mmu.translate(c, 1, 0x7000, AccessType.WRITE)
        self.assertTrue(r.cow)
        # 分裂（新私有帧 ppn1）
        ts = mmu.handle_write_fault(c, 0x7000)
        # ACK 前核 1 仍命中带 cow 的旧项：写依旧故障
        stale = mmu.translate(c, 1, 0x7000, AccessType.WRITE)
        self.assertEqual((stale.via, stale.cow), ("tlb", True))
        self.assertFalse(stale.ok)
        mmu.ack(0, ts.id)
        mmu.ack(1, ts.id)
        mmu.finish(ts.id)
        # ACK 后重走页表 -> 私有新帧，写成功
        done = mmu.translate(c, 1, 0x7000, AccessType.WRITE)
        self.assertTrue(done.ok)
        self.assertEqual(done.phys_addr, 1 * PAGE_SIZE)
        self.assertEqual(done.via, "walk")


class TestWriteFaultFailureAtomic(unittest.TestCase):
    def test_frame_exhausted_keeps_original_mapping(self):
        # 2 帧小池：ppn0 共享页，ppn1 占满，使分裂分配必然失败
        mem = PhysicalMemory(num_frames=2)
        mmu = MMU(mem)
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x0000, PERM_RW))   # ppn0
        c, tc = mmu.clone_address_space(p)         # ppn0 refs=2，无新帧
        settle(mmu, tc)
        settle(mmu, mmu.map(p, 0x0100, PERM_RW))   # ppn1，池满
        self.assertEqual(mem.used_frames(), 2)

        with self.assertRaises(FrameExhausted):
            mmu.handle_write_fault(p, 0x0000)
        # 失败后原映射保持：仍指旧帧、仍 COW、引用计数不变
        w = mmu.walk_table(p, 0x0000)
        self.assertEqual((w.ppn, w.cow), (0, True))
        self.assertEqual(mem.frame_refs(0), 2)
        self.assertEqual(mem.used_frames(), 2)
        # 子的共享映射同样完好
        self.assertEqual(mmu.walk_table(c, 0x0000).ppn, 0)
        # 仍可重试：写依旧是 COW 故障（而不是直接写穿共享帧）
        r = mmu.translate(p, 0, 0x0000, AccessType.WRITE)
        self.assertEqual((r.fault, r.cow), ("write", True))

    def test_clone_does_not_consume_frames_so_cannot_exhaust(self):
        # 克隆只增加引用，帧池再小也不会因克隆本身耗尽
        mem = PhysicalMemory(num_frames=1)
        mmu = MMU(mem)
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x0000, PERM_RW))
        c, tc = mmu.clone_address_space(p)
        settle(mmu, tc)
        self.assertEqual(mem.used_frames(), 1)
        self.assertEqual(mem.frame_refs(0), 2)
        self.assertEqual(mmu.walk_table(c, 0x0000).ppn, 0)


class TestCloneAndFaultInputValidation(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.p = self.mmu.create_address_space()
        settle(self.mmu, self.mmu.map(self.p, 0x1000, PERM_RW))

    def test_clone_invalid_asid(self):
        with self.assertRaises(InvalidAddressSpace):
            self.mmu.clone_address_space(9)
        with self.assertRaises(TypeError):
            self.mmu.clone_address_space(True)

    def test_clone_when_slots_full(self):
        for _ in range(3):
            settle(self.mmu, self.mmu.map(self.mmu.create_address_space(), 0x1000, PERM_RW))
        # 4 个活动空间已满（self.p + 3），克隆需要第 5 个槽位 -> AddressSpaceLimit
        with self.assertRaises(AddressSpaceLimit):
            self.mmu.clone_address_space(self.p)
        # 失败不影响父空间
        self.assertFalse(self.mmu.walk_table(self.p, 0x1000).cow)
        self.assertEqual(self.mmu.mem.frame_refs(0), 1)

    def test_write_fault_invalid_inputs(self):
        # 父页 0x1000 从未被克隆：它是普通可写页，不是 COW
        with self.assertRaises(NotCopyOnWrite):
            self.mmu.handle_write_fault(self.p, 0x1000)
        c, tc = self.mmu.clone_address_space(self.p)
        settle(self.mmu, tc)
        # 克隆后父子 0x1000 都是 COW；未映射地址报错
        with self.assertRaises(NotMapped):
            self.mmu.handle_write_fault(c, 0x2000)
        # 子中新映射的普通页不是 COW
        settle(self.mmu, self.mmu.map(c, 0x2000, PERM_RW))
        with self.assertRaises(NotCopyOnWrite):
            self.mmu.handle_write_fault(c, 0x2000)
        # 非对齐
        with self.assertRaises(ValueError):
            self.mmu.handle_write_fault(c, 0x1001)
        # bool / float 拒绝
        with self.assertRaises(TypeError):
            self.mmu.handle_write_fault(c, True)
        # 非法 ASID
        with self.assertRaises(InvalidAddressSpace):
            self.mmu.handle_write_fault(9, 0x1000)

    def test_write_fault_on_huge_leaf_rejected(self):
        # 未克隆的普通大页不是 COW，handle_write_fault 明确拒绝
        mmu = MMU()
        p = mmu.create_address_space()
        settle(mmu, mmu.map(p, 0x0000, PERM_RWX, huge=True))
        with self.assertRaises(NotCopyOnWrite):
            mmu.handle_write_fault(p, 0x0300)


if __name__ == "__main__":
    unittest.main()
