"""两级页表、大页、拆分保护与帧分配语义测试。

物理帧号期望值来自独立的“首位适配”推导（题面：线性帧池，小页取第一个空帧、
大页取第一个全空的 16 对齐块），不读取被测实现内部状态来构造期望。
"""

from __future__ import annotations

import unittest

from vmshoot import (
    AccessType,
    HUGE_PAGE_SIZE,
    MMU,
    MappingConflict,
    MappingExists,
    NotMapped,
    PAGE_SIZE,
    PERM_EXEC,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PERM_WRITE,
    PhysicalMemory,
    AddressSpaceLimit,
)


def ppn_of(wr) -> int:
    return wr.ppn


class TestAddressLifecycle(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()

    def test_asids_0_to_3_then_limit(self):
        ids = [self.mmu.create_address_space() for _ in range(4)]
        self.assertEqual(ids, [0, 1, 2, 3])
        with self.assertRaises(AddressSpaceLimit):
            self.mmu.create_address_space()

    def test_destroy_frees_slot(self):
        a = self.mmu.create_address_space()
        b = self.mmu.create_address_space()
        self.mmu.destroy_address_space(a)
        # 无未完成票据时可立即复用原编号（从最小编号分配）
        c = self.mmu.create_address_space()
        self.assertEqual(c, a)

    def test_invalid_asid(self):
        from vmshoot import InvalidAddressSpace
        with self.assertRaises(InvalidAddressSpace):
            self.mmu.walk_table(0, 0)
        self.mmu.create_address_space()
        self.mmu.destroy_address_space(0)
        with self.assertRaises(InvalidAddressSpace):
            self.mmu.walk_table(0, 0)
        with self.assertRaises(TypeError):
            self.mmu.walk_table(True, 0)


class TestSmallPages(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()

    def test_first_maps_get_first_fit_ppns_independently_expected(self):
        # 独立期望：空池首位适配 -> 第 1 页 ppn=0，第 2 页 ppn=1，第 3 页 ppn=2
        self.mmu.map(self.asid, 0x0000, PERM_RW)
        self.mmu.map(self.asid, 0x0100, PERM_RW)
        self.mmu.map(self.asid, 0x0500, PERM_RW)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x0000).ppn, 0)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x01FF).ppn, 1)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x0500).ppn, 2)

    def test_independent_phys_formula(self):
        self.mmu.map(self.asid, 0x2000, PERM_RWX)  # ppn 0
        wr = self.mmu.walk_table(self.asid, 0x20C0)
        # 独立公式 phys = ppn*256 + offset
        self.assertEqual(wr.phys_addr, wr.ppn * 256 + 0xC0)
        self.assertEqual(wr.phys_addr, 0xC0)

    def test_duplicate_map_rejected(self):
        self.mmu.map(self.asid, 0x0000, PERM_RW)
        with self.assertRaises(MappingExists):
            self.mmu.map(self.asid, 0x0000, PERM_RW)
        # 同页的非对齐地址不静默当作该页处理：直接以对齐错误拒绝
        with self.assertRaises(ValueError):
            self.mmu.map(self.asid, 0x0080, PERM_RW)

    def test_misaligned_map_rejected_without_state_change(self):
        used_before = self.mmu.mem.used_frames()
        with self.assertRaises(ValueError):
            self.mmu.map(self.asid, 0x0001, PERM_RW)
        self.assertEqual(self.mmu.mem.used_frames(), used_before)
        self.assertIsNone(self.mmu.walk_table(self.asid, 0x0000))

    def test_unmap_and_frame_reuse(self):
        self.mmu.map(self.asid, 0x0000, PERM_RW)  # ppn 0
        self.mmu.map(self.asid, 0x0100, PERM_RW)  # ppn 1
        tu = self.mmu.unmap(self.asid, 0x0000)
        # 无核进入过该空间：票据目标为空，finish 立即成功并回收隔离帧
        self.assertEqual(tu.targets, ())
        self.mmu.finish(tu.id)
        self.assertIsNone(self.mmu.walk_table(self.asid, 0x0000))
        self.mmu.map(self.asid, 0x0200, PERM_RW)  # 首位适配：重新拿到 ppn 0
        self.assertEqual(self.mmu.walk_table(self.asid, 0x0200).ppn, 0)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x0100).ppn, 1)

    def test_unmap_frame_not_reused_before_all_acks(self):
        """旧物理页必须等全部目标核 ACK 后才能复用。"""
        tm = self.mmu.map(self.asid, 0x0000, PERM_RW)  # ppn 0
        self.mmu.finish(tm.id)
        self.mmu.translate(self.asid, 0, 0x0000, AccessType.READ)  # 核0 进入并缓存
        tu = self.mmu.unmap(self.asid, 0x0000)
        self.assertEqual(tu.targets, (0,))  # 只有核0可能缓存
        # ACK 之前：帧仍被隔离占用，新映射只能拿到 ppn 1
        t2 = self.mmu.map(self.asid, 0x0200, PERM_RW)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x0200).ppn, 1)
        self.mmu.ack(0, tu.id)
        self.mmu.finish(tu.id)             # 此刻 ppn 0 才回收
        self.mmu.ack(0, t2.id)
        self.mmu.finish(t2.id)
        t3 = self.mmu.map(self.asid, 0x0300, PERM_RW)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x0300).ppn, 0)
        self.mmu.ack(0, t3.id)
        self.mmu.finish(t3.id)

    def test_unmap_not_mapped(self):
        with self.assertRaises(NotMapped):
            self.mmu.unmap(self.asid, 0x3000)

    def test_unmap_misaligned(self):
        self.mmu.map(self.asid, 0x0000, PERM_RW)
        with self.assertRaises(ValueError):
            self.mmu.unmap(self.asid, 0x0001)


class TestHugePages(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()

    def test_huge_uses_sixteen_aligned_frames(self):
        # 独立期望：大页首位适配 -> 起始 ppn 0，占用帧 0..15
        self.mmu.map(self.asid, 0x0000, PERM_RWX, huge=True)
        wr = self.mmu.walk_table(self.asid, 0x0FFF)
        self.assertTrue(wr.huge)
        self.assertEqual(wr.ppn, 0)
        self.assertEqual(wr.phys_addr, 15 * 256 + 255)
        self.assertEqual(self.mmu.mem.used_frames(), 16)

    def test_second_huge_starts_at_frame_16(self):
        self.mmu.map(self.asid, 0x0000, PERM_RW, huge=True)
        self.mmu.map(self.asid, 0x1000, PERM_RW, huge=True)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x1000).ppn, 16)
        self.assertEqual(self.mmu.mem.used_frames(), 32)

    def test_huge_alignment_required(self):
        with self.assertRaises(ValueError):
            self.mmu.map(self.asid, 0x0100, PERM_RW, huge=True)

    def test_small_map_conflicts_with_huge(self):
        self.mmu.map(self.asid, 0x2000, PERM_RW, huge=True)
        with self.assertRaises(MappingExists):
            self.mmu.map(self.asid, 0x2A00, PERM_RW)
        with self.assertRaises(MappingExists):
            self.mmu.map(self.asid, 0x2000, PERM_RW)

    def test_huge_conflicts_with_small_table(self):
        self.mmu.map(self.asid, 0x3100, PERM_RW)  # 先建小页，一级项已存在
        with self.assertRaises(MappingExists):
            self.mmu.map(self.asid, 0x3000, PERM_RW, huge=True)

    def test_unmap_huge_interior_refused_no_implicit_split(self):
        self.mmu.map(self.asid, 0x4000, PERM_RWX, huge=True)
        with self.assertRaises(MappingConflict):
            self.mmu.unmap(self.asid, 0x4300)
        # 失败后大页完好无损
        wr = self.mmu.walk_table(self.asid, 0x4300)
        self.assertIsNotNone(wr)
        self.assertTrue(wr.huge)

    def test_unmap_huge_base_releases_all_frames(self):
        self.mmu.map(self.asid, 0x0000, PERM_RW, huge=True)
        tu = self.mmu.unmap(self.asid, 0x0000)
        self.mmu.finish(tu.id)
        self.assertIsNone(self.mmu.walk_table(self.asid, 0x0000))
        self.assertEqual(self.mmu.mem.used_frames(), 0)


class TestHugeSplitProtect(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()

    def test_protect_middle_splits_and_preserves_others(self):
        self.mmu.map(self.asid, 0x5000, PERM_RWX, huge=True)  # ppn 0..15
        # 大页中间页（第 3 个小页，0x5300）改成只读
        self.mmu.protect(self.asid, 0x5300, PERM_READ)
        ms = self.mmu.mappings(self.asid)
        self.assertEqual(len(ms), 16)
        for va, size, perms, huge in ms:
            self.assertFalse(huge)
            self.assertEqual(size, PAGE_SIZE)
            if va == 0x5300:
                self.assertEqual(perms, PERM_READ)
            else:
                self.assertEqual(perms, PERM_RWX, f"0x{va:x} 的权限应保持 rwx")

    def test_split_preserves_phys_addresses(self):
        self.mmu.map(self.asid, 0x6000, PERM_RWX, huge=True)  # ppn 0..15
        before = {off: self.mmu.walk_table(self.asid, 0x6000 + off).phys_addr
                  for off in (0, 0x300, 0x3FF, 0xA07, 0xFFF)}
        self.mmu.protect(self.asid, 0x6300, PERM_READ)
        after = {off: self.mmu.walk_table(self.asid, 0x6000 + off).phys_addr
                 for off in (0, 0x300, 0x3FF, 0xA07, 0xFFF)}
        self.assertEqual(before, after)
        # 独立期望：拆分不换帧，0x6A07 仍解析到 ppn=10 的 0x07 偏移
        wr = self.mmu.walk_table(self.asid, 0x6A07)
        self.assertFalse(wr.huge)
        self.assertEqual(wr.ppn, 10)
        self.assertEqual(wr.phys_addr, 10 * 256 + 0x07)

    def test_split_does_not_change_frame_count(self):
        self.mmu.map(self.asid, 0x7000, PERM_RWX, huge=True)
        used = self.mmu.mem.used_frames()
        self.assertEqual(used, 16)
        self.mmu.protect(self.asid, 0x7F00, PERM_READ)
        self.assertEqual(self.mmu.mem.used_frames(), 16)  # 就地拆分，不增帧

    def test_multiple_protects_after_split(self):
        self.mmu.map(self.asid, 0x8000, PERM_RWX, huge=True)
        self.mmu.protect(self.asid, 0x8200, PERM_READ)
        self.mmu.protect(self.asid, 0x8400, PERM_RX)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x8200).perms, PERM_READ)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x8400).perms, PERM_READ | PERM_EXEC)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x8300).perms, PERM_RWX)

    def test_protect_at_huge_base_also_splits_single_page(self):
        # protect 语义是“单个 256B 页改权限”；落在大页基址也只改第 0 小页，
        # 其余 15 页保留原权限（不提供整体大页改权限的静默快捷路径）。
        self.mmu.map(self.asid, 0xC000, PERM_RWX, huge=True)
        self.mmu.protect(self.asid, 0xC000, PERM_READ)
        self.assertEqual(self.mmu.walk_table(self.asid, 0xC000).perms, PERM_READ)
        self.assertEqual(self.mmu.walk_table(self.asid, 0xC100).perms, PERM_RWX)
        self.assertEqual(self.mmu.walk_table(self.asid, 0xCF00).perms, PERM_RWX)
        self.assertEqual(len(self.mmu.mappings(self.asid)), 16)

    def test_protect_requires_alignment(self):
        self.mmu.map(self.asid, 0x9000, PERM_RWX, huge=True)
        with self.assertRaises(ValueError):
            self.mmu.protect(self.asid, 0x9001, PERM_READ)
        # 未改动
        self.assertEqual(self.mmu.walk_table(self.asid, 0x9001).perms, PERM_RWX)

    def test_protect_not_mapped_small_region(self):
        with self.assertRaises(NotMapped):
            self.mmu.protect(self.asid, 0xA000, PERM_READ)

    def test_small_protect(self):
        self.mmu.map(self.asid, 0xB000, PERM_RW)
        self.mmu.protect(self.asid, 0xB000, PERM_READ)
        self.assertEqual(self.mmu.walk_table(self.asid, 0xB000).perms, PERM_READ)


class TestPhysicalMemory(unittest.TestCase):
    def test_huge_first_fit_skips_hole_until_aligned_block(self):
        mem = PhysicalMemory(num_frames=64)
        mem.alloc_frame()      # ppn 0
        mem.alloc_huge()       # 0..15 不全空；第一个全空 16 对齐块为 16..31
        self.assertEqual(mem.used_frames(), 17)
        mem.free_frame(0)
        mem.free_huge(16)
        self.assertEqual(mem.used_frames(), 0)

    def test_exhaustion_is_definitive(self):
        from vmshoot import FrameExhausted
        mem = PhysicalMemory(num_frames=2)
        mem.alloc_frame()
        mem.alloc_frame()
        with self.assertRaises(FrameExhausted):
            mem.alloc_frame()
        with self.assertRaises(FrameExhausted):
            mem.alloc_huge()

    def test_double_free_rejected(self):
        mem = PhysicalMemory(num_frames=4)
        mem.alloc_frame()
        mem.free_frame(0)
        with self.assertRaises(ValueError):
            mem.free_frame(0)


if __name__ == "__main__":
    unittest.main()
