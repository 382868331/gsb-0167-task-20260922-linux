"""remap_range：同 ASID 映射区间平移的测试。

参考模型是**独立逐小页映射表**：``snapshot`` 把每个小页的
``(页基物理地址, 权限)`` 存进普通 dict，``ref_remap`` 用纯 dict 操作逐小页
搬移，不调用 vmshoot 的任何核心算法；每次操作后全量比对 256 个小页。
物理地址按“页基地址”比较，因此大页保持/拆小两种内部表示都等价可比；
大页是否保持由专门用例单独断言 ``walk.huge``。
"""

from __future__ import annotations

import unittest

from vmshoot import (
    MMU,
    AccessType,
    AsidBusy,
    MappingConflict,
    MappingExists,
    NotMapped,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PAGE_SIZE,
    HUGE_PAGE_SIZE,
    VADDR_SPACE,
)

SRC = 0x2000
DST = 0x8000


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        mmu.ack(0, t.id)
        mmu.ack(1, t.id)
        mmu.finish(t.id)


def snapshot(mmu: MMU, asid: int) -> dict[int, tuple[int, int]]:
    """独立逐小页参考表：vpn -> (该页基址对应的物理地址, 权限)。"""
    snap: dict[int, tuple[int, int]] = {}
    for vpn in range(VADDR_SPACE // PAGE_SIZE):
        w = mmu.walk_table(asid, vpn * PAGE_SIZE)
        if w is not None:
            snap[vpn] = (w.phys_addr, w.perms)
    return snap


def ref_remap(snap: dict[int, tuple[int, int]], src: int, dst: int, size: int):
    """纯 dict 逐小页平移：源键弹出、装到目标键。"""
    out = dict(snap)
    n = size // PAGE_SIZE
    sv, dv = src // PAGE_SIZE, dst // PAGE_SIZE
    for j in range(n):
        out[dv + j] = out.pop(sv + j)
    return out


class RemapBase(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()

    def assert_table(self, expected: dict[int, tuple[int, int]]) -> None:
        got = snapshot(self.mmu, self.asid)
        self.assertEqual(got, expected)

    def assert_unchanged(self, before: dict[int, tuple[int, int]],
                         frames_before: int, tickets_before: list[int]) -> None:
        """失败路径：页表（逐小页）、映射枚举、帧占用、票据列表全部不变。"""
        self.assert_table(before)
        self.assertEqual(self.mmu.mem.used_frames(), frames_before)
        self.assertEqual(self.mmu.pending_tickets(), tickets_before)


class TestRemapBasic(RemapBase):
    def test_small_range_moves_frames_and_perms(self):
        ts = [
            self.mmu.map(self.asid, SRC + 0 * PAGE_SIZE, PERM_RW),
            self.mmu.map(self.asid, SRC + 1 * PAGE_SIZE, PERM_RX),
            self.mmu.map(self.asid, SRC + 2 * PAGE_SIZE, PERM_READ),
        ]
        settle(self.mmu, *ts)
        ref = snapshot(self.mmu, self.asid)
        orig = [ref[(SRC // PAGE_SIZE) + j] for j in range(3)]

        t = self.mmu.remap_range(self.asid, SRC, DST, 3 * PAGE_SIZE)
        self.assertEqual(t.kind, "remap")
        self.assertEqual(t.asid, self.asid)
        self.assertEqual(t.vaddr, SRC)
        # 票据同时覆盖源与目标的全部小页 key
        self.assertEqual(
            set(t.keys),
            {(self.asid, SRC // PAGE_SIZE + j) for j in range(3)}
            | {(self.asid, DST // PAGE_SIZE + j) for j in range(3)},
        )
        settle(self.mmu, t)

        # 独立逐小页参考表核对全表
        self.assert_table(ref_remap(ref, SRC, DST, 3 * PAGE_SIZE))
        # finish 后源访问必须 fault，目标得到原物理地址与原权限
        for j in range(3):
            r_src = self.mmu.translate(self.asid, 0, SRC + j * PAGE_SIZE, AccessType.READ)
            self.assertFalse(r_src.ok)
            self.assertEqual(r_src.fault, "not_mapped")
            r_dst = self.mmu.translate(self.asid, 0, DST + j * PAGE_SIZE, AccessType.READ)
            self.assertTrue(r_dst.ok)
            self.assertEqual(r_dst.phys_addr, orig[j][0])
            self.assertEqual(r_dst.perms, orig[j][1])

    def test_remap_does_not_touch_frames(self):
        t1 = self.mmu.map(self.asid, SRC, PERM_RW)
        t2 = self.mmu.map(self.asid, SRC + PAGE_SIZE, PERM_RW)
        settle(self.mmu, t1, t2)
        frames = self.mmu.mem.used_frames()
        t = self.mmu.remap_range(self.asid, SRC, DST, 2 * PAGE_SIZE)
        # 纯地址重定位：不分配也不释放帧
        self.assertEqual(self.mmu.mem.used_frames(), frames)
        settle(self.mmu, t)
        self.assertEqual(self.mmu.mem.used_frames(), frames)


class TestRemapStaleTLB(RemapBase):
    def test_two_cores_cache_old_source(self):
        t1 = self.mmu.map(self.asid, SRC, PERM_RW)
        t2 = self.mmu.map(self.asid, SRC + PAGE_SIZE, PERM_RW)
        settle(self.mmu, t1, t2)
        # 双核都缓存旧源页
        for core in (0, 1):
            for off in (0, PAGE_SIZE):
                r = self.mmu.translate(self.asid, core, SRC + off, AccessType.READ)
                self.assertTrue(r.ok)
        orig_phys = self.mmu.walk_table(self.asid, SRC).phys_addr

        t = self.mmu.remap_range(self.asid, SRC, DST, 2 * PAGE_SIZE)
        # ACK 前：两核旧 TLB 项允许继续命中（陈旧翻译窗口）
        for core in (0, 1):
            r = self.mmu.translate(self.asid, core, SRC, AccessType.READ)
            self.assertTrue(r.ok)
            self.assertEqual(r.via, "tlb")
            self.assertEqual(r.phys_addr, orig_phys)
        # 核0 ACK 后：核0 重走页表即 fault；核1 仍旧
        self.mmu.ack(0, t.id)
        r0 = self.mmu.translate(self.asid, 0, SRC, AccessType.READ)
        self.assertFalse(r0.ok)
        self.assertEqual(r0.fault, "not_mapped")
        r1 = self.mmu.translate(self.asid, 1, SRC, AccessType.READ)
        self.assertTrue(r1.ok)
        self.assertEqual(r1.via, "tlb")
        # 核1 ACK + finish：双核源都 fault，目标得到原物理地址
        self.mmu.ack(1, t.id)
        self.mmu.finish(t.id)
        for core in (0, 1):
            r = self.mmu.translate(self.asid, core, SRC, AccessType.READ)
            self.assertFalse(r.ok)
            self.assertEqual(r.fault, "not_mapped")
            r = self.mmu.translate(self.asid, core, DST, AccessType.READ)
            self.assertTrue(r.ok)
            self.assertEqual(r.phys_addr, orig_phys)

    def test_dst_previously_cached_other_mapping(self):
        # 目标页曾经有过别的映射，核0 仍缓存着它（unmap 票据未 ACK）
        t_src = self.mmu.map(self.asid, SRC, PERM_RW)
        t_old = self.mmu.map(self.asid, DST, PERM_READ)
        settle(self.mmu, t_src, t_old)
        stale_phys = self.mmu.walk_table(self.asid, DST).phys_addr
        new_phys = self.mmu.walk_table(self.asid, SRC).phys_addr
        self.assertNotEqual(stale_phys, new_phys)
        self.mmu.translate(self.asid, 0, DST, AccessType.READ)  # 核0 缓存旧目标
        t_unmap = self.mmu.unmap(self.asid, DST)                # 故意不 ACK

        t = self.mmu.remap_range(self.asid, SRC, DST, PAGE_SIZE)
        # 票据同时覆盖源与目标：目标 vpn 必须在失效 key 中
        self.assertIn((self.asid, DST // PAGE_SIZE), t.keys)
        self.assertIn((self.asid, SRC // PAGE_SIZE), t.keys)
        # ACK 前：核0 在目标地址仍看到旧映射的陈旧项
        r = self.mmu.translate(self.asid, 0, DST, AccessType.READ)
        self.assertTrue(r.ok)
        self.assertEqual(r.via, "tlb")
        self.assertEqual(r.phys_addr, stale_phys)
        # 两张票据都 ACK + finish 后：目标得到搬来的原物理地址
        settle(self.mmu, t_unmap, t)
        r = self.mmu.translate(self.asid, 0, DST, AccessType.READ)
        self.assertTrue(r.ok)
        self.assertEqual(r.phys_addr, new_phys)
        r = self.mmu.translate(self.asid, 0, SRC, AccessType.READ)
        self.assertFalse(r.ok)
        self.assertEqual(r.fault, "not_mapped")


class TestRemapHuge(RemapBase):
    def test_cross_huge_boundary_partial_split(self):
        # 源区间跨两个大页的边界：0x3f00 在大页@0x3000 内，0x4000 在大页@0x4000 内
        t1 = self.mmu.map(self.asid, 0x3000, PERM_RWX, huge=True)
        t2 = self.mmu.map(self.asid, 0x4000, PERM_RX, huge=True)
        settle(self.mmu, t1, t2)
        ref = snapshot(self.mmu, self.asid)
        src, size = 0x3F00, 2 * PAGE_SIZE

        t = self.mmu.remap_range(self.asid, src, DST, size)
        # 部分覆盖 -> 两个大页的标记 key（0x30、0x40）都必须失效，
        # 即使 0x30 不在源区间小页范围内
        self.assertIn((self.asid, 0x3000 // PAGE_SIZE), t.keys)
        self.assertIn((self.asid, 0x4000 // PAGE_SIZE), t.keys)
        settle(self.mmu, t)

        self.assert_table(ref_remap(ref, src, DST, size))
        # 两个大页都被拆小：留下的 15+15 页保持原帧原权限
        for va in (0x3000, 0x4000 + PAGE_SIZE):
            w = self.mmu.walk_table(self.asid, va)
            self.assertIsNotNone(w)
            self.assertFalse(w.huge)
        # 搬走的是小页
        w = self.mmu.walk_table(self.asid, DST)
        self.assertFalse(w.huge)
        # 源两页 fault
        for va in (0x3F00, 0x4000):
            r = self.mmu.translate(self.asid, 0, va, AccessType.READ)
            self.assertFalse(r.ok)
            self.assertEqual(r.fault, "not_mapped")

    def test_partial_huge_stale_marker_invalidated(self):
        # 核0 缓存大页标记项；只搬大页后半段，票据须失效该标记
        t1 = self.mmu.map(self.asid, 0x4000, PERM_RWX, huge=True)
        settle(self.mmu, t1)
        self.mmu.translate(self.asid, 0, 0x4100, AccessType.READ)  # 填入大页标记项
        self.assertIn((self.asid, 0x4000 // PAGE_SIZE), self.mmu.tlb_keys(0))

        src = 0x4800  # 大页后半段（不在大页基址上）
        t = self.mmu.remap_range(self.asid, src, DST, 0x800)
        # ACK 前陈旧大页项仍可命中被搬走的地址
        r = self.mmu.translate(self.asid, 0, 0x4900, AccessType.READ)
        self.assertTrue(r.ok)
        self.assertEqual(r.via, "tlb")
        self.mmu.ack(0, t.id)
        self.assertNotIn((self.asid, 0x4000 // PAGE_SIZE), self.mmu.tlb_keys(0))
        self.mmu.ack(1, t.id)
        self.mmu.finish(t.id)
        # 大页已拆：留下的前半段仍可访问，搬走的后半段 fault
        self.assertTrue(self.mmu.translate(self.asid, 0, 0x4000, AccessType.READ).ok)
        self.assertFalse(self.mmu.translate(self.asid, 0, 0x4900, AccessType.READ).ok)

    def test_huge_kept_when_fully_aligned(self):
        t1 = self.mmu.map(self.asid, 0x4000, PERM_RWX, huge=True)
        settle(self.mmu, t1)
        base_ppn = self.mmu.walk_table(self.asid, 0x4000).ppn
        ref = snapshot(self.mmu, self.asid)

        t = self.mmu.remap_range(self.asid, 0x4000, DST, HUGE_PAGE_SIZE)
        settle(self.mmu, t)
        self.assert_table(ref_remap(ref, 0x4000, DST, HUGE_PAGE_SIZE))
        # 目标整段对齐 -> 保持原大页（同一批帧，同一个一级叶子）
        w = self.mmu.walk_table(self.asid, DST)
        self.assertTrue(w.huge)
        self.assertEqual(w.ppn, base_ppn)
        w = self.mmu.walk_table(self.asid, DST + 0xABC)
        self.assertTrue(w.huge)
        self.assertEqual(w.ppn, base_ppn)
        # 源大页基址 fault
        r = self.mmu.translate(self.asid, 0, 0x4000, AccessType.READ)
        self.assertFalse(r.ok)
        self.assertEqual(r.fault, "not_mapped")

    def test_huge_split_when_dst_not_huge_aligned(self):
        t1 = self.mmu.map(self.asid, 0x4000, PERM_RWX, huge=True)
        settle(self.mmu, t1)
        base_ppn = self.mmu.walk_table(self.asid, 0x4000).ppn
        ref = snapshot(self.mmu, self.asid)
        dst = DST + PAGE_SIZE  # 仅小页对齐，目标映像不是 4096 对齐

        t = self.mmu.remap_range(self.asid, 0x4000, dst, HUGE_PAGE_SIZE)
        settle(self.mmu, t)
        self.assert_table(ref_remap(ref, 0x4000, dst, HUGE_PAGE_SIZE))
        # 目标侧为 16 个小页，帧不变、不重新合并
        for j in range(16):
            w = self.mmu.walk_table(self.asid, dst + j * PAGE_SIZE)
            self.assertFalse(w.huge)
            self.assertEqual(w.ppn, base_ppn + j)


class TestRemapFailures(RemapBase):
    def _state(self):
        return snapshot(self.mmu, self.asid), self.mmu.mem.used_frames(), \
            self.mmu.pending_tickets()

    def test_dst_conflict_leaves_tables_unchanged(self):
        t1 = self.mmu.map(self.asid, SRC, PERM_RW)
        t2 = self.mmu.map(self.asid, SRC + PAGE_SIZE, PERM_RW)
        t3 = self.mmu.map(self.asid, DST + PAGE_SIZE, PERM_READ)  # 目标区间内的占用
        settle(self.mmu, t1, t2, t3)
        before = self._state()
        with self.assertRaises(MappingExists):
            self.mmu.remap_range(self.asid, SRC, DST, 2 * PAGE_SIZE)
        self.assert_unchanged(*before)

    def test_dst_conflict_with_huge_leaf_unchanged(self):
        t1 = self.mmu.map(self.asid, SRC, PERM_RW)
        t2 = self.mmu.map(self.asid, DST, PERM_RWX, huge=True)  # 大页占住目标区
        settle(self.mmu, t1, t2)
        before = self._state()
        with self.assertRaises(MappingExists):
            self.mmu.remap_range(self.asid, SRC, DST + 0x300, PAGE_SIZE)
        self.assert_unchanged(*before)

    def test_source_hole_leaves_tables_unchanged(self):
        t1 = self.mmu.map(self.asid, SRC, PERM_RW)
        t2 = self.mmu.map(self.asid, SRC + 2 * PAGE_SIZE, PERM_RW)  # 中间有洞
        settle(self.mmu, t1, t2)
        before = self._state()
        with self.assertRaises(NotMapped):
            self.mmu.remap_range(self.asid, SRC, DST, 3 * PAGE_SIZE)
        self.assert_unchanged(*before)

    def test_overlap_rejected_unchanged(self):
        t1 = self.mmu.map(self.asid, SRC, PERM_RW)
        t2 = self.mmu.map(self.asid, SRC + PAGE_SIZE, PERM_RW)
        settle(self.mmu, t1, t2)
        before = self._state()
        # 目标落入源区间内部
        with self.assertRaises(MappingConflict):
            self.mmu.remap_range(self.asid, SRC, SRC + PAGE_SIZE, 2 * PAGE_SIZE)
        # 源 == 目标
        with self.assertRaises(MappingConflict):
            self.mmu.remap_range(self.asid, SRC, SRC, PAGE_SIZE)
        self.assert_unchanged(*before)

    def test_invalid_args_rejected_unchanged(self):
        t1 = self.mmu.map(self.asid, SRC, PERM_RW)
        settle(self.mmu, t1)
        before = self._state()
        bad_calls = [
            lambda: self.mmu.remap_range(self.asid, SRC + 1, DST, PAGE_SIZE),      # 源未对齐
            lambda: self.mmu.remap_range(self.asid, SRC, DST + 1, PAGE_SIZE),      # 目标未对齐
            lambda: self.mmu.remap_range(self.asid, SRC, DST, 0),                  # size 为 0
            lambda: self.mmu.remap_range(self.asid, SRC, DST, 100),                # size 非小页倍数
            lambda: self.mmu.remap_range(self.asid, SRC, DST, True),               # bool size
            lambda: self.mmu.remap_range(self.asid, SRC, DST, 256.0),              # float size
            lambda: self.mmu.remap_range(self.asid, SRC, DST, float("nan")),       # NaN
            lambda: self.mmu.remap_range(self.asid, True, DST, PAGE_SIZE),         # bool 地址
            lambda: self.mmu.remap_range(self.asid, SRC, VADDR_SPACE - PAGE_SIZE,
                                         2 * PAGE_SIZE),                           # 目标越界
            lambda: self.mmu.remap_range(self.asid, VADDR_SPACE - PAGE_SIZE, DST,
                                         2 * PAGE_SIZE),                           # 源越界
        ]
        for call in bad_calls:
            with self.assertRaises((ValueError, TypeError)):
                call()
        self.assert_unchanged(*before)

    def test_failed_remap_then_valid_remap_works(self):
        t1 = self.mmu.map(self.asid, SRC, PERM_RW)
        t2 = self.mmu.map(self.asid, SRC + PAGE_SIZE, PERM_RX)
        t3 = self.mmu.map(self.asid, DST, PERM_READ)
        settle(self.mmu, t1, t2, t3)
        ref = snapshot(self.mmu, self.asid)
        # 先失败一次（目标冲突），状态不变
        with self.assertRaises(MappingExists):
            self.mmu.remap_range(self.asid, SRC, DST, 2 * PAGE_SIZE)
        self.assert_table(ref)
        # 换一个空闲目标后成功
        t = self.mmu.remap_range(self.asid, SRC, 0xC000, 2 * PAGE_SIZE)
        settle(self.mmu, t)
        self.assert_table(ref_remap(ref, SRC, 0xC000, 2 * PAGE_SIZE))


class TestRemapAsidReuse(unittest.TestCase):
    def test_pending_remap_ticket_blocks_asid_reuse(self):
        mmu = MMU()
        victim = mmu.create_address_space()
        others = [mmu.create_address_space() for _ in range(3)]  # 4 槽占满
        t1 = mmu.map(victim, SRC, PERM_RW)
        settle(mmu, t1)
        t = mmu.remap_range(victim, SRC, DST, PAGE_SIZE)  # 故意不 ACK/finish
        mmu.destroy_address_space(victim)
        mmu.destroy_address_space(others[0])
        # others[0] 的槽位空闲可用；victim 的号被未结束 remap 票据占据
        reused = mmu.create_address_space()
        self.assertEqual(reused, others[0])
        with self.assertRaises(AsidBusy):
            mmu.create_address_space()
        # 票据 finish 后编号立即可用
        settle(mmu, t)
        freed = mmu.create_address_space()
        self.assertEqual(freed, victim)


if __name__ == "__main__":
    unittest.main()
