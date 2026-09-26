"""A/D 位（访问位/脏位）与采样并清位接口的测试。

覆盖题面要求的场景：
* 两核读写同一页；
* 只命中 TLB 的访问也被记录；
* 采样与写交错（清位后的新访问不丢失，旧世代迟到回写不污染）；
* 大页标志检查；
* 独立账本：按成功访存事件与采样边界独立推算期望，与 sample_and_clear 对照；
* 物理地址与权限判定结果与上一迭代保持一致。
"""

from __future__ import annotations

import random
import unittest

from vmshoot import (
    MMU,
    AccessType,
    InvalidAddressSpace,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PERM_WRITE,
)


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        mmu.ack(0, t.id)
        mmu.ack(1, t.id)
        mmu.finish(t.id)


class FlagTestBase(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()

    def map(self, vaddr, perms, *, huge=False):
        settle(self.mmu, self.mmu.map(self.asid, vaddr, perms, huge=huge))

    def flags_of(self, result, vaddr):
        for p in result.pages:
            if p.vaddr == vaddr:
                return p
        return None


class TestBasicFlags(FlagTestBase):
    def test_walk_sets_A_and_write_sets_D(self):
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1010, AccessType.READ)
        res = self.mmu.sample_and_clear(self.asid)
        p = self.flags_of(res, 0x1000)
        self.assertTrue(p.accessed)
        self.assertFalse(p.dirty)
        # 清位后干净；再写入 -> A 和 D
        res2 = self.mmu.sample_and_clear(self.asid)
        self.assertEqual(res2.flagged(), [])
        self.mmu.translate(self.asid, 0, 0x1020, AccessType.WRITE)
        res3 = self.mmu.sample_and_clear(self.asid)
        p3 = self.flags_of(res3, 0x1000)
        self.assertTrue(p3.accessed)
        self.assertTrue(p3.dirty)

    def test_tlb_hit_only_records_flags(self):
        """只命中 TLB（不再走页表）的访问也必须被记录。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 1, 0x1000, AccessType.READ)  # 核1 填 TLB
        self.mmu.sample_and_clear(self.asid)
        # 接下来全部是 TLB 命中
        for off in (0x00, 0x40, 0x80):
            r = self.mmu.translate(self.asid, 1, 0x1000 + off, AccessType.READ)
            self.assertEqual(r.via, "tlb")
        res = self.mmu.sample_and_clear(self.asid)
        p = self.flags_of(res, 0x1000)
        self.assertTrue(p.accessed)
        self.assertFalse(p.dirty)
        # 纯 TLB 命中的写 -> D
        r = self.mmu.translate(self.asid, 1, 0x1008, AccessType.WRITE)
        self.assertEqual(r.via, "tlb")
        res2 = self.mmu.sample_and_clear(self.asid)
        p2 = self.flags_of(res2, 0x1000)
        self.assertTrue(p2.accessed)
        self.assertTrue(p2.dirty)

    def test_fault_accesses_leave_no_trace(self):
        """权限失败与不存在的页不留下成功访问痕迹。"""
        self.map(0x1000, PERM_READ)
        self.map(0x2000, PERM_RX)
        r1 = self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        self.assertFalse(r1.ok)
        self.assertEqual(r1.fault, "write")
        r2 = self.mmu.translate(self.asid, 0, 0x2000, AccessType.WRITE)
        self.assertFalse(r2.ok)
        r3 = self.mmu.translate(self.asid, 1, 0x3000, AccessType.READ)  # 未映射
        self.assertFalse(r3.ok)
        self.assertEqual(r3.fault, "not_mapped")
        res = self.mmu.sample_and_clear(self.asid)
        self.assertEqual(res.flagged(), [])

    def test_two_cores_read_write_same_page(self):
        """两核读写同页：A 由读贡献、D 由另一核的写贡献，合并到同一 PTE。"""
        self.map(0x1000, PERM_RW)
        # 双核都先缓存该页，随后全部 TLB 命中
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)
        self.mmu.translate(self.asid, 1, 0x1000, AccessType.READ)
        self.mmu.sample_and_clear(self.asid)
        self.mmu.translate(self.asid, 0, 0x1010, AccessType.READ)
        self.mmu.translate(self.asid, 1, 0x1020, AccessType.WRITE)
        self.mmu.translate(self.asid, 1, 0x1030, AccessType.READ)
        res = self.mmu.sample_and_clear(self.asid)
        p = self.flags_of(res, 0x1000)
        self.assertTrue(p.accessed)
        self.assertTrue(p.dirty)

    def test_huge_page_flags(self):
        """大页：任意 256B 子页的访问都归并到同一个大页 PTE 的标志上。"""
        self.map(0x4000, PERM_RWX, huge=True)
        self.mmu.translate(self.asid, 0, 0x4000, AccessType.READ)   # 核0 填大页项
        self.mmu.translate(self.asid, 1, 0x4100, AccessType.READ)   # 核1 填大页项
        self.mmu.sample_and_clear(self.asid)
        # 全部 TLB 命中，落在大页内不同小页
        self.mmu.translate(self.asid, 0, 0x4200, AccessType.READ)
        self.mmu.translate(self.asid, 1, 0x4F00, AccessType.WRITE)
        self.mmu.translate(self.asid, 1, 0x4800, AccessType.EXECUTE)
        res = self.mmu.sample_and_clear(self.asid)
        self.assertEqual(len(res.pages), 1)
        p = res.pages[0]
        self.assertEqual(p.vaddr, 0x4000)
        self.assertTrue(p.huge)
        self.assertTrue(p.accessed)
        self.assertTrue(p.dirty)

    def test_interleaved_sampling_and_writes(self):
        """采样与写交错：清位后的新访问进入下一世代，不丢失。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        r1 = self.mmu.sample_and_clear(self.asid)
        self.assertTrue(self.flags_of(r1, 0x1000).dirty)
        # 清位之后立刻写（TLB 仍热）——必须计入下一轮
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        r2 = self.mmu.sample_and_clear(self.asid)
        self.assertEqual(r2.generation, r1.generation + 1)
        p2 = self.flags_of(r2, 0x1000)
        self.assertTrue(p2.accessed)
        self.assertTrue(p2.dirty)
        # 没有访问的一轮：完全干净
        r3 = self.mmu.sample_and_clear(self.asid)
        self.assertEqual(r3.flagged(), [])
        self.assertEqual(r3.generation, r2.generation + 1)

    def test_late_writeback_does_not_pollute_next_generation(self):
        """旧世代的迟到回写（采样后旧 TLB 项才被淘汰）不能污染下一轮。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)  # TLB 待回写 D
        r1 = self.mmu.sample_and_clear(self.asid)                   # 合并 + 清位 + 推进世代
        self.assertTrue(self.flags_of(r1, 0x1000).dirty)
        # 该 TLB 项在采样之后才被淘汰：其旧世代标志必须被丢弃
        for i in range(8):
            self.map(0x2000 + i * 0x100, PERM_READ)
            self.mmu.translate(self.asid, 0, 0x2000 + i * 0x100, AccessType.READ)
        self.assertNotIn((self.asid, 0x1000 >> 8), self.mmu.tlb_keys(0))
        # 页表位没有被迟到回写重新置上
        w = self.mmu.walk_table(self.asid, 0x1000)
        self.assertFalse(w.accessed)
        self.assertFalse(w.dirty)
        r2 = self.mmu.sample_and_clear(self.asid)
        p2 = self.flags_of(r2, 0x1000)
        self.assertFalse(p2.accessed)
        self.assertFalse(p2.dirty)

    def test_lru_eviction_writes_back_flags(self):
        """未采样时 TLB 项被淘汰，标志应写回 PTE 而不是丢失。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        for i in range(8):
            self.map(0x2000 + i * 0x100, PERM_READ)
            self.mmu.translate(self.asid, 0, 0x2000 + i * 0x100, AccessType.READ)
        w = self.mmu.walk_table(self.asid, 0x1000)
        self.assertTrue(w.accessed)
        self.assertTrue(w.dirty)

    def test_ack_merges_flags_into_pte(self):
        """失效 ACK 移除 TLB 项时，其待回写标志先并入 PTE（映射仍在时）。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)  # 核0 待回写 D
        self.mmu.translate(self.asid, 1, 0x1000, AccessType.READ)   # 核1 待回写 A
        t = self.mmu.protect(self.asid, 0x1000, PERM_READ)
        settle(self.mmu, t)
        res = self.mmu.sample_and_clear(self.asid)
        p = self.flags_of(res, 0x1000)
        self.assertTrue(p.accessed)
        self.assertTrue(p.dirty)

    def test_phys_addr_and_perms_unchanged(self):
        """物理地址与权限判定结果不受 A/D 机制影响，且与 walk_table 一致。"""
        self.map(0x1000, PERM_RW)
        self.map(0x2000, PERM_READ)
        self.map(0x4000, PERM_RWX, huge=True)
        before = [
            self.mmu.translate(self.asid, 0, 0x1010, AccessType.READ),
            self.mmu.translate(self.asid, 1, 0x1020, AccessType.WRITE),
            self.mmu.translate(self.asid, 0, 0x2000, AccessType.WRITE),   # fault
            self.mmu.translate(self.asid, 1, 0x4ABC, AccessType.EXECUTE),
        ]
        self.mmu.sample_and_clear(self.asid)
        self.mmu.sample_and_clear(self.asid)
        after = [
            self.mmu.translate(self.asid, 0, 0x1010, AccessType.READ),
            self.mmu.translate(self.asid, 1, 0x1020, AccessType.WRITE),
            self.mmu.translate(self.asid, 0, 0x2000, AccessType.WRITE),
            self.mmu.translate(self.asid, 1, 0x4ABC, AccessType.EXECUTE),
        ]
        for b, a in zip(before, after):
            self.assertEqual(b.ok, a.ok)
            self.assertEqual(b.phys_addr, a.phys_addr)
            self.assertEqual(b.fault, a.fault)
            self.assertEqual(b.perms, a.perms)
        for va in (0x1010, 0x2000, 0x4ABC):
            w = self.mmu.walk_table(self.asid, va)
            r = self.mmu.translate(self.asid, 0, va, AccessType.READ)
            self.assertEqual(r.phys_addr, w.phys_addr)
            self.assertEqual(r.perms, w.perms)

    def test_sample_errors(self):
        with self.assertRaises(InvalidAddressSpace):
            self.mmu.sample_and_clear(3)
        with self.assertRaises(TypeError):
            self.mmu.sample_and_clear(True)
        with self.assertRaises(TypeError):
            self.mmu.sample_generation(1.0)
        self.mmu.destroy_address_space(self.asid)
        with self.assertRaises(InvalidAddressSpace):
            self.mmu.sample_and_clear(self.asid)

    def test_generation_advances_per_space(self):
        other = self.mmu.create_address_space()
        self.map(0x1000, PERM_RW)
        g0 = self.mmu.sample_generation(self.asid)
        self.mmu.sample_and_clear(self.asid)
        self.mmu.sample_and_clear(self.asid)
        self.assertEqual(self.mmu.sample_generation(self.asid), g0 + 2)
        self.assertEqual(self.mmu.sample_generation(other), 0)  # 互不影响
        self.mmu.destroy_address_space(other)


class TestLedger(FlagTestBase):
    """独立账本：测试侧自行记录成功访存事件，按采样边界推算期望标志。"""

    def test_event_ledger_matches_samples(self):
        # 固定布局：小页 rw / ro / rx + 大页 rwx
        layout = [
            (0x1000, PERM_RW, False),
            (0x2000, PERM_READ, False),
            (0x3000, PERM_RX, False),
            (0x5000, PERM_RW, False),
            (0x8000, PERM_RWX, True),
        ]
        for va, perms, huge in layout:
            self.map(va, perms, huge=huge)
        bases = [va for va, _, _ in layout]
        huge_of = {va: huge for va, _, huge in layout}

        rng = random.Random(20260926)
        accesses = [AccessType.READ, AccessType.WRITE, AccessType.EXECUTE]
        # 账本：当前采样轮内 {映射起始 vaddr: [A, D]}
        ledger: dict[int, list[bool]] = {}

        def note(vaddr, access, ok):
            if not ok:
                return
            base = vaddr & ~0xFFF if huge_of.get(vaddr & ~0xFF00, False) else vaddr & ~0xFF
            # 大页覆盖整个 4096 对齐区
            for b in bases:
                size = 0x1000 if huge_of[b] else 0x100
                if b <= vaddr < b + size:
                    base = b
                    break
            else:
                return  # 未映射地址，无痕迹
            e = ledger.setdefault(base, [False, False])
            e[0] = True
            if access is AccessType.WRITE:
                e[1] = True

        rounds = 0
        for step in range(400):
            if step > 0 and step % 40 == 0:
                res = self.mmu.sample_and_clear(self.asid)
                got = {p.vaddr: [p.accessed, p.dirty] for p in res.flagged()}
                self.assertEqual(got, ledger, f"第 {rounds} 轮采样与账本不符")
                ledger.clear()
                rounds += 1
            core = rng.randrange(2)
            base = rng.choice(bases + [0xF000])  # 0xF000 未映射
            size = 0x1000 if huge_of.get(base, False) else 0x100
            va = base + rng.randrange(size)
            access = rng.choice(accesses)
            r = self.mmu.translate(self.asid, core, va, access)
            note(va, access, r.ok)
            # 物理地址一致性：成功结果必须与无 TLB 遍历器一致
            if r.ok:
                w = self.mmu.walk_table(self.asid, va)
                self.assertEqual(r.phys_addr, w.phys_addr)
                self.assertEqual(r.perms, w.perms)
        # 收尾再核对一轮
        res = self.mmu.sample_and_clear(self.asid)
        got = {p.vaddr: [p.accessed, p.dirty] for p in res.flagged()}
        self.assertEqual(got, ledger)
        self.assertGreaterEqual(rounds, 8)


if __name__ == "__main__":
    unittest.main()
