"""A/D 位与采样接口测试。

覆盖：walk/TLB 命中两条路径的置位、fault 不留痕迹、双核合并、
采样与写交错的世代隔离、迟到回写丢弃、大页标志，以及独立账本对照。
"""

from __future__ import annotations

import random
import unittest

from vmshoot import (
    MMU,
    AccessType,
    PERM_READ,
    PERM_RW,
    PERM_RX,
    PERM_RWX,
    PAGE_SIZE,
    TLB_SIZE,
)


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        mmu.ack(0, t.id)
        mmu.ack(1, t.id)
        mmu.finish(t.id)


class AccessedDirtyBase(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()

    def map(self, vaddr, perms, *, huge=False):
        settle(self.mmu, self.mmu.map(self.asid, vaddr, perms, huge=huge))

    def pte(self, vaddr):
        return self.mmu._space(self.asid).find_pte(vaddr)


class TestFlagSetting(AccessedDirtyBase):
    def test_walk_sets_accessed_and_dirty_on_write(self):
        self.map(0x1000, PERM_RW)
        r = self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        self.assertEqual(r.via, "walk")
        pte = self.pte(0x1000)
        self.assertTrue(pte.accessed)
        self.assertTrue(pte.dirty)

    def test_read_does_not_set_dirty(self):
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)
        pte = self.pte(0x1000)
        self.assertTrue(pte.accessed)
        self.assertFalse(pte.dirty)

    def test_tlb_hit_also_records_flags(self):
        """TLB 命中（无 page walk）也必须留下 A/D 痕迹。"""
        self.map(0x1000, PERM_RW)
        first = self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)
        self.assertEqual(first.via, "walk")
        # 清掉 walk 路径直接置的位，制造“只有 TLB 命中”的干净起点
        self.mmu.sample_and_clear(self.asid)
        hit = self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        self.assertEqual(hit.via, "tlb")
        # PTE 尚未被直接置位（标志挂在 TLB 项上）
        self.assertFalse(self.pte(0x1000).accessed)
        # 采样时合并回来
        res = self.mmu.sample_and_clear(self.asid)
        self.assertTrue(res.pages[0x1000].accessed)
        self.assertTrue(res.pages[0x1000].dirty)

    def test_faults_leave_no_trace(self):
        """权限失败与不存在页的访问不留下成功访问痕迹。"""
        self.map(0x1000, PERM_READ)
        r = self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        self.assertFalse(r.ok)
        self.assertEqual(r.fault, "write")
        miss = self.mmu.translate(self.asid, 0, 0x2000, AccessType.READ)
        self.assertFalse(miss.ok)
        self.assertEqual(miss.fault, "not_mapped")
        res = self.mmu.sample_and_clear(self.asid)
        self.assertFalse(res.pages[0x1000].accessed)
        self.assertFalse(res.pages[0x1000].dirty)
        self.assertNotIn(0x2000, res.pages)

    def test_fault_after_protection_leaves_no_trace(self):
        """降权后权限失败的访问不置位。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)  # 填 TLB
        settle(self.mmu, self.mmu.protect(self.asid, 0x1000, PERM_READ))
        self.mmu.sample_and_clear(self.asid)
        r = self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        self.assertFalse(r.ok)
        res = self.mmu.sample_and_clear(self.asid)
        self.assertFalse(res.pages[0x1000].accessed)
        self.assertFalse(res.pages[0x1000].dirty)


class TestSampling(AccessedDirtyBase):
    def test_two_cores_same_page_merge(self):
        """两核读写同页，采样时两个核的挂起标志都要合并回来。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1010, AccessType.READ)
        self.mmu.translate(self.asid, 1, 0x1020, AccessType.WRITE)
        self.mmu.sample_and_clear(self.asid)
        # 现在两核都只靠 TLB 命中
        self.assertEqual(self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ).via, "tlb")
        self.assertEqual(self.mmu.translate(self.asid, 1, 0x1000, AccessType.WRITE).via, "tlb")
        res = self.mmu.sample_and_clear(self.asid)
        self.assertTrue(res.pages[0x1000].accessed)
        self.assertTrue(res.pages[0x1000].dirty)

    def test_generation_isolation_and_late_writeback(self):
        """采样与写交错：旧世代的迟到回写（TLB 逐出）不污染新一轮。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)
        self.mmu.sample_and_clear(self.asid)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)  # tlb 命中，pend 置位
        res1 = self.mmu.sample_and_clear(self.asid)
        self.assertTrue(res1.pages[0x1000].dirty)
        self.assertEqual(res1.generation, self.mmu.generation(self.asid) - 1)
        # 新世代不碰 0x1000；用其它页把它的 TLB 项挤出去，触发“迟到回写”
        for i in range(1, TLB_SIZE + 2):
            self.map(0x1000 + i * PAGE_SIZE, PERM_READ)
            self.mmu.translate(self.asid, 0, 0x1000 + i * PAGE_SIZE, AccessType.READ)
        res2 = self.mmu.sample_and_clear(self.asid)
        self.assertFalse(res2.pages[0x1000].accessed)
        self.assertFalse(res2.pages[0x1000].dirty)

    def test_access_after_clear_not_lost(self):
        """清位之后、下一世代的新访问不能丢。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)
        res1 = self.mmu.sample_and_clear(self.asid)
        self.assertTrue(res1.pages[0x1000].accessed)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        res2 = self.mmu.sample_and_clear(self.asid)
        self.assertTrue(res2.pages[0x1000].accessed)
        self.assertTrue(res2.pages[0x1000].dirty)

    def test_ack_invalidation_writes_back_pending_flags(self):
        """ACK 失效把 TLB 项移除前，挂起标志先合并回页表。"""
        self.map(0x1000, PERM_RW)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)
        self.mmu.sample_and_clear(self.asid)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)  # tlb 命中，pend_d
        t = self.mmu.protect(self.asid, 0x1000, PERM_RX)
        self.mmu.ack(0, t.id)  # 移除核 0 的 TLB 项 -> 应回写 D
        self.assertTrue(self.pte(0x1000).dirty)
        self.mmu.ack(1, t.id)
        self.mmu.finish(t.id)

    def test_huge_page_flags(self):
        """大页：标志落在一级叶子上，TLB 命中也要记录。"""
        self.map(0x4000, PERM_RWX, huge=True)
        r = self.mmu.translate(self.asid, 0, 0x4100, AccessType.WRITE)
        self.assertEqual(r.via, "walk")
        pte = self.pte(0x4000)
        self.assertTrue(pte.accessed and pte.dirty)
        self.mmu.translate(self.asid, 1, 0x4200, AccessType.READ)  # 核 1 也填上 TLB
        self.mmu.sample_and_clear(self.asid)
        hit = self.mmu.translate(self.asid, 1, 0x4F00, AccessType.READ)
        self.assertEqual(hit.via, "tlb")
        res = self.mmu.sample_and_clear(self.asid)
        self.assertTrue(res.pages[0x4000].huge)
        self.assertTrue(res.pages[0x4000].accessed)
        self.assertFalse(res.pages[0x4000].dirty)  # 本轮只有读

    def test_sample_result_covers_all_mappings(self):
        self.map(0x1000, PERM_RW)
        self.map(0x2000, PERM_READ)
        self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)
        res = self.mmu.sample_and_clear(self.asid)
        self.assertEqual(set(res.pages), {0x1000, 0x2000})
        self.assertEqual(set(res.flagged()), {0x1000})

    def test_sample_rejects_bad_asid(self):
        from vmshoot import InvalidAddressSpace
        with self.assertRaises(InvalidAddressSpace):
            self.mmu.sample_and_clear(3)  # 未创建


class TestLedger(unittest.TestCase):
    """独立账本对照：按成功访存事件与采样边界独立记账，与采样结果比对。"""

    def test_random_interleaved_ledger(self):
        rng = random.Random(20260926)
        mmu = MMU()
        asid = mmu.create_address_space()
        pages = [0x1000 + i * PAGE_SIZE for i in range(6)]
        perms = {}
        for v in pages:
            p = rng.choice([PERM_READ, PERM_RW, PERM_RX, PERM_RWX])
            t = mmu.map(asid, v, p)
            mmu.ack(0, t.id); mmu.ack(1, t.id); mmu.finish(t.id)
            perms[v] = p

        # 账本：{vaddr: [accessed, dirty]}，只记成功访问
        ledger = {v: [False, False] for v in pages}
        for step in range(400):
            if rng.random() < 0.15:
                # 采样边界：对照账本后清零
                res = mmu.sample_and_clear(asid)
                for v in pages:
                    self.assertEqual(res.pages[v].accessed, ledger[v][0], f"step {step} A {v:#x}")
                    self.assertEqual(res.pages[v].dirty, ledger[v][1], f"step {step} D {v:#x}")
                    ledger[v] = [False, False]
            else:
                v = rng.choice(pages)
                core = rng.randrange(2)
                access = rng.choice([AccessType.READ, AccessType.WRITE, AccessType.EXECUTE])
                r = mmu.translate(asid, core, v + rng.randrange(PAGE_SIZE), access)
                expect_ok = bool(perms[v] & int(access))
                self.assertEqual(r.ok, expect_ok)
                if r.ok:
                    ledger[v][0] = True
                    if access is AccessType.WRITE:
                        ledger[v][1] = True
        # 收尾再对一次
        res = mmu.sample_and_clear(asid)
        for v in pages:
            self.assertEqual(res.pages[v].accessed, ledger[v][0])
            self.assertEqual(res.pages[v].dirty, ledger[v][1])


if __name__ == "__main__":
    unittest.main()
