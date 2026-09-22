"""ASID 生命周期、失效票据状态机与跨核陈旧 TLB。"""

from __future__ import annotations

import unittest

import asid_vm as vm
from tests._helpers import settle


class TestAsidReuse(unittest.TestCase):
    def test_alloc_min_and_limit(self) -> None:
        sys = vm.MemorySystem()
        ids = [sys.alloc_asid() for _ in range(vm.MAX_ASIDS)]
        self.assertEqual(ids, [0, 1, 2, 3])
        with self.assertRaises(vm.AsidExhaustedError):
            sys.alloc_asid()

    def test_reuse_after_destroy_finished(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()  # 0
        for _ in range(3):
            sys.alloc_asid()
        t, _ = sys.map(asid, 0x0000, phys=0x1000)
        # 制造陈旧项，使销毁票据需要核 0 ack
        sys.translate(0, asid, 0x0000, vm.Access.READ)
        settle(sys, t)
        dt = sys.free_asid(asid)
        # finish 前：编号不可复用
        with self.assertRaises(vm.AsidExhaustedError):
            sys.alloc_asid()
        sys.ack(0, dt)
        sys.finish(dt)
        # finish 后：编号 0 被重新分配，且是干净空间
        reused = sys.alloc_asid()
        self.assertEqual(reused, 0)
        with self.assertRaises(vm.NotPresentFault):
            sys.translate(0, reused, 0x0000, vm.Access.READ)
        self.assertEqual(sys.used_phys_bytes(), 0)

    def test_reuse_allowed_without_stale_entries(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0000)
        sys.finish(t)  # 无目标核，可立即 finish
        dt = sys.free_asid(asid)
        sys.finish(dt)
        self.assertEqual(sys.alloc_asid(), asid)

    def test_destroy_blocked_by_outstanding_ticket(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0000, phys=0x2000)
        sys.translate(0, asid, 0x0000, vm.Access.READ)
        # t 未 finish（核 0 还挂着）
        with self.assertRaises(vm.PendingFinishError):
            sys.free_asid(asid)
        settle(sys, t)

    def test_reused_asid_tlb_key_not_poisoned(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0100, phys=0x3000)
        sys.translate(0, asid, 0x0100, vm.Access.READ)
        settle(sys, t)  # ack 清掉核 0 的陈旧项
        dt = sys.free_asid(asid)
        settle(sys, dt)
        new = sys.alloc_asid()
        t2, pa2 = sys.map(new, 0x0100, phys=0x4000)
        settle(sys, t2)
        self.assertEqual(sys.translate(0, new, 0x0100, vm.Access.READ), 0x4000)


class TestTicketLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.sys = vm.MemorySystem()
        self.asid = self.sys.alloc_asid()

    def test_finish_waits_for_all_target_cores(self) -> None:
        t, _ = self.sys.map(self.asid, 0x0000, phys=0x5000)
        self.sys.translate(0, self.asid, 0x0000, vm.Access.READ)
        self.sys.translate(1, self.asid, 0x0000, vm.Access.READ)
        settle(self.sys, t)
        # 现在两核都缓存了 0x0000 -> 0x5000
        pt = self.sys.protect(self.asid, 0x0000, (False, False, False))
        self.assertEqual(pt.target_cores, frozenset({0, 1}))
        # ack 前：finish 必须等待
        with self.assertRaises(vm.PendingFinishError):
            self.sys.finish(pt)
        self.sys.ack(0, pt)
        with self.assertRaises(vm.PendingFinishError):
            self.sys.finish(pt)
        self.sys.ack(1, pt)
        self.sys.finish(pt)  # 不再等待

    def test_ack_rules(self) -> None:
        t, _ = self.sys.map(self.asid, 0x0000, phys=0x6000)
        self.sys.translate(0, self.asid, 0x0000, vm.Access.READ)
        settle(self.sys, t)
        ut = self.sys.unmap(self.asid, 0x0000)
        self.assertEqual(ut.target_cores, frozenset({0}))
        # 非目标核 ack 被拒绝
        with self.assertRaises(vm.NotTicketTargetError):
            self.sys.ack(1, ut)
        self.sys.ack(0, ut)
        # 重复 ack
        with self.assertRaises(vm.AlreadyAckedError):
            self.sys.ack(0, ut)
        self.sys.finish(ut)
        # finish 后票据不可再用
        with self.assertRaises(vm.UnknownTicketError):
            self.sys.ack(0, ut)
        with self.assertRaises(vm.UnknownTicketError):
            self.sys.finish(ut)

    def test_finished_ticket_releases_asid(self) -> None:
        a0 = self.asid
        for _ in range(vm.MAX_ASIDS - 1):
            self.sys.alloc_asid()
        t, _ = self.sys.map(a0, 0x0000, phys=0x7000)
        self.sys.translate(1, a0, 0x0000, vm.Access.READ)
        settle(self.sys, t)
        dt = self.sys.free_asid(a0)
        with self.assertRaises(vm.AsidExhaustedError):
            self.sys.alloc_asid()
        self.sys.ack(1, dt)
        self.sys.finish(dt)
        self.assertEqual(self.sys.alloc_asid(), a0)


class TestStaleEntries(unittest.TestCase):
    def test_stale_entry_served_before_ack_removed_after(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0000, (True, True, True), phys=0x8000)
        sys.translate(0, asid, 0x0000, vm.Access.READ)
        settle(sys, t)
        # 页表改为只读，票据指向核 0
        pt = sys.protect(asid, 0x0000, (True, False, False))
        # ack 前：核 0 陈旧项仍允许写
        self.assertEqual(sys.translate(0, asid, 0x0010, vm.Access.WRITE),
                         0x8010)
        self.assertIn((asid, 0x0000), sys.tlb_keys(0))
        sys.ack(0, pt)
        # ack 后：陈旧项消失，写被页表权限拒绝
        self.assertNotIn((asid, 0x0000), sys.tlb_keys(0))
        with self.assertRaises(vm.WriteFault):
            sys.translate(0, asid, 0x0010, vm.Access.WRITE)
        sys.finish(pt)

    def test_other_core_keeps_its_own_stale_entry(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0000, phys=0x9000)
        sys.translate(0, asid, 0x0000, vm.Access.READ)
        sys.translate(1, asid, 0x0000, vm.Access.READ)
        settle(sys, t)
        ut = sys.unmap(asid, 0x0000)
        self.assertEqual(ut.target_cores, frozenset({0, 1}))
        sys.ack(0, ut)
        # 核 0 已失效：NotPresent；核 1 仍陈旧命中
        with self.assertRaises(vm.NotPresentFault):
            sys.translate(0, asid, 0x0000, vm.Access.READ)
        self.assertEqual(sys.translate(1, asid, 0x0000, vm.Access.READ), 0x9000)
        sys.ack(1, ut)
        with self.assertRaises(vm.NotPresentFault):
            sys.translate(1, asid, 0x0000, vm.Access.READ)
        sys.finish(ut)

    def test_different_asids_do_not_shoot_each_other(self) -> None:
        sys = vm.MemorySystem()
        a0 = sys.alloc_asid()
        a1 = sys.alloc_asid()
        t0, _ = sys.map(a0, 0x0000, phys=0xA000)
        t1, _ = sys.map(a1, 0x0000, phys=0xB000)
        settle(sys, t0)
        settle(sys, t1)
        sys.translate(0, a0, 0x0000, vm.Access.READ)
        sys.translate(0, a1, 0x0000, vm.Access.READ)
        ut = sys.unmap(a0, 0x0000)
        # 票据只影响 a0 的 key；a1 的缓存必须保留
        self.assertEqual(ut.target_cores, frozenset({0}))
        sys.ack(0, ut)
        sys.finish(ut)
        self.assertNotIn((a0, 0x0000), sys.tlb_keys(0))
        self.assertIn((a1, 0x0000), sys.tlb_keys(0))
        self.assertEqual(sys.translate(0, a1, 0x0000, vm.Access.READ), 0xB000)


if __name__ == "__main__":
    unittest.main()
