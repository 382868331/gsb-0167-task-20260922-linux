"""大页拆分（局部改权限/解除必须保留其他小页）、跨页访问拒绝、原子性。"""

from __future__ import annotations

import unittest

import asid_vm as vm
from tests.reference import RefSpace
from tests._helpers import assert_ref_matches, settle


class TestHugeSplit(unittest.TestCase):
    def test_partial_protect_splits_and_preserves_others(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        # 大页 0x2000-0x2FFF -> 0x20000，rwx
        t, _ = sys.map(asid, 0x2000, (True, True, True), huge=True,
                       phys=0x20000)
        settle(sys, t)
        # 中间两小页（0x2400, 0x2500）改为只读
        ref = RefSpace()
        ref.map_huge(0x2000, 0x20000, (True, True, True))
        pt = sys.protect(asid, 0x2400, (True, False, False), length=512)
        ref.protect(0x2400, 512, (True, False, False))
        settle(sys, pt)

        # 其他 14 小页：物理地址与原权限原样保留
        for j in range(16):
            va = 0x2000 + j * 256
            w = sys.direct_walk(asid, va)
            self.assertFalse(w.huge, f"页 {j} 应已拆成小页")
            self.assertEqual(w.phys_base, 0x20000 + j * 256)
            if j in (4, 5):
                self.assertEqual(
                    (w.perms.readable, w.perms.writable, w.perms.executable),
                    (True, False, False))
            else:
                self.assertEqual(
                    (w.perms.readable, w.perms.writable, w.perms.executable),
                    (True, True, True))
        assert_ref_matches(self, sys, asid, ref)

    def test_partial_unmap_splits_and_preserves_frame(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x3000, huge=True, phys=0x40000)
        settle(sys, t)
        self.assertEqual(sys.used_phys_bytes(), 4096)
        # 解除首小页：其余 15 页保留，4096 帧不归还
        ut = sys.unmap(asid, 0x3000, 256)
        settle(sys, ut)
        self.assertEqual(sys.used_phys_bytes(), 4096)
        with self.assertRaises(vm.NotPresentFault):
            sys.direct_walk(asid, 0x3000)
        w = sys.direct_walk(asid, 0x3100)
        self.assertEqual(w.phys_base, 0x40100)
        # 再解除其余 15 页后整帧归还
        ut2 = sys.unmap(asid, 0x3100, 256 * 15)
        settle(sys, ut2)
        self.assertEqual(sys.used_phys_bytes(), 0)

    def test_whole_huge_protect_keeps_leaf(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x5000, huge=True, phys=0x50000)
        settle(sys, t)
        pt = sys.protect(asid, 0x5000, (False, False, True), length=4096)
        settle(sys, pt)
        w = sys.direct_walk(asid, 0x5ABC)
        self.assertTrue(w.huge)
        self.assertEqual((w.perms.readable, w.perms.writable,
                          w.perms.executable), (False, False, True))

    def test_whole_huge_unmap_returns_frame(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x6000, huge=True, phys=0x60000)
        settle(sys, t)
        ut = sys.unmap(asid, 0x6000, 4096)
        settle(sys, ut)
        self.assertEqual(sys.used_phys_bytes(), 0)
        with self.assertRaises(vm.NotPresentFault):
            sys.direct_walk(asid, 0x6000)

    def test_unmap_hole_fails_atomically(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x7000, phys=0x70000)
        settle(sys, t)
        t, _ = sys.map(asid, 0x7200, phys=0x70200)
        settle(sys, t)
        with self.assertRaises(vm.NotMappedError):
            sys.unmap(asid, 0x7000, 3 * 256)  # 0x7100 是空洞
        # 失败无部分变更：两张小页仍在
        self.assertEqual(sys.direct_walk(asid, 0x7000).phys_base, 0x70000)
        self.assertEqual(sys.direct_walk(asid, 0x7200).phys_base, 0x70200)

    def test_mixed_layout_matches_reference(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        ref = RefSpace()
        t, _ = sys.map(asid, 0x8000, huge=True, phys=0x80000)
        settle(sys, t)
        ref.map_huge(0x8000, 0x80000, (True, True, True))
        # 局部保护 + 局部解除交错
        pt = sys.protect(asid, 0x8300, (True, False, True), length=256)
        ref.protect(0x8300, 256, (True, False, True))
        settle(sys, pt)
        ut = sys.unmap(asid, 0x8F00, 256)
        ref.unmap(0x8F00, 256)
        settle(sys, ut)
        pt2 = sys.protect(asid, 0x8000, (False, True, False), length=256)
        ref.protect(0x8000, 256, (False, True, False))
        settle(sys, pt2)
        assert_ref_matches(self, sys, asid, ref)


class TestCrossPage(unittest.TestCase):
    def test_cross_small_page_rejected_even_inside_huge(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x1000, huge=True, phys=0x90000)
        settle(sys, t)
        # 起始 0x10F0 长度 32 -> 跨到 0x110F；即使都在同一大页内也拒绝
        with self.assertRaises(vm.CrossPageFault) as cm:
            sys.check_access(0, asid, 0x10F0, 32, vm.Access.READ)
        self.assertEqual(cm.exception.va, 0x10F0)
        # 恰好落在页内（结束于页内最后一字节）允许
        self.assertEqual(
            sys.check_access(0, asid, 0x10F0, 16, vm.Access.READ), 0x900F0)
        # 长度 1 总是允许
        self.assertEqual(
            sys.check_access(0, asid, 0x10FF, 1, vm.Access.EXECUTE), 0x900FF)

    def test_cross_page_permission_fault_still_raised(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0000, (False, True, True), phys=0xA0000)
        settle(sys, t)
        with self.assertRaises(vm.ReadFault):
            sys.check_access(0, asid, 0x0000, 1, vm.Access.READ)

    def test_cross_page_unmapped_raises_not_present(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        with self.assertRaises(vm.NotPresentFault):
            sys.check_access(0, asid, 0x00F0, 4, vm.Access.READ)

    def test_cross_page_validation(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        with self.assertRaises(vm.VmValueError):
            sys.check_access(0, asid, 0, 0, vm.Access.READ)
        with self.assertRaises(vm.VmTypeError):
            sys.check_access(0, asid, 0, True, vm.Access.READ)


class TestRangeValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.sys = vm.MemorySystem()
        self.asid = self.sys.alloc_asid()

    def test_length_must_be_multiple_of_page(self) -> None:
        with self.assertRaises(vm.AlignmentError):
            self.sys.unmap(self.asid, 0x0000, 100)
        with self.assertRaises(vm.AlignmentError):
            self.sys.protect(self.asid, 0x0000, (True, True, True), length=1)

    def test_range_cannot_cross_l1_region(self) -> None:
        with self.assertRaises(vm.AddressRangeError):
            self.sys.unmap(self.asid, 0x0F00, 512)  # 跨 0x1000

    def test_negative_and_oversize_length(self) -> None:
        with self.assertRaises(vm.VmValueError):
            self.sys.unmap(self.asid, 0x0000, -256)
        with self.assertRaises(vm.AddressRangeError):
            self.sys.unmap(self.asid, 0xFF00, 512)


if __name__ == "__main__":
    unittest.main()
