"""基本映射/翻译、TLB 填充与 LRU、输入校验。"""

from __future__ import annotations

import math
import unittest

import asid_vm as vm
from tests.reference import RefSpace
from tests._helpers import assert_ref_matches, settle


class TestMapTranslate(unittest.TestCase):
    def test_small_map_and_translate(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        ticket, pa = sys.map(asid, 0x0000, phys=0x1000)
        settle(sys, ticket)
        self.assertEqual(pa, 0x1000)
        # 冷核、热核（TLB）两种路径都应得到同一物理地址
        self.assertEqual(sys.translate(0, asid, 0x0000, vm.Access.READ), 0x1000)
        self.assertEqual(sys.translate(0, asid, 0x00FF, vm.Access.READ), 0x10FF)
        self.assertEqual(sys.translate(1, asid, 0x0080, vm.Access.WRITE), 0x1080)
        self.assertIn((asid, 0x0000), sys.tlb_keys(0))

    def test_huge_page_translation(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        ticket, pa = sys.map(asid, 0x1000, huge=True, phys=0x20000)
        settle(sys, ticket)
        self.assertEqual(pa, 0x20000)
        for j in range(16):
            va = 0x1000 + j * 256
            self.assertEqual(sys.translate(0, asid, va, vm.Access.READ),
                             0x20000 + j * 256)
            self.assertEqual(sys.translate(0, asid, va + 255, vm.Access.EXECUTE),
                             0x20000 + j * 256 + 255)
        # 大页 TLB 只有一项（key 按小页页号，每小页首次访问会各自填充）
        self.assertGreaterEqual(len(sys.tlb_keys(0)), 1)

    def test_auto_allocated_phys_are_aligned_and_distinct(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        tickets = []
        pas = set()
        for va in (0x0000, 0x0100, 0x4000):
            t, pa = sys.map(asid, va)
            tickets.append(t)
            self.assertEqual(pa % 256, 0)
            pas.add(pa)
        t, pa = sys.map(asid, 0x8000, huge=True)
        tickets.append(t)
        self.assertEqual(pa % 4096, 0)
        pas.add(pa)
        self.assertEqual(len(pas), 4)
        for t in tickets:
            settle(sys, t)

    def test_map_conflict_is_atomic(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0200, phys=0x3000)
        settle(sys, t)
        before = sys.used_phys_bytes()
        with self.assertRaises(vm.MappingExistsError):
            sys.map(asid, 0x0200)
        with self.assertRaises(vm.MappingExistsError):
            sys.map(asid, 0x0200, huge=False, phys=0x9000)
        # 失败未留下物理帧
        self.assertEqual(sys.used_phys_bytes(), before)
        # 大页与已有小页冲突
        with self.assertRaises(vm.MappingExistsError):
            sys.map(asid, 0x0000, huge=True)

    def test_external_frame_conflict_is_atomic(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0000, phys=0x1000)
        settle(sys, t)
        before = sys.used_phys_bytes()
        # 新一级区域（将触发新建二级表），但物理帧重叠 -> 失败且无空表残留
        with self.assertRaises(vm.MappingExistsError):
            sys.map(asid, 0x1000, phys=0x1000)
        self.assertEqual(sys.used_phys_bytes(), before)
        with self.assertRaises(vm.NotPresentFault):
            sys.direct_walk(asid, 0x1000)
        # 随后在该区域的正常映射仍可成功
        t2, pa = sys.map(asid, 0x1000, phys=0x2000)
        settle(sys, t2)
        self.assertEqual(pa, 0x2000)

    def test_not_present_faults(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        for access in vm.Access:
            with self.assertRaises(vm.NotPresentFault) as cm:
                sys.translate(0, asid, 0x1234, access)
            self.assertEqual(cm.exception.va, 0x1234)
            self.assertEqual(cm.exception.access, access)


class TestPermissions(unittest.TestCase):
    def test_rwx_faults_match_reference(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        t, _ = sys.map(asid, 0x0500, (True, False, False), phys=0x4000)
        settle(sys, t)
        self.assertEqual(sys.translate(0, asid, 0x0500, vm.Access.READ), 0x4000)
        with self.assertRaises(vm.WriteFault):
            sys.translate(0, asid, 0x0500, vm.Access.WRITE)
        with self.assertRaises(vm.ExecuteFault):
            sys.translate(0, asid, 0x05FF, vm.Access.EXECUTE)

        t = sys.protect(asid, 0x0500, (False, True, False))
        # ack 前旧 TLB（可读快照）仍允许读 —— 陈旧窗口
        self.assertEqual(sys.translate(0, asid, 0x0501, vm.Access.READ), 0x4001)
        settle(sys, t)
        with self.assertRaises(vm.ReadFault):
            sys.translate(0, asid, 0x0501, vm.Access.READ)
        self.assertEqual(sys.translate(0, asid, 0x0501, vm.Access.WRITE), 0x4001)

    def test_full_table_matches_reference(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        ref = RefSpace()
        layout = [
            (0x0000, False, 0x10000, (True, True, True)),
            (0x0300, False, 0x10400, (True, False, True)),
            (0x0F00, False, 0x10800, (False, False, True)),
            (0x4000, True, 0x20000, (True, True, False)),
        ]
        tickets = []
        for va, huge, phys, perms in layout:
            t, got = sys.map(asid, va, perms, huge=huge, phys=phys)
            tickets.append(t)
            if huge:
                ref.map_huge(va, phys, perms)
            else:
                ref.map_small(va, phys, perms)
        for t in tickets:
            settle(sys, t)
        assert_ref_matches(self, sys, asid, ref)


class TestTLBLRU(unittest.TestCase):
    def test_lru_eviction_eight_entries(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        tickets = []
        for i in range(8):
            va = i * 0x1000  # 不同一级区域，避免互相覆盖
            t, pa = sys.map(asid, va, phys=0x50000 + i * 256)
            tickets.append(t)
            sys.translate(0, asid, va, vm.Access.READ)  # 填充 TLB
        for t in tickets:
            settle(sys, t)
        self.assertEqual(len(sys.tlb_keys(0)), 8)
        # 访问第 9 个页 -> LRU 队首（va=0）被淘汰
        t, _ = sys.map(asid, 0x9000, phys=0x51000)
        settle(sys, t)
        sys.translate(0, asid, 0x9000, vm.Access.READ)
        keys = sys.tlb_keys(0)
        self.assertEqual(len(keys), 8)
        self.assertNotIn((asid, 0x0000), keys)
        self.assertIn((asid, 0x9000), keys)

    def test_lru_touch_refreshes_order(self) -> None:
        sys = vm.MemorySystem()
        asid = sys.alloc_asid()
        tickets = []
        for i in range(8):
            va = (i + 2) * 0x1000
            t, _ = sys.map(asid, va, phys=0x60000 + i * 256)
            tickets.append(t)
            sys.translate(0, asid, va, vm.Access.READ)
        # 触碰最老的一项，它应升到队尾而不在下一次淘汰中消失
        sys.translate(0, asid, 0x2000, vm.Access.READ)
        t, _ = sys.map(asid, 0xA000, phys=0x61000)
        settle(sys, t)
        sys.translate(0, asid, 0xA000, vm.Access.READ)
        keys = sys.tlb_keys(0)
        self.assertIn((asid, 0x2000), keys)
        self.assertNotIn((asid, 0x3000), keys)  # 原次老项被淘汰
        for t in tickets:
            settle(sys, t)

    def test_asid_is_part_of_tlb_key(self) -> None:
        sys = vm.MemorySystem()
        a0 = sys.alloc_asid()
        a1 = sys.alloc_asid()
        t0, _ = sys.map(a0, 0x0000, phys=0x70000)
        t1, _ = sys.map(a1, 0x0000, phys=0x71000)
        settle(sys, t0)
        settle(sys, t1)
        self.assertEqual(sys.translate(0, a0, 0x0000, vm.Access.READ), 0x70000)
        self.assertEqual(sys.translate(0, a1, 0x0000, vm.Access.READ), 0x71000)
        keys = sys.tlb_keys(0)
        self.assertIn((a0, 0x0000), keys)
        self.assertIn((a1, 0x0000), keys)


class TestInputValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.sys = vm.MemorySystem()
        self.asid = self.sys.alloc_asid()

    def test_bool_rejected_as_int(self) -> None:
        with self.assertRaises(vm.VmTypeError):
            self.sys.translate(True, self.asid, 0, vm.Access.READ)
        with self.assertRaises(vm.VmTypeError):
            self.sys.map(self.asid, False)

    def test_nan_infinity_rejected(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(vm.VmTypeError):
                self.sys.translate(0, self.asid, bad, vm.Access.READ)
            with self.assertRaises(vm.VmTypeError):
                self.sys.map(self.asid, bad)

    def test_out_of_range(self) -> None:
        with self.assertRaises(vm.AddressRangeError):
            self.sys.translate(0, self.asid, 0x10000, vm.Access.READ)
        with self.assertRaises(vm.AddressRangeError):
            self.sys.translate(0, self.asid, -1, vm.Access.READ)
        with self.assertRaises(vm.AddressRangeError):
            self.sys.translate(2, self.asid, 0, vm.Access.READ)

    def test_alignment(self) -> None:
        with self.assertRaises(vm.AlignmentError):
            self.sys.map(self.asid, 0x1)
        with self.assertRaises(vm.AlignmentError):
            self.sys.map(self.asid, 0x100, huge=True)

    def test_unknown_asid(self) -> None:
        with self.assertRaises(vm.UnknownAsidError):
            self.sys.translate(0, 9, 0, vm.Access.READ)

    def test_bad_permissions(self) -> None:
        with self.assertRaises(vm.VmTypeError):
            self.sys.map(self.asid, 0x0000, (True, True))
        with self.assertRaises(vm.VmTypeError):
            self.sys.map(self.asid, 0x0100, "rwx")

    def test_bad_access_enum(self) -> None:
        with self.assertRaises(vm.VmTypeError):
            self.sys.translate(0, self.asid, 0, "read")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
