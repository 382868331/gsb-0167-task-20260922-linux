"""常量与输入校验测试。"""

from __future__ import annotations

import math
import unittest

from vmshoot.primitives import (
    HUGE_PAGE_SIZE,
    L1_SIZE,
    L2_SIZE,
    PAGE_SIZE,
    PERM_EXEC,
    PERM_READ,
    PERM_RWX,
    PERM_WRITE,
    VADDR_SPACE,
    split_vaddr,
    validate_int,
    validate_perms,
    validate_vaddr,
)


class TestConstants(unittest.TestCase):
    def test_fixed_geometry(self):
        self.assertEqual(PAGE_SIZE, 256)
        self.assertEqual(L1_SIZE, 16)
        self.assertEqual(L2_SIZE, 16)
        self.assertEqual(HUGE_PAGE_SIZE, 4096)
        self.assertEqual(VADDR_SPACE, 65536)

    def test_split_vaddr_examples(self):
        # 0x1abc -> L1=1, L2=10(0xa), off=0xbc
        self.assertEqual(split_vaddr(0x1ABC), (1, 10, 0xBC))
        self.assertEqual(split_vaddr(0x0000), (0, 0, 0))
        self.assertEqual(split_vaddr(0xFFFF), (15, 15, 255))
        self.assertEqual(split_vaddr(0x0300), (0, 3, 0))


class TestValidateInt(unittest.TestCase):
    def test_rejects_bool(self):
        with self.assertRaises(TypeError):
            validate_int(True, "x")
        with self.assertRaises(TypeError):
            validate_int(False, "x")

    def test_rejects_float_including_1_0(self):
        for v in (1.0, 2.5, -1.0):
            with self.assertRaises(TypeError):
                validate_int(v, "x")

    def test_rejects_nan_infinity_with_value_error(self):
        with self.assertRaises(ValueError):
            validate_int(math.nan, "x")
        with self.assertRaises(ValueError):
            validate_int(math.inf, "x")
        with self.assertRaises(ValueError):
            validate_int(-math.inf, "x")

    def test_rejects_str_none(self):
        with self.assertRaises(TypeError):
            validate_int("3", "x")
        with self.assertRaises(TypeError):
            validate_int(None, "x")

    def test_negative_and_upper_bound(self):
        with self.assertRaises(ValueError):
            validate_int(-1, "x")
        self.assertEqual(validate_int(0, "x"), 0)
        with self.assertRaises(ValueError):
            validate_int(11, "x", max_value=10)
        self.assertEqual(validate_int(10, "x", max_value=10), 10)

    def test_error_message_mentions_param_name(self):
        try:
            validate_int(True, "core_id")
        except TypeError as e:
            self.assertIn("core_id", str(e))
        else:
            self.fail("应当抛 TypeError")

    def test_validate_vaddr_bounds(self):
        self.assertEqual(validate_vaddr(0xFFFF), 0xFFFF)
        with self.assertRaises(ValueError):
            validate_vaddr(0x10000)

    def test_validate_perms(self):
        self.assertEqual(validate_perms(PERM_RWX), 7)
        with self.assertRaises(ValueError):
            validate_perms(0)
        with self.assertRaises(ValueError):
            validate_perms(8)  # 未知位
        with self.assertRaises(TypeError):
            validate_perms(True)


if __name__ == "__main__":
    unittest.main()
