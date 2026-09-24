"""protect_range（范围改权限）测试。

对照模型 :class:`FlatPerms` 是**独立**的逐小页权限/帧表参考实现：256 个
小页槽各存 ``[ppn, perms]``，另有一个“当前仍为整大页”的集合。它不调用
:mod:`vmshoot` 的页表/TLB 算法得到期望值（大页拆分对它只是清除一个标记，
与被测实现的两级页表结构完全不同），只复用题面常量与异常类型。
"""

from __future__ import annotations

import random
import unittest

from vmshoot import (
    MMU,
    AccessType,
    InvalidAddressSpace,
    InvalidationsPending,
    MappingConflict,
    MappingExists,
    NotMapped,
    PERM_EXEC,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PERM_WRITE,
    PhysicalMemory,
)

NUM_FRAMES = 256 * 4 + 64
NEED_OF = {AccessType.READ: PERM_READ, AccessType.WRITE: PERM_WRITE,
           AccessType.EXECUTE: PERM_EXEC}


class FlatPerms:
    """独立参考：逐小页 ``[ppn, perms]`` 表 + 自模拟帧池（首位适配）。"""

    def __init__(self, asid: int):
        self.asid = asid
        self.used = [False] * NUM_FRAMES
        self.small: list[list | None] = [None] * 256  # vpn -> [ppn, perms]
        self.huge: set[int] = set()                   # 仍保持整大页形式的 l1 下标

    # ---- 帧（独立实现首位适配） -------------------------------------------
    def _alloc1(self) -> int:
        for i, u in enumerate(self.used):
            if not u:
                self.used[i] = True
                return i
        raise AssertionError("ref: 帧耗尽")

    def _alloc16(self) -> int:
        for s in range(0, NUM_FRAMES - 15, 16):
            if not any(self.used[s + j] for j in range(16)):
                for j in range(16):
                    self.used[s + j] = True
                return s
        raise AssertionError("ref: 无法分配大页帧")

    # ---- 结构操作（镜像 MMU 公开语义，独立实现） --------------------------
    def map(self, va: int, perms: int, huge: bool):
        l1i, vpn = va >> 12, va >> 8
        if huge:
            if va & 0xFFF:
                raise ValueError("align")
            if l1i in self.huge or any(self.small[(l1i << 4) + j] for j in range(16)):
                raise MappingExists("ref")
            base = self._alloc16()
            for j in range(16):
                self.small[(l1i << 4) + j] = [base + j, perms]
            self.huge.add(l1i)
            return
        if va & 0xFF:
            raise ValueError("align")
        if l1i in self.huge or self.small[vpn] is not None:
            raise MappingExists("ref")
        self.small[vpn] = [self._alloc1(), perms]

    def unmap(self, va: int):
        l1i, vpn = va >> 12, va >> 8
        if l1i in self.huge:
            if va & 0xFFF:
                raise MappingConflict("ref huge interior")
            for j in range(16):
                self.used[self.small[(l1i << 4) + j][0]] = False
                self.small[(l1i << 4) + j] = None
            self.huge.discard(l1i)
            return
        ent = self.small[vpn]
        if ent is None:
            raise NotMapped("ref")
        self.used[ent[0]] = False
        self.small[vpn] = None

    def protect(self, va: int, perms: int):
        l1i, vpn = va >> 12, va >> 8
        if l1i in self.huge:
            self.huge.discard(l1i)  # 拆分：逐小页表本无结构变化
            self.small[vpn][1] = perms
            return
        ent = self.small[vpn]
        if ent is None:
            raise NotMapped("ref")
        ent[1] = perms

    def protect_range(self, s: int, e: int, perms: int) -> tuple:
        """返回期望的失效 key 序列（按被测约定的同样规则独立推导）。"""
        for vpn in range(s >> 8, e >> 8):
            if self.small[vpn] is None:
                raise NotMapped("ref hole")
        keys: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for l1i in range(s >> 12, ((e - 1) >> 12) + 1):
            region = l1i << 12
            lo, hi = max(s, region), min(e, region + 0x1000)
            if l1i in self.huge:
                if not (lo == region and hi == region + 0x1000):
                    self.huge.discard(l1i)  # 部分覆盖 -> 拆分
                cand = [(self.asid, (l1i << 4) + j) for j in range(16)]
            else:
                cand = [(self.asid, vpn) for vpn in range(lo >> 8, hi >> 8)]
            for k in cand:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        for vpn in range(s >> 8, e >> 8):
            self.small[vpn][1] = perms
        return tuple(keys)

    # ---- 期望值 -----------------------------------------------------------
    def expected(self, vpn: int):
        """``(ppn, perms, huge)`` 或 None。"""
        ent = self.small[vpn]
        if ent is None:
            return None
        return (ent[0], ent[1], (vpn >> 4) in self.huge)

    def used_frames(self) -> int:
        return sum(1 for u in self.used if u)


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        for core in (0, 1):
            mmu.ack(core, t.id)
        mmu.finish(t.id)


def settle_all(mmu: MMU, asid) -> None:
    for tid in mmu.pending_tickets(asid):
        for core in (0, 1):
            mmu.ack(core, tid)
        mmu.finish(tid)


class ProtectRangeBase(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU(PhysicalMemory(NUM_FRAMES))
        self.asid = self.mmu.create_address_space()
        self.ref = FlatPerms(self.asid)

    def tearDown(self):
        # 别名测试会手工装配共享帧，destroy 会重复释放；这里统一不销毁。
        pass

    # ---- 对照工具 ----------------------------------------------------------
    def assert_sweep_matches_ref(self):
        """全部 256 个小页：无 TLB 遍历结果必须等于独立逐小页表。"""
        for vpn in range(256):
            va = vpn * 256 + 5
            exp = self.ref.expected(vpn)
            wr = self.mmu.walk_table(self.asid, va)
            if exp is None:
                self.assertIsNone(wr, f"vpn={vpn:#x} 应为未映射")
            else:
                eppn, eperms, ehuge = exp
                self.assertIsNotNone(wr, f"vpn={vpn:#x} 应有映射")
                self.assertEqual(wr.perms, eperms, f"vpn={vpn:#x} 权限")
                self.assertEqual(wr.huge, ehuge, f"vpn={vpn:#x} 大页标记")
                self.assertEqual(wr.phys_addr, eppn * 256 + (va & 0xFF),
                                 f"vpn={vpn:#x} 物理地址")
        self.assertEqual(self.mmu.mem.used_frames(), self.ref.used_frames())

    def map_both(self, va, perms, *, huge=False):
        self.ref.map(va, perms, huge)
        t = self.mmu.map(self.asid, va, perms, huge=huge)
        settle(self.mmu, t)


class TestPartialHugeDowngrade(ProtectRangeBase):
    """题面指定的主场景：双核先缓存旧权限 -> 部分大页降权 -> 逐核 ACK ->
    最终拒绝写入。"""

    def test_two_core_partial_huge_downgrade(self):
        self.map_both(0x4000, PERM_RWX, huge=True)
        # 两个核先缓存旧权限（大页标记项）
        for core in (0, 1):
            r = self.mmu.translate(self.asid, core, 0x4600, AccessType.WRITE)
            self.assertTrue(r.ok)
        self.assertEqual(
            self.mmu.translate(self.asid, 0, 0x4600, AccessType.WRITE).via, "tlb")
        phys_before = self.mmu.walk_table(self.asid, 0x4600).phys_addr
        frames_before = self.mmu.mem.used_frames()

        # 部分大页降权：0x4500..0x4700（两页）rwx -> r--
        t = self.mmu.protect_range(self.asid, 0x4500, 0x4700, PERM_READ)
        self.assertEqual(t.kind, "protect_range")
        self.assertEqual(len(t.keys), 16)  # 部分覆盖大页：整区域 16 个 key
        self.ref.protect_range(0x4500, 0x4700, PERM_READ)

        # 页表立即生效：无 TLB 遍历器立刻看到新权限，大页已拆分
        wr = self.mmu.walk_table(self.asid, 0x4600)
        self.assertEqual((wr.perms, wr.huge), (PERM_READ, False))
        # 物理地址与帧占用不变
        self.assertEqual(self.mmu.walk_table(self.asid, 0x4600).phys_addr, phys_before)
        self.assertEqual(self.mmu.mem.used_frames(), frames_before)

        # ACK 前：双核旧 TLB 项仍放行写（陈旧翻译是协议允许的窗口）
        for core in (0, 1):
            r = self.mmu.translate(self.asid, core, 0x4600, AccessType.WRITE)
            self.assertTrue(r.ok, f"core{core} 旧 TLB 应仍放行")
            self.assertEqual(r.via, "tlb")
        with self.assertRaises(InvalidationsPending):
            self.mmu.finish(t.id)

        # 逐核确认：核0 ACK 后核0 立即拒绝，核1 仍旧放行
        self.mmu.ack(0, t.id)
        r0 = self.mmu.translate(self.asid, 0, 0x4600, AccessType.WRITE)
        self.assertEqual((r0.ok, r0.fault), (False, "write"))
        r1 = self.mmu.translate(self.asid, 1, 0x4600, AccessType.WRITE)
        self.assertTrue(r1.ok, "核1 未 ACK，旧项应仍放行")

        # 全部确认后：任何旧 TLB 权限都不能再放行被禁止的写
        self.mmu.ack(1, t.id)
        self.mmu.finish(t.id)
        for core in (0, 1):
            r = self.mmu.translate(self.asid, core, 0x4600, AccessType.WRITE)
            self.assertEqual((r.ok, r.fault), (False, "write"))
            r = self.mmu.translate(self.asid, core, 0x4600, AccessType.READ)
            self.assertTrue(r.ok, "只读降权后读仍应允许")

        # 区间外的同一大页邻居保持 rwx（且已随拆分变为小页形式）
        for va in (0x4000, 0x4400, 0x4700, 0x4F00):
            wr = self.mmu.walk_table(self.asid, va)
            self.assertEqual(wr.perms, PERM_RWX, f"va={va:#x} 邻居权限不应改变")
            self.assertFalse(wr.huge)
        self.assert_sweep_matches_ref()


class TestRangeShapes(ProtectRangeBase):
    def test_span_small_and_huge_boundary(self):
        """区间横跨小页区与大页区边界。"""
        self.map_both(0x3E00, PERM_RW)
        self.map_both(0x3F00, PERM_RW)
        self.map_both(0x4000, PERM_RWX, huge=True)
        frames_before = self.mmu.mem.used_frames()

        # 0x3f00..0x4200：1 个小页 + 大页头两页 -> 大页部分覆盖，须拆分
        keys = self.ref.protect_range(0x3F00, 0x4200, PERM_RX)
        t = self.mmu.protect_range(self.asid, 0x3F00, 0x4200, PERM_RX)
        self.assertEqual(t.keys, keys)  # 1 个小页 key + 大页区域 16 个 key
        self.assertEqual(len(t.keys), 17)
        settle(self.mmu, t)

        self.assertEqual(self.mmu.mem.used_frames(), frames_before)  # 不增帧
        self.assert_sweep_matches_ref()
        # 物理地址在拆分前后连续：抽查大页内偏移
        self.assertEqual(self.mmu.walk_table(self.asid, 0x4ABC).phys_addr & 0xFFF,
                         0xABC)

    def test_whole_huge_stays_huge(self):
        """整段大页统一改权限：保留大页形式，不拆分。"""
        self.map_both(0x4000, PERM_RWX, huge=True)
        keys = self.ref.protect_range(0x4000, 0x5000, PERM_RX)
        t = self.mmu.protect_range(self.asid, 0x4000, 0x5000, PERM_RX)
        self.assertEqual(t.keys, keys)
        self.assertEqual(len(t.keys), 16)
        settle(self.mmu, t)
        wr = self.mmu.walk_table(self.asid, 0x4800)
        self.assertTrue(wr.huge, "整段改权限应保留大页形式")
        self.assertEqual(wr.perms, PERM_RX)
        self.assertIn((0x4000, 4096, PERM_RX, True), self.mmu.mappings(self.asid))
        self.assert_sweep_matches_ref()

    def test_range_wider_than_huge_keeps_huge(self):
        """区间比大页还大（两侧是小页）：大页被整段覆盖，仍保留大页形式。"""
        self.map_both(0x3F00, PERM_RW)
        self.map_both(0x4000, PERM_RWX, huge=True)
        self.map_both(0x5000, PERM_RW)
        self.ref.protect_range(0x3F00, 0x5100, PERM_READ)
        t = self.mmu.protect_range(self.asid, 0x3F00, 0x5100, PERM_READ)
        settle(self.mmu, t)
        self.assertTrue(self.mmu.walk_table(self.asid, 0x4400).huge)
        self.assert_sweep_matches_ref()

    def test_full_address_space_range(self):
        """整个 64KiB 地址空间一次改权限（end 可取 0x10000）。"""
        for l1i in range(16):
            self.map_both(l1i * 0x1000, PERM_RWX, huge=True)
        self.ref.protect_range(0x0000, 0x10000, PERM_READ)
        t = self.mmu.protect_range(self.asid, 0x0000, 0x10000, PERM_READ)
        self.assertEqual(len(t.keys), 256)
        settle(self.mmu, t)
        for l1i in range(16):
            self.assertTrue(self.mmu.walk_table(self.asid, l1i * 0x1000).huge)
        self.assert_sweep_matches_ref()

    def test_write_restore(self):
        """解除写权限后再恢复。"""
        self.map_both(0x3000, PERM_RW)
        self.ref.protect_range(0x3000, 0x3100, PERM_READ)
        settle(self.mmu, self.mmu.protect_range(self.asid, 0x3000, 0x3100, PERM_READ))
        r = self.mmu.translate(self.asid, 0, 0x3000, AccessType.WRITE)
        self.assertEqual((r.ok, r.fault), (False, "write"))
        self.assertTrue(self.mmu.translate(self.asid, 0, 0x3000, AccessType.READ).ok)

        self.ref.protect_range(0x3000, 0x3100, PERM_RW)
        settle(self.mmu, self.mmu.protect_range(self.asid, 0x3000, 0x3100, PERM_RW))
        r = self.mmu.translate(self.asid, 0, 0x3000, AccessType.WRITE)
        self.assertTrue(r.ok, "恢复写权限后写应重新放行")
        self.assert_sweep_matches_ref()


class TestAlias(ProtectRangeBase):
    def test_alias_not_affected(self):
        """同一物理页在别处的别名映射不随 protect_range 改权限。"""
        self.map_both(0x1000, PERM_RW)
        self.map_both(0x2000, PERM_RW)
        # 测试专用装配：让 0x2000 成为 0x1000 同一物理帧的别名
        sp = self.mmu._spaces[self.asid]
        src, dst = sp.l1[0x1000 >> 12][0], sp.l1[0x2000 >> 12][0]
        self.assertNotEqual(src.ppn, dst.ppn)
        dst.ppn = src.ppn
        self.ref.small[0x20][0] = self.ref.small[0x10][0]

        pa = self.mmu.walk_table(self.asid, 0x1000).phys_addr
        pb = self.mmu.walk_table(self.asid, 0x2000).phys_addr
        self.assertEqual(pa, pb, "别名应解析到同一物理地址")

        self.ref.protect_range(0x1000, 0x1100, PERM_READ)
        settle(self.mmu, self.mmu.protect_range(self.asid, 0x1000, 0x1100, PERM_READ))

        self.assertEqual(self.mmu.walk_table(self.asid, 0x1000).perms, PERM_READ)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x2000).perms, PERM_RW,
                         "别名映射的权限不应被改动")
        r = self.mmu.translate(self.asid, 0, 0x2000, AccessType.WRITE)
        self.assertTrue(r.ok, "别名仍可写")
        self.assertEqual(r.phys_addr, pa, "别名物理地址不变")
        r = self.mmu.translate(self.asid, 0, 0x1000, AccessType.WRITE)
        self.assertEqual((r.ok, r.fault), (False, "write"))
        self.assert_sweep_matches_ref()


class TestFailureAtomicity(ProtectRangeBase):
    def test_hole_in_middle_no_side_effects(self):
        """区间中间有空洞：报错且页表/票据/TLB 全部不变。"""
        self.map_both(0x1000, PERM_RW)
        self.map_both(0x1400, PERM_RW)  # 0x1100..0x13ff 是空洞
        self.map_both(0x8000, PERM_RWX, huge=True)
        # 让双核 TLB 缓存一些项，验证失败调用连 TLB 都不碰
        for core in (0, 1):
            self.mmu.translate(self.asid, core, 0x1000, AccessType.READ)
            self.mmu.translate(self.asid, core, 0x8000, AccessType.READ)
        snap = (
            self.mmu.mappings(self.asid),
            self.mmu.mem.used_frames(),
            self.mmu.tlb_keys(0),
            self.mmu.tlb_keys(1),
            self.mmu.pending_tickets(),
        )
        with self.assertRaises(NotMapped):
            self.mmu.protect_range(self.asid, 0x1000, 0x1500, PERM_READ)
        self.assertEqual(snap, (
            self.mmu.mappings(self.asid),
            self.mmu.mem.used_frames(),
            self.mmu.tlb_keys(0),
            self.mmu.tlb_keys(1),
            self.mmu.pending_tickets(),
        ))
        # 空洞在第一个字节（整区域未映射）也同样无副作用
        with self.assertRaises(NotMapped):
            self.mmu.protect_range(self.asid, 0x5000, 0x6100, PERM_READ)
        self.assertEqual(snap[0], self.mmu.mappings(self.asid))
        self.assert_sweep_matches_ref()

    def test_invalid_args_no_side_effects(self):
        self.map_both(0x1000, PERM_RW)
        snap = (self.mmu.mappings(self.asid), self.mmu.pending_tickets())
        bad_calls = [
            (ValueError, (0x1001, 0x2000, PERM_READ)),        # start 未对齐
            (ValueError, (0x1000, 0x2001, PERM_READ)),        # end 未对齐
            (ValueError, (0x1000, 0x1000, PERM_READ)),        # 空区间
            (ValueError, (0x2000, 0x1000, PERM_READ)),        # 反向区间
            (ValueError, (0x1000, 0x10100, PERM_READ)),       # end 越界
            (ValueError, (0x10000, 0x10100, PERM_READ)),      # start 越界
            (ValueError, (0x1000, 0x1100, 0)),                # 权限为 0
            (ValueError, (0x1000, 0x1100, 0x8)),              # 未知权限位
            (TypeError, (True, 0x1100, PERM_READ)),           # bool start
            (TypeError, (0x1000, False, PERM_READ)),          # bool end
            (TypeError, (0x1000, 0x1100, 1.0)),               # float 权限
            (TypeError, (1.5, 0x1100, PERM_READ)),            # float start
            (ValueError, (float("nan"), 0x1100, PERM_READ)),  # NaN
            (ValueError, (0x1000, 0x1100, float("inf"))),     # Infinity 权限
        ]
        for exc, args in bad_calls:
            with self.subTest(args=args):
                with self.assertRaises(exc):
                    self.mmu.protect_range(self.asid, *args)
        with self.assertRaises(InvalidAddressSpace):
            self.mmu.protect_range(99, 0x1000, 0x1100, PERM_READ)
        self.assertEqual(snap, (self.mmu.mappings(self.asid), self.mmu.pending_tickets()))
        self.assert_sweep_matches_ref()


class TestRandomAgainstFlatOracle(ProtectRangeBase):
    """固定种子随机操作序列：被测 MMU vs 独立逐小页表。"""

    def test_five_fixed_seeds(self):
        for seed in (7, 42, 20260925, 31337, 999983):
            with self.subTest(seed=seed):
                self._run_one(seed, 300)

    def _run_one(self, seed: int, steps: int):
        rng = random.Random(seed)
        for step in range(steps):
            r = rng.random()
            with self.subTest(step=step):
                if r < 0.25:
                    va = rng.randrange(256) * 256
                    perms = rng.choice((PERM_READ, PERM_RW, PERM_RWX, PERM_RX))
                    self._op(lambda: self.ref.map(va, perms, False),
                             lambda: self.mmu.map(self.asid, va, perms))
                elif r < 0.35:
                    va = rng.randrange(16) * 4096
                    perms = rng.choice((PERM_READ, PERM_RW, PERM_RWX, PERM_RX))
                    self._op(lambda: self.ref.map(va, perms, True),
                             lambda: self.mmu.map(self.asid, va, perms, huge=True))
                elif r < 0.45:
                    va = rng.randrange(256) * 256
                    self._op(lambda: self.ref.unmap(va),
                             lambda: self.mmu.unmap(self.asid, va))
                elif r < 0.55:
                    va = rng.randrange(256) * 256
                    perms = rng.choice((PERM_READ, PERM_RW, PERM_RWX, PERM_RX))
                    self._op(lambda: self.ref.protect(va, perms),
                             lambda: self.mmu.protect(self.asid, va, perms))
                elif r < 0.85:
                    a = rng.randrange(256) * 256
                    b = rng.randrange(256) * 256
                    s, e = min(a, b), max(a, b)
                    if s == e:
                        e = min(s + 256, 0x10000)
                        if s == e:
                            s = e - 256
                    perms = rng.choice((PERM_READ, PERM_RW, PERM_RWX, PERM_RX,
                                        PERM_WRITE))
                    self._op_range(s, e, perms)
                else:
                    self._spot_translate(rng)
            settle_all(self.mmu, self.asid)
            self.assert_sweep_matches_ref()

    # ---- 操作执行：先参考机后被测，异常类型必须一致 -------------------------
    def _op(self, ref_fn, sut_fn):
        try:
            ref_fn()
        except Exception as exc:
            with self.assertRaises(type(exc)):
                sut_fn()
            return
        t = sut_fn()
        settle(self.mmu, t)

    def _op_range(self, s, e, perms):
        try:
            keys = self.ref.protect_range(s, e, perms)
        except Exception as exc:
            with self.assertRaises(type(exc)):
                self.mmu.protect_range(self.asid, s, e, perms)
            return
        t = self.mmu.protect_range(self.asid, s, e, perms)
        self.assertEqual(t.kind, "protect_range")
        self.assertEqual(t.keys, keys, f"range=[{s:#x},{e:#x}) 失效 key 不一致")
        settle(self.mmu, t)

    def _spot_translate(self, rng):
        """全部票据已 settle，TLB 与页表一致，可与逐小页表直接对照。"""
        va = rng.randrange(0x10000)
        acc = rng.choice((AccessType.READ, AccessType.WRITE, AccessType.EXECUTE))
        core = rng.randrange(2)
        ent = self.ref.small[va >> 8]
        r = self.mmu.translate(self.asid, core, va, acc)
        if ent is None:
            self.assertEqual((r.ok, r.fault), (False, "not_mapped"))
            return
        if ent[1] & NEED_OF[acc]:
            self.assertTrue(r.ok, f"va={va:#x} {acc.name} 应成功")
            self.assertEqual(r.phys_addr, ent[0] * 256 + (va & 0xFF))
        else:
            self.assertEqual((r.ok, r.fault), (False, acc.name.lower()))


if __name__ == "__main__":
    unittest.main()
