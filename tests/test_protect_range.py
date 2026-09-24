"""protect_range（ASID + 半开区间范围权限修改）测试。

期望值来自一个**独立的逐小页参考模型** :class:`PageOracle`：它直接维护
``vpn -> (ppn, perms)`` 的小页权限表与物理帧占用表，不读取、不查询被测页表
（不用 mmu.walk_table / mmu.mappings 来产生期望值；那些只作为补充交叉核对）。

覆盖：

* 正常路径：纯小页区间、整段大页（保留大页形式）、部分大页降权（拆分）；
* 跨大小页边界；解除写权限后恢复；同一物理帧的不同别名不连带改权限；
* 区间中间有空洞时无任何副作用（页表、TLB、票据均不变）；
* 非法参数；票据协议：逐核 ACK、finish 阻塞、全部确认后旧 TLB 权限不再放行；
* 演示场景：两核先缓存旧权限 -> 部分大页降权 -> 逐核确认 -> 最终拒绝写入。
"""

from __future__ import annotations

import unittest

from vmshoot import (
    MMU,
    AccessType,
    InvalidationsPending,
    NotMapped,
    PAGE_SIZE,
    PERM_EXEC,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PERM_WRITE,
    TicketError,
    VMShootError,
    perm_to_str,
)

NUM_SMALL_PAGES = 256          # 16 位虚址 / 256
PAGES_PER_HUGE = 16


class PageOracle:
    """独立逐小页真相表：vpn -> [ppn, perms]，外加已用 ppn 集合。

    大页以 16 个 vpn 共享同一批连续 ppn 表示；它自己决定帧号（独立的首位
    适配扫描），与被测实现的帧分配规则一致但代码完全独立。
    """

    def __init__(self):
        self.pages: dict[int, list] = {}
        self.used: set[int] = set()

    def alloc_small(self) -> int:
        ppn = 0
        while ppn in self.used:
            ppn += 1
        self.used.add(ppn)
        return ppn

    def alloc_huge(self) -> int:
        base = 0
        while any(base + j in self.used for j in range(PAGES_PER_HUGE)):
            base += PAGES_PER_HUGE
        for j in range(PAGES_PER_HUGE):
            self.used.add(base + j)
        return base

    def map_small(self, va: int, perms: int) -> None:
        vpn = va >> 8
        assert vpn not in self.pages
        self.pages[vpn] = [self.alloc_small(), perms]

    def map_huge(self, va: int, perms: int) -> None:
        base_vpn = va >> 8
        base_ppn = self.alloc_huge()
        for j in range(PAGES_PER_HUGE):
            assert base_vpn + j not in self.pages
            self.pages[base_vpn + j] = [base_ppn + j, perms]

    def unmap_small(self, va: int) -> None:
        vpn = va >> 8
        self.used.discard(self.pages.pop(vpn)[0])

    def unmap_huge(self, va: int) -> None:
        base_vpn = va >> 8
        for j in range(PAGES_PER_HUGE):
            self.used.discard(self.pages.pop(base_vpn + j)[0])

    # ---- 被测操作的独立镜像 -------------------------------------------------
    def check_range(self, start: int, end: int) -> None:
        """与 protect_range 相同的预检语义：区间内每个小页必须已映射。"""
        for vpn in range(start >> 8, end >> 8):
            if vpn not in self.pages:
                raise NotMapped(f"oracle hole at vpn {vpn}")

    def protect_range(self, start: int, end: int, perms: int) -> None:
        self.check_range(start, end)
        for vpn in range(start >> 8, end >> 8):
            self.pages[vpn][1] = perms

    def expected_perms(self, va: int) -> int:
        return self.pages[va >> 8][1]

    def expected_ppn(self, va: int) -> int:
        return self.pages[va >> 8][0]


def settle(mmu: MMU, ticket) -> None:
    mmu.ack(0, ticket.id)
    mmu.ack(1, ticket.id)
    mmu.finish(ticket.id)


def settle_all(mmu: MMU) -> None:
    for tid in list(mmu.pending_tickets()):
        mmu.ack(0, tid)
        mmu.ack(1, tid)
        mmu.finish(tid)


def map_small_settled(mmu, asid, oracle: PageOracle, va: int, perms: int):
    oracle.map_small(va, perms)
    settle(mmu, mmu.map(asid, va, perms))


def map_huge_settled(mmu, asid, oracle: PageOracle, va: int, perms: int):
    oracle.map_huge(va, perms)
    settle(mmu, mmu.map(asid, va, perms, huge=True))


class _OracleCase(unittest.TestCase):
    """带独立参考表的公共断言工具。"""

    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()
        self.ora = PageOracle()

    def assertTableMatchesOracle(self):
        """逐小页对照：walk_table（小页展开视角）必须与独立权限表一致。

        注意：这里的期望值取自 PageOracle，被测页表只负责被核对，不产生期望。
        """
        for vpn in range(NUM_SMALL_PAGES):
            va = vpn * PAGE_SIZE
            wr = self.mmu.walk_table(self.asid, va)
            if vpn not in self.ora.pages:
                self.assertIsNone(wr, f"vpn={vpn} 应为空洞")
                continue
            self.assertIsNotNone(wr, f"vpn={vpn} 应已映射")
            exp_ppn, exp_perms = self.ora.pages[vpn]
            # 大页的 WalkResult.ppn 是区域基帧号；小页则是该页自己的帧号
            if wr.huge:
                self.assertEqual(wr.ppn, exp_ppn & ~15, f"vpn={vpn} 大页基帧号")
            else:
                self.assertEqual(wr.ppn, exp_ppn, f"vpn={vpn} ppn")
            self.assertEqual(wr.perms, exp_perms,
                             f"vpn={vpn} perms 期望 {perm_to_str(exp_perms)} "
                             f"实得 {perm_to_str(wr.perms)}")
            # 物理地址（含页内偏移）两种形式下都应一致
            self.assertEqual(wr.phys_addr, exp_ppn * PAGE_SIZE + (va & 0xFF))

    def assertPhysUnchanged(self, ppns_before: dict):
        """物理帧映射在操作前后必须完全不变（只改权限，不搬数据）。

        用物理地址对照（页基址处物理地址 = ppn * 256），大页/小页形式通用。
        """
        for vpn, exp in ppns_before.items():
            wr = self.mmu.walk_table(self.asid, vpn * PAGE_SIZE)
            self.assertIsNotNone(wr)
            self.assertEqual(wr.phys_addr, exp * PAGE_SIZE,
                             f"vpn={vpn} 物理帧被改变")

    def assertAccess(self, core, va, access, expect_ok: bool, *, via: str | None = None):
        r = self.mmu.translate(self.asid, core, va, access)
        self.assertEqual(r.ok, expect_ok,
                         f"va=0x{va:x} {access.name}: 期望 ok={expect_ok}，"
                         f"实得 fault={r.fault} via={r.via}")
        if expect_ok:
            self.assertEqual(r.phys_addr,
                             self.ora.expected_ppn(va) * PAGE_SIZE + (va & 0xFF))
        if via is not None:
            self.assertEqual(r.via, via)
        return r


class TestProtectRangeSmallPages(_OracleCase):
    def test_plain_small_page_range_downgrade_and_restore(self):
        for va in (0x1000, 0x1100, 0x1200, 0x1300):
            map_small_settled(self.mmu, self.asid, self.ora, va, PERM_RWX)
        # 解除写权限（覆盖 3 页，第 4 页不动）
        t = self.mmu.protect_range(self.asid, 0x1000, 0x1300, PERM_RX)
        self.ora.protect_range(0x1000, 0x1300, PERM_RX)
        self.assertEqual(t.kind, "protect_range")
        settle(self.mmu, t)
        self.assertTableMatchesOracle()
        for va in (0x1000, 0x1100, 0x1200):
            self.assertAccess(0, va, AccessType.READ, True)
            self.assertAccess(0, va, AccessType.WRITE, False)
        # 区间外页保持 rwx
        self.assertAccess(0, 0x1300, AccessType.WRITE, True)
        # 恢复写权限
        t2 = self.mmu.protect_range(self.asid, 0x1000, 0x1300, PERM_RWX)
        self.ora.protect_range(0x1000, 0x1300, PERM_RWX)
        settle(self.mmu, t2)
        self.assertTableMatchesOracle()
        for va in (0x1000, 0x1100, 0x1200):
            self.assertAccess(1, va, AccessType.WRITE, True)

    def test_full_huge_range_keeps_huge_form(self):
        map_huge_settled(self.mmu, self.asid, self.ora, 0x4000, PERM_RWX)
        ppns_before = {vpn: self.ora.pages[vpn][0] for vpn in range(0x40, 0x50)}
        t = self.mmu.protect_range(self.asid, 0x4000, 0x5000, PERM_READ)
        self.ora.protect_range(0x4000, 0x5000, PERM_READ)
        settle(self.mmu, t)
        # 仍是大页叶子
        wr = self.mmu.walk_table(self.asid, 0x4000)
        self.assertTrue(wr.huge)
        self.assertTableMatchesOracle()
        self.assertPhysUnchanged(ppns_before)
        self.assertAccess(0, 0x4fff, AccessType.WRITE, False)

    def test_partial_huge_split_only_inside_changes(self):
        map_huge_settled(self.mmu, self.asid, self.ora, 0x4000, PERM_RWX)
        ppns_before = {vpn: self.ora.pages[vpn][0] for vpn in range(0x40, 0x50)}
        # 只把大页中间 0x4300..0x44ff（含）两页降为只读
        t = self.mmu.protect_range(self.asid, 0x4300, 0x4500, PERM_READ)
        self.ora.protect_range(0x4300, 0x4500, PERM_READ)
        settle(self.mmu, t)
        wr = self.mmu.walk_table(self.asid, 0x4300)
        self.assertFalse(wr.huge, "部分覆盖必须拆成小页")
        self.assertTableMatchesOracle()
        self.assertPhysUnchanged(ppns_before)
        # 区间内只读
        for va in (0x4300, 0x4400):
            self.assertAccess(0, va, AccessType.WRITE, False)
            self.assertAccess(0, va, AccessType.READ, True)
        # 区间外 14 页仍 rwx（物理地址、执行权限都保留）
        for va in (0x4000, 0x4200, 0x4500, 0x4f00):
            self.assertAccess(1, va, AccessType.WRITE, True)
            self.assertAccess(1, va, AccessType.EXECUTE, True)

    def test_sub_page_range_inside_huge_single_page(self):
        map_huge_settled(self.mmu, self.asid, self.ora, 0x0000, PERM_RW)
        # 大页内单个小页 [0x0a00, 0x0b00)
        t = self.mmu.protect_range(self.asid, 0x0a00, 0x0b00, PERM_READ)
        self.ora.protect_range(0x0a00, 0x0b00, PERM_READ)
        settle(self.mmu, t)
        self.assertFalse(self.mmu.walk_table(self.asid, 0x0a00).huge)
        self.assertTableMatchesOracle()

    def test_range_across_small_and_huge_boundary(self):
        """跨大小页边界：前面是小页表区域、中间一个大页、后面再小页。"""
        # 0x3000..0x3fff 区域铺小页（与 0x4000 大页相邻）
        for va in range(0x3000, 0x4000, 0x100):
            map_small_settled(self.mmu, self.asid, self.ora, va, PERM_RWX)
        map_huge_settled(self.mmu, self.asid, self.ora, 0x4000, PERM_RWX)
        for va in range(0x5000, 0x5400, 0x100):
            map_small_settled(self.mmu, self.asid, self.ora, va, PERM_RWX)
        ppns_before = dict(self._snapshot_ppns())
        # 区间 0x3f00..0x5100：1 个小页 + 整个大页（恰好整段覆盖）+ 1 个小页
        t = self.mmu.protect_range(self.asid, 0x3f00, 0x5100, PERM_READ)
        self.ora.protect_range(0x3f00, 0x5100, PERM_READ)
        settle(self.mmu, t)
        self.assertTableMatchesOracle()
        self.assertPhysUnchanged(ppns_before)
        self.assertTrue(self.mmu.walk_table(self.asid, 0x4000).huge)
        # 边界两侧
        self.assertAccess(0, 0x3e00, AccessType.WRITE, True)
        self.assertAccess(0, 0x3f00, AccessType.WRITE, False)
        self.assertAccess(0, 0x4abc, AccessType.WRITE, False)
        self.assertAccess(0, 0x5000, AccessType.WRITE, False)
        self.assertAccess(0, 0x5100, AccessType.WRITE, True)

    def test_range_partial_huge_at_4k_boundary_then_into_small(self):
        """区间在 4096 边界处只覆盖大页的后半，再跨入小页区域。"""
        map_huge_settled(self.mmu, self.asid, self.ora, 0x4000, PERM_RWX)
        for va in range(0x5000, 0x5300, 0x100):
            map_small_settled(self.mmu, self.asid, self.ora, va, PERM_RWX)
        ppns_before = dict(self._snapshot_ppns())
        t = self.mmu.protect_range(self.asid, 0x4800, 0x5200, PERM_READ)
        self.ora.protect_range(0x4800, 0x5200, PERM_READ)
        settle(self.mmu, t)
        self.assertTableMatchesOracle()
        self.assertPhysUnchanged(ppns_before)
        # 大页被拆分：前半 rwx，后半只读
        self.assertAccess(0, 0x4700, AccessType.WRITE, True)
        self.assertAccess(0, 0x4800, AccessType.WRITE, False)
        self.assertAccess(0, 0x4f00, AccessType.WRITE, False)
        # 跨入的小页只读，其后的小页仍 rwx
        self.assertAccess(0, 0x5000, AccessType.WRITE, False)
        self.assertAccess(0, 0x5100, AccessType.WRITE, False)
        self.assertAccess(0, 0x5200, AccessType.WRITE, True)

    def _snapshot_ppns(self):
        return ((vpn, rec[0]) for vpn, rec in self.ora.pages.items())

    def test_downgrade_then_restore_exec_bit(self):
        map_small_settled(self.mmu, self.asid, self.ora, 0x6000, PERM_RWX)
        t = self.mmu.protect_range(self.asid, 0x6000, 0x6100, PERM_RW)
        self.ora.protect_range(0x6000, 0x6100, PERM_RW)
        settle(self.mmu, t)
        self.assertAccess(0, 0x6000, AccessType.EXECUTE, False)
        self.assertAccess(0, 0x6000, AccessType.WRITE, True)
        t2 = self.mmu.protect_range(self.asid, 0x6000, 0x6100, PERM_RWX)
        self.ora.protect_range(0x6000, 0x6100, PERM_RWX)
        settle(self.mmu, t2)
        self.assertAccess(1, 0x6000, AccessType.EXECUTE, True)


class TestProtectRangeAlias(_OracleCase):
    def test_alias_of_same_physical_frame_not_changed(self):
        """同一物理页的别名映射：改一个虚页权限，别名权限与物理数据保持不变。"""
        # 手动构造：先映射两个小页，再用底层 PTE 把第二个 PTE 指向同一帧，
        # 形成别名。oracle 也记录同一 ppn。
        map_small_settled(self.mmu, self.asid, self.ora, 0x1000, PERM_RWX)
        map_small_settled(self.mmu, self.asid, self.ora, 0x2000, PERM_RWX)
        alias_ppn = self.ora.pages[0x1000 >> 8][0]
        sp = self.mmu._spaces[self.asid]
        # 0x2000: L1=2, L2=0 —— 把该 PTE 的帧号改成与 0x1000 相同，形成别名
        sp.l1[2][0].ppn = alias_ppn
        self.ora.pages[0x2000 >> 8][0] = alias_ppn  # 独立表同步记录别名事实
        settle_all(self.mmu)
        # 只降权 0x1000
        t = self.mmu.protect_range(self.asid, 0x1000, 0x1100, PERM_READ)
        self.ora.pages[0x1000 >> 8][1] = PERM_READ
        settle(self.mmu, t)
        self.assertTableMatchesOracle()
        # 别名 0x2000 仍 rwx 且仍指向同一物理帧
        self.assertAccess(0, 0x2000, AccessType.WRITE, True)
        wr_alias = self.mmu.walk_table(self.asid, 0x2000)
        self.assertEqual(wr_alias.ppn, alias_ppn)
        wr_main = self.mmu.walk_table(self.asid, 0x1000)
        self.assertEqual(wr_main.ppn, alias_ppn)
        self.assertEqual(wr_main.perms, PERM_READ)

    def test_alias_inside_split_huge_not_changed(self):
        """大页拆分降权后，区间外小页恰为某帧的别名时，别名权限不变。"""
        map_huge_settled(self.mmu, self.asid, self.ora, 0x4000, PERM_RWX)
        map_small_settled(self.mmu, self.asid, self.ora, 0x9000, PERM_RWX)
        target_ppn = self.ora.pages[0x4500 >> 8][0]
        # 让 0x9000 成为大页内 0x4500 帧的别名
        self.mmu._spaces[self.asid].l1[9][0].ppn = target_ppn
        self.ora.pages[0x9000 >> 8][0] = target_ppn
        settle_all(self.mmu)
        # 降权区间只含 0x4300..0x44ff，不含 0x4500
        t = self.mmu.protect_range(self.asid, 0x4300, 0x4500, PERM_READ)
        self.ora.protect_range(0x4300, 0x4500, PERM_READ)
        settle(self.mmu, t)
        self.assertTableMatchesOracle()
        self.assertAccess(0, 0x9000, AccessType.WRITE, True)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x9000).ppn, target_ppn)
        self.assertEqual(self.mmu.walk_table(self.asid, 0x4500).perms, PERM_RWX)


class TestProtectRangeHoleAtomic(_OracleCase):
    def _snapshot_state(self):
        return (
            [(m[0], m[1], m[2], m[3]) for m in self.mmu.mappings(self.asid)],
            self.mmu.mem.used_frames(),
            [list(self.mmu.tlb_keys(c)) for c in range(2)],
            sorted(self.mmu.pending_tickets()),
        )

    def test_hole_in_middle_of_small_range_changes_nothing(self):
        for va in (0x1000, 0x1100, 0x1300):  # 0x1200 是空洞
            map_small_settled(self.mmu, self.asid, self.ora, va, PERM_RW)
        before = self._snapshot_state()
        with self.assertRaises(NotMapped):
            self.mmu.protect_range(self.asid, 0x1000, 0x1400, PERM_READ)
        self.assertEqual(self._snapshot_state(), before)
        # 没有产生票据；oracle 也不改
        self.assertEqual(self.mmu.pending_tickets(), [])
        # 原权限仍可写
        self.assertAccess(0, 0x1300, AccessType.WRITE, True)

    def test_hole_inside_partial_huge_changes_nothing(self):
        map_huge_settled(self.mmu, self.asid, self.ora, 0x4000, PERM_RWX)
        # 区间跨入一个完全空的 4096 区域（0x5000）
        map_small_settled(self.mmu, self.asid, self.ora, 0x6000, PERM_RW)
        before = self._snapshot_state()
        with self.assertRaises(NotMapped):
            self.mmu.protect_range(self.asid, 0x4f00, 0x6100, PERM_READ)
        self.assertEqual(self._snapshot_state(), before)
        # 大页没有被拆分
        self.assertTrue(self.mmu.walk_table(self.asid, 0x4f00).huge)
        self.assertAccess(0, 0x4f00, AccessType.WRITE, True)

    def test_partial_small_table_with_unmapped_slot_changes_nothing(self):
        # 二级表存在但中间有 None 槽位
        map_small_settled(self.mmu, self.asid, self.ora, 0x7000, PERM_RWX)
        map_small_settled(self.mmu, self.asid, self.ora, 0x7300, PERM_RWX)
        before = self._snapshot_state()
        with self.assertRaises(NotMapped):
            self.mmu.protect_range(self.asid, 0x7000, 0x7400, PERM_READ)
        self.assertEqual(self._snapshot_state(), before)
        self.assertEqual(self.ora.expected_perms(0x7300), PERM_RWX)
        self.assertAccess(0, 0x7300, AccessType.EXECUTE, True)

    def test_invalid_arguments_change_nothing(self):
        map_small_settled(self.mmu, self.asid, self.ora, 0x1000, PERM_RW)
        before = self._snapshot_state()
        bad_calls = [
            (0x1000, 0x1000, PERM_READ),       # 空区间
            (0x1100, 0x1000, PERM_READ),       # 反向
            (0x1080, 0x1100, PERM_READ),       # start 未对齐
            (0x1000, 0x1180, PERM_READ),       # end 未对齐
            (0x1000, 0x1100, 0),               # 权限 0
            (0x1000, 0x1100, 8),               # 未知权限位
        ]
        for s, e, p in bad_calls:
            with self.subTest(s=s, e=e, p=p):
                with self.assertRaises((ValueError, VMShootError)):
                    self.mmu.protect_range(self.asid, s, e, p)
        # bool 拒绝、float 拒绝
        with self.assertRaises(TypeError):
            self.mmu.protect_range(self.asid, True, 0x1100, PERM_READ)
        with self.assertRaises(TypeError):
            self.mmu.protect_range(self.asid, 0x1000, float(0x1100), PERM_READ)
        # end 超界
        with self.assertRaises(ValueError):
            self.mmu.protect_range(self.asid, 0xff00, 0x10100, PERM_READ)
        self.assertEqual(self._snapshot_state(), before)

    def test_bad_asid_raises(self):
        with self.assertRaises(Exception):
            self.mmu.protect_range(7, 0x0, 0x100, PERM_READ)


class TestProtectRangeTicketProtocol(_OracleCase):
    def test_demo_two_cores_cache_old_then_ack_one_by_one(self):
        """题面演示场景：两核先缓存旧权限，部分大页降权，逐核确认，最终拒绝写。"""
        map_huge_settled(self.mmu, self.asid, self.ora, 0x4000, PERM_RWX)
        # 两核都缓存旧（大页）权限
        self.assertAccess(0, 0x4300, AccessType.WRITE, True, via="walk")
        self.assertAccess(0, 0x4300, AccessType.WRITE, True, via="tlb")
        self.assertAccess(1, 0x4300, AccessType.WRITE, True, via="walk")
        self.assertAccess(1, 0x4300, AccessType.WRITE, True, via="tlb")
        # 页表先改、票据发出
        t = self.mmu.protect_range(self.asid, 0x4300, 0x4500, PERM_READ)
        self.ora.protect_range(0x4300, 0x4500, PERM_READ)
        self.assertEqual(len(t.keys), 16)  # 拆分大页：整个 4096 区域 16 个 key
        # ACK 前：两核旧 TLB 项仍放行写（基线协议允许的窗口）
        self.assertAccess(0, 0x4300, AccessType.WRITE, True, via="tlb")
        self.assertAccess(1, 0x4300, AccessType.WRITE, True, via="tlb")
        # finish 被阻塞
        with self.assertRaises(InvalidationsPending) as cm:
            self.mmu.finish(t.id)
        self.assertEqual(set(cm.exception.outstanding), {0, 1})
        # 核0 ACK：核0 立即拒绝写（重走页表），核1 仍旧
        self.mmu.ack(0, t.id)
        self.assertAccess(0, 0x4300, AccessType.WRITE, False, via="walk")
        self.assertAccess(1, 0x4300, AccessType.WRITE, True, via="tlb")
        with self.assertRaises(InvalidationsPending):
            self.mmu.finish(t.id)
        # 核1 ACK + finish
        self.mmu.ack(1, t.id)
        self.mmu.finish(t.id)
        self.assertAccess(1, 0x4300, AccessType.WRITE, False, via="walk")
        # 全部目标核确认后：任何旧 TLB 权限都不能再放行；两核、两页都拒绝写
        self.assertAccess(0, 0x4300, AccessType.WRITE, False)
        self.assertAccess(1, 0x4400, AccessType.WRITE, False)
        self.assertAccess(0, 0x4300, AccessType.READ, True)
        # 区间外仍可写
        self.assertAccess(0, 0x4200, AccessType.WRITE, True)
        self.assertAccess(1, 0x4500, AccessType.WRITE, True)
        self.assertTableMatchesOracle()

    def test_stale_huge_entry_at_outside_vpn_invalidated(self):
        """核把大页项缓存在区间外的小页 vpn 上，票据仍须把它失效。"""
        map_huge_settled(self.mmu, self.asid, self.ora, 0x4000, PERM_RWX)
        # 核0 在 0x4f00（区间外）上缓存大页标记项
        self.assertAccess(0, 0x4f00, AccessType.READ, True, via="walk")
        # 只降权大页头部两页
        t = self.mmu.protect_range(self.asid, 0x4000, 0x4200, PERM_READ)
        self.ora.protect_range(0x4000, 0x4200, PERM_READ)
        settle(self.mmu, t)
        # 旧大页标记项已被删除：0x4000 重走页表 -> 只读
        self.assertAccess(0, 0x4000, AccessType.WRITE, False, via="walk")
        # 0x4f00 重走页表（拆分后的小页）仍 rwx
        self.assertAccess(0, 0x4f00, AccessType.WRITE, True)

    def test_ticket_acks_and_lifecycle(self):
        for va in (0x1000, 0x1100):
            map_small_settled(self.mmu, self.asid, self.ora, va, PERM_RW)
        t = self.mmu.protect_range(self.asid, 0x1000, 0x1200, PERM_READ)
        self.ora.protect_range(0x1000, 0x1200, PERM_READ)
        self.assertEqual(self.mmu.pending_tickets(), [t.id])
        # 未知票据
        with self.assertRaises(TicketError):
            self.mmu.ack(0, t.id + 999)
        self.mmu.ack(0, t.id)
        with self.assertRaises(TicketError):
            self.mmu.ack(0, t.id)  # 重复 ACK
        self.mmu.ack(1, t.id)
        self.mmu.finish(t.id)
        with self.assertRaises(TicketError):
            self.mmu.finish(t.id)  # 已关闭
        self.assertEqual(self.mmu.pending_tickets(), [])

    def test_full_huge_range_stale_entry_also_flushed(self):
        """整段大页保留大页形式改权限：旧大页 TLB 项同样必须失效。"""
        map_huge_settled(self.mmu, self.asid, self.ora, 0x8000, PERM_RWX)
        self.assertAccess(0, 0x8abc, AccessType.WRITE, True, via="walk")
        self.assertAccess(0, 0x8abc, AccessType.WRITE, True, via="tlb")
        t = self.mmu.protect_range(self.asid, 0x8000, 0x9000, PERM_RX)
        self.ora.protect_range(0x8000, 0x9000, PERM_RX)
        # ACK 前旧项仍放行写
        self.assertAccess(0, 0x8abc, AccessType.WRITE, True, via="tlb")
        settle(self.mmu, t)
        self.assertAccess(0, 0x8abc, AccessType.WRITE, False, via="walk")
        self.assertTrue(self.mmu.walk_table(self.asid, 0x8abc).huge)

    def test_no_second_confirmation_mechanism_same_ticket_protocol(self):
        map_small_settled(self.mmu, self.asid, self.ora, 0x1000, PERM_RW)
        t = self.mmu.protect_range(self.asid, 0x1000, 0x1100, PERM_READ)
        self.ora.protect_range(0x1000, 0x1100, PERM_READ)
        # 返回对象就是普通 Ticket：复用既有 ack/finish，无额外 API
        self.assertTrue(hasattr(t, "id") and hasattr(t, "acks") and hasattr(t, "pending"))
        self.assertEqual(t.targets, (0, 1))
        settle(self.mmu, t)


if __name__ == "__main__":
    unittest.main()
