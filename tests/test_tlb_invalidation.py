"""带 ASID 的 LRU TLB、陈旧项、失效票据协议与跨页拒绝测试。"""

from __future__ import annotations

import unittest

from vmshoot import (
    MMU,
    AccessType,
    AsidBusy,
    CrossPageAccess,
    InvalidAddressSpace,
    InvalidationsPending,
    L2_SIZE,
    PAGE_SIZE,
    PERM_EXEC,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PERM_WRITE,
    TLB_SIZE,
    TicketError,
)


def warm(mmu, asid, core, vaddr, access=AccessType.READ):
    """让某核翻译一次，确保 TLB 填充。"""
    return mmu.translate(asid, core, vaddr, access)


def settle_targets(mmu, *tickets):
    """只对票据的实际目标核 ACK 并 finish（跳过非目标核）。"""
    for t in tickets:
        for core in t.targets:
            mmu.ack(core, t.id)
        mmu.finish(t.id)


class TestTLBLru(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()
        # 映射 9 个不同小页，用于淘汰；用 map_pages 不可用，逐个 map
        self.addrs = [i * 0x1000 for i in range(TLB_SIZE + 1)]
        for va in self.addrs:
            self.mmu.map(self.asid, va, PERM_RWX)

    def _settle_tickets(self):
        for tid in list(self.mmu.pending_tickets()):
            t = self.mmu.notifications(tid)[0]
            for core in t.notified:
                self.mmu.ack(core, tid)
            self.mmu.finish(tid)

    def test_capacity_eight_and_lru_eviction(self):
        self._settle_tickets()
        for va in self.addrs[:TLB_SIZE]:
            warm(self.mmu, self.asid, 0, va)
        self.assertEqual(self.mmu.tlb_len(0), TLB_SIZE)
        # 第 9 个不同页 -> 淘汰最久未用的第 0 页
        warm(self.mmu, self.asid, 0, self.addrs[TLB_SIZE])
        keys = [vpn for _, vpn in self.mmu.tlb_keys(0)]
        self.assertEqual(len(keys), TLB_SIZE)
        self.assertNotIn(self.addrs[0] >> 8, keys)
        self.assertIn(self.addrs[TLB_SIZE] >> 8, keys)

    def test_lru_touch_refreshes_order(self):
        self._settle_tickets()
        for va in self.addrs[:TLB_SIZE]:
            warm(self.mmu, self.asid, 0, va)
        # 重新访问最旧的第 0 页，把它刷新为最新
        warm(self.mmu, self.asid, 0, self.addrs[0])
        warm(self.mmu, self.asid, 0, self.addrs[TLB_SIZE])  # 触发淘汰
        keys = [vpn for _, vpn in self.mmu.tlb_keys(0)]
        self.assertIn(self.addrs[0] >> 8, keys)       # 被刷新，存活
        self.assertNotIn(self.addrs[1] >> 8, keys)    # 原来的次旧者被淘汰


class TestTLBAsidIsolation(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.a = self.mmu.create_address_space()
        self.b = self.mmu.create_address_space()
        # 两个地址空间映射同一虚页
        self.ta = self.mmu.map(self.a, 0x1000, PERM_RW)
        self.tb = self.mmu.map(self.b, 0x1000, PERM_RW)
        # 此时没有任何核进入过任一空间：票据目标为空，无需发失效请求
        for t in (self.ta, self.tb):
            self.assertEqual(t.targets, ())
            self.mmu.finish(t.id)

    def test_same_vpn_distinct_keys_and_frames(self):
        ra = warm(self.mmu, self.a, 0, 0x1000)
        rb = warm(self.mmu, self.b, 0, 0x1000)
        self.assertTrue(ra.ok and rb.ok)
        self.assertNotEqual(ra.phys_addr, rb.phys_addr)  # 帧独立
        keys = self.mmu.tlb_keys(0)
        self.assertIn((self.a, 0x10), keys)
        self.assertIn((self.b, 0x10), keys)
        self.assertEqual(self.mmu.tlb_len(0), 2)

    def test_unmap_one_asid_does_not_invalidate_other(self):
        warm(self.mmu, self.a, 0, 0x1000)
        warm(self.mmu, self.b, 0, 0x1000)
        t = self.mmu.unmap(self.a, 0x1000)
        # 只有核0进入过这两个空间：只通知核0，核1是无关核不等待
        self.assertEqual(t.targets, (0,))
        self.mmu.ack(0, t.id)
        self.mmu.finish(t.id)
        # B 仍可命中（key 含 ASID，未被失效）
        rb = self.mmu.translate(self.b, 0, 0x1000, AccessType.READ)
        self.assertTrue(rb.ok)
        self.assertEqual(rb.via, "tlb")
        # A 已失效
        ra = self.mmu.translate(self.a, 0, 0x1000, AccessType.READ)
        self.assertFalse(ra.ok)
        self.assertEqual(ra.fault, "not_mapped")


class TestStaleAndTicketProtocol(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()

    def test_stale_entry_observable_before_ack_gone_after(self):
        t = self.mmu.map(self.asid, 0x1000, PERM_RW)
        self.mmu.finish(t.id)
        # 两个核都预热到同一映射
        warm(self.mmu, self.asid, 0, 0x1000)
        warm(self.mmu, self.asid, 1, 0x1000)
        tu = self.mmu.unmap(self.asid, 0x1000)
        # ACK 前：页表已无映射，但两核旧 TLB 项仍允许存在并命中
        s0 = self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ)
        s1 = self.mmu.translate(self.asid, 1, 0x1000, AccessType.READ)
        self.assertTrue(s0.ok and s0.via == "tlb")
        self.assertTrue(s1.ok and s1.via == "tlb")
        # finish 必须等待所有目标核
        with self.assertRaises(InvalidationsPending) as cm:
            self.mmu.finish(tu.id)
        self.assertEqual(set(cm.exception.outstanding), {0, 1})
        # 只 ACK 核 0：核 0 旧项消失，核 1 仍旧
        self.mmu.ack(0, tu.id)
        self.assertEqual(
            self.mmu.translate(self.asid, 0, 0x1000, AccessType.READ).fault,
            "not_mapped",
        )
        stale = self.mmu.translate(self.asid, 1, 0x1000, AccessType.READ)
        self.assertTrue(stale.ok and stale.via == "tlb")
        with self.assertRaises(InvalidationsPending):
            self.mmu.finish(tu.id)
        # 核 1 也 ACK 后才能 finish
        self.mmu.ack(1, tu.id)
        done = self.mmu.finish(tu.id)
        self.assertTrue(done.done)
        self.assertEqual(self.mmu.tlb_len(0), 0)
        self.assertEqual(self.mmu.tlb_len(1), 0)
        self.assertEqual(
            self.mmu.translate(self.asid, 1, 0x1000, AccessType.READ).fault,
            "not_mapped",
        )

    def test_stale_permission_before_ack_then_fault_after(self):
        t = self.mmu.map(self.asid, 0x2000, PERM_RWX)
        self.mmu.finish(t.id)
        warm(self.mmu, self.asid, 0, 0x2000, AccessType.READ)
        tp = self.mmu.protect(self.asid, 0x2000, PERM_READ)
        # 只有核0进入过：只通知核0（核1不被无关等待）
        self.assertEqual(tp.targets, (0,))
        # ACK 前旧项仍带 w/x
        self.assertTrue(self.mmu.translate(self.asid, 0, 0x2000, AccessType.WRITE).ok)
        settle_targets(self.mmu, tp)
        # ACK 后重走页表 -> write fault
        r = self.mmu.translate(self.asid, 0, 0x2000, AccessType.WRITE)
        self.assertFalse(r.ok)
        self.assertEqual(r.fault, "write")
        # read 仍然成功
        self.assertTrue(self.mmu.translate(self.asid, 0, 0x2000, AccessType.READ).ok)

    def test_finish_unknown_or_double_finish(self):
        t = self.mmu.map(self.asid, 0x3000, PERM_RW)
        with self.assertRaises(TicketError):
            self.mmu.finish(999)
        with self.assertRaises(TicketError):
            self.mmu.ack(0, 999)
        # 无核进入过：无目标核，非目标核 ACK 是非法操作
        with self.assertRaises(TicketError):
            self.mmu.ack(0, t.id)
        self.mmu.finish(t.id)
        with self.assertRaises(TicketError):
            self.mmu.finish(t.id)  # 已关闭
        with self.assertRaises(TicketError):
            self.mmu.ack(0, t.id)  # 重复 ACK

    def test_bad_core_id(self):
        t = self.mmu.map(self.asid, 0x3000, PERM_RW)
        for bad in (2, -1):
            with self.assertRaises(ValueError):
                self.mmu.ack(bad, t.id)
        with self.assertRaises(TypeError):
            self.mmu.ack(True, t.id)


class TestHugeStaleAfterSplit(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()

    def test_huge_cached_on_both_cores_split_invalidates(self):
        t = self.mmu.map(self.asid, 0x4000, PERM_RWX, huge=True)
        self.mmu.finish(t.id)
        # 两核在大页不同偏移预热，填入大页标记 key (asid, vpn&~15)
        warm(self.mmu, self.asid, 0, 0x4100)
        warm(self.mmu, self.asid, 1, 0x4300)
        self.assertEqual(self.mmu.tlb_len(0), 1)
        self.assertEqual(self.mmu.tlb_len(1), 1)
        # 大页中间改权限 -> 拆分
        tp = self.mmu.protect(self.asid, 0x4300, PERM_READ)
        self.assertEqual(len(tp.keys), L2_SIZE)
        # 两核都进入过：双核都在实际通知范围内
        self.assertEqual(tp.targets, (0, 1))
        # ACK 前核 1 用旧大页项仍可写 0x4300（陈旧）
        stale = self.mmu.translate(self.asid, 1, 0x4300, AccessType.WRITE)
        self.assertTrue(stale.ok)
        self.assertEqual(stale.via, "tlb")
        # 全部 ACK
        settle_targets(self.mmu, tp)
        # 0x4300 重走页表 -> 只读 -> write fault
        r = self.mmu.translate(self.asid, 1, 0x4300, AccessType.WRITE)
        self.assertFalse(r.ok)
        self.assertEqual(r.fault, "write")
        # 其他小页仍 rwx
        ok = self.mmu.translate(self.asid, 1, 0x4100, AccessType.EXECUTE)
        self.assertTrue(ok.ok)
        self.assertEqual(ok.via, "walk")  # 旧大页项被失效后重新填入小页项


class TestTranslateFaultsAndCrossPage(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.asid = self.mmu.create_address_space()
        t = self.mmu.map(self.asid, 0x5000, PERM_RX)
        self.mmu.finish(t.id)

    def test_not_mapped_fault(self):
        r = self.mmu.translate(self.asid, 0, 0x6000, AccessType.READ)
        self.assertFalse(r.ok)
        self.assertEqual(r.fault, "not_mapped")
        self.assertIsNone(r.phys_addr)

    def test_permission_faults(self):
        r = self.mmu.translate(self.asid, 0, 0x5000, AccessType.WRITE)
        self.assertEqual(r.fault, "write")
        self.assertTrue(self.mmu.translate(self.asid, 0, 0x5000, AccessType.READ).ok)
        self.assertTrue(self.mmu.translate(self.asid, 0, 0x5000, AccessType.EXECUTE).ok)

    def test_cross_small_page_rejected(self):
        # 起点偏移 250，长度 10：越过 256 边界
        with self.assertRaises(CrossPageAccess):
            self.mmu.translate(self.asid, 0, 0x50FA, AccessType.READ, length=10)

    def test_within_page_at_boundary_ok(self):
        r = self.mmu.translate(self.asid, 0, 0x50FF, AccessType.READ, length=1)
        self.assertTrue(r.ok)

    def test_huge_cross_page_allows_within_4096(self):
        t = self.mmu.map(self.asid, 0x7000, PERM_RWX, huge=True)
        self.mmu.finish(t.id)
        # 大页内跨小页边界但不跨 4096 大页边界：允许
        r = self.mmu.translate(self.asid, 0, 0x7FF0, AccessType.READ, length=16)
        self.assertTrue(r.ok)
        # setUp 已占用帧 0；大页首位适配取 16..31，故物理地址 = 16*256 + 0xFF0
        self.assertEqual(r.phys_addr, 16 * 256 + 0xFF0)
        # 跨大页边界：拒绝
        with self.assertRaises(CrossPageAccess):
            self.mmu.translate(self.asid, 0, 0x7FF0, AccessType.READ, length=17)

    def test_bad_length(self):
        with self.assertRaises(ValueError):
            self.mmu.translate(self.asid, 0, 0x5000, AccessType.READ, length=0)
        with self.assertRaises(ValueError):
            self.mmu.translate(self.asid, 0, 0x5000, AccessType.READ, length=4097)
        with self.assertRaises(TypeError):
            self.mmu.translate(self.asid, 0, 0x5000, AccessType.READ, length=True)
        with self.assertRaises(TypeError):
            self.mmu.translate(self.asid, 0, 0x5000, 1)  # access 必须是枚举


class TestAsidReuseWithPendingTickets(unittest.TestCase):
    def test_destroy_with_pending_ticket_blocks_reuse(self):
        mmu = MMU()
        a = mmu.create_address_space()
        t = mmu.map(a, 0x1000, PERM_RW)  # 票据未 finish
        mmu.destroy_address_space(a)
        # 槽位被未完成票据占用，不能复用同号；其余 3 个槽位仍可分配
        self.assertEqual(mmu.create_address_space(), 1)
        self.assertEqual(mmu.create_address_space(), 2)
        self.assertEqual(mmu.create_address_space(), 3)
        # 4 个槽位：0 被僵尸票据占据，1..3 活动 -> 再创建触发 AsidBusy
        with self.assertRaises(AsidBusy) as cm:
            mmu.create_address_space()
        self.assertEqual(cm.exception.asid, 0)
        self.assertEqual(cm.exception.ticket_ids, [t.id])
        # 无任何核进入过该空间：票据无目标核，ACK 不改变僵尸状态；
        # 真正决定槽位释放的是 finish。
        with self.assertRaises(TicketError):
            mmu.ack(0, t.id)
        with self.assertRaises(AsidBusy):
            mmu.create_address_space()
        mmu.finish(t.id)
        # 现在 0 号可复用（需要先销毁一个活动空间腾出整体名额）
        mmu.destroy_address_space(1)
        self.assertEqual(mmu.create_address_space(), 0)

    def test_destroy_flushes_tlb_and_kills_translations(self):
        mmu = MMU()
        a = mmu.create_address_space()
        t = mmu.map(a, 0x1000, PERM_RW)
        mmu.finish(t.id)
        warm(mmu, a, 0, 0x1000)
        warm(mmu, a, 1, 0x1000)
        mmu.destroy_address_space(a)
        self.assertEqual(mmu.tlb_len(0), 0)
        self.assertEqual(mmu.tlb_len(1), 0)
        with self.assertRaises(InvalidAddressSpace):
            mmu.translate(a, 0, 0x1000, AccessType.READ)

    def test_ticket_pending_list_filtered_by_asid(self):
        mmu = MMU()
        a = mmu.create_address_space()
        b = mmu.create_address_space()
        ta = mmu.map(a, 0x1000, PERM_RW)
        tb = mmu.map(b, 0x1000, PERM_RW)
        self.assertEqual(mmu.pending_tickets(a), [ta.id])
        self.assertEqual(set(mmu.pending_tickets()), {ta.id, tb.id})


class TestSelectiveShootdown(unittest.TestCase):
    """只通知确实可能缓存过该地址空间的核；不漏通知、不省必要检查。"""

    def setUp(self):
        self.mmu = MMU()
        self.a = self.mmu.create_address_space()
        self.b = self.mmu.create_address_space()

    def _settle(self, t):
        settle_targets(self.mmu, t)

    def test_no_core_entered_means_no_invalidation_request(self):
        tm = self.mmu.map(self.a, 0x1000, PERM_RW)
        self.assertEqual(tm.targets, ())
        n = self.mmu.notifications(tm.id)[0]
        self.assertEqual(n.notified, ())
        self.assertEqual(n.skipped, (0, 1))
        self.mmu.finish(tm.id)  # 无目标核：无需等待

    def test_only_entered_cores_are_notified(self):
        tm = self.mmu.map(self.a, 0x1000, PERM_RW)
        self.mmu.finish(tm.id)
        # 核1 从未进入 A
        warm(self.mmu, self.a, 0, 0x1000)
        tu = self.mmu.unmap(self.a, 0x1000)
        self.assertEqual(tu.targets, (0,))
        n = self.mmu.notifications(tu.id)[0]
        self.assertEqual(n.notified, (0,))
        self.assertEqual(n.skipped, (1,))  # 减少的是对核1 的无关失效请求
        # finish 只等待核0：核1 的 ACK 既不需要也不合法
        with self.assertRaises(TicketError):
            self.mmu.ack(1, tu.id)
        self.mmu.ack(0, tu.id)
        self.mmu.finish(tu.id)

    def test_context_switch_then_first_translation_records_participant(self):
        tm = self.mmu.map(self.a, 0x1000, PERM_RW)
        self.mmu.finish(tm.id)
        # 显式切换：核1 第一次进入 A，即便还没翻译，也要保守通知
        self.mmu.switch_address_space(1, self.a)
        self.assertEqual(self.mmu.participants(self.a), (1,))
        tp = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.assertEqual(tp.targets, (1,))
        self._settle(tp)
        # 翻译同样隐式登记“第一次进入”
        tm2 = self.mmu.map(self.b, 0x2000, PERM_RW)
        self.mmu.finish(tm2.id)
        warm(self.mmu, self.b, 0, 0x2000)
        self.assertEqual(self.mmu.participants(self.b), (0,))

    def test_switch_away_and_stale_ack_changes_scope(self):
        tm = self.mmu.map(self.a, 0x1000, PERM_RW)
        self.mmu.finish(tm.id)
        warm(self.mmu, self.a, 0, 0x1000)
        warm(self.mmu, self.a, 1, 0x1000)
        # 第一次改权限：两核都可能缓存，双核通知
        tp1 = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.assertEqual(tp1.targets, (0, 1))
        self.mmu.ack(0, tp1.id)       # 核0 旧确认先到
        self.mmu.ack(1, tp1.id)       # 核1 的旧确认迟到
        self.mmu.finish(tp1.id)
        # 两核切走（核0->B，核1->无空间），且 A 的 TLB 项已被各自 ACK 删除
        self.mmu.switch_address_space(0, self.b)
        self.mmu.switch_address_space(1, None)
        self.assertEqual(self.mmu.participants(self.a), ())
        # A 已无参与核：再次改权限不发任何失效请求
        tmw = self.mmu.map(self.a, 0x1100, PERM_RW)  # 新页 map 同样无需通知
        self.assertEqual(tmw.targets, ())
        self.mmu.finish(tmw.id)

    def test_unacked_core_stays_in_scope_until_late_ack_arrives(self):
        tm = self.mmu.map(self.a, 0x1000, PERM_RW)
        self.mmu.finish(tm.id)
        warm(self.mmu, self.a, 0, 0x1000)
        warm(self.mmu, self.a, 1, 0x1000)
        tu = self.mmu.unmap(self.a, 0x1000)
        self.assertEqual(tu.targets, (0, 1))
        # 核1 切走但确认未到：旧确认迟到前不能把它移出范围
        self.mmu.switch_address_space(1, self.b)
        tm2 = self.mmu.map(self.a, 0x2000, PERM_RW)
        self.assertEqual(tm2.targets, (0, 1))  # 核1 仍欠 unmap 的 ACK，保守通知
        self.mmu.ack(0, tu.id)
        self.mmu.ack(1, tu.id)  # 迟到的旧确认
        self.mmu.finish(tu.id)
        # 迟到确认到达后，核1 仍欠 tm2 的 ACK（它被保守地纳入了 tm2 目标），
        # 所以范围暂时还含核1；直到它对 tm2 也确认后才退出。
        self.assertEqual(self.mmu.participants(self.a), (0, 1))
        settle_targets(self.mmu, tm2)
        self.assertEqual(self.mmu.participants(self.a), (0,))

    def test_stale_entry_after_lru_eviction_and_switch_shrinks_scope(self):
        tm = self.mmu.map(self.a, 0x1000, PERM_RW)
        self.mmu.finish(tm.id)
        warm(self.mmu, self.a, 1, 0x1000)
        self.assertEqual(self.mmu.participants(self.a), (1,))
        # 先用 B 的 8 个不同页（票据此时无目标核）填满映射
        for i in range(TLB_SIZE):
            t = self.mmu.map(self.b, i * 0x1000, PERM_RW)
            self.mmu.finish(t.id)
        # 核1 连续翻译 B 的 8 个页：第一次翻译隐式进入 B，并把 A 的项淘汰干净
        for i in range(TLB_SIZE):
            warm(self.mmu, self.b, 1, i * 0x1000)
        self.assertFalse(any(k[0] == self.a for k in self.mmu.tlb_keys(1)))
        # A 的 TLB 项已被淘汰，且 translate 隐式把核1 当前空间改成 B：
        # A 已无任何参与证据，参与范围收缩为空。
        self.assertEqual(self.mmu.current_address_space(1), self.b)
        self.assertEqual(self.mmu.participants(self.a), ())


class TestPostAckWalkVerification(unittest.TestCase):
    """验收点：无 TLB 遍历器在全部 ACK 后的核对。"""

    def test_walker_matches_authoritative_state_after_all_acks(self):
        mmu = MMU()
        a = mmu.create_address_space()
        t1 = mmu.map(a, 0x0000, PERM_RW)
        t2 = mmu.map(a, 0x4000, PERM_RWX, huge=True)
        # 预热陈旧项
        mmu.translate(a, 0, 0x0000, AccessType.READ)
        mmu.translate(a, 1, 0x4300, AccessType.WRITE)
        tu = mmu.unmap(a, 0x0000)
        tp = mmu.protect(a, 0x4300, PERM_READ)
        # 对实际目标核 ACK + finish
        for t in (t1, t2, tu, tp):
            settle_targets(mmu, t)
        # 无 TLB 遍历器是页表真相的权威：
        self.assertIsNone(mmu.walk_table(a, 0x0000))
        w = mmu.walk_table(a, 0x4300)
        self.assertFalse(w.huge)
        self.assertEqual(w.perms, PERM_READ)
        # 小页占帧 0；大页首位适配取 16..31，拆分后 0x4300（第 3 小页）-> ppn 19
        self.assertEqual(w.ppn, 19)
        self.assertEqual(w.phys_addr, 19 * 256)
        # 且遍历器结论与双核 translate 一致（陈旧窗口已关闭）
        for core in (0, 1):
            self.assertEqual(
                mmu.translate(a, core, 0x0000, AccessType.READ).fault, "not_mapped")
            self.assertEqual(
                mmu.translate(a, core, 0x4300, AccessType.WRITE).fault, "write")
            self.assertTrue(
                mmu.translate(a, core, 0x4400, AccessType.WRITE).ok)
        # 无遗留票据
        self.assertEqual(mmu.pending_tickets(a), [])


if __name__ == "__main__":
    unittest.main()
