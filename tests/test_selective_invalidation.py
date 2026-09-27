"""选择性失效：参与范围维护、实际通知核、迟到旧确认与旧帧延迟复用。"""

from __future__ import annotations

import unittest

from vmshoot import (
    MMU,
    AccessType,
    InvalidationsPending,
    PERM_RW,
    PERM_RWX,
    PERM_READ,
    TicketError,
)


def settle(mmu, ticket):
    if ticket.closed:
        return ticket
    for core in ticket.notified_cores:
        mmu.ack(core, ticket.id)
    mmu.finish(ticket.id)
    return ticket


class TestParticipantTracking(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.a = self.mmu.create_address_space()
        self.b = self.mmu.create_address_space()
        self.ta = self.mmu.map(self.a, 0x1000, PERM_RWX)
        self.tb = self.mmu.map(self.b, 0x1000, PERM_RWX)
        settle(self.mmu, self.ta)
        settle(self.mmu, self.tb)

    def test_fresh_space_ticket_targets_nobody(self):
        t = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.assertEqual(t.notified_cores, ())
        self.assertTrue(t.closed)
        self.assertEqual(self.mmu.pending_tickets(self.a), [])
        self.assertEqual(self.mmu.participants(self.a), ())

    def test_first_entry_registers_participant(self):
        self.assertEqual(self.mmu.participants(self.a), ())
        self.mmu.switch_to(0, self.a)
        self.assertEqual(self.mmu.participants(self.a), (0,))
        self.assertEqual(self.mmu.current_address_space(0), self.a)
        t = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.assertEqual(t.notified_cores, (0,))

    def test_translate_fill_registers_participant_without_explicit_switch(self):
        # 直接在核 1 上做一次成功翻译（遍历填充）即证明它缓存过该空间
        r = self.mmu.translate(self.a, 1, 0x1000, AccessType.READ)
        self.assertTrue(r.ok)
        self.assertEqual(self.mmu.participants(self.a), (1,))
        self.assertEqual(self.mmu.current_address_space(1), self.a)
        t = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.assertEqual(t.notified_cores, (1,))

    def test_switch_away_shrinks_only_when_no_keys_remain(self):
        self.mmu.switch_to(0, self.a)
        warm = self.mmu.translate(self.a, 0, 0x1000, AccessType.READ)
        self.assertEqual(warm.via, "walk")
        self.mmu.switch_to(0, self.b)  # TLB 仍保留 A 的 key（ASID 隔离）
        self.assertEqual(self.mmu.participants(self.a), (0,))
        self.assertEqual(self.mmu.participants(self.b), (0,))
        t = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.assertEqual(t.notified_cores, (0,))  # 切出但有缓存 -> 仍通知

    def test_core_never_entered_is_not_notified(self):
        self.mmu.switch_to(0, self.a)
        self.mmu.translate(self.a, 0, 0x1000, AccessType.READ)
        t = self.mmu.protect(self.a, 0x1000, PERM_READ)
        # 核 1 从未进入 A：只通知核 0，减少的是对核 1 的无关失效请求
        self.assertEqual(t.notified_cores, (0,))
        with self.assertRaises(InvalidationsPending) as cm:
            self.mmu.finish(t.id)
        self.assertEqual(cm.exception.outstanding, (0,))
        self.mmu.ack(0, t.id)
        self.mmu.finish(t.id)
        self.assertEqual(self.mmu.pending_tickets(self.a), [])

    def test_shrink_after_ack_when_switched_away_and_emptied(self):
        self.mmu.switch_to(0, self.a)
        self.mmu.translate(self.a, 0, 0x1000, AccessType.READ)
        t = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.mmu.switch_to(0, self.b)  # 显式切走，给参与范围收缩提供证据
        self.mmu.ack(0, t.id)
        self.mmu.finish(t.id)
        # ACK 清掉唯一 key 且核 0 已切到 B -> 可证明移出 A 的参与范围
        self.assertEqual(self.mmu.participants(self.a), ())
        t2 = self.mmu.protect(self.a, 0x1000, PERM_RW)
        self.assertEqual(t2.notified_cores, ())
        self.assertTrue(t2.closed)


class TestLateAck(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.a = self.mmu.create_address_space()
        t0 = self.mmu.map(self.a, 0x1000, PERM_RWX)
        settle(self.mmu, t0)

    def test_late_ack_extends_inflight_later_tickets(self):
        # 核 0 进入并缓存；核 1 从未进入 A
        self.mmu.switch_to(0, self.a)
        self.mmu.translate(self.a, 0, 0x1000, AccessType.READ)
        t1 = self.mmu.protect(self.a, 0x1000, PERM_RW)
        self.assertEqual(t1.notified_cores, (0,))
        t2 = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.assertEqual(t2.notified_cores, (0,))
        # 核 1 对 t1 送来“迟到的旧确认”（说明它确实缓存过该空间）
        self.mmu.ack(1, t1.id)
        self.assertEqual(self.mmu.participants(self.a), (0, 1))
        # 在途票据 t2 必须把核 1 补进目标，绝不漏通知；该确认同时算作响应
        self.assertEqual(set(t2.notified_cores), {0, 1})
        self.assertIn(1, t1.acks)
        self.assertIn(1, t2.acks)
        self.mmu.ack(0, t1.id)
        self.mmu.finish(t1.id)
        with self.assertRaises(InvalidationsPending):
            self.mmu.finish(t2.id)
        self.mmu.ack(0, t2.id)
        self.mmu.finish(t2.id)

    def test_ticket_after_late_ack_includes_core(self):
        self.mmu.switch_to(0, self.a)
        self.mmu.translate(self.a, 0, 0x1000, AccessType.READ)
        t1 = self.mmu.protect(self.a, 0x1000, PERM_RW)
        self.mmu.ack(1, t1.id)       # 核 1 迟到确认
        self.mmu.ack(0, t1.id)
        self.mmu.finish(t1.id)
        # 此后新票据：核 1 已是确证参与核，直接进入目标快照
        t2 = self.mmu.protect(self.a, 0x1000, PERM_READ)
        self.assertEqual(set(t2.notified_cores), {0, 1})

    def test_duplicate_ack_still_rejected_after_late_join(self):
        self.mmu.switch_to(0, self.a)
        t1 = self.mmu.protect(self.a, 0x1000, PERM_RW)
        self.mmu.ack(1, t1.id)
        with self.assertRaises(TicketError):
            self.mmu.ack(1, t1.id)  # 重复 ACK


class TestDeferredFrameReuse(unittest.TestCase):
    def test_old_frame_kept_until_notified_cores_finish(self):
        mmu = MMU()
        a = mmu.create_address_space()
        t0 = mmu.map(a, 0x1000, PERM_RW)
        settle(mmu, t0)
        mmu.switch_to(1, a)
        mmu.translate(a, 1, 0x1000, AccessType.READ)
        tu = mmu.unmap(a, 0x1000)
        self.assertEqual(tu.notified_cores, (1,))
        # finish 前旧帧不能复用：新映射首位适配拿到 ppn 1 而不是 ppn 0
        tn = mmu.map(a, 0x2000, PERM_RW)
        settle(mmu, tn)
        self.assertEqual(mmu.walk_table(a, 0x2000).ppn, 1)
        with self.assertRaises(InvalidationsPending):
            mmu.finish(tu.id)
        mmu.ack(1, tu.id)
        mmu.finish(tu.id)
        tn2 = mmu.map(a, 0x3000, PERM_RW)
        settle(mmu, tn2)
        self.assertEqual(mmu.walk_table(a, 0x3000).ppn, 0)  # 旧帧此时才复用

    def test_huge_frames_deferred_too(self):
        mmu = MMU()
        a = mmu.create_address_space()
        th = mmu.map(a, 0x4000, PERM_RWX, huge=True)
        settle(mmu, th)
        mmu.switch_to(0, a)
        mmu.translate(a, 0, 0x4abc, AccessType.READ)
        tu = mmu.unmap(a, 0x4000)
        self.assertEqual(tu.notified_cores, (0,))
        self.assertEqual(mmu.mem.used_frames(), 16)  # finish 前 16 帧仍占
        mmu.ack(0, tu.id)
        mmu.finish(tu.id)
        self.assertEqual(mmu.mem.used_frames(), 0)


if __name__ == "__main__":
    unittest.main()
