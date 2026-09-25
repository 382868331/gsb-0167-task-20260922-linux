"""remap_range 的专项测试。

参考模型 :class:`SmallPageModel` 是**独立逐小页映射表**：每个 ASID 用
``dict[vpn -> (ppn, perms)]`` 表示全部映射（大页展开成 16 个连续小页项），
帧池是独立的位图首位分配器。它不调用 :mod:`vmshoot` 的任何代码取期望值，
只复用题面常量与异常类型。remap 在参考模型上就是“逐页把字典项从源 vpn
搬到目标 vpn”，并独立校验有洞/冲突。

覆盖题面点名的场景：
* 两个核都缓存旧源，ACK 前陈旧翻译可见，finish 后源必须 fault、目标得原物理址；
* 目标地址曾经缓存过别的映射（含 ACK 前的陈旧目标项）；
* 搬迁区间跨大页边界；源大页部分搬走时的拆分；整张大页对齐平移时保持大页；
* 目标冲突 / 源有洞 / 区间重叠 / 参数非法 -> 拒绝且**页表、帧位图、TLB 完全不变**；
* remap 票据未 finish 时阻止 ASID 复用；
* 固定种子的随机序列：参考模型与被测实现逐页差分，失败操作双方状态都不变。
"""

from __future__ import annotations

import copy
import random
import unittest

from vmshoot import (
    MMU,
    AccessType,
    AsidBusy,
    InvalidationsPending,
    MappingExists,
    NotMapped,
    PERM_EXEC,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    PERM_WRITE,
    PhysicalMemory,
    TicketError,
)
from vmshoot.primitives import (
    HUGE_PAGE_SIZE,
    L2_SIZE,
    PAGE_SIZE,
    PERM_RX,
    VADDR_SPACE,
)

NPAGES_TOTAL = VADDR_SPACE // PAGE_SIZE  # 256
ACCESSES = (AccessType.READ, AccessType.WRITE, AccessType.EXECUTE)
NEED_OF = {
    AccessType.READ: PERM_READ,
    AccessType.WRITE: PERM_WRITE,
    AccessType.EXECUTE: PERM_EXEC,
}


# ===========================================================================
# 独立参考模型：逐小页 dict + 独立帧池，不借用被测实现的任何算法
# ===========================================================================
class SmallPageModel:
    def __init__(self, num_frames: int = 1088):
        self.used = bytearray(num_frames)
        # pages[asid][vpn] = (ppn, perms)；huge_bases[asid] = 16 对齐 vpn 集合
        self.pages: dict[int, dict[int, tuple[int, int]]] = {}
        self.huge_bases: dict[int, set[int]] = {}
        self.next_asid = 0

    def alloc_small(self) -> int:
        for i, b in enumerate(self.used):
            if not b:
                self.used[i] = 1
                return i
        raise AssertionError("ref frames exhausted")

    def alloc_huge(self) -> int:
        n = len(self.used)
        for s in range(0, n - L2_SIZE + 1, L2_SIZE):
            if all(self.used[s + j] == 0 for j in range(L2_SIZE)):
                for j in range(L2_SIZE):
                    self.used[s + j] = 1
                return s
        raise AssertionError("ref huge frames exhausted")

    def create(self) -> int:
        a = self.next_asid
        self.next_asid += 1
        self.pages[a] = {}
        self.huge_bases[a] = set()
        return a

    def map_small(self, va: int, perms: int) -> None:
        vpn = va // PAGE_SIZE
        if vpn in self.pages[self.cur]:
            raise AssertionError("ref map exists")
        ppn = self.alloc_small()
        self.pages[self.cur][vpn] = (ppn, perms)

    def map_huge(self, va: int, perms: int) -> None:
        base = va // PAGE_SIZE
        if any(base + j in self.pages[self.cur] for j in range(L2_SIZE)):
            raise AssertionError("ref huge map exists")
        ppn0 = self.alloc_huge()
        for j in range(L2_SIZE):
            self.pages[self.cur][base + j] = (ppn0 + j, perms)
        self.huge_bases[self.cur].add(base)

    cur = 0  # 测试里只有一个活动 ASID，用固定槽即可

    def remap(self, src: int, dst: int, length: int) -> None:
        """参考 remap：逐小页搬字典项，与被测相同的检查次序与错误类型。"""
        n = length // PAGE_SIZE
        pages = self.pages[self.cur]
        for i in range(n):
            if (src // PAGE_SIZE + i) not in pages:
                raise NotMapped("ref hole")
        for i in range(n):
            if (dst // PAGE_SIZE + i) in pages:
                raise MappingExists("ref conflict")
        for i in range(n):
            sv = src // PAGE_SIZE + i
            dv = dst // PAGE_SIZE + i
            pages[dv] = pages.pop(sv)
        # remap 后的大页归属：
        # * 未被源区间触及的大页：保持；
        # * 整 16 页都在搬迁区间内、且目标 4096 对齐：基址平移（保持大页）；
        # * 只搬走一部分：拆分，大页标记消失（不重新合并）。
        src_lo, src_hi = src // PAGE_SIZE, src // PAGE_SIZE + n
        hb = self.huge_bases[self.cur]
        kept: set[int] = set()
        for base in hb:
            touched = any(src_lo <= base + j < src_hi for j in range(L2_SIZE))
            whole = all(src_lo <= base + j < src_hi for j in range(L2_SIZE))
            if not touched:
                kept.add(base)
            elif whole and dst % HUGE_PAGE_SIZE == 0:
                kept.add(dst // PAGE_SIZE + (base - src_lo))
        hb.clear()
        hb.update(kept)

    def full_state(self) -> tuple:
        """全量状态快照：256 个小页的 (ppn, perms) + 大页基集合 + 帧位图副本。"""
        pages = self.pages[self.cur]
        return (
            tuple(pages.get(v) for v in range(NPAGES_TOTAL)),
            frozenset(self.huge_bases[self.cur]),
            bytes(self.used),
        )


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        for c in range(2):
            mmu.ack(c, t.id)
        mmu.finish(t.id)


class TestRemapRange(unittest.TestCase):
    def setUp(self):
        self.mmu = MMU()
        self.a = self.mmu.create_address_space()
        self.ref = SmallPageModel()
        self.ref.create()

    # ---- 对照工具 -----------------------------------------------------------
    def assert_matches_ref(self, msg: str = ""):
        pages = self.ref.pages[self.ref.cur]
        for vpn in range(NPAGES_TOTAL):
            va = vpn * PAGE_SIZE
            w = self.mmu.walk_table(self.a, va)
            exp = pages.get(vpn)
            if exp is None:
                self.assertIsNone(w, f"vpn={vpn} {msg}")
            else:
                self.assertIsNotNone(w, f"vpn={vpn} {msg}")
                # 大页 WalkResult.ppn 是基帧号；逐页帧号用物理地址归一化。
                got_ppn = w.phys_addr // PAGE_SIZE
                self.assertEqual((got_ppn, w.perms), exp, f"vpn={vpn} {msg}")
        # 大页标记必须与参考模型的 huge 集合一致
        for base in sorted(self.ref.huge_bases[self.ref.cur]):
            w = self.mmu.walk_table(self.a, base * PAGE_SIZE)
            self.assertTrue(w.huge, f"base vpn={base} 应保持大页 {msg}")
        # 非大页位置必须不是大页
        huge_pages = {
            b + j for b in self.ref.huge_bases[self.ref.cur] for j in range(L2_SIZE)
        }
        for vpn, exp in pages.items():
            if vpn not in huge_pages:
                w = self.mmu.walk_table(self.a, vpn * PAGE_SIZE)
                self.assertFalse(w.huge, f"vpn={vpn} 不应是大页 {msg}")
        # 物理帧占用位图逐位一致
        self.assertEqual(self.mmu.mem.used_frames(), sum(self.ref.used), msg)
        self.assertEqual(bytes(self.mmu.mem._used), bytes(self.ref.used), msg)

    def snapshot_sut(self):
        """深快照被测机全部可观察状态（页表/帧/TLB/票据号）。"""
        tables = []
        for l1e in self.mmu._spaces[self.a].l1:
            if l1e is None:
                tables.append(None)
            elif isinstance(l1e, list):
                tables.append([None if p is None else (p.ppn, p.perms, p.huge)
                               for p in l1e])
            else:
                tables.append(("H", l1e.ppn, l1e.perms))
        return (
            tables,
            bytes(self.mmu.mem._used),
            [list((k, tuple(v)) for k, v in tlb.items())
             for tlb in self.mmu._tlb],
            self.mmu._next_ticket,
            sorted(self.mmu._tickets),
        )

    # ---- 1. 基本平移：物理页与权限不变，源消失、目标得到原地址 -------------
    def test_small_page_move_preserves_ppn_and_perms(self):
        t = self.mmu.map(self.a, 0x1000, PERM_RW)
        settle(self.mmu, t)
        self.ref.map_small(0x1000, PERM_RW)
        before = self.mmu.walk_table(self.a, 0x1000)
        frames_before = self.mmu.mem.used_frames()

        tr = self.mmu.remap_range(self.a, 0x1000, 0x9000, 0x100)
        self.assertEqual(tr.kind, "remap")
        self.assertEqual((tr.vaddr, tr.dst, tr.length), (0x1000, 0x9000, 0x100))
        settle(self.mmu, tr)
        self.ref.remap(0x1000, 0x9000, 0x100)
        self.assert_matches_ref()

        self.assertIsNone(self.mmu.walk_table(self.a, 0x1000))
        w = self.mmu.walk_table(self.a, 0x9000)
        self.assertEqual((w.ppn, w.perms), (before.ppn, before.perms))
        self.assertEqual(self.mmu.mem.used_frames(), frames_before)  # 零帧增减

    def test_multi_page_permissions_carried_per_page(self):
        perms_of = {0x1000: PERM_RW, 0x1100: PERM_READ, 0x1200: PERM_RX,
                    0x1300: PERM_RWX}
        for va, p in perms_of.items():
            settle(self.mmu, self.mmu.map(self.a, va, p))
            self.ref.map_small(va, p)
        tr = self.mmu.remap_range(self.a, 0x1000, 0x9000, 0x400)
        settle(self.mmu, tr)
        self.ref.remap(0x1000, 0x9000, 0x400)
        self.assert_matches_ref()
        for i, (_, p) in enumerate(perms_of.items()):
            w = self.mmu.walk_table(self.a, 0x9000 + i * PAGE_SIZE)
            self.assertEqual(w.perms, p)

    # ---- 2. 大页：整段对齐平移保持大页；部分搬走则拆分 ---------------------
    def test_whole_huge_aligned_move_keeps_huge(self):
        t = self.mmu.map(self.a, 0x4000, PERM_RWX, huge=True)
        settle(self.mmu, t)
        self.ref.map_huge(0x4000, PERM_RWX)
        before = self.mmu.walk_table(self.a, 0x4000)

        tr = self.mmu.remap_range(self.a, 0x4000, 0x8000, HUGE_PAGE_SIZE)
        # 源 16 页 ∪ 目标 16 页 = 32 个失效 key
        self.assertEqual(len(tr.keys), 2 * L2_SIZE)
        settle(self.mmu, tr)
        self.ref.remap(0x4000, 0x8000, HUGE_PAGE_SIZE)
        self.assert_matches_ref()

        self.assertIsNone(self.mmu.walk_table(self.a, 0x4000))
        w = self.mmu.walk_table(self.a, 0x8ABC)
        self.assertTrue(w.huge)
        self.assertEqual(w.ppn, before.ppn)
        self.assertEqual(w.phys_addr,
                         before.ppn * PAGE_SIZE + (0xABC & (HUGE_PAGE_SIZE - 1)))

    def test_partial_huge_move_splits_source(self):
        t = self.mmu.map(self.a, 0xA000, PERM_RWX, huge=True)
        settle(self.mmu, t)
        self.ref.map_huge(0xA000, PERM_RWX)
        base_ppn = self.mmu.walk_table(self.a, 0xA000).ppn

        # 搬走大页内 vpn 下标 1..2（0xa100、0xa200）到 0x6000
        tr = self.mmu.remap_range(self.a, 0xA100, 0x6000, 2 * PAGE_SIZE)
        settle(self.mmu, tr)
        # 手工更新参考：先在参考上展开大页，再逐页搬
        pages = self.ref.pages[0]
        pages.pop(0xA100 // PAGE_SIZE)
        pages.pop(0xA200 // PAGE_SIZE)
        pages[0x6000 // PAGE_SIZE] = (base_ppn + 1, PERM_RWX)
        pages[0x6100 // PAGE_SIZE] = (base_ppn + 2, PERM_RWX)
        self.ref.huge_bases[0].clear()  # 拆分后源不再是大页
        self.assert_matches_ref("大页部分搬走")

        # 搬走的两页源 not_mapped；同大页其余 14 页帧号/权限不变且变小页
        self.assertIsNone(self.mmu.walk_table(self.a, 0xA100))
        self.assertIsNone(self.mmu.walk_table(self.a, 0xA200))
        for off in (0x000, 0x300, 0x400, 0xF00):
            w = self.mmu.walk_table(self.a, 0xA000 + off)
            self.assertFalse(w.huge)
            self.assertEqual((w.ppn, w.perms),
                             (base_ppn + off // PAGE_SIZE, PERM_RWX))

    def test_whole_huge_to_unaligned_dst_spans_two_blocks(self):
        # 整张大页（16 页）搬到非 4096 对齐目标：目标跨两个 L1 块，
        # 不满足保持条件 -> 源拆分、目标全部装小页，物理址逐页不变。
        t = self.mmu.map(self.a, 0x4000, PERM_RWX, huge=True)
        settle(self.mmu, t)
        self.ref.map_huge(0x4000, PERM_RWX)
        base_ppn = self.mmu.walk_table(self.a, 0x4000).ppn
        tr = self.mmu.remap_range(self.a, 0x4000, 0x9100, HUGE_PAGE_SIZE)
        settle(self.mmu, tr)
        # 参考：源 16 页移到 vpn 0x91..0xA0，且不再保留大页标记
        pages = self.ref.pages[0]
        for q in range(L2_SIZE):
            pages.pop(0x40 + q)
            pages[0x91 + q] = (base_ppn + q, PERM_RWX)
        self.ref.huge_bases[0].clear()
        self.assert_matches_ref("整张大页非对齐平移")
        # 跨 0x9xxx/0xAxxx 块边界处的连续物理址
        self.assertEqual(self.mmu.walk_table(self.a, 0x9F00).ppn, base_ppn + 14)
        self.assertEqual(self.mmu.walk_table(self.a, 0xA000).ppn, base_ppn + 15)
        self.assertFalse(self.mmu.walk_table(self.a, 0x9100).huge)
        self.assertIsNone(self.mmu.walk_table(self.a, 0x4000))
        self.assertIsNone(self.mmu.walk_table(self.a, 0x4F00))

    def test_range_crosses_huge_boundary(self):
        # 两张相邻大页 0x4000 / 0x5000；搬 0x4f00（第一大页末页）和
        # 0x5000（第二大页首页）到 0x9000，跨大页边界且两张大页都要拆分。
        t1 = self.mmu.map(self.a, 0x4000, PERM_RWX, huge=True)
        t2 = self.mmu.map(self.a, 0x5000, PERM_RW, huge=True)
        settle(self.mmu, t1, t2)
        self.ref.map_huge(0x4000, PERM_RWX)
        self.ref.map_huge(0x5000, PERM_RW)
        ppn4f = self.mmu.walk_table(self.a, 0x4F00).phys_addr // PAGE_SIZE
        ppn50 = self.mmu.walk_table(self.a, 0x5000).phys_addr // PAGE_SIZE

        tr = self.mmu.remap_range(self.a, 0x4F00, 0x9000, 2 * PAGE_SIZE)
        settle(self.mmu, tr)
        pages = self.ref.pages[0]
        pages.pop(0x4F00 // PAGE_SIZE)
        pages.pop(0x5000 // PAGE_SIZE)
        pages[0x9000 // PAGE_SIZE] = (ppn4f, PERM_RWX)
        pages[0x9100 // PAGE_SIZE] = (ppn50, PERM_RW)
        self.ref.huge_bases[0].clear()
        self.assert_matches_ref("跨大页边界搬迁")

        self.assertIsNone(self.mmu.walk_table(self.a, 0x4F00))
        self.assertIsNone(self.mmu.walk_table(self.a, 0x5000))
        # 两张大页的其他 30 页仍在，物理址/权限保持
        self.assertEqual(self.mmu.walk_table(self.a, 0x4000).ppn, ppn4f - 15)
        self.assertEqual(self.mmu.walk_table(self.a, 0x5F00).perms, PERM_RW)
        # 目标两页各自带上各自的原帧与原权限
        self.assertEqual(self.mmu.walk_table(self.a, 0x9000).ppn, ppn4f)
        self.assertEqual(self.mmu.walk_table(self.a, 0x9100).perms, PERM_RW)

    # ---- 3. TLB：双核旧源、目标曾缓存别的映射、ACK 窗口、finish 后故障 -----
    def test_both_cores_stale_source_then_fault_after_finish(self):
        settle(self.mmu, self.mmu.map(self.a, 0x3000, PERM_RW))
        r0 = self.mmu.translate(self.a, 0, 0x3000, AccessType.READ)
        r1 = self.mmu.translate(self.a, 1, 0x3000, AccessType.READ)
        self.assertTrue(r0.ok and r1.ok)
        self.assertEqual(r0.phys_addr, r1.phys_addr)

        tr = self.mmu.remap_range(self.a, 0x3000, 0xC000, PAGE_SIZE)
        # ACK 前：两个核的旧源 TLB 项都允许继续命中（协议允许的陈旧窗口）
        for core in (0, 1):
            stale = self.mmu.translate(self.a, core, 0x3000, AccessType.READ)
            self.assertEqual((stale.via, stale.phys_addr), ("tlb", r0.phys_addr))
        # finish 在双核 ACK 前被阻塞
        with self.assertRaises(InvalidationsPending):
            self.mmu.finish(tr.id)

        self.mmu.ack(0, tr.id)
        # 单核 ACK：核 0 源必须重走页表 -> not_mapped；核 1 仍旧
        self.assertEqual(
            self.mmu.translate(self.a, 0, 0x3000, AccessType.READ).fault,
            "not_mapped")
        stale1 = self.mmu.translate(self.a, 1, 0x3000, AccessType.READ)
        self.assertEqual(stale1.via, "tlb")
        self.mmu.ack(1, tr.id)
        self.mmu.finish(tr.id)

        # finish 后：两个核访问源都必须 fault；目标得到原物理地址
        for core in (0, 1):
            gone = self.mmu.translate(self.a, core, 0x3000, AccessType.READ)
            self.assertFalse(gone.ok)
            self.assertEqual(gone.fault, "not_mapped")
            hit = self.mmu.translate(self.a, core, 0xC000, AccessType.READ)
            self.assertTrue(hit.ok)
            self.assertEqual(hit.phys_addr, r0.phys_addr)

    def test_target_previously_cached_another_mapping(self):
        # 先在目标 0x7000 放一个映射并让核 0 缓存，再 unmap 但不让其 ACK，
        # 制造“目标缓存着旧映射”的窗口；随后把源搬到同一目标。
        t_old = self.mmu.map(self.a, 0x7000, PERM_READ)
        settle(self.mmu, t_old)
        old = self.mmu.translate(self.a, 0, 0x7000, AccessType.READ)
        t_unmap = self.mmu.unmap(self.a, 0x7000)  # 故意不 ACK

        settle(self.mmu, self.mmu.map(self.a, 0x2000, PERM_RW))
        src = self.mmu.translate(self.a, 0, 0x2000, AccessType.READ)
        tr = self.mmu.remap_range(self.a, 0x2000, 0x7000, PAGE_SIZE)

        # ACK 前核 0 仍可能在 0x7000 命中 unmap 前的旧 TLB 项
        stale = self.mmu.translate(self.a, 0, 0x7000, AccessType.READ)
        self.assertEqual((stale.via, stale.phys_addr), ("tlb", old.phys_addr))
        # 页表真相已是新映射
        self.assertEqual(self.mmu.walk_table(self.a, 0x7000).ppn,
                         src.phys_addr // PAGE_SIZE)

        # remap 票据的目标 key 与 unmap 票据的 key 相同；ack remap 也摘掉旧项
        self.mmu.ack(0, tr.id)
        self.mmu.ack(1, tr.id)
        self.mmu.finish(tr.id)
        new = self.mmu.translate(self.a, 0, 0x7000, AccessType.READ)
        self.assertEqual((new.via, new.phys_addr), ("walk", src.phys_addr))
        # 收尾旧票据，不影响结果
        self.mmu.ack(0, t_unmap.id)
        self.mmu.ack(1, t_unmap.id)
        self.mmu.finish(t_unmap.id)

    def test_stale_huge_entry_invalidated_on_partial_split(self):
        # 大页只搬一部分：核 0 缓存的是旧大页标记项（16 对齐 key）。
        settle(self.mmu, self.mmu.map(self.a, 0x6000, PERM_RWX, huge=True))
        old = self.mmu.translate(self.a, 0, 0x6100, AccessType.WRITE)
        tr = self.mmu.remap_range(self.a, 0x6100, 0x3000, PAGE_SIZE)
        # 票据必须覆盖源大页的全部 16 个 key（旧大页 TLB 项挂在基 key 上）
        keys = {v for _, v in tr.keys}
        self.assertTrue(set(range(0x60, 0x70)).issubset(keys))
        # ACK 前旧大页项仍让被搬走页陈旧命中
        stale = self.mmu.translate(self.a, 0, 0x6100, AccessType.READ)
        self.assertEqual(stale.via, "tlb")
        self.assertEqual(stale.phys_addr, old.phys_addr)
        self.mmu.ack(0, tr.id)
        self.mmu.ack(1, tr.id)
        self.mmu.finish(tr.id)
        gone = self.mmu.translate(self.a, 0, 0x6100, AccessType.READ)
        self.assertEqual(gone.fault, "not_mapped")
        moved = self.mmu.translate(self.a, 0, 0x3000, AccessType.READ)
        self.assertEqual(moved.phys_addr, old.phys_addr)

    def test_stale_huge_entry_at_destination_block_evicted(self):
        # 目标块 0x9000..0x9fff 曾是一张大页；拆除后核 0 仍缓存旧大页标记项。
        t_old = self.mmu.map(self.a, 0x9000, PERM_RWX, huge=True)
        settle(self.mmu, t_old)
        stale = self.mmu.translate(self.a, 0, 0x9300, AccessType.READ)
        t_rm = self.mmu.unmap(self.a, 0x9000)  # 不 ACK：大页标记项残留在核 0
        settle(self.mmu, self.mmu.map(self.a, 0x2000, PERM_RW))
        src = self.mmu.translate(self.a, 0, 0x2000, AccessType.READ)

        # 只把一个小页搬进该大页块的内部位置 0x9300
        tr = self.mmu.remap_range(self.a, 0x2000, 0x9300, PAGE_SIZE)
        self.mmu.ack(0, tr.id)
        self.mmu.ack(1, tr.id)
        self.mmu.finish(tr.id)

        # 新映射必须生效：目标得到源物理地址
        new = self.mmu.translate(self.a, 0, 0x9300, AccessType.READ)
        self.assertEqual(new.phys_addr, src.phys_addr)
        # 旧大页标记项必须已被摘掉：同块内未映射的邻页不能再陈旧命中
        for off in (0x9000, 0x9200, 0x9400, 0x9F00):
            r = self.mmu.translate(self.a, 0, off, AccessType.READ)
            self.assertEqual(r.fault, "not_mapped", f"off=0x{off:x} 命中了残留大页项")
        # 旧 unmap 票据收尾
        settle(self.mmu, t_rm)
        _ = stale


    def assert_unchanged_on_failure(self, fn, exc_type):
        snap = self.snapshot_sut()
        frames = self.mmu.mem.used_frames()
        with self.assertRaises(exc_type):
            fn()
        self.assertEqual(self.snapshot_sut(), snap, "失败后可观察状态必须完全不变")
        self.assertEqual(self.mmu.mem.used_frames(), frames)

    def test_target_conflict_leaves_everything_unchanged(self):
        for va, p in ((0x1000, PERM_RW), (0x1100, PERM_RW), (0x8000, PERM_READ)):
            settle(self.mmu, self.mmu.map(self.a, va, p))
        # 目标首页空闲、次页被占 -> 冲突
        self.assert_unchanged_on_failure(
            lambda: self.mmu.remap_range(self.a, 0x1000, 0x7F00, 2 * PAGE_SIZE),
            MappingExists)
        # 目标整段落入一张现有大页
        settle(self.mmu, self.mmu.map(self.a, 0xB000, PERM_RWX, huge=True))
        self.assert_unchanged_on_failure(
            lambda: self.mmu.remap_range(self.a, 0x1000, 0xB100, PAGE_SIZE),
            MappingExists)

    def test_source_hole_leaves_everything_unchanged(self):
        settle(self.mmu, self.mmu.map(self.a, 0x1000, PERM_RW))
        settle(self.mmu, self.mmu.map(self.a, 0x1200, PERM_RW))  # 0x1100 是洞
        self.assert_unchanged_on_failure(
            lambda: self.mmu.remap_range(self.a, 0x1000, 0x9000, 3 * PAGE_SIZE),
            NotMapped)

    def test_overlapping_src_dst_rejected(self):
        for va in (0x1000, 0x1100, 0x1200):
            settle(self.mmu, self.mmu.map(self.a, va, PERM_RW))
        # 任何重叠都视为目标碰到现有映射
        self.assert_unchanged_on_failure(
            lambda: self.mmu.remap_range(self.a, 0x1000, 0x1100, 0x200),
            MappingExists)
        self.assert_unchanged_on_failure(
            lambda: self.mmu.remap_range(self.a, 0x1100, 0x1000, 0x200),
            MappingExists)

    def test_bad_arguments_rejected_without_changes(self):
        settle(self.mmu, self.mmu.map(self.a, 0x1000, PERM_RW))
        bad = [
            (0x1001, 0x9000, PAGE_SIZE),       # src 未对齐
            (0x1000, 0x9001, PAGE_SIZE),       # dst 未对齐
            (0x1000, 0x9000, 300),             # length 非页整数倍
            (0x1000, 0x9000, 0),               # length 为 0
            (0x1000, 0x9000, -256),            # length 为负
            (0xFF00, 0x9000, 0x200),           # 源越界
            (0x9000, 0xFF00, 0x200),           # 目标越界
        ]
        for s, d, n in bad:
            with self.subTest(s=s, d=d, n=n):
                self.assert_unchanged_on_failure(
                    lambda s=s, d=d, n=n:
                        self.mmu.remap_range(self.a, s, d, n), ValueError)
        # bool/float 输入沿用全库的类型语义
        with self.assertRaises(TypeError):
            self.mmu.remap_range(self.a, 0x1000, 0x9000, True)
        with self.assertRaises(TypeError):
            self.mmu.remap_range(self.a, True, 0x9000, PAGE_SIZE)

    # ---- 5. 票据协议：重复 ACK / ASID 复用被未结束 remap 票据阻止 ----------
    def test_ticket_acks_and_asid_reuse(self):
        settle(self.mmu, self.mmu.map(self.a, 0x1000, PERM_RW))
        tr = self.mmu.remap_range(self.a, 0x1000, 0x9000, PAGE_SIZE)
        self.assertEqual(self.mmu.pending_tickets(self.a), [tr.id])
        self.mmu.ack(0, tr.id)
        with self.assertRaises(TicketError):
            self.mmu.ack(0, tr.id)  # 重复 ACK
        with self.assertRaises(InvalidationsPending):
            self.mmu.finish(tr.id)

        # 占满其余三个 ASID 后销毁 a；其 remap 票据未 finish -> 0 号不能复用
        others = [self.mmu.create_address_space() for _ in range(3)]
        self.mmu.destroy_address_space(self.a)
        with self.assertRaises(AsidBusy) as cm:
            self.mmu.create_address_space()
        self.assertEqual(cm.exception.asid, self.a)
        self.assertEqual(cm.exception.ticket_ids, [tr.id])
        # finish 后编号立即可复用
        self.mmu.ack(1, tr.id)
        self.mmu.finish(tr.id)
        reused = self.mmu.create_address_space()
        self.assertEqual(reused, self.a)
        for a in others:
            self.mmu.destroy_address_space(a)

    # ---- 6. 固定种子随机差分：成功搬迁双方逐页一致；失败双方状态不变 -------
    def test_random_against_small_page_reference(self):
        rng = random.Random(20260925)
        ref = SmallPageModel()
        ref.create()
        mmu = MMU(PhysicalMemory(1088))
        asid = mmu.create_address_space()
        # 预置一批随机小页/大页映射（全部先 settle，避免陈旧 TLB 干扰翻译对照）
        for _ in range(28):
            if rng.random() < 0.35:
                base = rng.randrange(L2_SIZE) * L2_SIZE  # 大页对齐 vpn
                if any(base + j in ref.pages[0] for j in range(L2_SIZE)):
                    continue
                perms = rng.choice((PERM_READ, PERM_RW, PERM_RWX, PERM_RX))
                settle(mmu, mmu.map(asid, base * PAGE_SIZE, perms, huge=True))
                ref.map_huge(base * PAGE_SIZE, perms)
            else:
                vpn = rng.randrange(NPAGES_TOTAL)
                if vpn in ref.pages[0]:
                    continue
                perms = rng.choice((PERM_READ, PERM_WRITE, PERM_EXEC,
                                    PERM_RW, PERM_RWX, PERM_RX))
                settle(mmu, mmu.map(asid, vpn * PAGE_SIZE, perms))
                ref.map_small(vpn * PAGE_SIZE, perms)

        def state_of(mmu, asid):
            out = {}
            for vpn in range(NPAGES_TOTAL):
                w = mmu.walk_table(asid, vpn * PAGE_SIZE)
                if w is not None:
                    out[vpn] = (w.phys_addr // PAGE_SIZE, w.perms, w.huge)
            return out

        for step in range(400):
            pages = ref.pages[0]
            n = rng.randrange(1, 21)  # 1..20 小页：可跨边界、可部分搬、可整张大页
            src_vpn = rng.randrange(NPAGES_TOTAL - n + 1)
            dst_vpn = rng.randrange(NPAGES_TOTAL - n + 1)
            src, dst = src_vpn * PAGE_SIZE, dst_vpn * PAGE_SIZE
            length = n * PAGE_SIZE

            src_ok = all(src_vpn + i in pages for i in range(n))
            dst_ok = all(dst_vpn + i not in pages for i in range(n))

            if not src_ok:
                with self.assertRaises(NotMapped):
                    mmu.remap_range(asid, src, dst, length)
                continue
            if not dst_ok:
                with self.assertRaises(MappingExists):
                    mmu.remap_range(asid, src, dst, length)
                continue

            # 成功路径：双方各自 remap，再逐页 + 大页标记全量对照
            t = mmu.remap_range(asid, src, dst, length)
            ref.remap(src, dst, length)
            settle(mmu, t)

            got = state_of(mmu, asid)
            for vpn in range(NPAGES_TOTAL):
                exp = pages.get(vpn)
                if exp is None:
                    self.assertNotIn(vpn, got, f"step={step} vpn={vpn}")
                else:
                    self.assertEqual((got[vpn][0], got[vpn][1]), exp,
                                     f"step={step} vpn={vpn}")
            huge_vpns = {b + j for b in ref.huge_bases[0] for j in range(L2_SIZE)}
            for vpn, (_, _, huge) in got.items():
                self.assertEqual(
                    huge, vpn in huge_vpns,
                    f"step={step} vpn={vpn} 大页标记与参考不一致")
            # 大页基址处的物理址必须连续 16 对齐
            for base in ref.huge_bases[0]:
                self.assertTrue(got[base][0] % L2_SIZE == 0)
            # 每一步物理帧总数恒定（remap 永不分配/释放帧）
            self.assertEqual(mmu.mem.used_frames(), sum(ref.used))

        # 终态全量对照（含大页标记）
        for vpn in range(NPAGES_TOTAL):
            w = mmu.walk_table(asid, vpn * PAGE_SIZE)
            exp = ref.pages[0].get(vpn)
            if exp is None:
                self.assertIsNone(w)
            else:
                self.assertEqual((w.phys_addr // PAGE_SIZE, w.perms), exp)


if __name__ == "__main__":
    unittest.main()
