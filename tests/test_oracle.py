"""差分测试：独立参考模型 vs 被测 MMU。

参考模型 :class:`RefMMU` 是按题面契约**独立重写**的朴素实现（dict/list 风格、
独立的帧池与独立的 LRU TLB），不调用 :mod:`vmshoot` 的任何核心算法来取得期望值；
只复用异常类型与题面常量。随后用固定种子的随机操作序列同时驱动两个实现，
逐步比较：票据内容、翻译结果、无 TLB 遍历结果与物理帧占用。
"""

from __future__ import annotations

import random
import unittest
from collections import OrderedDict

from vmshoot import (
    MMU,
    AccessType,
    AddressSpaceLimit,
    AsidBusy,
    CrossPageAccess,
    FrameExhausted,
    InvalidAddressSpace,
    InvalidationsPending,
    MappingConflict,
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

NUM_FRAMES = 256 * 4 + 64  # 与被测默认帧池一致
PERMS_CHOICES = (PERM_READ, PERM_WRITE, PERM_EXEC, PERM_RW, PERM_RWX,
                 PERM_READ | PERM_EXEC)
ACCESSES = (AccessType.READ, AccessType.WRITE, AccessType.EXECUTE)
NEED_OF = {AccessType.READ: PERM_READ, AccessType.WRITE: PERM_WRITE,
           AccessType.EXECUTE: PERM_EXEC}


class RefMMU:
    """按题面独立实现的朴素参考机（与被测实现不共享任何算法代码）。"""

    def __init__(self):
        self.used = [False] * NUM_FRAMES
        # l1 项：None / ("H", base_ppn, perms) / [16 个 (ppn, perms) 或 None]
        self.spaces: dict[int, list] = {}
        self.tlb: list[OrderedDict] = [OrderedDict() for _ in range(2)]
        self.tickets: dict[int, dict] = {}
        self.next_id = 1
        self.current: list[int | None] = [None, None]
        self.participants: dict[int, set] = {}

    # ---- 帧：独立首位适配 ----------------------------------------------
    def alloc_small(self) -> int:
        for i, u in enumerate(self.used):
            if not u:
                self.used[i] = True
                return i
        raise FrameExhausted("ref exhausted")

    def alloc_huge(self) -> int:
        for s in range(0, NUM_FRAMES - 15, 16):
            if all(not self.used[s + j] for j in range(16)):
                for j in range(16):
                    self.used[s + j] = True
                return s
        raise FrameExhausted("ref huge exhausted")

    # ---- ASID -----------------------------------------------------------
    def create(self) -> int:
        active = set(self.spaces)
        if len(active) >= 4:
            raise AddressSpaceLimit("ref limit")
        for cand in range(4):
            if cand in active:
                continue
            if any(t["asid"] == cand for t in self.tickets.values()):
                continue
            self.spaces[cand] = [None] * 16
            self.participants.setdefault(cand, set())
            return cand
        raise AsidBusy(0, [])

    def destroy(self, asid) -> None:
        l1 = self.spaces[asid]
        for e in l1:
            if isinstance(e, tuple):
                _, base, _ = e
                for j in range(16):
                    self.used[base + j] = False
            elif isinstance(e, list):
                for pte in e:
                    if pte is not None:
                        self.used[pte[0]] = False
        del self.spaces[asid]
        for tlb in self.tlb:
            for k in [k for k in tlb if k[0] == asid]:
                del tlb[k]
        for c in range(2):
            if self.current[c] == asid:
                self.current[c] = None
        self.recompute(asid)

    # ---- 参与范围（独立实现，与被测相同的保守三条件） ------------------
    def recompute(self, asid) -> None:
        kept = set()
        for c in range(2):
            if (self.current[c] == asid
                    or any(k[0] == asid for k in self.tlb[c])
                    or any(t["asid"] == asid and c in t["targets"]
                           and c not in t["acks"] for t in self.tickets.values())):
                kept.add(c)
        if kept:
            self.participants[asid] = kept
        else:
            self.participants.pop(asid, None)

    def note_entry(self, core, asid) -> None:
        self.participants.setdefault(asid, set()).add(core)

    def switch(self, core, asid) -> None:
        old = self.current[core]
        if asid is not None and asid not in self.spaces:
            raise InvalidAddressSpace("ref bad switch")
        self.current[core] = asid
        if asid is not None:
            self.note_entry(core, asid)
            self.recompute(asid)
        if old is not None and old != asid:
            self.recompute(old)

    # ---- 遍历 -----------------------------------------------------------
    def walk(self, asid: int, va: int):
        l1i, l2i = (va >> 12) & 0xF, (va >> 8) & 0xF
        e = self.spaces[asid][l1i]
        if e is None:
            return None
        if isinstance(e, tuple):
            _, base, perms = e
            return (base, perms, True)  # 大页返回基帧号；偏移由 translate 处理
        pte = e[l2i]
        return None if pte is None else (pte[0], pte[1], False)

    # ---- 结构操作 -------------------------------------------------------
    def issue(self, asid, kind, va, keys,
              free_small=(), free_huge=()) -> int:
        self.recompute(asid)
        targets = tuple(sorted(self.participants.get(asid, set())))
        tid = self.next_id
        self.next_id += 1
        self.tickets[tid] = {"asid": asid, "kind": kind, "va": va,
                             "keys": tuple(keys), "acks": set(),
                             "targets": targets,
                             "free_small": tuple(free_small),
                             "free_huge": tuple(free_huge)}
        return tid

    @staticmethod
    def keys_for(asid, va, huge):
        if huge:
            base = (va & ~0xFFF) >> 8
            return tuple((asid, base + j) for j in range(16))
        return ((asid, va >> 8),)

    def map(self, asid, va, perms, huge) -> int:
        l1i, l2i = (va >> 12) & 0xF, (va >> 8) & 0xF
        l1 = self.spaces[asid]
        if huge:
            if va & 0xFFF:
                raise ValueError("align")
            if l1[l1i] is not None:
                raise MappingExists("ref")
            base = self.alloc_huge()
            l1[l1i] = ("H", base, perms)
            return self.issue(asid, "map", va, self.keys_for(asid, va, True))
        if va & 0xFF:
            raise ValueError("align")
        e = l1[l1i]
        if isinstance(e, tuple):
            raise MappingExists("ref under huge")
        if e is not None and e[l2i] is not None:
            raise MappingExists("ref exists")
        ppn = self.alloc_small()
        if e is None:
            e = [None] * 16
            l1[l1i] = e
        e[l2i] = (ppn, perms)
        return self.issue(asid, "map", va, self.keys_for(asid, va, False))

    def unmap(self, asid, va) -> int:
        if va & 0xFF:
            raise ValueError("align")
        l1i, l2i = (va >> 12) & 0xF, (va >> 8) & 0xF
        e = self.spaces[asid][l1i]
        if isinstance(e, tuple):
            if l2i != 0:
                raise MappingConflict("ref huge interior")
            _, base, _ = e
            self.spaces[asid][l1i] = None
            return self.issue(asid, "unmap", va, self.keys_for(asid, va, True),
                              free_huge=(base,))
        if e is None or e[l2i] is None:
            raise NotMapped("ref")
        ppn = e[l2i][0]
        e[l2i] = None
        if all(x is None for x in e):
            self.spaces[asid][l1i] = None
        return self.issue(asid, "unmap", va, self.keys_for(asid, va, False),
                          free_small=(ppn,))

    def protect(self, asid, va, perms) -> int:
        if va & 0xFF:
            raise ValueError("align")
        l1i, l2i = (va >> 12) & 0xF, (va >> 8) & 0xF
        e = self.spaces[asid][l1i]
        if isinstance(e, tuple):
            _, base, old_perms = e
            table = [(base + j, old_perms) for j in range(16)]
            table[l2i] = (base + l2i, perms)
            self.spaces[asid][l1i] = table
            return self.issue(asid, "protect", va, self.keys_for(asid, va & ~0xFFF, True))
        if e is None or e[l2i] is None:
            raise NotMapped("ref")
        e[l2i] = (e[l2i][0], perms)
        return self.issue(asid, "protect", va, self.keys_for(asid, va, False))

    # ---- 票据 -----------------------------------------------------------
    def ack(self, core, tid) -> None:
        t = self.tickets.get(tid)
        if t is None:
            raise TicketError("ref no ticket")
        if core not in t["targets"]:
            raise TicketError("ref non-target ack")
        if core in t["acks"]:
            raise TicketError("ref dup ack")
        for k in t["keys"]:
            self.tlb[core].pop(k, None)
        t["acks"].add(core)
        self.recompute(t["asid"])

    def finish(self, tid) -> None:
        t = self.tickets.get(tid)
        if t is None:
            raise TicketError("ref finished")
        asid = t["asid"]
        if any(c not in t["acks"] for c in t["targets"]):
            raise InvalidationsPending(
                tid, tuple(c for c in t["targets"] if c not in t["acks"]))
        for ppn in t["free_small"]:
            self.used[ppn] = False
        for base in t["free_huge"]:
            for j in range(16):
                self.used[base + j] = False
        del self.tickets[tid]
        self.recompute(asid)

    # ---- 翻译 -----------------------------------------------------------
    def translate(self, asid, core, va, need_bit, length):
        if isinstance(length, bool):
            raise TypeError("ref length bool")
        if length <= 0 or length > 4096:
            raise ValueError("ref length")
        if va + length - 1 > 0xFFFF:
            raise ValueError("ref range")
        vpn = va >> 8
        tlb = self.tlb[core]
        entry, via = None, "walk"
        k = (asid, vpn)
        if k in tlb:
            tlb.move_to_end(k)
            entry, via = tlb[k], "tlb"
        else:
            hk = (asid, vpn & ~15)
            if hk in tlb and tlb[hk][2]:
                tlb.move_to_end(hk)
                entry, via = tlb[hk], "tlb"
        if entry is None:
            w = self.walk(asid, va)
            if w is None:
                return (False, None, "not_mapped", "walk")
            entry = w
            if self.current[core] != asid:
                self.current[core] = asid
            self.note_entry(core, asid)
            key = (asid, vpn if not w[2] else vpn & ~15)
            if key in tlb:
                tlb.move_to_end(key)
            else:
                tlb[key] = entry
                if len(tlb) > 8:
                    tlb.popitem(last=False)
        ppn, perms, huge = entry
        region = 4096 if huge else 256
        if (va & (region - 1)) + length > region:
            raise CrossPageAccess("ref cross")
        if not (perms & need_bit):
            fault = {PERM_READ: "read", PERM_WRITE: "write",
                     PERM_EXEC: "execute"}[need_bit]
            return (False, None, fault, via)
        off = va & (0xFFF if huge else 0xFF)
        return (True, ppn * 256 + off, None, via)

    def used_frames(self) -> int:
        return sum(1 for x in self.used if x)


def _ref_create(m):
    return m.create() if isinstance(m, RefMMU) else m.create_address_space()


def _ref_destroy(m, asid):
    return m.destroy(asid) if isinstance(m, RefMMU) else m.destroy_address_space(asid)


class TestDifferential(unittest.TestCase):
    @staticmethod
    def _small_va(rng):
        return rng.randrange(256) * 256

    @staticmethod
    def _huge_va(rng):
        return rng.randrange(16) * 4096

    def _assert_walks_match(self, sut: MMU, ref: RefMMU):
        """无 TLB 遍历器：全部 256 个小页基址上的结果必须与参考机一致。"""
        self.assertEqual(sorted(sut._spaces), sorted(ref.spaces))
        for asid in sorted(ref.spaces):
            for vpn in range(256):
                va = vpn * 256 + 7
                rw = ref.walk(asid, va)
                sw = sut.walk_table(asid, va)
                if rw is None:
                    self.assertIsNone(sw, f"asid={asid} va=0x{va:x}")
                else:
                    self.assertEqual((sw.ppn, sw.perms, sw.huge), rw,
                                     f"asid={asid} va=0x{va:x}")
                    expected_off = va & 0xFFF if rw[2] else va & 0xFF
                    self.assertEqual(sw.phys_addr, rw[0] * 256 + expected_off)

    def _assert_tlb_match(self, sut: MMU, ref: RefMMU):
        for core in range(2):
            self.assertEqual(list(sut._tlb[core].keys()), list(ref.tlb[core].keys()))
            for k in ref.tlb[core]:
                self.assertEqual(sut._tlb[core][k], ref.tlb[core][k])

    def _assert_participants_match(self, sut: MMU, ref: RefMMU):
        asids = set(sut._spaces) | set(ref.spaces) | set(ref.tickets)
        for a in asids:
            if a in sut._spaces:
                got = set(sut.participants(a))
            else:
                sut._recompute_participants(a)
                got = set(sut._participants.get(a, set()))
            self.assertEqual(got, ref.participants.get(a, set()), f"participants asid={a}")
        self.assertEqual([sut.current_address_space(c) for c in range(2)],
                         list(ref.current))

    # ---- 双跑比较器 ------------------------------------------------------
    def _cmp_void(self, fn, sut, ref):
        try:
            fn(ref)
        except Exception as exc:
            with self.assertRaises(type(exc)):
                fn(sut)
            return
        fn(sut)

    def _cmp_id(self, fn, sut, ref):
        try:
            rv = fn(ref)
        except Exception as exc:
            with self.assertRaises(type(exc)):
                fn(sut)
            return
        sv = fn(sut)
        self.assertEqual(sv, rv)

    def _cmp_ticket(self, fn, sut, ref):
        try:
            rid = fn(ref)
        except Exception as exc:
            with self.assertRaises(type(exc)):
                fn(sut)
            return
        st = fn(sut)
        rt = ref.tickets[rid]
        self.assertEqual(st.id, rid)
        self.assertEqual(st.keys, rt["keys"])
        self.assertEqual(st.targets, rt["targets"])
        self.assertEqual(st.asid, rt["asid"])

    def _cmp_translate(self, sut, ref, asid, core, va, acc, length):
        try:
            rv = ref.translate(asid, core, va, NEED_OF[acc], length)
        except Exception as exc:
            with self.assertRaises(type(exc)):
                sut.translate(asid, core, va, acc, length)
            return
        sr = sut.translate(asid, core, va, acc, length)
        self.assertEqual((sr.ok, sr.phys_addr, sr.fault, sr.via), rv,
                         f"asid={asid} core={core} va=0x{va:x} {acc.name} len={length}")

    def test_random_sequence_matches_reference(self):
        self._run_one(20260922, 600)

    def test_ten_more_fixed_seeds(self):
        # 固定种子集合：可复现，不依赖随机环境
        for seed in (1, 2, 7, 42, 100, 31337, 65535, 80808, 999983, 20260923):
            with self.subTest(seed=seed):
                self._run_one(seed, 400)

    def _run_one(self, seed: int, steps: int):
        rng = random.Random(seed)
        sut = MMU(PhysicalMemory(NUM_FRAMES))
        ref = RefMMU()
        for step in range(steps):
            r = rng.random()
            active = sorted(ref.spaces)
            with self.subTest(step=step):
                if r < 0.07:
                    self._cmp_id(lambda m: _ref_create(m), sut, ref)
                elif r < 0.12 and active:
                    a = rng.choice(active)
                    self._cmp_void(lambda m, a=a: _ref_destroy(m, a), sut, ref)
                elif r < 0.18 and active:
                    # 显式上下文切换（含切到 None）：更新参与范围
                    core = rng.randrange(2)
                    new_asid = rng.choice(active + [None])
                    def _switch(m, core=core, new_asid=new_asid):
                        if isinstance(m, RefMMU):
                            m.switch(core, new_asid)
                        else:
                            m.switch_address_space(core, new_asid)
                    self._cmp_void(_switch, sut, ref)
                elif r < 0.46 and active:
                    a = rng.choice(active)
                    huge = rng.random() < 0.35
                    va = self._huge_va(rng) if huge else self._small_va(rng)
                    perms = rng.choice(PERMS_CHOICES)
                    self._cmp_ticket(
                        lambda m, a=a, va=va, p=perms, h=huge:
                            m.map(a, va, p, h) if isinstance(m, RefMMU)
                            else m.map(a, va, p, huge=h), sut, ref)
                elif r < 0.56 and active:
                    a = rng.choice(active)
                    self._cmp_ticket(
                        lambda m, a=a, va=self._small_va(rng): m.unmap(a, va), sut, ref)
                elif r < 0.68 and active:
                    a = rng.choice(active)
                    va = self._small_va(rng)
                    perms = rng.choice(PERMS_CHOICES)
                    self._cmp_ticket(
                        lambda m, a=a, va=va, p=perms: m.protect(a, va, p), sut, ref)
                elif r < 0.90 and active:
                    a = rng.choice(active)
                    core = rng.randrange(2)
                    va = rng.randrange(0x10000)
                    acc = rng.choice(ACCESSES)
                    roll = rng.random()
                    if roll < 0.12:
                        length = rng.choice([0, 4097, True])  # 非法长度
                    elif roll < 0.35:
                        length = rng.randrange(2, 32)        # 可能跨页
                    else:
                        length = 1
                    self._cmp_translate(sut, ref, a, core, va, acc, length)
                elif ref.tickets:
                    tid = rng.choice(sorted(ref.tickets))
                    if rng.random() < 0.6:
                        core = rng.randrange(2)
                        self._cmp_void(lambda m, c=core, t=tid: m.ack(c, t), sut, ref)
                    else:
                        self._cmp_void(lambda m, t=tid: m.finish(t), sut, ref)
                else:
                    self._assert_walks_match(sut, ref)
                    self._assert_participants_match(sut, ref)

        # 收尾：ACK 并 finish 全部票据，做最终全量核对
        for tid in sorted(ref.tickets):
            for core in list(ref.tickets[tid]["targets"]):
                if core not in ref.tickets[tid]["acks"]:
                    ref.ack(core, tid)
                    sut.ack(core, tid)
            ref.finish(tid)
            sut.finish(tid)
        self.assertEqual(sut.pending_tickets(), [])
        self._assert_walks_match(sut, ref)
        self._assert_tlb_match(sut, ref)
        self._assert_participants_match(sut, ref)
        self.assertEqual(sut.mem.used_frames(), ref.used_frames())


if __name__ == "__main__":
    unittest.main()
