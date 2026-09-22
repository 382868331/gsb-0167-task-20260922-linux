#!/usr/bin/env python3
"""固定输入演示：ASID 两级页表 + 每核 TLB 失效的“失效前/后”对照。

运行：python demo.py（约 8 秒内，纯标准库，无网络/无 sleep/无预录结果）。

演示场景（全部数据在本文件内确定性给出）：
1. 正常结果：4096 大页映射，两核各自翻译并缓存 TLB；
2. 失效前窗口：大页中间两小页改为只读后，ack 之前旧 TLB 仍允许写；
3. 失效完成：核 0 ack 后其陈旧项被清除，同一写访问被真实拒绝（WriteFault）；
   核 1 尚未 ack，陈旧项仍可命中——展示逐核 shootdown；
4. 边界拒绝：跨 256 字节页边界的访问在同一大页内也被拒绝（CrossPageFault）；
5. ASID 复用：销毁票据 finish 后编号被新空间重新取得。
"""

from __future__ import annotations

import time

import asid_vm as vm


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    start = time.perf_counter()
    sys = vm.MemorySystem()
    asid = sys.alloc_asid()
    print(f"参数: 页={vm.PAGE_SIZE}B 大页={vm.HUGE_SIZE}B "
          f"核数={vm.CORE_COUNT} 每核TLB={vm.TLB_SIZE}项 "
          f"最大ASID={vm.MAX_ASIDS} 虚拟空间=0x{vm.VIRT_SIZE:x}")

    # --- 1) 正常映射与翻译 ------------------------------------------------
    hr("1. 正常结果：映射 4096 大页，两核翻译填充 TLB")
    ticket, phys = sys.map(asid, 0x2000, (True, True, True),
                           huge=True, phys=0x40000)
    sys.finish(ticket)  # 此刻无陈旧项，目标核集合为空，可立即 finish
    for core in (0, 1):
        pa = sys.translate(core, asid, 0x2410, vm.Access.READ)
        print(f"  核{core} 读 va=0x2410 -> pa=0x{pa:x}（已缓存进 TLB）")
    print(f"  核0 TLB keys: {[(a, hex(v)) for a, v in sys.tlb_keys(0)]}")

    # --- 2)/3) 大页中间改权限：失效前 vs 失效后 ---------------------------
    hr("2. 中间两小页 0x2400-0x25ff 改只读；先看票据，再逐核 ack")
    pticket = sys.protect(asid, 0x2400, (True, False, False), length=512)
    print(f"  票据 #{pticket.id} 目标核={sorted(pticket.target_cores)} "
          f"区间={[(hex(a), hex(b)) for a, b in pticket.ranges]}")
    print("  -- 失效前（ack 之前，旧 TLB 快照允许存在）--")
    for core in (0, 1):
        pa = sys.translate(core, asid, 0x2410, vm.Access.WRITE)
        print(f"  核{core} 写 va=0x2410 -> pa=0x{pa:x}  [陈旧 TLB 放行]")

    print("  -- 核 0 ack：清除核 0 的陈旧项 --")
    sys.ack(0, pticket)
    try:
        sys.translate(0, asid, 0x2410, vm.Access.WRITE)
    except vm.WriteFault as exc:
        print(f"  核0 写 va=0x{exc.va:x} -> 实际拒绝: {type(exc).__name__}: "
              f"{exc}")
    pa = sys.translate(1, asid, 0x2410, vm.Access.WRITE)
    print(f"  核1 尚未 ack，写 va=0x2410 -> pa=0x{pa:x}  [陈旧 TLB 仍放行]")
    print("  finish 前等待所有目标核：")
    try:
        sys.finish(pticket)
    except vm.PendingFinishError as exc:
        print(f"    -> {type(exc).__name__}: {exc}")
    sys.ack(1, pticket)
    sys.finish(pticket)
    print("  核1 ack 后 finish 完成；大页已拆成 16 小页，"
          "其余 14 页保留 rwx 与原物理地址：")
    for j in (0, 3, 4, 5, 15):
        w = sys.direct_walk(asid, 0x2000 + j * 256)
        print(f"    小页[{j:2d}] va=0x{0x2000 + j * 256:x} "
              f"pa=0x{w.phys_base:x} 权限={w.perms.as_rwx()} huge={w.huge}")

    # --- 4) 边界：跨页访问即使在同一大页内也拒绝 --------------------------
    hr("3. 边界拒绝：20 字节访问从 0x27F0 跨到 0x2803（跨 256B 页边界）")
    try:
        sys.check_access(0, asid, 0x27F0, 20, vm.Access.READ)
    except vm.CrossPageFault as exc:
        print(f"  -> 实际触发 {type(exc).__name__}: {exc}")
    pa = sys.check_access(0, asid, 0x27F0, 16, vm.Access.READ)
    print(f"  同一起始地址长度 16（落在页内）则正常 -> pa=0x{pa:x}")

    # --- 5) ASID 复用 -----------------------------------------------------
    hr("4. ASID 复用：销毁地址空间，票据 finish 前后编号可用性")
    ids = [sys.alloc_asid() for _ in range(vm.MAX_ASIDS - 1)]  # 占满 1,2,3
    print(f"  已占用 ASID: {sorted(ids + [asid])}")
    dt = sys.free_asid(asid)
    print(f"  销毁票据 #{dt.id} 未 finish 时再分配 -> ", end="")
    try:
        sys.alloc_asid()
    except vm.AsidExhaustedError as exc:
        print(f"{type(exc).__name__}: {exc}")
    for core in sorted(dt.target_cores):
        sys.ack(core, dt)
    sys.finish(dt)
    reused = sys.alloc_asid()
    print(f"  全部 ack 并 finish 后重新分配得到 ASID={reused}（干净空间）")
    try:
        sys.translate(0, reused, 0x2000, vm.Access.READ)
    except vm.NotPresentFault as exc:
        print(f"  新空间翻译旧地址 -> {type(exc).__name__}（无历史映射残留）")

    elapsed = time.perf_counter() - start
    print(f"\n演示完成，实际耗时 {elapsed:.3f} 秒（无 sleep、无预录结果）。")


if __name__ == "__main__":
    main()
