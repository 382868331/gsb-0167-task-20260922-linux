"""固定输入演示：页表项 A/D 位与“采样并清位”接口（本迭代新增功能）。

直接运行：python demo_iteration.py（纯标准库，离线，毫秒级完成，无 sleep、无预录）。

演示内容：
1. 正常路径：两核读写同页（含只命中 TLB 的访问），采样并清位收割 A/D 标志；
2. 大页标志：大页内不同子页的访问归并到同一个大页 PTE；
3. 采样与写交错：清位后的新访问进入下一世代，旧 TLB 项的迟到回写不污染；
4. 真实触发的边界：权限失败/未映射访问不留痕迹；对已销毁 ASID 采样抛
   InvalidAddressSpace。
"""

from __future__ import annotations

from vmshoot import (
    MMU,
    AccessType,
    InvalidAddressSpace,
    PERM_READ,
    PERM_RW,
    PERM_RWX,
    perm_to_str,
)


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        mmu.ack(0, t.id)
        mmu.ack(1, t.id)
        mmu.finish(t.id)


def show_sample(label: str, res) -> None:
    flagged = res.flagged()
    print(f"  {label}: 收割世代 gen={res.generation}，映射页 {len(res.pages)} 个，"
          f"置位页 {len(flagged)} 个")
    for p in flagged:
        kind = "大页" if p.huge else "小页"
        print(f"    {kind} @0x{p.vaddr:04x}  A={int(p.accessed)} D={int(p.dirty)}")
    if not flagged:
        print("    （无任何置位）")


def main() -> None:
    mmu = MMU()
    asid = mmu.create_address_space()
    print(f"[setup] ASID={asid}，2 核，每核 8 项 LRU TLB；页表项带 A/D 位")

    # ---- 1. 两核读写同页 + 只命中 TLB --------------------------------------
    hr("1. 正常结果：两核读写同页，TLB 命中也记账")
    settle(mmu, mmu.map(asid, 0x1000, PERM_RW))
    r = mmu.translate(asid, 0, 0x1000, AccessType.READ)
    print(f"  核0 读 0x1000  via={r.via}（首次走页表，直接置 PTE 的 A 位）")
    r = mmu.translate(asid, 1, 0x1010, AccessType.READ)
    print(f"  核1 读 0x1010  via={r.via}（核1 首次，走页表）")
    r = mmu.translate(asid, 1, 0x1020, AccessType.WRITE)
    print(f"  核1 写 0x1020  via={r.via}（TLB 命中，标志先记在核1 的待回写区）")
    r = mmu.translate(asid, 0, 0x1030, AccessType.WRITE)
    print(f"  核0 写 0x1030  via={r.via}（TLB 命中）")
    res = mmu.sample_and_clear(asid)
    print("  sample_and_clear：先让两核把 TLB 待回写标志合并回页表，再快照+清位")
    show_sample("第 1 轮采样", res)

    # ---- 2. 大页标志 --------------------------------------------------------
    hr("2. 大页标志：子页访问归并到同一个大页 PTE")
    settle(mmu, mmu.map(asid, 0x4000, PERM_RWX, huge=True))
    mmu.translate(asid, 0, 0x4000, AccessType.READ)
    mmu.translate(asid, 1, 0x4100, AccessType.READ)
    mmu.sample_and_clear(asid)  # 清掉填 TLB 时的痕迹，只看接下来的纯 TLB 命中
    mmu.translate(asid, 0, 0x4200, AccessType.EXECUTE)
    mmu.translate(asid, 1, 0x4F00, AccessType.WRITE)
    show_sample("大页采样", mmu.sample_and_clear(asid))

    # ---- 3. 采样与写交错 + 迟到回写 -----------------------------------------
    hr("3. 采样与写交错：新访问不丢失，旧世代迟到回写不污染")
    mmu.translate(asid, 0, 0x1000, AccessType.WRITE)
    res = mmu.sample_and_clear(asid)
    show_sample("交错第 1 轮（写入后采样）", res)
    mmu.translate(asid, 0, 0x1000, AccessType.WRITE)  # 清位后立刻再写
    res = mmu.sample_and_clear(asid)
    show_sample("交错第 2 轮（清位后的新写被计入）", res)
    # 制造迟到回写：某页在 TLB 里挂着旧标志被采样收割后，该项才被淘汰
    mmu.translate(asid, 0, 0x1000, AccessType.WRITE)  # 核0 待回写 D（旧世代）
    res = mmu.sample_and_clear(asid)
    show_sample("交错第 3 轮（收割旧标志）", res)
    for i in range(8):  # 填满核0 TLB，把 0x1000 的项挤出去 -> 触发迟到回写
        settle(mmu, mmu.map(asid, 0x2000 + i * 0x100, PERM_READ))
        mmu.translate(asid, 0, 0x2000 + i * 0x100, AccessType.READ)
    w = mmu.walk_table(asid, 0x1000)
    print(f"  0x1000 的 TLB 项采样后才被淘汰；无 TLB 遍历器核对："
          f"A={int(w.accessed)} D={int(w.dirty)}（迟到回写被丢弃，未污染）")
    res = mmu.sample_and_clear(asid)
    p = [p for p in res.pages if p.vaddr == 0x1000][0]
    print(f"  下一轮采样中 0x1000：A={int(p.accessed)} D={int(p.dirty)}（保持干净）")

    # ---- 4. 真实触发的边界 --------------------------------------------------
    hr("4. 边界：失败访问不留痕迹；已销毁空间不可采样")
    settle(mmu, mmu.map(asid, 0xE000, PERM_READ))
    r1 = mmu.translate(asid, 0, 0xE000, AccessType.WRITE)   # 权限失败
    r2 = mmu.translate(asid, 0, 0xF000, AccessType.READ)    # 未映射
    print(f"  写只读页 -> fault={r1.fault}；读未映射页 -> fault={r2.fault}")
    res = mmu.sample_and_clear(asid)
    p = [p for p in res.pages if p.vaddr == 0xE000][0]
    print(f"  采样核对 0xE000：A={int(p.accessed)} D={int(p.dirty)}（失败访问无痕迹）")
    mmu.destroy_address_space(asid)
    try:
        mmu.sample_and_clear(asid)
    except InvalidAddressSpace as e:
        print(f"  已销毁 ASID 采样被拒绝：{type(e).__name__}: {e}")

    print("\n演示完成：以上 A/D 标志、世代号与异常均为库实时计算结果。")


if __name__ == "__main__":
    main()
