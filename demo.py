"""固定输入演示：ASID 页表 / TLB 失效的正常路径与真实触发的拒绝边界。

直接运行：``python demo.py``（纯标准库，离线，毫秒级完成，无 sleep、无预录结果）。

演示内容：
1. 小页/大页映射与正常翻译（真实计算物理地址）；
2. unmap 的失效票据：ACK 前双核陈旧 TLB 项仍可见、finish 被阻塞，
   全部 ACK + finish 后翻译变为 not_mapped；
3. 大页中间改权限：ACK 前陈旧写仍成功，ACK 后目标页 write fault、邻页保持 rwx；
3b/3c. protect_range 范围改权限：两个核先缓存旧权限，部分大页降权后逐核
   ACK、最终拒绝写入；整段大页统一改权限保留大页形式；
4. 真实触发的边界：跨页访问抛 CrossPageAccess；未完成票据阻塞 ASID 复用 (AsidBusy)。
"""

from __future__ import annotations

from vmshoot import (
    MMU,
    AccessType,
    AsidBusy,
    CrossPageAccess,
    InvalidAddressSpace,
    InvalidationsPending,
    PERM_RW,
    PERM_RWX,
    PERM_RX,
    PERM_READ,
    perm_to_str,
)


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        mmu.ack(0, t.id)
        mmu.ack(1, t.id)
        mmu.finish(t.id)


def show(label: str, result) -> None:
    if result.ok:
        print(f"  {label}: OK  vaddr=0x{result.vaddr:04x} -> phys=0x{result.phys_addr:04x}"
              f"  perms={perm_to_str(result.perms)}  via={result.via}")
    else:
        print(f"  {label}: FAULT={result.fault}  vaddr=0x{result.vaddr:04x}  via={result.via}")


def main() -> None:
    mmu = MMU()
    asid = mmu.create_address_space()
    print(f"[setup] 创建地址空间 ASID={asid}（最多 4 个），2 核，每核 8 项 LRU TLB")

    # ---- 1. 正常映射与翻译 ------------------------------------------------
    hr("1. 正常结果：小页 + 大页映射与翻译")
    t_small = mmu.map(asid, 0x1000, PERM_RW)
    t_huge = mmu.map(asid, 0x4000, PERM_RWX, huge=True)
    settle(mmu, t_small, t_huge)
    print("  map 小页 @0x1000 rw-（票据已双核 ACK + finish）")
    print("  map 大页 @0x4000 rwx，覆盖 0x4000..0x4fff（票据已双核 ACK + finish）")
    show("核0 读 0x1020", mmu.translate(asid, 0, 0x1020, AccessType.READ))
    show("核0 再次读 0x1020", mmu.translate(asid, 0, 0x1020, AccessType.READ))  # 命中 TLB
    show("核1 取指 0x4abc", mmu.translate(asid, 1, 0x4ABC, AccessType.EXECUTE))
    show("核0 写 0x10ff", mmu.translate(asid, 0, 0x10FF, AccessType.WRITE))

    # ---- 2. unmap 失效：ACK 前后 ------------------------------------------
    hr("2. 失效前/后：unmap 0x1000 的票据协议")
    # 先让双核都缓存该页
    mmu.translate(asid, 0, 0x1000, AccessType.READ)
    mmu.translate(asid, 1, 0x1000, AccessType.READ)
    tu = mmu.unmap(asid, 0x1000)
    print(f"  页表已更新；票据 #{tu.id} 发出，目标核={list(tu.targets)}，"
          f"失效 keys={[k for k in tu.keys]}")
    show("[失效前] 核0 读 0x1000（旧 TLB 项允许存在）",
         mmu.translate(asid, 0, 0x1000, AccessType.READ))
    show("[失效前] 核1 读 0x1000（旧 TLB 项允许存在）",
         mmu.translate(asid, 1, 0x1000, AccessType.READ))
    try:
        mmu.finish(tu.id)
    except InvalidationsPending as e:
        print(f"  finish 被阻塞：{e}")
    mmu.ack(0, tu.id)
    print("  核0 已 ACK -> 其 TLB 旧项移除")
    show("[部分 ACK] 核0 读 0x1000", mmu.translate(asid, 0, 0x1000, AccessType.READ))
    show("[部分 ACK] 核1 读 0x1000（仍旧）",
         mmu.translate(asid, 1, 0x1000, AccessType.READ))
    mmu.ack(1, tu.id)
    mmu.finish(tu.id)
    print("  核1 已 ACK；finish 成功")
    show("[失效后] 核1 读 0x1000", mmu.translate(asid, 1, 0x1000, AccessType.READ))

    # ---- 3. 大页中间改权限 ------------------------------------------------
    hr("3. 大页中间改权限：0x4300 rwx -> r--（大页就地拆成 16 小页）")
    mmu.translate(asid, 0, 0x4300, AccessType.WRITE)  # 核0 预填大页项
    tp = mmu.protect(asid, 0x4300, PERM_READ)
    print(f"  页表已拆分并更新；票据 #{tp.id} 覆盖 {len(tp.keys)} 个小页 key")
    show("[失效前] 核0 写 0x4300（旧大页 TLB 项仍可写）",
         mmu.translate(asid, 0, 0x4300, AccessType.WRITE))
    settle(mmu, tp)
    show("[失效后] 核0 写 0x4300（目标页只读）",
         mmu.translate(asid, 0, 0x4300, AccessType.WRITE))
    show("[失效后] 核0 读 0x4300", mmu.translate(asid, 0, 0x4300, AccessType.READ))
    show("[失效后] 核1 写 0x4400（其他映射保持 rwx）",
         mmu.translate(asid, 1, 0x4400, AccessType.WRITE))
    show("[失效后] 核1 取指 0x4f00（其他映射保持 rwx）",
         mmu.translate(asid, 1, 0x4F00, AccessType.EXECUTE))
    # 全部 ACK 后，用无 TLB 遍历器独立核对页表真相
    w_gone = mmu.walk_table(asid, 0x1000)
    w_tgt = mmu.walk_table(asid, 0x4300)
    w_other = mmu.walk_table(asid, 0x4F00)
    print(f"  无 TLB 遍历器核对：0x1000 -> {w_gone}；"
          f"0x4300 -> ppn={w_tgt.ppn} perms={perm_to_str(w_tgt.perms)} huge={w_tgt.huge}；"
          f"0x4f00 -> perms={perm_to_str(w_other.perms)}")

    # ---- 3b. 范围改权限：部分大页降权（protect_range） ---------------------
    hr("3b. protect_range：两个核先缓存旧权限，部分大页降权后逐核确认")
    t_h2 = mmu.map(asid, 0x8000, PERM_RWX, huge=True)
    settle(mmu, t_h2)
    print("  map 大页 @0x8000 rwx（票据已完成）")
    for core in (0, 1):
        mmu.translate(asid, core, 0x8600, AccessType.WRITE)  # 双核缓存旧权限
    print("  核0/核1 均已把 0x8600 的旧权限（rwx）缓存进各自 TLB")
    phys_before = mmu.walk_table(asid, 0x8600).phys_addr
    tr = mmu.protect_range(asid, 0x8500, 0x8700, PERM_READ)
    print(f"  protect_range [0x8500, 0x8700) rwx->r--：大页就地拆成 16 小页，"
          f"票据 #{tr.id}（kind={tr.kind}）覆盖 {len(tr.keys)} 个失效 key；"
          f"物理地址 0x{phys_before:04x} 不变")
    show("[失效前] 核0 写 0x8600（旧 TLB 项仍放行）",
         mmu.translate(asid, 0, 0x8600, AccessType.WRITE))
    show("[失效前] 核1 写 0x8600（旧 TLB 项仍放行）",
         mmu.translate(asid, 1, 0x8600, AccessType.WRITE))
    try:
        mmu.finish(tr.id)
    except InvalidationsPending as e:
        print(f"  finish 被阻塞：{e}")
    mmu.ack(0, tr.id)
    print("  核0 已 ACK -> 其 TLB 旧项移除")
    show("[核0 ACK] 核0 写 0x8600（立即拒绝）",
         mmu.translate(asid, 0, 0x8600, AccessType.WRITE))
    show("[核0 ACK] 核1 写 0x8600（未 ACK，仍旧放行）",
         mmu.translate(asid, 1, 0x8600, AccessType.WRITE))
    mmu.ack(1, tr.id)
    mmu.finish(tr.id)
    print("  核1 已 ACK；finish 成功——此后任何旧 TLB 权限都不能再放行写")
    show("[全部 ACK] 核1 写 0x8600（最终拒绝）",
         mmu.translate(asid, 1, 0x8600, AccessType.WRITE))
    show("[全部 ACK] 核1 读 0x8600（只读仍允许）",
         mmu.translate(asid, 1, 0x8600, AccessType.READ))
    show("[全部 ACK] 核0 写 0x8800（区间外邻居保持 rwx）",
         mmu.translate(asid, 0, 0x8800, AccessType.WRITE))

    # ---- 3c. 整段大页统一改权限：保留大页形式 ------------------------------
    hr("3c. protect_range 整段覆盖大页：保留大页形式")
    t_h3 = mmu.map(asid, 0xC000, PERM_RWX, huge=True)
    settle(mmu, t_h3)
    tr2 = mmu.protect_range(asid, 0xC000, 0xD000, PERM_RX)
    settle(mmu, tr2)
    w = mmu.walk_table(asid, 0xC800)
    print(f"  protect_range [0xc000, 0xd000) rwx->r-x：huge={w.huge}（保留大页），"
          f"perms={perm_to_str(w.perms)}")
    show("核0 写 0xc800（降权后拒绝）", mmu.translate(asid, 0, 0xC800, AccessType.WRITE))
    show("核0 取指 0xc800（r-x 允许）", mmu.translate(asid, 0, 0xC800, AccessType.EXECUTE))

    # ---- 4a. 真实触发的拒绝：跨页访问 -------------------------------------
    hr("4. 真实触发的拒绝/异常边界")
    t_ro = mmu.map(asid, 0x2000, PERM_READ)
    settle(mmu, t_ro)
    try:
        # 0x20fa 起 20 字节：0xfa + 20 = 270 > 256，越过小页边界
        mmu.translate(asid, 0, 0x20FA, AccessType.READ, length=20)
    except CrossPageAccess as e:
        print(f"  跨页访问被拒绝：{type(e).__name__}: {e}")
    # 对照：同样长度在大页内部跨小页边界是允许的
    ok = mmu.translate(asid, 0, 0x4FF0, AccessType.READ, length=16)
    print(f"  对照：大页内 0x4ff0 起 16 字节未跨 4096 边界 -> "
          f"OK phys=0x{ok.phys_addr:04x}")

    # ---- 4b. 真实触发的拒绝：未完成票据阻塞 ASID 复用 ---------------------
    a1 = mmu.create_address_space()
    a2 = mmu.create_address_space()
    a3 = mmu.create_address_space()
    th = mmu.map(a1, 0x0000, PERM_RW)   # 故意不 finish
    mmu.destroy_address_space(a1)       # 销毁空间，但票据 #th 未完成
    print(f"  ASID {a1} 已销毁，但其票据 #{th.id} 未 ACK/finish；"
          f"其余槽位 {a2},{a3} 正常分配")
    try:
        mmu.create_address_space()  # 4 槽位中 1/2/3 活动，0 被僵尸票据占据
    except AsidBusy as e:
        print(f"  ASID 复用被拒绝：{type(e).__name__}: {e}")
    mmu.ack(0, th.id)
    mmu.ack(1, th.id)
    mmu.finish(th.id)
    mmu.destroy_address_space(a2)
    reused = mmu.create_address_space()
    print(f"  票据 #{th.id} 完成并腾出名额后，ASID 号可复用 -> 新 ASID={reused}")

    # ---- 4c. 销毁后再访问即非法地址空间 -----------------------------------
    try:
        mmu.translate(a3, 0, 0x0000, AccessType.READ)
        mmu.destroy_address_space(a3)
        mmu.translate(a3, 0, 0x0000, AccessType.READ)
    except InvalidAddressSpace as e:
        print(f"  销毁后翻译被拒绝：{type(e).__name__}: {e}")

    print("\n演示完成：以上物理地址、fault 与异常均为库实时计算结果。")


if __name__ == "__main__":
    main()
