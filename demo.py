"""固定输入演示：ASID 页表 / TLB 失效的正常路径与真实触发的拒绝边界。

直接运行：``python demo.py``（纯标准库，离线，毫秒级完成，无 sleep、无预录结果）。

演示内容：
1. 小页/大页映射与正常翻译（真实计算物理地址）；
2. unmap 的失效票据：ACK 前双核陈旧 TLB 项仍可见、finish 被阻塞，
   全部 ACK + finish 后翻译变为 not_mapped；
3. 大页中间改权限：ACK 前陈旧写仍成功，ACK 后目标页 write fault、邻页保持 rwx；
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
    PERM_READ,
    perm_to_str,
)


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        for core in t.targets:  # 只等待实际被通知的核
            mmu.ack(core, t.id)
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

    # ---- 1. 映射时尚无核进入：不发任何失效请求 ----------------------------
    hr("1. 正常结果：建映射（无核进入过 -> 不发失效请求）")
    t_small = mmu.map(asid, 0x1000, PERM_RW)
    t_huge = mmu.map(asid, 0x4000, PERM_RWX, huge=True)
    for t in (t_small, t_huge):
        n = mmu.notifications(t.id)[0]
        print(f"  map 票据 #{t.id}：实际通知核={list(n.notified)}，"
              f"跳过核={list(n.skipped)}（没有任何核可能缓存过 ASID={asid}）")
        mmu.finish(t.id)
    print("  小页 @0x1000 rw-；大页 @0x4000 rwx，覆盖 0x4000..0x4fff")
    show("核1 取指 0x4abc", mmu.translate(asid, 1, 0x4ABC, AccessType.EXECUTE))
    show("核1 再次取指 0x4abc", mmu.translate(asid, 1, 0x4ABC, AccessType.EXECUTE))  # 命中 TLB
    show("核0 读 0x1020", mmu.translate(asid, 0, 0x1020, AccessType.READ))

    # ---- 2. unmap：只有真正缓存过的核被通知；旧页延迟复用 -----------------
    hr("2. 选择性失效：unmap 大页 0x4000 只通知核1，旧页 finish 后才复用")
    other = mmu.create_address_space()
    to = mmu.map(other, 0x9000, PERM_RW)
    mmu.finish(to.id)
    # 核0 切到 other 并连续翻译 8 个页：把它 TLB 中 ASID=0 的那一项淘汰干净
    mmu.switch_address_space(0, other)  # 核0 第一次进入另一个地址空间
    for i in range(8):
        t = mmu.map(other, i * 0x1000, PERM_RW)
        mmu.ack(0, t.id)  # 核0 已在 other 中：被保守纳入目标
        mmu.finish(t.id)
    for i in range(8):
        mmu.translate(other, 0, i * 0x1000, AccessType.READ)
    print(f"  核1 留在 ASID={asid}（缓存过大页 0x4000）；核0 已切到 ASID={other}")
    print(f"  核0 TLB 中已无 ASID={asid} 的项，参与范围={list(mmu.participants(asid))}")
    tu = mmu.unmap(asid, 0x4000)  # 摘表：16 个旧帧进入隔离，暂不复用
    n = mmu.notifications(tu.id)[0]
    print(f"  页表已更新；票据 #{tu.id} 实际通知核={list(n.notified)}，"
          f"跳过核={list(n.skipped)}")
    print("  减少的是对核0 的无关失效请求；判定基于参与证据（当前空间/TLB/欠账），不漏通知")
    show("[失效前] 核1 取指 0x4abc（旧大页 TLB 项允许存在）",
         mmu.translate(asid, 1, 0x4ABC, AccessType.EXECUTE))
    try:
        mmu.finish(tu.id)
    except InvalidationsPending as e:
        print(f"  finish 被阻塞：{e}")
    mmu.ack(1, tu.id)
    mmu.finish(tu.id)
    print("  核1 已 ACK；finish 成功，16 个旧物理帧此刻才回收（未被提前复用）")
    show("[失效后] 核1 取指 0x4abc", mmu.translate(asid, 1, 0x4ABC, AccessType.EXECUTE))

    # ---- 3. 旧确认迟到后参与范围收缩 --------------------------------------
    hr("3. 旧确认迟到与范围收缩：protect 小页 0x1000")
    show("[改权限前] 核1 写 0x1080（按 rw- 成功，旧项带写权限）",
         mmu.translate(asid, 1, 0x1080, AccessType.WRITE))
    tp = mmu.protect(asid, 0x1000, PERM_READ)
    n = mmu.notifications(tp.id)[0]
    print(f"  页表已更新；票据 #{tp.id} 实际通知核={list(n.notified)}，"
          f"跳过核={list(n.skipped)}（核0 已切走且无 ASID={asid} 缓存/欠账）")
    show("[失效前] 核1 写 0x1080（旧 TLB 项仍带写权限）",
         mmu.translate(asid, 1, 0x1080, AccessType.WRITE))
    settle(mmu, tp)
    print("  ACK（含迟到旧确认）到达后参与范围重估；后续票据不再无关等待已切走的核")
    show("[失效后] 核1 写 0x1080（只读）",
         mmu.translate(asid, 1, 0x1080, AccessType.WRITE))
    show("[失效后] 核1 读 0x1080", mmu.translate(asid, 1, 0x1080, AccessType.READ))
    # 无 TLB 遍历器独立核对页表真相
    w_gone = mmu.walk_table(asid, 0x4000)
    w_tgt = mmu.walk_table(asid, 0x1000)
    print(f"  无 TLB 遍历器核对：0x4000 -> {w_gone}；"
          f"0x1000 -> ppn={w_tgt.ppn} perms={perm_to_str(w_tgt.perms)} huge={w_tgt.huge}")

    # ---- 4a. 真实触发的拒绝：跨页访问 -------------------------------------
    hr("4. 真实触发的拒绝/异常边界")
    t_ro = mmu.map(asid, 0x2000, PERM_READ)
    settle(mmu, t_ro)
    try:
        # 0x20fa 起 20 字节：0xfa + 20 = 270 > 256，越过小页边界
        mmu.translate(asid, 1, 0x20FA, AccessType.READ, length=20)
    except CrossPageAccess as e:
        print(f"  跨页访问被拒绝：{type(e).__name__}: {e}")
    # 对照：长度 1 在页边界地址是允许的
    ok = mmu.translate(asid, 1, 0x20FF, AccessType.READ, length=1)
    print(f"  对照：0x20ff 起 1 字节不跨页 -> OK phys=0x{ok.phys_addr:04x}")

    # ---- 4b. 真实触发的拒绝：未完成票据阻塞 ASID 复用 ---------------------
    mmu.destroy_address_space(other)  # 回收演示用空间，腾出槽位
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
    # th 无目标核（没有核进入过 a1）：直接 finish 即释放槽位
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
