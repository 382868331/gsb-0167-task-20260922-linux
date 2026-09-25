"""固定输入演示：ASID 页表 / TLB 失效的正常路径与真实触发的拒绝边界。

直接运行：``python demo.py``（纯标准库，离线，毫秒级完成，无 sleep、无预录结果）。

演示内容：
1. 小页/大页映射与正常翻译（真实计算物理地址）；
2. unmap 的失效票据：ACK 前双核陈旧 TLB 项仍可见、finish 被阻塞，
   全部 ACK + finish 后翻译变为 not_mapped；
3. 大页中间改权限：ACK 前陈旧写仍成功，ACK 后目标页 write fault、邻页保持 rwx；
4. 真实触发的边界：跨页访问抛 CrossPageAccess；未完成票据阻塞 ASID 复用 (AsidBusy)。
5. remap_range：整段映射平移，ACK 前双核旧源仍可命中，finish 后源 fault、
   目标得到原物理地址；目标被占用时真实触发 MappingExists 且页表不变。
"""

from __future__ import annotations

from vmshoot import (
    MMU,
    AccessType,
    AsidBusy,
    CrossPageAccess,
    InvalidAddressSpace,
    InvalidationsPending,
    MappingExists,
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

    # ---- 5. remap_range：整段映射平移 --------------------------------------
    hr("5. remap_range：0x6000 起 3 个小页平移到 0xa000（帧与权限不变）")
    mmu2 = MMU()
    a = mmu2.create_address_space()
    t_a = mmu2.map(a, 0x6000, PERM_RW)
    t_b = mmu2.map(a, 0x6100, PERM_RX)
    t_c = mmu2.map(a, 0x6200, PERM_READ)
    settle(mmu2, t_a, t_b, t_c)
    # 双核缓存旧源页
    mmu2.translate(a, 0, 0x6000, AccessType.READ)
    mmu2.translate(a, 1, 0x6100, AccessType.READ)
    before = [mmu2.walk_table(a, 0x6000 + j * 0x100).phys_addr for j in range(3)]
    print(f"  源三页物理地址（页基）: {[hex(p) for p in before]}")
    tr = mmu2.remap_range(a, 0x6000, 0xA000, 3 * 0x100)
    print(f"  页表已平移；票据 #{tr.id} 覆盖 {len(tr.keys)} 个 key（源+目标）")
    show("[失效前] 核0 读 0x6000（旧 TLB 项允许存在）",
         mmu2.translate(a, 0, 0x6000, AccessType.READ))
    show("[失效前] 核1 读 0x6100（旧 TLB 项允许存在）",
         mmu2.translate(a, 1, 0x6100, AccessType.READ))
    settle(mmu2, tr)
    show("[失效后] 核0 读 0x6000（源必须 fault）",
         mmu2.translate(a, 0, 0x6000, AccessType.READ))
    for j in range(3):
        show(f"[失效后] 核1 读 0x{0xA000 + j * 0x100:04x}（目标得到原物理地址）",
             mmu2.translate(a, 1, 0xA000 + j * 0x100, AccessType.READ))
    after = [mmu2.walk_table(a, 0xA000 + j * 0x100).phys_addr for j in range(3)]
    print(f"  目标三页物理地址（页基）: {[hex(p) for p in after]}  "
          f"-> 与源完全一致: {before == after}")

    # ---- 5b. 真实触发的拒绝：目标被占用，页表完全不变 ----------------------
    t_occ = mmu2.map(a, 0xC000, PERM_RW)
    settle(mmu2, t_occ)
    n_maps = len(mmu2.mappings(a))
    try:
        mmu2.remap_range(a, 0xA000, 0xC000, 0x100)  # 0xc000 已被占用
    except MappingExists as e:
        print(f"  目标冲突被拒绝：{type(e).__name__}: {e}")
    print(f"  失败后页表完全不变：映射数 {n_maps} == {len(mmu2.mappings(a))}，"
          f"帧占用 {mmu2.mem.used_frames()} 帧")

    print("\n演示完成：以上物理地址、fault 与异常均为库实时计算结果。")


if __name__ == "__main__":
    main()
