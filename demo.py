"""固定输入演示：按参与范围选择性发送 TLB 失效，并保证旧物理页不提前复用。

直接运行：``python demo.py``（纯标准库，离线，毫秒级完成，无 sleep、无预录结果）。

演示内容：
1. 全新地址空间：无核进入过时，map/unmap 票据通知核为空，旧帧可立即回收；
2. 只有核 0 进入并缓存过该空间：unmap 实际只通知核 0（不打扰核 1），
   且旧物理帧在核 0 ACK + finish 前不会被新映射复用；
3. 迟到的旧确认：从未进入过的核 1 对旧票据送来迟到 ACK，参与范围扩大，
   后续票据把核 1 补进实际通知集合（不漏通知）；
4. 真实触发的边界：finish 等待未响应核抛 InvalidationsPending。
"""

from __future__ import annotations

from vmshoot import (
    MMU,
    AccessType,
    InvalidationsPending,
    PERM_RW,
    PERM_RWX,
    PERM_READ,
    perm_to_str,
)


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def settle(mmu: MMU, ticket) -> None:
    """只对票据实际通知的核收 ACK；无目标核的票据发票即完成。"""
    if ticket.closed:
        print(f"  票据 #{ticket.id} 实际通知核=[]：无需任何 ACK")
        return
    for core in ticket.notified_cores:
        mmu.ack(core, ticket.id)
    mmu.finish(ticket.id)
    print(f"  票据 #{ticket.id} 实际通知核={list(ticket.notified_cores)}：均已 ACK 并 finish")


def main() -> None:
    mmu = MMU()
    asid = mmu.create_address_space()
    print(f"[setup] 创建地址空间 ASID={asid}，2 核，每核 8 项 LRU TLB（key 含 ASID）")

    # ---- 1. 无核进入：通知为空、旧帧立即可回收 ----------------------------
    hr("1. 全新空间：没有核可能缓存过时，失效请求为空")
    tm = mmu.map(asid, 0x1000, PERM_RW)
    settle(mmu, tm)
    tu = mmu.unmap(asid, 0x1000)
    settle(mmu, tu)
    print(f"  unmap 后占用帧数={mmu.mem.used_frames()}（无参与核，旧帧摘表即回收）")

    # ---- 2. 只有核 0 参与：只通知核 0，旧帧延迟到 finish -------------------
    hr("2. 只有核 0 缓存过该空间：选择性通知 + 旧物理页不提前复用")
    tm2 = mmu.map(asid, 0x1000, PERM_RW)
    settle(mmu, tm2)
    mmu.switch_to(0, asid)
    r0 = mmu.translate(asid, 0, 0x1000, AccessType.READ)
    print(f"  核0 进入并翻译 0x1000 -> phys=0x{r0.phys_addr:04x} via={r0.via}；"
          f"参与核={list(mmu.participants(asid))}（核1 从未进入）")
    tu2 = mmu.unmap(asid, 0x1000)
    print(f"  unmap 票据 #{tu2.id} 实际通知核={list(tu2.notified_cores)}"
          f"（减少的是对核1的无关失效请求，不是省掉必要检查）")
    try:
        mmu.finish(tu2.id)
    except InvalidationsPending as e:
        print(f"  finish 等待实际通知的核：{type(e).__name__}: 未响应核={list(e.outstanding)}")
    print(f"  finish 前占用帧数={mmu.mem.used_frames()}：旧帧 ppn=0 仍被保留，不能复用")
    tm3 = mmu.map(asid, 0x2000, PERM_RW)
    settle(mmu, tm3)
    w3 = mmu.walk_table(asid, 0x2000)
    print(f"  期间新映射 0x2000 拿到 ppn={w3.ppn}（首位适配跳过仍保留的 ppn=0）")
    mmu.ack(0, tu2.id)
    mmu.finish(tu2.id)
    print(f"  核0 ACK + finish 后占用帧数={mmu.mem.used_frames()}：旧帧现在才允许回收")
    tm4 = mmu.map(asid, 0x3000, PERM_RW)
    settle(mmu, tm4)
    w4 = mmu.walk_table(asid, 0x3000)
    print(f"  之后新映射 0x3000 拿到 ppn={w4.ppn}（旧物理页此刻才被复用）")

    # ---- 3. 迟到的旧确认扩大参与范围 --------------------------------------
    hr("3. 迟到的旧确认：依据更新，后续票据不漏通知")
    tp1 = mmu.protect(asid, 0x3000, PERM_RWX)
    settle(mmu, tp1)
    mmu.translate(asid, 0, 0x3000, AccessType.READ)  # 核0 缓存
    tp2 = mmu.protect(asid, 0x3000, PERM_RW)
    print(f"  protect 票据 #{tp2.id} 发票时实际通知核={list(tp2.notified_cores)}")
    mmu.ack(1, tp2.id)  # 核1 的确认迟到：证明它确实缓存过该空间
    print(f"  收到核1 的迟到旧 ACK 后，参与核={list(mmu.participants(asid))}，"
          f"票据 #{tp2.id} 实际通知核={list(tp2.notified_cores)}（在途范围已保守扩大）")
    tp3 = mmu.protect(asid, 0x3000, PERM_READ)
    print(f"  下一张票据 #{tp3.id} 实际通知核={list(tp3.notified_cores)}（核1 不再漏）")
    for core in (0, 1):
        if core not in tp2.acks:
            mmu.ack(core, tp2.id)
    mmu.finish(tp2.id)
    settle(mmu, tp3)
    w5 = mmu.walk_table(asid, 0x3000)
    print(f"  全部完成后页表真相：0x3000 perms={perm_to_str(w5.perms)} ppn={w5.ppn}")

    print("\n演示完成：实际通知核均为库实时计算结果；只省无关请求，必要等待与旧帧保护不变。")


if __name__ == "__main__":
    main()
