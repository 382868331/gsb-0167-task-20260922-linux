"""protect_range 演示：两核缓存旧权限 -> 部分大页降权 -> 逐核确认 -> 最终拒绝写入。

直接运行：``python demo_protect_range.py``（纯标准库，离线，毫秒级完成，
无 sleep、无预录结果；不调用任何 OS 内存保护接口）。

时间线（权限生效时点）：
  T0  两核各自访问 0x4300，把大页旧权限 (rwx) 缓存进各自 TLB；
  T1  protect_range 修改页表并发出失效票据 —— 从此刻起，**重走页表**的访问
      一律按新权限裁决；但已缓存旧项的核仍可按旧权限放行（协议允许的窗口）；
  T2  核0 ACK：核0 的旧 TLB 项被删除，核0 写 0x4300 立即被拒绝；核1 仍旧；
  T3  核1 ACK + finish：全部目标核确认完成，此后**任何**旧 TLB 权限都不能
      再放行被禁止的访问。
"""

from __future__ import annotations

from vmshoot import (
    MMU,
    AccessType,
    InvalidationsPending,
    NotMapped,
    PERM_READ,
    PERM_RWX,
    perm_to_str,
)

HUGE_BASE = 0x4000          # 大页 0x4000..0x4fff（16 个小页）
TARGET_LO, TARGET_HI = 0x4300, 0x4500   # 半开区间：大页中间两个小页


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def show(label: str, result) -> None:
    if result.ok:
        print(f"  {label}: OK  -> phys=0x{result.phys_addr:04x}"
              f"  perms={perm_to_str(result.perms)}  via={result.via}")
    else:
        print(f"  {label}: FAULT={result.fault}  via={result.via}")


def main() -> None:
    mmu = MMU()
    asid = mmu.create_address_space()
    print(f"[setup] ASID={asid}；在 0x{HUGE_BASE:04x} 映射 4096 字节大页，权限 rwx")
    t0 = mmu.map(asid, HUGE_BASE, PERM_RWX, huge=True)
    for c in (0, 1):
        mmu.ack(c, t0.id)
    mmu.finish(t0.id)

    hr("T0  两核先把旧权限缓存进各自 TLB")
    show("核0 写 0x4300（walk 后填入 TLB）",
         mmu.translate(asid, 0, 0x4300, AccessType.WRITE))
    show("核0 再写 0x4300（命中 TLB）",
         mmu.translate(asid, 0, 0x4300, AccessType.WRITE))
    show("核1 写 0x4300（walk 后填入 TLB）",
         mmu.translate(asid, 1, 0x4300, AccessType.WRITE))
    show("核1 再写 0x4300（命中 TLB）",
         mmu.translate(asid, 1, 0x4300, AccessType.WRITE))

    hr("T1  protect_range [0x4300, 0x4500) rwx -> r--（部分大页降权）")
    ticket = mmu.protect_range(asid, TARGET_LO, TARGET_HI, PERM_READ)
    print(f"  页表已更新：大页就地拆成 16 个小页，仅区间内 2 页改为 r--，"
          f"其余 14 页保持 rwx；物理帧不变")
    print(f"  返回失效票据 #{ticket.id}（kind={ticket.kind}），"
          f"目标核={list(ticket.targets)}，覆盖 {len(ticket.keys)} 个 TLB key")
    print("  —— 从此时起，重走页表的访问按新权限裁决；旧 TLB 项在 ACK 前仍有效 ——")
    show("[ACK 前] 核0 写 0x4300（旧 TLB 项仍放行）",
         mmu.translate(asid, 0, 0x4300, AccessType.WRITE))
    show("[ACK 前] 核1 写 0x4300（旧 TLB 项仍放行）",
         mmu.translate(asid, 1, 0x4300, AccessType.WRITE))
    try:
        mmu.finish(ticket.id)
    except InvalidationsPending as e:
        print(f"  finish 被阻塞（须等全部目标核 ACK）：{e}")

    hr("T2  核0 确认（逐核 ACK）")
    mmu.ack(0, ticket.id)
    print("  核0 已 ACK：其 TLB 中该区域旧项被删除")
    show("核0 写 0x4300（重走页表 -> 拒绝）",
         mmu.translate(asid, 0, 0x4300, AccessType.WRITE))
    show("核1 写 0x4300（未 ACK，旧项仍放行）",
         mmu.translate(asid, 1, 0x4300, AccessType.WRITE))

    hr("T3  核1 确认 + finish：全部目标核确认完成")
    mmu.ack(1, ticket.id)
    mmu.finish(ticket.id)
    print(f"  票据 #{ticket.id} 已 finish；此后任何旧 TLB 权限都不能再放行")
    show("核1 写 0x4300（最终拒绝写入）",
         mmu.translate(asid, 1, 0x4300, AccessType.WRITE))
    show("核0 写 0x4400（区间内另一页，同样拒绝）",
         mmu.translate(asid, 0, 0x4400, AccessType.WRITE))
    show("核1 读 0x4300（新权限仍允许读）",
         mmu.translate(asid, 1, 0x4300, AccessType.READ))
    show("核0 写 0x4200（区间外小页，保持 rwx）",
         mmu.translate(asid, 0, 0x4200, AccessType.WRITE))
    show("核1 写 0x4f00（区间外小页，保持 rwx）",
         mmu.translate(asid, 1, 0x4F00, AccessType.WRITE))
    w_in = mmu.walk_table(asid, 0x4300)
    w_out = mmu.walk_table(asid, 0x4F00)
    print(f"  无 TLB 遍历器核对：0x4300 -> perms={perm_to_str(w_in.perms)} "
          f"huge={w_in.huge}；0x4f00 -> perms={perm_to_str(w_out.perms)}")

    hr("附  区间中间有空洞：整个操作被拒绝且无副作用")
    t1 = mmu.map(asid, 0x8000, PERM_RWX)   # 0x8100 留作空洞
    t2 = mmu.map(asid, 0x8200, PERM_RWX)
    for t in (t1, t2):
        for c in (0, 1):
            mmu.ack(c, t.id)
        mmu.finish(t.id)
    n_tickets = len(mmu.pending_tickets())
    try:
        mmu.protect_range(asid, 0x8000, 0x8300, PERM_READ)
    except NotMapped as e:
        print(f"  protect_range 被拒绝：{type(e).__name__}: {e}")
    print(f"  无副作用：未产生新票据（仍为 {n_tickets} 张待处理），"
          f"0x8000 权限仍为 {perm_to_str(mmu.walk_table(asid, 0x8000).perms)}")

    print("\n演示完成：以上物理地址、fault 与异常均为库实时计算结果。")


if __name__ == "__main__":
    main()
