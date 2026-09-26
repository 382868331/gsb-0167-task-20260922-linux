"""固定输入演示：页表项 A/D 位、TLB 挂起标志合并与采样世代。

直接运行：``python demo_iteration.py``（纯标准库，离线，毫秒级完成，
无 sleep、无预录结果）。

演示内容：
1. walk 与纯 TLB 命中两条路径都会置 A（写再置 D）；
2. 两核读写同页，sample_and_clear 先合并各核 TLB 再清位、推进世代；
3. 大页标志检查；
4. 边界情况一：权限失败 / 未映射的访问不留下任何痕迹；
5. 边界情况二：采样后旧 TLB 项被逐出，其“迟到的旧世代回写”被丢弃，
   不污染新一轮采样。
"""

from __future__ import annotations

from vmshoot import MMU, AccessType, PERM_READ, PERM_RW, PERM_RWX, PAGE_SIZE, TLB_SIZE


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def settle(mmu: MMU, *tickets) -> None:
    for t in tickets:
        mmu.ack(0, t.id)
        mmu.ack(1, t.id)
        mmu.finish(t.id)


def show_sample(res) -> None:
    print(f"采样世代 {res.generation} 封存，覆盖 {len(res.pages)} 个映射页：")
    for vaddr, f in sorted(res.pages.items()):
        kind = "大页" if f.huge else "小页"
        print(f"  vaddr=0x{vaddr:04x} [{kind}] A={int(f.accessed)} D={int(f.dirty)}")


def main() -> None:
    mmu = MMU()
    asid = mmu.create_address_space()
    settle(mmu, mmu.map(asid, 0x1000, PERM_RW))
    settle(mmu, mmu.map(asid, 0x2000, PERM_READ))
    settle(mmu, mmu.map(asid, 0x4000, PERM_RWX, huge=True))

    hr("1. walk 与纯 TLB 命中都置位")
    r = mmu.translate(asid, 0, 0x1000, AccessType.WRITE)
    print(f"核0 写 0x1000: via={r.via} phys=0x{r.phys_addr:x}  (walk 直接置 PTE)")
    mmu.sample_and_clear(asid)  # 清位，制造干净起点
    hit = mmu.translate(asid, 0, 0x1040, AccessType.WRITE)
    print(f"核0 再写 0x1040: via={hit.via}  (纯 TLB 命中，标志先挂在 TLB 项上)")
    res = mmu.sample_and_clear(asid)
    f = res.pages[0x1000]
    print(f"采样合并后: A={int(f.accessed)} D={int(f.dirty)}  <- TLB 命中同样被记录")

    hr("2. 两核读写同页 + 采样世代推进")
    mmu.translate(asid, 0, 0x1000, AccessType.READ)   # 核0 TLB 命中
    mmu.translate(asid, 1, 0x1000, AccessType.READ)   # 核1 walk
    mmu.translate(asid, 1, 0x1000, AccessType.WRITE)  # 核1 TLB 命中
    res = mmu.sample_and_clear(asid)
    print(f"两个核的挂起标志都已合并；当前世代推进到 {mmu.generation(asid)}")
    show_sample(res)

    hr("3. 大页标志检查")
    mmu.translate(asid, 0, 0x4100, AccessType.WRITE)
    mmu.translate(asid, 1, 0x4F00, AccessType.EXECUTE)
    res = mmu.sample_and_clear(asid)
    f = res.pages[0x4000]
    print(f"大页 0x4000 (4096B): huge={f.huge} A={int(f.accessed)} D={int(f.dirty)}")

    hr("4. 边界：fault 访问不留痕迹")
    bad1 = mmu.translate(asid, 0, 0x2000, AccessType.WRITE)   # 只读页上写
    bad2 = mmu.translate(asid, 0, 0x3000, AccessType.READ)    # 未映射页
    print(f"只读页写: fault={bad1.fault}; 未映射读: fault={bad2.fault}")
    res = mmu.sample_and_clear(asid)
    f = res.pages[0x2000]
    print(f"采样: 0x2000 A={int(f.accessed)} D={int(f.dirty)}（均为 0，无成功访问痕迹）")

    hr("5. 边界：迟到的旧世代回写不污染新一轮")
    mmu.translate(asid, 0, 0x1000, AccessType.WRITE)  # TLB 命中，挂起 D
    res = mmu.sample_and_clear(asid)
    print(f"世代 {res.generation}: 0x1000 D={int(res.pages[0x1000].dirty)}（已结算）")
    # 新世代不碰 0x1000，用其它页把它的 TLB 项挤出 -> 触发迟到回写
    for i in range(1, TLB_SIZE + 2):
        v = 0x1000 + i * PAGE_SIZE
        settle(mmu, mmu.map(asid, v, PERM_READ))
        mmu.translate(asid, 0, v, AccessType.READ)
    res = mmu.sample_and_clear(asid)
    f = res.pages[0x1000]
    print(f"世代 {res.generation}: 0x1000 A={int(f.accessed)} D={int(f.dirty)}"
          f"  <- 旧世代迟到回写被丢弃，新一轮保持干净")

    print("\n演示完成：全部数值均为本次运行真实计算。")


if __name__ == "__main__":
    main()
