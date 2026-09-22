"""测试辅助：票据结清与“参考模型 vs 被测核心”全量对账。"""

from __future__ import annotations

import asid_vm as vm
from tests.reference import RefNotPresent, RefPermission, RefSpace

ACCESS = [
    (vm.Access.READ, "r"),
    (vm.Access.WRITE, "w"),
    (vm.Access.EXECUTE, "x"),
]
FAULT = {
    "r": vm.ReadFault,
    "w": vm.WriteFault,
    "x": vm.ExecuteFault,
}


def settle(sys: vm.MemorySystem, ticket: vm.Ticket) -> None:
    """按真实顺序完成票据：每个目标核 ack，最后 finish。"""
    for core in sorted(ticket.target_cores):
        sys.ack(core, ticket)
    sys.finish(ticket)


def assert_ref_matches(testcase, sys: vm.MemorySystem, asid: int,
                       ref: RefSpace, *, offsets=(0, 1, 7, 200, 255)) -> None:
    """无 TLB 遍历 + 冷核翻译 与参考模型逐页/逐权限对账。

    参考模型独立给出期望值；本函数只把被测结果与之比较。
    """
    # 1) 无 TLB 遍历器：direct_walk 必须与参考页表逐项一致
    for page_va in range(0, vm.VIRT_SIZE, vm.PAGE_SIZE):
        ref_entry = ref.pages.get(page_va)
        if ref_entry is None:
            with testcase.assertRaises(vm.NotPresentFault):
                sys.direct_walk(asid, page_va)
        else:
            ref_phys, ref_perms = ref_entry
            walked = sys.direct_walk(asid, page_va)
            # 大页 WalkResult 给出 4096 帧基址；按二级索引还原该小页基址
            page_base = walked.phys_base + (
                (page_va & 0xF00) if walked.huge else 0)
            testcase.assertEqual(page_base, ref_phys)
            testcase.assertEqual(
                (walked.perms.readable, walked.perms.writable,
                 walked.perms.executable), ref_perms)

    # 2) 冷核（核 1，每次翻译前清空 TLB）翻译结果必须与参考一致
    for page_va in ref.all_pages():
        for off in offsets:
            va = page_va + off
            for access, ref_access in ACCESS:
                sys.tlbs[1].clear()
                try:
                    expected = ref.translate(va, ref_access)
                except RefNotPresent:
                    with testcase.assertRaises(vm.NotPresentFault):
                        sys.translate(1, asid, va, access)
                    continue
                except RefPermission as exc:
                    with testcase.assertRaises(FAULT[exc.kind]):
                        sys.translate(1, asid, va, access)
                    continue
                got = sys.translate(1, asid, va, access)
                testcase.assertEqual(got, expected)
    sys.tlbs[1].clear()


def _walked_present(sys, asid, va):  # pragma: no cover - 占位防御
    try:
        sys.direct_walk(asid, va)
    except vm.NotPresentFault:
        return False
    return True
