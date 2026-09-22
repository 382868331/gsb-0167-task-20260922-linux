"""独立参考模型：测试专用，不 import 被测核心的任何实现逻辑。

参考模型只用最朴素的字典簿记“小页号 -> [物理基址, 权限]”，大页在参考侧
直接展开为 16 个小页项。物理帧由测试用确定性地址显式给出（被测核心也通过
``phys=`` 接收同一地址，仅用于放置模拟数据；期望值全部在本文件内独立
计算，绝不调用 asid_vm 取得期望）。
"""

from __future__ import annotations

import math

PAGE = 256
HUGE = 4096


class RefFault(Exception):
    pass


class RefNotPresent(RefFault):
    pass


class RefPermission(RefFault):
    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind  # "r" / "w" / "x"


class RefSpace:
    """单地址空间参考：page_va -> [phys, (r, w, x)]。"""

    def __init__(self) -> None:
        self.pages: dict[int, list] = {}

    def map_small(self, va: int, phys: int,
                  perms: tuple[bool, bool, bool]) -> None:
        assert va % PAGE == 0 and va not in self.pages
        self.pages[va] = [phys, perms]

    def map_huge(self, va: int, phys: int,
                 perms: tuple[bool, bool, bool]) -> None:
        assert va % HUGE == 0
        for j in range(16):
            p = va + j * PAGE
            assert p not in self.pages
            self.pages[p] = [phys + j * PAGE, perms]

    def protect(self, va: int, length: int,
                perms: tuple[bool, bool, bool]) -> None:
        for off in range(0, length, PAGE):
            p = va + off
            assert p in self.pages, "参考侧：保护未映射页"
            self.pages[p][1] = perms

    def unmap(self, va: int, length: int) -> None:
        for off in range(0, length, PAGE):
            p = va + off
            assert p in self.pages, "参考侧：解除未映射页"
            del self.pages[p]

    def translate(self, va: int, access: str) -> int:
        page_va = va & ~(PAGE - 1)
        entry = self.pages.get(page_va)
        if entry is None:
            raise RefNotPresent(page_va)
        phys, (r, w, x) = entry
        if not {"r": r, "w": w, "x": x}[access]:
            raise RefPermission(access)
        return phys + (va & (PAGE - 1))

    def perms_of(self, va: int) -> tuple[bool, bool, bool]:
        return self.pages[va & ~(PAGE - 1)][1]

    def all_pages(self) -> list[int]:
        return sorted(self.pages)


def is_finite_int(value: object) -> bool:
    """与被测契约平行的独立判定：真整数（拒 bool）、NaN/Infinity 不算。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False
