# asid_vm —— ASID 下的两级页表与 TLB 失效模拟

一个**纯模拟、可导入**的 Python 标准库模块：两级页表（每级 16 项、页 256 字节、
一级叶支持 4096 字节大页）、2 个核各持 8 项 LRU TLB（key 含 ASID）、
基于“失效票据（shootdown ticket）”的逐核 TLB 失效协议。

**不进行任何真实 OS 映射**：物理地址只是整数，物理帧由库内分配器簿记，
全程不调用 `mmap`/`mprotect` 等系统调用。

## 运行环境

- Python **3.14.7**（仅使用标准库：`dataclasses`、`enum`、`collections`、
  `typing`、`unittest` 等）
- Linux 原生运行，**无需安装任何第三方依赖**，无需网络、账号、密钥、
  数据库、中间件、Docker。

## 准确命令

```bash
# 必要测试（独立参考模型 + 42 个用例）
python -m unittest discover -s tests -v

# 固定输入演示（真实计算，无 sleep、无预录结果；实测约 0.04 秒）
python demo.py
```

## 地址格式与固定参数

虚拟地址 16 位（空间 64 KiB）：

```
| 15..12 一级索引(4bit,16项) | 11..8 二级索引(4bit,16项) | 7..0 页内偏移 |
```

| 常量 | 值 | 含义 |
|---|---|---|
| `PAGE_SIZE` | 256 | 小页字节数 |
| `HUGE_SIZE` | 4096 | 大页字节数（= 16 小页） |
| `L1_ENTRIES` / `L2_ENTRIES` | 16 / 16 | 每级页表项数 |
| `VIRT_SIZE` | 65536 | 虚拟地址空间 |
| `PHYS_SIZE` | 1 MiB | 模拟物理内存上限 |
| `CORE_COUNT` | 2 | 核数 |
| `TLB_SIZE` | 8 | 每核 TLB 项数（LRU） |
| `MAX_ASIDS` | 4 | 同时存在的地址空间上限 |

## 公开接口（`asid_vm` 包）

### 生命周期
- `MemorySystem()`：构造系统（2 核 TLB、帧分配器、票据存储）。
- `alloc_asid() -> int`：分配最小可用 ASID（0..3）。被释放但**仍挂有未完成
  票据**的编号不能复用；4 个编号都不可用时抛 `AsidExhaustedError`。
- `free_asid(asid) -> Ticket`：销毁整个地址空间，归还全部物理帧并产生全目标
  失效票据。该 ASID 存在未完成票据时抛 `PendingFinishError`（无部分变更）。

### 映射操作（先改页表，再产生票据）
- `map(asid, va, perms=None, *, huge=False, phys=None) -> (Ticket, int)`：
  小页要求 `va` 256 对齐；大页要求 4096 对齐且一级表项为空。`perms` 为
  `Permissions(r,w,x)` 或三元组，默认 `rwx`。`phys` 省略时由内部分配器
  给出；显式给出时校验对齐、越界与重叠。已映射抛 `MappingExistsError`。
- `unmap(asid, va, length=256) -> Ticket`：解除区间映射。
- `protect(asid, va, perms, length=256) -> Ticket`：修改区间权限。
  - 区间要求：`va` 256 对齐、`length` 为 256 的正整数倍、不跨 4096 一级
    区域边界；区间内任一文件前未映射则整体失败（`NotMappedError`），不留部分变更。
  - **大页局部操作先拆成 16 个小页**：被覆盖小页应用解除/新权限，其余小页
    原样保留物理地址与权限。16 个小页共享同一个 4096 物理帧，最后一个小页
    消失时整帧才归还分配器。完整覆盖 4096 的操作直接改/删大页叶，不拆分。

### 翻译与访问
- `translate(core, asid, va, access) -> int`：返回物理地址。
  - 未映射抛 `NotPresentFault`；权限不足按访问类型抛
    `ReadFault` / `WriteFault` / `ExecuteFault`。
  - TLB 未命中时遍历页表并填充（8 项 LRU，key = `(asid, va & ~0xFF)`）。
- `check_access(core, asid, va, length, access) -> int`：校验一次多字节访问。
  **跨越 256 字节页边界一律抛 `CrossPageFault`**，即使两页都在同一大页内
  （本题访问模型以小页为粒度，调用方需自行拆跨页访问）。
- `direct_walk(asid, va) -> WalkResult`：**无 TLB 遍历器**，只读页表、
  不碰任何缓存，供验收核对。大页结果的 `phys_base` 为 4096 帧基址
  （小页物理基址 = 帧基址 + 二级索引×256），`huge=True`。

### 失效票据协议（`Ticket`）
- 票据字段：`id, asid, kind, ranges, target_cores, acked, finished`。
- **表先更新、票据后产生**：产生票据时快照各核命中受影响区间的 TLB key，
  这些核构成 `target_cores`。
- `ack(core, ticket)`：目标核确认，**此时**清除该核快照内的陈旧 key。
  ack 之前旧缓存允许继续存在并被命中（模拟真实 shootdown 延迟窗口）。
  非目标核 ack 抛 `NotTicketTargetError`；重复 ack 抛 `AlreadyAckedError`。
- `finish(ticket)`：所有目标核 ack 后才能完成，否则抛 `PendingFinishError`
  （即 finish 必须等待所有目标核）。完成后票据归档，ASID 复用阻塞解除。
- 目标核集合为空时（如映射在全新区域、无任何陈旧缓存）可立即 finish。

## 错误语义（均可定位：消息带参数名与取值）

- 所有异常继承 `VmError`；类型错误 `VmTypeError`，取值错误 `VmValueError`。
- **整数参数拒绝 `bool`**（即使 `bool` 是 `int` 子类）；地址/长度等数值
  拒绝浮点，**NaN 与 ±Infinity 一律拒绝**；不做静默真值纠正。
- 权限位只接受真正的 `bool` 或 `0/1`。
- 非法输入不静默纠正；批次/有状态操作失败时不留下未说明的部分变更
  （map 冲突、unmap/protect 遇空洞、外部帧重叠均为先校验后提交）。

## 设计取舍

1. **TLB 以小页页号为 key 粒度**：大页在某小页首次被访问时填充对应小页 key，
   即一个大页最多占用 16 个 TLB 项。这样失效区间数学最简单（区间均为
   256 的整数倍），且不影响正确性；代价是大页不能用一项 TLB 覆盖 4096。
2. **陈旧窗口的语义是“允许旧值”而非“保证旧值”**：ack 前陈旧项若被 LRU
   自然淘汰，再次访问会遍历到**新**页表。ack 清除快照 key 时若该位置已被
   新遍历项占用，会保守地一并清除（只多一次遍历，绝不放过陈旧数据）。
3. **跨页访问直接拒绝**，不模拟跨页的两段翻译（见上 `check_access`）。
4. 失效目标核集合按“票据产生时确实持有命中项的核”计算；表已先更新，
   此后新产生的 TLB 项一定来自新表，故不可能产生新的陈旧项。
5. 物理帧分配为模拟首块适配（对齐 + 区间合并簿记），与真实内存无关。

## 验证策略（独立参考）

- `tests/reference.py` 是**独立参考模型**：用朴素字典
  `小页号 -> [物理基址, (r,w,x)]` 簿记，大页在参考侧展开为 16 项；
  所有期望值（物理地址、r/w/x 故障类别）由参考模型独立给出，
  **从不调用被测核心计算期望**；被测核心与参考模型的逐项对账
  （含无 TLB 的 `direct_walk`）才是断言主体，“往返一致”仅作补充。
- 覆盖题面验收点：ASID 复用（销毁票据 finish 前后）、大页中间改权限拆分、
  不同核陈旧项逐核失效、跨页访问拒绝、票据状态机、LRU 淘汰与刷新、
  失败原子性、输入校验（bool/NaN/Infinity/越界/对齐）。

## 已知限制 / 非目标

- 仅实现本题契约的有限子集，**不宣称兼容任何真实 ISA/行业标准**
  （无真实页表格式、无缓存一致性协议、无指令集语义）。
- 不模拟访问位/脏位、页表项元数据、TLB 填洞的硬件时序或并发中断；
  票据 ack/finish 为显式调用的协作模型，非异步中断。
- 不模拟数据内容，只翻译地址与判定权限。
- 无浮点计算，故无浮点误差口径问题；地址运算为精确整数。
- 不包含网站、CLI 产品、报表或长时压力测试；终端命令仅用于运行测试与演示。
