# vmshoot — ASID 下的页表、TLB 失效与写时复制模拟库

纯 Python 标准库实现的**离线**可复用模块：模拟带地址空间标识（ASID）的两级
页表、大页、每核 LRU TLB、“先改表、发失效票据、逐核 ACK、finish 等待全部
目标核”的 TLB 失效协议，以及**地址空间克隆 + 小页写时复制（COW）**：克隆
共享普通页（帧引用计数 +1），写入时经显式写故障事件分裂，解除映射只减
引用计数、最后引用释放时帧才回收。不调用任何 OS 内存映射（无 `mmap`）、
无第三方依赖、无网络/账号/密钥/数据库/中间件。

## 运行环境

* **Python 版本：3.14.7**（仅使用 3.10+ 通用语法，3.14.7 实测）
* **依赖：无**（只用 Python 标准库，`requirements.txt` 为空说明性文件）
* **平台：Linux 原生运行**，无需 Docker、无需编译

## 准确命令

在仓库根目录（本文件所在目录）执行：

```bash
# 全部必要测试（72 个）
python -m unittest discover -s tests -v

# 固定输入演示（约 0.07 秒，远低于 90 秒；无 sleep、无预录结果）
python demo.py
```

库以包形式导入：`import vmshoot`（或 `from vmshoot import MMU, AccessType, ...`）。

## 体系结构子集（题面固定，非行业标准兼容实现）

| 项 | 取值 |
|---|---|
| 小页大小 | 256 字节，页内偏移 8 位 |
| 页表 | 两级，每级 16 项；L1 索引 4 位、L2 索引 4 位 |
| 虚拟地址宽度 | 16 位（`0x0000..0xffff`，共 65536 字节） |
| 大页 | 一级叶子，4096 字节（= 16 个连续小页），需 4096 对齐 |
| 核数 | 2，核号 `0`、`1` |
| 每核 TLB | 8 项，严格 LRU；key = `(asid, vpn)` |
| 地址空间 | 最多 4 个活动空间，ASID 取 `0..3` |
| 物理内存 | 位图模拟的帧池（默认 1088 帧，可构造时指定），每帧带引用计数与 256 字节内容后备 |

地址拆分：`L1 = (vaddr >> 12) & 0xf`，`L2 = (vaddr >> 8) & 0xf`，
`offset = vaddr & 0xff`。大页 TLB 项以 16 对齐的 vpn 作为 key（覆盖该 16 个小页）。

权限为位向量：`PERM_READ=1`、`PERM_WRITE=2`、`PERM_EXEC=4`，可按位或
（另有 `PERM_RW`/`PERM_RX`/`PERM_RWX`）。

## 公开接口

### 构造与地址空间

```python
from vmshoot import MMU, AccessType, PhysicalMemory

mmu = MMU(PhysicalMemory(num_frames=1088))  # 参数可省略
asid = mmu.create_address_space()           # -> 0..3；满则 AddressSpaceLimit
mmu.destroy_address_space(asid)
```

### map / unmap / protect（先更新页表，再返回失效票据）

```python
t1 = mmu.map(asid, 0x1000, PERM_RW)                  # 小页（256 对齐）
t2 = mmu.map(asid, 0x4000, PERM_RWX, huge=True)     # 大页（4096 对齐）
t3 = mmu.unmap(asid, 0x1000)                         # 小页起始；大页须给大页起始
t4 = mmu.protect(asid, 0x4300, PERM_READ)            # 大页中间地址 -> 自动拆分
```

* **大页局部保护**：`protect` 的语义是“单个 256B 页改权限”。其地址落在大页内
  （含大页 4096 对齐基址）时，就地拆成 16 个小页——目标小页用新权限，其余 15 页
  保留原权限与原帧（物理地址不变、不增帧）。本库不提供“整体大页改权限”的隐式
  快捷路径：需要整体改时，对 16 个小页分别 protect。
* `unmap` **不隐式拆分大页**：大页内部地址抛 `MappingConflict`；请在大页起始
  unmap，或先 `protect` 拆分再逐页 unmap。

### 失效票据协议（`Ticket`）

```python
mmu.ack(core_id, ticket.id)   # 某核确认：立即从该核 TLB 删除票据所列 key
mmu.finish(ticket.id)         # 所有目标核都 ACK 才成功，否则 InvalidationsPending
mmu.pending_tickets(asid=None)
```

* 目标核恒为 `{0,1}`。**ACK 之前旧 TLB 项允许存在**（其他核可观察到陈旧翻译）；
  重复 ACK、ACK 未知票据、finish 未知/已完成票据均抛 `TicketError`。
* `destroy_address_space` 会同步冲刷该 ASID 的全部 TLB 项并释放页表/帧；但
  **未 finish 的票据保留为“僵尸票据”**：仍须逐核 ACK + finish，在此之前该 ASID
  号不能被新空间复用（`AsidBusy`，异常中带票据号），全部 finish 后编号立即可用。

### 翻译与无 TLB 遍历器

```python
r = mmu.translate(asid, core_id, vaddr, AccessType.READ, length=1)
# r.ok=True  -> r.phys_addr；r.perms；r.via ∈ {"tlb","walk"}
# r.ok=False -> r.fault ∈ {"not_mapped","read","write","execute","cow"}
wr = mmu.walk_table(asid, vaddr)   # 绕过 TLB 的纯页表遍历：None 或 WalkResult
```

`translate` 对未映射和权限不足**返回 fault 结果而不抛异常**；`length` 为 1..4096，
访问越过其所在页（小页 256 / 大页 4096）边界时抛 `CrossPageAccess`（大页内跨
小页边界、但不跨 4096 边界是允许的）。`fault="cow"` 表示写访问命中写时复制
共享页（见下节）。

诊断：`mmu.tlb_keys(core_id)`（LRU 顺序的 key 列表）、`mmu.tlb_len(core_id)`、
`mmu.mappings(asid)`（`(起始虚址, 字节数, 权限, 是否大页)` 升序枚举）。

### 地址空间克隆与写时复制（COW）

```python
res = mmu.clone_address_space(asid)   # -> CloneResult(child_asid, ticket)
t = mmu.handle_write_fault(asid, 0x1000)   # 显式处理一次 COW 写故障 -> Ticket
```

* **克隆共享普通页**：`clone_address_space` 为每个小页建立子空间 PTE，父子
  指向**同一物理帧**（帧引用计数 +1），两侧 PTE 都打 COW 标记；**大页不参与
  共享**，克隆时分配新的 16 帧块并逐帧复制内容（深拷贝，见“设计取舍”）。
  返回 `CloneResult`：新子空间 ASID + 父空间失效票据（`kind="clone"`，
  覆盖父空间全部小页 key；ACK 前父空间陈旧可写 TLB 项仍允许命中——与既有
  陈旧窗口语义一致）。
* **写故障是显式、串行事件**：对 COW 共享页的写访问报 `fault="cow"`
  （`translate`）或抛 `MemoryAccessFault(fault="cow")`（`write_bytes`），
  模拟机**不会**自动分裂；调用方须显式调用 `handle_write_fault(asid, vaddr)`
  （256 对齐）：分配新帧、复制源帧 256 字节内容、旧帧引用计数 -1，换 PTE、
  清 COW 标记，并返回该小页的失效票据（`kind="cow"`）。目标未映射抛
  `NotMapped`；大页或非 COW 页抛 `InvalidWriteFault`。
* **引用计数与回收**：`unmap` / `destroy_address_space` 只把帧引用计数减 1，
  计数归零时帧才真正回到帧池（内容同时清零）。诊断：`mmu.mem.frame_refs(ppn)`。
* **失败时原映射保持**：`handle_write_fault` 先分配新帧再改表，帧池耗尽抛
  `FrameExhausted` 时 PTE、COW 标记、引用计数与内容全不变；
  `clone_address_space` 在槽位不足（`AddressSpaceLimit`/`AsidBusy`）或帧池
  耗尽（`FrameExhausted`）时完整回滚——父空间页表/引用计数不变、不安装
  子空间、不产生票据。

### 字节级内容访问（核对共享/分裂语义）

```python
mmu.write_bytes(asid, 0x1000, b"data")   # 按页表写入（不经过 TLB），返回字节数
b = mmu.read_bytes(asid, 0x1000, 4)      # 按页表读取
```

* 物理内存为每帧维护 256 字节内容后备（惰性创建、回收时清零），用于核对
  COW 的“共享可见、分裂分叉”语义；单次读写不得跨页（小页 256 / 大页 4096
  边界），跨页抛 `CrossPageAccess`。
* 未映射 / 权限不足抛 `MemoryAccessFault`（`.fault` 取值同 translate）；
  写 COW 共享页抛 `MemoryAccessFault(fault="cow")`——须先 `handle_write_fault`。
* 内容写路径**不经过 TLB**，因此不受“ACK 前陈旧项”窗口影响：陈旧窗口只
  影响 `translate` 的结果，不影响实际内容写入的安全性。

### 输入上限与错误语义

* 虚拟地址：`0..0xffff`；`length`：`1..4096`；ASID：`0..3`；核号：`0..1`；
  权限位：非零、只含 r/w/x（`1..7`）；活动地址空间最多 4 个。
* **整数参数拒绝 `bool`**（`True` 不是 1）与 `float`（含 `1.0`）；
  `NaN` / `±Infinity` 以 `ValueError` 拒绝；类型不符为 `TypeError`。
  所有错误消息带参数名（如 `vaddr`、`core_id`、`length`），不静默纠正。
* 对齐/越界/重映射等给可定位的具体异常类型（见下“异常”）。
* 有状态操作失败时不留未说明的部分变更：`map` 先分配帧再安装页表项；大页拆分
  是纯指针安装、不分配资源故不可能中途失败；`clone_address_space` 失败时完整
  回滚（父空间零变化）；`handle_write_fault` 先分配新帧再改表，耗尽即整体失败。
* 帧分配做确定性全扫描首位适配，找不到即 `FrameExhausted`——这是当前占用状态下的
  **证明无解**；本模块没有搜索预算类算法，因此不存在“预算耗尽”这一状态。

### 异常层次

```
VMShootError
├── AddressSpaceLimit       # 4 个活动地址空间已满
├── AsidBusy                # 空闲 ASID 号被未 finish 的票据占用，不能复用
├── InvalidAddressSpace     # ASID 不存在/已销毁
├── MappingExists           # 重复映射 / 与大页冲突
├── NotMapped               # unmap/protect/写故障 的目标未映射
├── MappingConflict         # 大页内部 unmap 等结构性冲突（不做隐式拆分）
├── FrameExhausted          # 物理帧确定无解
├── InvalidationsPending    # finish 时仍有核未 ACK（outstanding 给出核列表）
├── TicketError             # 票据未知/已关闭/重复 ACK
├── InvalidWriteFault       # 写故障目标不是 COW 共享小页（大页/私有页）
├── MemoryAccessFault       # read_bytes/write_bytes 访问失败（.fault 给出类别）
└── CrossPageAccess         # 跨页访问
```

## 设计取舍

1. **大页拆分就地完成**：大页本就占用 16 个连续 16 对齐帧，拆分只是把一级叶子
   换成指向同一批帧的 16 个小页 PTE，因此物理地址与“内容”自然保留，且无资源
   分配、无失败窗口。
2. **失效粒度**：小页票据列 1 个 `(asid, vpn)`；大页/拆分票据列出 16 个小页 key，
   使“表已更新但 ACK 未到”窗口里其他核可能新填入的小页项也能被精确失效。
3. **TLB 陈旧是可观察语义**：ACK 到达前，命中旧项的翻译仍按旧权限/旧物理地址
   成功（`via="tlb"`），这是协议允许的窗口；ACK 即删除，下一次访问重走页表。
4. **ASID 复用约束用僵尸票据实现**：销毁空间不等同于撤销正在进行的失效；票据
   生命周期独立，槽位复用严格等待 finish。
5. **克隆只共享小页，大页深拷贝**：大页共享会让“局部 protect 就地拆分”的语义
   无法定义（一侧拆分会牵动另一侧），且本题不交付大页拆分；因此克隆为大页
   分配新帧块并复制内容，COW 只发生在 256 字节小页粒度。
6. **写故障显式串行处理**：`handle_write_fault` 是同步、可重入安全的串行事件
   （先分配、再复制、后换表），不存在并发竞争窗口；帧池耗尽时整体失败且
   原映射保持，没有部分变更。
7. **COW 共享页不进 TLB**：共享页的读写每次都走页表（`via="walk"`），写访问
   才能稳定报 `cow` fault；克隆票据负责把父空间克隆前缓存的旧项失效掉。
8. **帧内容只用于核对**：物理内存为每帧维护 256 字节后备，目的是让“共享可见、
   分裂分叉、回收清零”可被测试与演示真实核对；不模拟访问位/脏位。
9. **不模拟真实硬件一致性**：失效目标核固定为全部 2 核，票据为同步 API 模拟，
   不涉及真实中断/IPI 时序与并发线程。

9. **不模拟真实硬件一致性**：失效目标核固定为全部 2 核，票据为同步 API 模拟，
   不涉及真实中断/IPI 时序与并发线程。
10. **无浮点运算**：全部结果为整数地址/枚举/字节串，因此不存在浮点误差；
    本库没有需要声明误差口径的数值输出。

## 测试

`python -m unittest discover -s tests -v`，共 72 个：

* `test_primitives.py`：固定常量、地址拆分、NaN/Infinity/bool/float 拒绝；
* `test_page_tables.py`：小页/大页映射、首位适配帧号（独立公式推导期望）、
  重复映射、unmap 与帧复用、大页拆分后权限/物理地址/帧数、`FrameExhausted`；
* `test_tlb_invalidation.py`：8 项 LRU 淘汰与刷新、ASID key 隔离、ACK 前后陈旧
  项、finish 阻塞、大页拆分的跨核失效、read/write/execute fault、跨页拒绝、
  ASID 复用被未完成票据阻塞；
* `test_cow_clone.py`：克隆共享小页与引用计数（手算帧号/计数）、父子双写分裂
  与内容分叉、最后引用释放回收、大页克隆深拷贝、克隆票据 ACK 前的陈旧窗口、
  链式克隆（祖/父/子）、`handle_write_fault` 帧耗尽时原映射保持、克隆中途
  耗尽的完整回滚、非法目标/非法输入的明确错误，以及**独立参考机 `RefCOW`**
  （自实现引用计数帧池/内容字典/页表/TLB，不调用被测核心）的 3 个固定种子
  随机序列差分——逐步核对票据、翻译、遍历结果、全部帧的引用计数与页内容；
* `test_oracle.py`：既有独立参考模型 `RefMMU` 的 11 个固定种子差分（不含克隆，
  继续覆盖原有 map/unmap/protect/票据/TLB 行为）。

## 已知限制 / 非目标

* 只实现题面规定的 16 位虚址、两级 16 项页表等固定子集，**不兼容**任何真实
  指令集/行业标准页表格式，也不模拟页表基址寄存器、硬件遍历器、缓存一致性协议。
* COW 只覆盖 256 字节小页：**大页克隆为深拷贝**，不做大页共享、也不做
  “大页写故障时拆成小页”的按需拆分（本次迭代明确不交付大页拆分）。
* 写故障为**显式同步事件**：没有异步缺页票据、没有多核 TLB 一致性协议
  （IPI/应答时序），`handle_write_fault` 一次性完成分裂并返回失效票据。
* 无访问位/脏位；物理地址与帧内容是模拟值，不触碰真实内存。
* 失效目标核固定为全部 2 核，不支持选择性核掩码。
* 默认帧池 1088 帧（4×256+64）；构造 `MMU(PhysicalMemory(num_frames=...))` 可改。
* 不提供持久化、CLI、Web 界面（`demo.py` 仅为固定输入的最小演示）。
