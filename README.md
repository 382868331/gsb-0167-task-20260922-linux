# vmshoot — ASID 下的页表、TLB 失效与 A/D 位采样模拟库

纯 Python 标准库实现的**离线**可复用模块：模拟带地址空间标识（ASID）的两级
页表、大页、每核 LRU TLB，“先改表、发失效票据、逐核 ACK、finish 等待全部
目标核”的 TLB 失效协议，以及页表项 **A/D 位（访问位/脏位）与采样并清位**
接口。不调用任何 OS 内存映射（无 `mmap`）、无第三方依赖、
无网络/账号/密钥/数据库/中间件。

## 运行环境

* **Python 版本：3.14.7**（仅使用 3.10+ 通用语法，3.14.7 实测）
* **依赖：无**（只用 Python 标准库，`requirements.txt` 为空说明性文件）
* **平台：Linux 原生运行**，无需 Docker、无需编译

## 环境准备（venv）

不需要任何第三方依赖，venv 只是为了隔离解释器，可任选其一：

```bash
# 方式一：直接用系统 Python 3.14.7
python3 -m unittest discover -s tests -v

# 方式二：建一个干净 venv（无包可装，激活即可）
python3.14 -m venv .venv
. .venv/bin/activate
```

## 准确命令

在仓库根目录（本文件所在目录）执行：

```bash
# 全部必要测试（72 个）
python -m unittest discover -s tests -v

# 上一迭代的失效协议演示（约 0.05 秒；无 sleep、无预录结果）
python demo.py

# 本迭代新增：A/D 位与采样演示（约 0.05 秒，远低于 90 秒；真实计算、无 sleep）
python demo_iteration.py
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
| 物理内存 | 位图模拟的帧池（默认 1088 帧，可构造时指定） |

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
# r.ok=False -> r.fault ∈ {"not_mapped","read","write","execute"}
wr = mmu.walk_table(asid, vaddr)   # 绕过 TLB 的纯页表遍历：None 或 WalkResult
```

`translate` 对未映射和权限不足**返回 fault 结果而不抛异常**；`length` 为 1..4096，
访问越过其所在页（小页 256 / 大页 4096）边界时抛 `CrossPageAccess`（大页内跨
小页边界、但不跨 4096 边界是允许的）。

诊断：`mmu.tlb_keys(core_id)`（LRU 顺序的 key 列表）、`mmu.tlb_len(core_id)`、
`mmu.mappings(asid)`（`(起始虚址, 字节数, 权限, 是否大页)` 升序枚举）。

### A/D 位与采样并清位（本迭代新增）

每个页表项（PTE，含大页一级叶子）带 `accessed`（A 位）与 `dirty`（D 位）：

* **成功访问才置 A，成功写入再置 D**；权限失败（read/write/execute fault）与
  未映射（not_mapped）的访问不留下任何痕迹；
* **TLB 命中同样记账**：命中时标志先记在该核的“待回写标志”区（按 TLB key
  隔离），在采样合并、该项被 ACK 失效或 LRU 淘汰时写回 PTE；page walk 路径
  直接置 PTE 位；
* 物理地址与权限判定结果与上一迭代完全一致，A/D 机制不改变翻译结果。

```python
res = mmu.sample_and_clear(asid)   # 采样并清位
res.generation                     # 本次收割的世代号（采样后空间世代已 +1）
res.pages                          # 全部映射页的 PageFlags(vaddr, huge, accessed, dirty)
res.flagged()                      # 只列出 A 或 D 置位的页
mmu.sample_generation(asid)        # 当前世代号（下次将收割的世代）
```

`sample_and_clear` 的步骤与**线性化点**：对本模拟的同步执行模型，整个调用是
一个原子操作，线性化点即调用本身——调用前已成功的访问属于被收割的世代，
调用返回后的访问属于下一世代。内部顺序为：

1. **合并**：让两个核把该 ASID 的 TLB 待回写标志写回 PTE，并把各核该项的
   待回写标志清空、世代戳推进到新世代；
2. **快照**：读取所有映射页 PTE 的 A/D 位；
3. **清位**：PTE 的 A/D 位清零；
4. **推进采样世代**（`generation += 1`，按地址空间各自计数）。

由此保证两条关键性质：

* **迟到的旧世代回写不污染下一轮**：采样后旧 TLB 项才被淘汰/失效时，其标志
  世代戳已过期，写回时被直接丢弃（可用 `walk_table` 核对 PTE 位未被重置）；
* **清位期间的新访问不丢失**：清位后的访问在 TLB 里以新世代重新记账（或在
  walk 时直接置 PTE 位），进入下一轮采样。

错误语义：`asid` 非法/已销毁抛 `InvalidAddressSpace`；`bool`/`float` 等
非 int 参数抛 `TypeError`（与全库一致的校验规则）。采样不改变映射、权限、
TLB 内容与票据状态，只读清 A/D 位并推进世代。

### 输入上限与错误语义

* 虚拟地址：`0..0xffff`；`length`：`1..4096`；ASID：`0..3`；核号：`0..1`；
  权限位：非零、只含 r/w/x（`1..7`）；活动地址空间最多 4 个。
* **整数参数拒绝 `bool`**（`True` 不是 1）与 `float`（含 `1.0`）；
  `NaN` / `±Infinity` 以 `ValueError` 拒绝；类型不符为 `TypeError`。
  所有错误消息带参数名（如 `vaddr`、`core_id`、`length`），不静默纠正。
* 对齐/越界/重映射等给可定位的具体异常类型（见下“异常”）。
* 有状态操作失败时不留未说明的部分变更：`map` 先分配帧再安装页表项；大页拆分
  是纯指针安装、不分配资源故不可能中途失败。
* 帧分配做确定性全扫描首位适配，找不到即 `FrameExhausted`——这是当前占用状态下的
  **证明无解**；本模块没有搜索预算类算法，因此不存在“预算耗尽”这一状态。

### 异常层次

```
VMShootError
├── AddressSpaceLimit       # 4 个活动地址空间已满
├── AsidBusy                # 空闲 ASID 号被未 finish 的票据占用，不能复用
├── InvalidAddressSpace     # ASID 不存在/已销毁
├── MappingExists           # 重复映射 / 与大页冲突
├── NotMapped               # unmap/protect 的目标未映射
├── MappingConflict         # 大页内部 unmap 等结构性冲突（不做隐式拆分）
├── FrameExhausted          # 物理帧确定无解
├── InvalidationsPending    # finish 时仍有核未 ACK（outstanding 给出核列表）
├── TicketError             # 票据未知/已关闭/重复 ACK
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
5. **不模拟字节内容**：物理内存只维护帧占用位图与地址翻译，不存放页内数据字节
   （本题契约只要求翻译、权限与失效）。
6. **无浮点运算**：全部结果为整数地址/枚举，因此不存在浮点误差；本库没有需要
   声明误差口径的数值输出。
7. **A/D 位双账本**：PTE 位是权威存储；TLB 命中只更新该核的待回写标志
   （模拟真实硬件中 TLB 项缓存 A/D、未必即时写回），合并点有三处——采样、
   ACK 失效、LRU 淘汰。写回前校验世代戳与 ppn：旧世代标志丢弃，重映射后
   旧映射的痕迹也不会落到新映射上。
8. **采样按地址空间独立**：世代计数器在每个地址空间上，采样一个 ASID 不影响
   其他 ASID 的标志与世代。

## 测试

`python -m unittest discover -s tests -v`，共 72 个：

* `test_primitives.py`：固定常量、地址拆分、NaN/Infinity/bool/float 拒绝；
* `test_page_tables.py`：小页/大页映射、首位适配帧号（独立公式推导期望）、
  重复映射、unmap 与帧复用、大页拆分后权限/物理地址/帧数、`FrameExhausted`；
* `test_tlb_invalidation.py`：8 项 LRU 淘汰与刷新、ASID key 隔离、ACK 前后陈旧
  项、finish 阻塞、大页拆分的跨核失效、read/write/execute fault、跨页拒绝、
  ASID 复用被未完成票据阻塞；
* `test_accessed_dirty.py`：A/D 位与采样——两核读写同页、只命中 TLB 的记账、
  采样与写交错、迟到旧世代回写不污染、LRU 淘汰/ACK 时的标志写回、大页标志、
  失败访问无痕迹、物理地址与权限结果不变、采样错误语义，以及**独立账本**
  测试（测试侧按成功访存事件与采样边界自行推算期望标志，固定种子 400 步
  随机序列逐轮对照）；
* `test_oracle.py`：**独立参考模型** `RefMMU`（自行实现页表/帧池/LRU/票据，
  不调用被测核心算法取期望值），用 11 个固定种子（600 + 10×400 步）的随机
  操作序列双跑差分，逐步核对票据、翻译结果、无 TLB 遍历结果、TLB 内容与帧占用。

## 已知限制 / 非目标

* 只实现题面规定的 16 位虚址、两级 16 项页表等固定子集，**不兼容**任何真实
  指令集/行业标准页表格式，也不模拟页表基址寄存器、硬件遍历器、缓存一致性协议。
* 无实际字节读写；A/D 位为模拟标志（不是真实页表硬件），物理地址是模拟值，
  不触碰真实内存。
* 失效目标核固定为全部 2 核，不支持选择性核掩码；票据为同步 API 模拟，不涉及
  真实中断/IPI 时序与并发线程。
* 默认帧池 1088 帧（4×256+64）；构造 `MMU(PhysicalMemory(num_frames=...))` 可改。
* 不提供持久化、CLI、Web 界面（`demo.py` 仅为固定输入的最小演示）。
