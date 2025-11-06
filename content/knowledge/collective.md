---
title: "集合通信"
date: 2025-11-05T15:30:46+08:00
draft: false
description: ""
---

集合通信

# 集合通信的定义
集合通信（Collective Communication）是一种涉及一组进程或节点的通信模式，所有参与进程协同完成数据传输、同步或规约等操作，常用于分布式计算、AI训练等场景。它包括以下核心特点：

1. 多对多通信：通信涉及一个进程组（或通信域）中的所有成员。
2. 同步性：所有参与者必须同时调用相同的集合通信操作。
3. 无需指定标签（tag）：与点对点通信不同，集合通信不需要指定消息标签。
4. 常用于梯度同步、参数更新、数据广播等场景。

与集合通信相对的是点对点通信：P2P（point to point）。它只涉及两个进程，一个接收者，一个发送者，常用操作只有send和recv。通信双方需要明确指定对方身份，还需要指定消息标签tag来区分不同消息。

P2P通信即点对点通信，是底层基础通信机制，聚焦于单一发送方与接收方间的数据传递，像快递点对点精准投送，有明确收发两端，依序完成信息交换，是分布式计算里进程间“一对一”交互基石；集合通信则是“多对多”协同作业，进程组集体参与，像大型乐团合奏，依特定规则同步数据、规约结果，常见广播、分发、收集、规约操作，用于梯度同步、参数更新，强调群体协作与数据一致性，二者差异关键在于通信规模、参与角色及数据流向，P2P通信是“个体对接”，集合通信是“群体共舞”，分别在分布式系统不同层面发挥关键作用。实际上集合通信也可以看作是P2P通信的上层抽象，集合通信可以通过P2P通信来实现。

## 常用集合通信操作
| 操作           | 通信模式 | 主要功能描述                                                                 | 典型用途示例                      |
|----------------|----------|------------------------------------------------------------------------------|-----------------------------------|
| Broadcast      | 1→N      | 根进程把同一份数据复制到所有进程内存                                        | 分发模型权重、超参数              |
| Scatter        | 1→N      | 根进程将数据分块，第 i 块发给 rank-i                                        | 划分训练样本、矩阵行分块          |
| Gather         | N→1      | 根进程按秩收集各进程发来的块并顺序拼接                                      | 收集各卡损失、梯度片段            |
| AllGather      | N→N      | Gather 的全员版，拼接结果最终人手一份                                       | 拼接完整特征、样本标签            |
| Reduce         | N→1      | 所有进程数据按算子规约到根进程                                              | 全局求和损失、统计量              |
| AllReduce  | N→N      | 先做 Reduce 再 Broadcast，结果全员一致                                      | 梯度同步、参数更新（最常用）      |
| ReduceScatter  | N→N      | 全局规约后把结果切片，第 i 片留在 rank-i                                    | 大模型分片优化器状态、梯度切片    |
| AlltoAll       | N↔N      | 每进程与每进程交换一块数据，实现全连接数据重排                              | MoE 专家并行、矩阵转置            |
| Barrier        | N⇄       | 纯同步，无数据搬运；所有进程到达后才继续                                    | 训练阶段切换、调试同步点          |

## 集合通信操作详解
以NCCL ALL Reduce操作为例：

ncclResult_tncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
| # | 参数名 | 输入/输出 | 必须内容 | 常见坑/提示 |
|---|--------|-----------|----------|-------------|
| ① | `sendbuff` | **IN** | 本进程要贡献的 **设备内存** 首地址（GPU） | 必须指向 **GPU 内存** 且与 `datatype` 对齐；可以和 ② 相同（in-place）。 |
| ② | `recvbuff` | **OUT** | 本进程接收 **规约结果** 的 GPU 内存首地址 | 大小 ≥ `count * sizeof(type)`；in-place 时传 `sendbuff` 即可。 |
| ③ | `count` | **IN** | 元素个数（**不是字节数**） | 所有进程必须传 **相同值**，否则行为未定义。 |
| ④ | `datatype` | **IN** | `ncclFloat16`, `ncclFloat32`, `ncclInt32` … | 所有进程类型必须一致；NCCL 不支持任意用户自定义类型。 |
| ⑤ | `op` | **IN** | `ncclSum`, `ncclProd`, `ncclMax`, `ncclMin` … | 仅支持内置算子；如需自定义，只能用 `ncclRedOpCreatePreMulSum` 等有限扩展。 |
| ⑥ | `comm` | **IN** | 事先通过 `ncclCommInitRank` 创建好的 **通信域** | 必须保证 **所有属于同一 comm 的进程都参与** 且 **顺序一致**，否则死锁。 |
| ⑦ | `stream` | **IN** | 本次通信要 **排队** 的 CUDA stream | 通信与计算并行时，务必保证 `sendbuff`/`recvbuff` 在该流上 **已就绪**；多流注意数据依赖。 |

对于集合通信操作，有这些事实需要明确：

1. 设备群中哪些进程参与到通信当中是由通信域来决定的。所有通信域的进程必须调用同一个操作。
2. 此操作中的流这个概念是CUDA独有的，并不属于集合通信里面。CUDA stream将所有GPU操作放进串行队列里面执行，这个队列就叫做stream，一条流是CPU交给GPU的命令队列。NCCL 内部最终也是调用 CUDA kernel 做 reduce、copy、P2P 等，因此必须告诉它 “把 kernel 插到哪个队列里去”。收到并行度硬件槽位限制，一般真正同时执行的只有几十条硬件流，多余流被驱动时分复用。

# 调用集合通信原语之后发生了什么

1. 调用集合通信操作之后，经过参数检查之后，NCCL将这次通信的所有信息打包成一个work item，enqueue进 enqueue 进用户 CUDA stream。排队等待前面所有任务先完成。 work item是对一次通信任务的描述，是 NCCL 内部结构体，保存了「操作类型、数据指针、大小、环形拓扑」等信息。当轮到此work item时发射此kernel。

2. NCCL设备kernel启动，CUDA driver 把 预编译的 kernel 扔进 GPU compute queue。

3. NCCL在初始化阶段会探测硬件拓扑。然后在Ring， Tree和NVLS，CollNet等几种算法里面挑选一种最快的，也可以使用环境变量强制指定。这时数据就准备进行通信了。

4. 一条数据进入到Kernel之后会被逐步切分，chunk - slice -packet。 首先数据在kernel入口就带有总的内存大小参数，因此，算法层面会将这块数据切分成多个chunk。然后进入到流水线层，每个block再把chunk切成slice，形成双缓冲流水线，计算与传输重叠。最后，每个chunk再被隐式切成Packet DMA事务。

5. 最后一个packet写回done flag，kernel结束。同stream后续任务自动等它完成。跨流用event同步。

# 拓扑

## NVLink与NVSwitch
NVLink是Nvidia私有高速串行总线，替代PCIe做GPU-GPU或者GPU-CPU直联。
- GPU-GPU P2P 内存拷贝、P2P 原子
- 比 PCIe 带宽高 4-10 倍、延迟低 2-3 倍
- 也能连 IBM Power CPU（OpenCAPI）或 Grace CPU（Chip-to-Chip）

NVSwitch
ASIC交换芯片，功能类似于以太网Switch，专门管理NVLink信号。一颗 NVSwitch 提供 18 端口 × 8 lane（Ampere 代），每端口 50 GB/s → 芯片总带宽 900 GB/s（双向 1.8 Tb/s）。

NVSwitch的额外能力
- 硬件多播（multicast）：一次写，8 卡同时收；
- 硬件规约（reduction unit）：支持 fp16/fp32/bf16 累加/求最大，在交换机里完成，节省 GPU SM 周期；
- 延迟 < 200 ns（芯片内部 cross-bar），比 CPU 访存还快。

NVLink相当于高速点对点串行链路，卡-卡或卡-CPU 直联。NVSwitch相当于NVLink的交换机，把多条 NVLink 拼成 全连接网络，还能在芯片里做 多播 & 累加。

## PCIe树状拓扑
PCIe 树状拓扑就是主板把多张 GPU 挂在 一条或多条 PCIe 根端口（Root Port） 下，形成 典型的“根-桥-设备”树结构。PCIe 树状拓扑是“最基础、最通用”的多 GPU 连接方式；所有更高阶拓扑（NVLink 环、NVSwitch、SXM）都是在 PCIe 树之外额外铺的“高速专线”

CPU Root Port
├─ PCIe Switch/Up-bridge
│  ├─ GPU0
│  ├─ GPU1
│  └─ GPU2
└─ 另一条 Root Port
   ├─ NIC
   └─ GPU3


一般来说，这种机制由于带宽/并发/功能三短板让它在算法层抉择时被淘汰掉。如果不存在专用高速专线，那么会利用这一拓扑，NCCL在软件层面将卡号0——1——2——3——0排成一个逻辑环。每一步邻居间通信依靠PCIe P2P完成。

## Ring单向环
物理假设：
- GPU卡间用NVLink单链路或者PCIe直连
- 无硬件多播能力，只能靠接力。

以4卡 256MB数据为例进行Reduce Scatter操作
- step 1： 0-1 64M 1-2 64M 2-3 64M 3-0 64M
- step 2： 每卡把收到的64M数据累加，同时转发下一环
- step 3： 每卡继续累加，同时新数据转发下一环

这种情况下不需要NVSwitch，是最小可行拓扑，只要有P2P NVLink就能成环。在一些2-4卡服务器上一般不会带有NVLink，因此通过PCIe树形成逻辑环。

分层环

## Tree双二叉树 大集群默认
树状拓扑是高性能计算里 “延迟换带宽” 的经典做法——把节点组织成树，先逐级汇总到根，再反向广播下来；深度仅 log₂N，远快于 O(N) 的 Ring。NCCL 的 Tree 算法 就是 双二叉树（Double Binary Tree）的硬件落地版。

物理假设：
- 跨机Fat-Tree EDR/HDR InfiniBand，或 DGX A100 机内 NVLink 双树
- NCCL构建两棵互补二叉树，树高 ⌈log₂N⌉，无共用非叶节点 → 带宽 ×2

## NVLS NVLink Switch硬件多播 
物理假设：
- NCSwitch全连接8卡，每卡12条NVLink
- Switch芯片内置reduction unit


## CollNet 融合网络，多轨NIC场景
物理假设：
- 每节点 ≥2 HDR/NDR NIC → Rail-optimized 网络；
- 交换机支持 SHARP v2/v3（网侧规约）。


## 拓扑结构总结



# chunk，Slice与Packet
NCCL在初始化阶段会用性能模型计算一套让链路保持忙碌的尺寸表。kernel在启动后只按查表尺寸循环。

## 尺寸表计算
首先根据拓扑图，每条边的实测带宽+延迟，每条链路并发段以及消息大小段，对每条边、每档消息计算出最优pipeline深度，得到对应的
- chunkSize   = 消息 / nChannels          (32 MB 级)
- sliceSize   = chunk / NCCL_STEPS        (256 KB 级)
- packetSize  = 64 KB                     (硬件常量)

## channel
channel时并发DMA管道个数，是 NCCL 自己抽象出来的“并行环/并行树”条数。
在物理底层层面：
- 1 条 channel → 1 个 GPU-DMA 描述符队列 → 占用 1 个 copy engine 槽位；
- 1 条 channel 在 机内 对应 1 组 NVLink lane（可 1 lane 也可 4 lane）；
- 机间 对应 1 个 NIC 端口 或 1 个 QP。
在软件层面：
- 每条 channel 完全独立：自己的 send/recv buffer、自己的邻居 rank、自己的 reduce kernel；
- 数据被 round-robin 拆成 nChannels 块 → 同时起飞 → 带宽线性叠加。

channel总结：nChannels 就是 NCCL 自己抽象的“并发 DMA 管道”条数：初始化时按“copy engine × 链路数”算，上限 32；每条 channel 独立发/收/reduce，数据被 round-robin 切片，带宽线性叠加。


## NCCL_STEPS
NCCL_STEPS 是 NCCL 在 流水线通信 里用来 切 chunk 的 slot 数量常量，不是环境变量，源码写死为 8，用户一般不动。

- 把 1 个 chunk 再切成 8 片 → 8 个独立 slot
- 8 个 slot 轮转 → 双缓冲 + 流水线
    - slot i 在 compute（reduce）时
    - slot i+1 在 copy engine（DMA）
    - → 计算与传输重叠，pipeline 不断流

## Packet
PacketSize是 硬件 DMA 引擎的“单次最大有效载荷”极限。到了这一阶段，数据包将通过DMA发送出去。

# 一点有趣的小知识
CUDA的流与shader之间毫无关系，图形渲染与CUDA走的完全是两套命令。虽然它们共享同一套计算资源。GPU 只有 一套底层机器指令集（SASS） 和 一组全局计算/拷贝引擎。
图形驱动 和 CUDA 驱动 各自把高层 API 翻译成 同一份 ISA 的不同 入口格式，然后由 GPU 的 统一调度器 去抢硬件资源。

图形侧：OpenGL/DX12/Vulkan 的 command packet → 驱动内部的 graphics/compute queue → 压入 GPU host-interface queue → 解码成 warp。

CUDA 侧：CUDA API → CUDA user-mode driver → cubin → 同样压入 host-interface queue → 解码成 warp。