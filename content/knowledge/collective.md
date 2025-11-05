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

## Ring单向环
物理假设 GPU卡间用NVLink单链路或者PCIe直连，无硬件多播能力，只能靠接力。

以4卡 256MB数据为例进行Reduce Scatter操作
- step 1： 0-1 64M 1-2 64M 2-3 64M 3-0 64M
- step 2： 每卡把收到的64M数据累加，同时转发下一环
- step 3： 每卡继续累加，同时新数据转发下一环




# chunk，Slice与Packet


# 一点有趣的小知识
CUDA的流与shader之间毫无关系，图形渲染与CUDA走的完全是两套命令。虽然它们共享同一套计算资源。GPU 只有 一套底层机器指令集（SASS） 和 一组全局计算/拷贝引擎。
图形驱动 和 CUDA 驱动 各自把高层 API 翻译成 同一份 ISA 的不同 入口格式，然后由 GPU 的 统一调度器 去抢硬件资源。

图形侧：OpenGL/DX12/Vulkan 的 command packet → 驱动内部的 graphics/compute queue → 压入 GPU host-interface queue → 解码成 warp。

CUDA 侧：CUDA API → CUDA user-mode driver → cubin → 同样压入 host-interface queue → 解码成 warp。