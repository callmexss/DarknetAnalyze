# DarknetAnalyze

使用 Python 分析 Tor 和 Freenet。

   * [DarknetAnalyze](#darknetanalyze)
      * [Freenet 介绍](#freenet-介绍)
         * [覆盖网络](#覆盖网络)
         * [路由](#路由)
         * [匿名性原理](#匿名性原理)
      * [流量分析](#流量分析)
         * [针对 Tor 的主动流量分析](#针对-tor-的主动流量分析)
         * [针对 Tor 的被动流量分析](#针对-tor-的被动流量分析)
         * [针对 Freenet 的主动流量分析](#针对-freenet-的主动流量分析)
         * [针对 Freenet 的被动流量分析](#针对-freenet-的被动流量分析)
      * [模拟针对 Freenet 的被动攻击](#模拟针对-freenet-的被动攻击)


## Freenet 介绍

### 覆盖网络

![覆盖网络的表示](./pictures/overlay-network.png)

名词解释：

- Freenet：一个分布式的文件存储检索系统，为数据的存储检索操作提供匿名性保护。
- 节点：一个共享磁盘空间和网络带宽的主机。
- 位置：表示节点在覆盖网络中的位置，用一个有效长度为 16 的浮点数表示，大小在 0 到 1 之间。
- 插入：向 Freenet 网络中插入文件。
- 检索：向 Freenet 请求文件。
- 文件key：用于插入和检索文件的二进制表示字符串，类似 Internet 中的 URI。
- FNP: Freenet 网络中的传输层协议。
- HTL: hops to live，FNP 中的一个字段，类比IP中的 TTL，用来限定请求长度。
- UID: 每一个 FNP 消息都有一个 UID 字段，用来标识请求，防止路由环路和回溯时寻路。

**位置**：

![](./pictures/location-distribution.png)

**Freenet URI:**

![](./pictures/freenet-key.png)

**文件组织形式**：

![](./pictures/file-organization.png)

### 路由

![](./pictures/nodes-typology.png)

网络拓扑——基于节点位置的小世界网络。

**检索**：

![](./pictures/request-process.png)

![](./pictures/request-example.png)

路由算法——基于位置的深度优先搜索。

**插入**：

![](./pictures/insert-process.png)

![](./pictures/insert-example.png)

随机插入一条消息到网络中，消息对应的位置值为 0.2950809848542828，成功插入到节点：

```text
store at location 0.2951634042520589
store at location 0.2911546594280774
store at location 0.3000989833987089
store at location 0.29382775921623394
```

**从不同位置检索刚才插入的文件**：

![](./pictures/request-example-0.png)

![](./pictures/request-example-1.png)

![](./pictures/request-example-2.png)

![](./pictures/request-example-3.png)

![](./pictures/request-example-4.png)

因为网络中有冗余存储，所以可以从不同位置的节点取到文件。

### 匿名性原理

主要是基于路由算法实现匿名特性：

- 每一条路由请求基于位置，除了节点的邻居节点，其他节点不知道请求节点的 IP 地址。
- HTL 并不是完全递减，而是在 18（最大值）和 1（最小值）处分别以 50% 和 25% 的概率减 1。
- 每条请求只使用 UID 来标识，UID 在 HTL 为 15 时变更一次。

## 流量分析

### 针对 Tor 的主动流量分析

![](./pictures/tor.png)

给定以下条件：

```
# pub 代表 RSA 公钥，pri代表 RSA 私钥
Alice: alice.pub alice.pri
Enter: enter.pub enter.pri
Delay: delay.pub delay.pri
Exit: exit.pub exit.pri
```

那么 Alice 首先加密一条消息：

$$ En(msg) = En_{enter.pub}(En_{delay.pub}(En_{exit.pub}(msg))))$$

然后根据提前建立好的链路发送消息，每到一个节点使用各自的 RSA 私钥解密一层，最终在出口节点解密成原文发送到目的地。

![](./pictures/flow-watermark.png)

> 流量水印，也称主动网络流水印，是一种主动网络流量分析技术。该技术通过某种方式可以改变发送端发送流中的一些特征，使之隐蔽携带一些特殊标记信息，即水印。该标记信息无法被路由器等常规网络硬件探测到，但可以被特殊的 嗅探设备发现，从而关联流的发送端和接收端。 

一个流量水印的例子：

- 1000：代表休眠时间，单位为毫秒。
- 101001000：代表水印模式，值为 1 时发送数据包，值为 0 时休眠，此处的模式表示一个周期内发送的三个数据包的时间间隔分别为 1，2，3 秒。

![](./pictures/flow_with_watermark_1000_101001000.png)

![](./pictures/flow_with_watermark_1000_101001000_modify.png)

![](./pictures/flow_with_watermark_1000_101001000_modify_2.png)

下图为几种不同时隙和发送模式的流量水印。

![](./pictures/flow_with_all.png)

### 针对 Tor 的被动流量分析

方法和流量水印攻击差不多，只是不再人为添加流量特征。

针对 Tor 的被动流量分析：

![](./pictures/2018-11-21-16-22_field_time_via_scapy.png)
![](./pictures/2018-11-21-16-22_no_filter.png)

### 针对 Freenet 的主动流量分析

上传端添加不同时隙的流量水印（斜率越来越大），下载端未见显著变化：

![](./pictures/freenet-watermark-test.png)

### 针对 Freenet 的被动流量分析

因为 Freenet 使用 UDP 协议作为传输层协议，无法使用同样的方法分析 Freenet 网络，关于 Freenet 的被动流量分析使用修改其源码并记录日志的方法。

流经一个节点的请求的位置分布：

![](./pictures/key_loc_distribution_line.png)

流经节点的请求的 HTL 分布：

![](./pictures/io_htl_distribution.png)



## 模拟针对 Freenet 的被动攻击

还是最开始的网络拓扑图，假设网络中有 30% 节点被攻击者控制：

![](./pictures/bad-nodes-with-target.png)

蓝色节点为攻击者节点，黄色节点为目标节点。攻击节点可以记录代理请求的信息（uid，htl，location）。

模拟从目标节点发起的请求，根据日志统计代理请求数目最多的节点：

```text
target: 0.00028349557720475094

[(0.48121497126960155, 4),
 (0.5298683372650849, 2),
 (0.00028349557720475094, 2),
 (0.4708016549364684, 2),
 (0.2585449757184801, 2),
 (0.021666505332342934, 1),
 (0.4707638359586981, 1),
 (0.5195603388432348, 1),
 (0.15303569226992653, 1)]
```

![](./pictures/possible-nodes.png)

如果目标节点出现在可疑节点的前三位，则视为被发现。

按照这个规则测试5次，每次模拟攻击 100 次，内部请求 1 次，绘制不同比例恶意节点下目标节点被发现的概率曲线图：

![](./pictures/freenet-passive-100.png)

模拟攻击 10000 次：

![](./pictures/freenet-passive-10000.png)

模拟攻击 10000 次，内部分别请求不同次数（1，2，3，4，5）：

![](./pictures/freenet-passive-10000-change.png)
