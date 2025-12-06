---
title: "NetworkSystemChapter2"
date: 2025-12-06T16:15:56+08:00
draft: false
description: ""
---

{{<katex>}}
矩阵相关的知识点

# Lectures On Network Systems Chapter 2 Notes

## 线性系统与约当标准形

### 离散时间线性系统
一个方阵定义了一个离散时间线性系统

$$
x(k+1)=Ax(k),\space x(0)=x_0
$$

序列\({x(k)}_{k\in Z_{\ge 0}}\)定义了一个轨迹，也就是系统的演化过程。我们关心的问题是，一个动力系统的解，随着时间趋于无穷大时，是否会收敛到某个极限值，以及最终收敛值是多少。即研究解的渐进行为
- 是否存在极限
- 极限是多少

数学定义：
矩阵\(A\in R^{n \times n}\)是
1. 半收敛的，如果\(lim_{k \rightarrow \infin}A^k\)存在
2. 收敛的或Schur稳定矩阵，如果A半收敛且\(lim_{k \rightarrow \infin}A^k=0_{n\times n}\)

如果A半收敛，那么显然有\(lim_{k \rightarrow \infin}x(k)=A_{\infin}x_0\)

