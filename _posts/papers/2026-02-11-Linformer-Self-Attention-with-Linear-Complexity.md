---
layout: default
title: "Linformer: Self-Attention with Linear Complexity"
date: 2026-02-11 09:00:00 +0900
categories: papers
permalink: /20260211/Linformer-Self-Attention-with-Linear-Complexity.html
---

# Linformer: Self-Attention with Linear Complexity

> [Paper Link](https://arxiv.org/abs/2006.04768)

## 기존 self-attention

$$
\text{head} = \operatorname{Attention}(Q,K,V) = \underbrace{\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V}_{P \in \mathbb{R}^{n\times n}}
$$

- $Q, K, V \in \mathbb{R}^{n\times d}$ 
- $QK^T \in \mathbb{R}^{n \times n}$ 
- 시간 복잡도: $O(n^2)$ 

## Transformer and Self-Attention

$$
P = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}} \right)
$$

$$
P \approx U\Sigma V^T,\ \text{rank} \approx k \ll n
$$

위 수식이 성립하면 $n \times n$ 과정이 필요 없다.  

$$
PV \approx (U\Sigma)(V^TV)
$$

### Key/Value 길이를 축소하자

Query는 그대로 두고, Key/Value를 시퀀스 차원에서 선형 투영한다.  
- $E \in \mathbb{R}^{k \times n}$  
- $F \in \mathbb{R}^{k \times n}$  

$$
\begin{align}
K^\prime &= EK\quad(k \times d)\\
V^\prime &= FV\quad(k \times d)
\end{align}
$$

#### 일반적인 linear projection
일반적으로 생각하는 linear projection은 feature 공간으로 투영한다.  

$$
x \in \mathbb{R}^d \rightarrow Wx \in \mathbb{R}^{d^\prime}
$$

#### Linformer linear projection
##### 원래 Key/Value 구조

$$
K \in \mathbb{R}^{n \times d}
$$

- $n$ : 토큰 개수 (시퀀스 길이)
- $d$ : embedding 차원  

> 각 feature 차원마다 길이 $n$ 짜리 신호가 있음.

##### 시퀀스 압축

$$
K^\prime = EK\quad\text{where}\quad E\in \mathbb{R}^{k\times n}
$$

결과  

$$
K^\prime \in \mathbb{R}^{k \times d}
$$

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O \tag{1}
$$

$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i) = \text{softmax}\left[\frac{QW^Q_i(KW^K_i)^T}{\sqrt{d_k}} \right]VW^V_i \tag{2}
$$

$\text{where}\ W^Q_i, W^K_i \in \mathbb{R}^{d_m \times d_k}, W^V_i \in \mathbb{R}^{d_m\times d_v}, W^O \in \mathbb{R}^{hd_v\times d_m}$  

여기서 우리는 행렬 $E$ 를 알면 되는데 이를 신경망으로 학습시키면 된다.