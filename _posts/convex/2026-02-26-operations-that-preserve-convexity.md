---
layout: default
title: "Operations that preserve convexity"
date: 2026-02-26 09:00:00 +0900
categories: convex_optimization
permalink: /20260226/operations-that-preserve-convexity.html
---

# Operations that preserve convexity

convex set의 convexity를 유지하는 연산에 대해서 알아보자.  

Convexity를 유지하는 연산
- Intersection
- Affine functions
- Perspective function
- Linear-fractional functions

## Intersection
Convex set의 교집합은 convex이다. 즉, $S_1$ 과 $S_2$ 이 convex라면, $S_1 \cap S_2$ 은 convex이다.  
Set의 convexity는 무한한 halfspace의 교집합으로 표현 가능하며 그 반대도 성립한다.  
즉, closed convex set $S$ 는 $S$ 를 포함하는 모든 halfspace의 교집합으로 다음과 같이 정의할 수 있다.  

$$
S = \bigcap \{\mathcal{H} \mid \mathcal{H}\ \text{halfspace,}\ S \subseteq \mathcal{H}\}
$$

## Affine functions
$A \in \mathbb{R}^{m\times n}$ 이고 $b \in \mathbb{R}^m$ 일 때, $f: \mathbb{R}^n \mapsto \mathbb{R}^m$ 인 $f(x) = Ax + b$ 을 affine function이라 한다.  
이때, $C \subseteq \mathbb{R}^n$ 가 convex이고 $D \subseteq \mathbb{R}^m$ 가 convex이면  
- affine image인 $f(C) = \{f(x) \mid x \in C\}$ 는 convex이다.  
- affine preimage인 $f^{-1}(D) = \{x \mid f(x) \in D\}$ 는 convex이다.  

Affine function인 scaling and translation, projection, sum of two sets, partial sum of set과 같은 연산을 convex set에 적용하면 결과는 convex set이다.  

### 예시
Linear matrix inequality의 해집합 $\{x \mid x_1A_1 + \dots + x_m A_m \preceq B\}(\ \text{with}\ A_i, B\in S^n)$ 도 convex이다.

## Perspective function
Perspective function은 카메라에 상이 맺히는 것과 같이 멀리 있는 물체는 작게, 가까이 있는 물체는 크게 원근에 따라 상을 만드는 함수이다. 따라서, 피사체는 $R^{n+1}$ 차원의 공간에 있고 상은 $R^n$ 차원의 평면에 맺히게 된다.  

![pin-hole camera](https://convex-optimization-for-all.github.io/img/chapter_img/chapter02/02.03_03_pine_hole.png)
