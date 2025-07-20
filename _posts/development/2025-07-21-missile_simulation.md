---
layout: default
title:  "유도 법칙에 대한 고찰: 최적 제어 관점에서 바라본 비례 항법 시뮬레이션"
date:   2025-07-21 17:06:59 +0900
categories: development
permalink: /20250721/missile_simulation.html
---

# 유도 법칙에 대한 고찰: 최적 제어 관점에서 바라본 비례 항법 시뮬레이션

## 제작 동기

**"백문이 불여일견(百聞不如一見)"** 이라는 말처럼, 저는 글보다 그래프나 시뮬레이션 같은 시각 자료로 공부할 때 더 쉽게 이해하는 편입니다.  
최근 가짜연구소에서 `Convex Optimization`을 공부하다가, 책에 나온 수식이 실제로 어떻게 작동하는지 직접 확인해보고 싶어 시뮬레이션을 제작하게 되었습니다.

## 1. 시뮬레이션 동적 시스템 모델링

시뮬레이션의 2차원 평면 운동을 상태 공간 모델로 표현해 봅시다.  

미사일의 상태 벡터 $\mathbf{x}(t)$를 위치와 속도로 정의합니다.
$$
\mathbf{x}(t) = 
\begin{bmatrix}
p_x(t) \\
p_y(t) \\
v_x(t) \\
v_y(t)
\end{bmatrix}
\in \mathbb{R}^4
$$

제어 입력 $\mathbf{u}(t)$는 미사일이 생성하는 추력 가속도 벡터입니다.
$$
\mathbf{u}(t) = 
\begin{bmatrix}
a_{thrust, x}(t) \\
a_{thrust, y}(t)
\end{bmatrix}
$$

이를 기반으로 시스템 동역학은 다음과 같은 비선형 상미분방정식(ODE)으로 표현됩니다.
$$
\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t)) =
\begin{bmatrix}
v_x(t) \\
v_y(t) \\
\frac{u_x(t) + F_{drag, x}(\mathbf{v})}{m} \\
\frac{u_y(t) + F_{drag, y}(\mathbf{v})}{m}
\end{bmatrix}
$$

여기서 항력($F_{drag}$)은 속도 벡터 $\mathbf{v}$에 대한 비선형 함수 $-C_d |\mathbf{v}| \mathbf{v}$로, Nonlinear 합니다.  

### 왜 Nonlinear인가?

Linear와 Nonlinear를 구분하는 쉬운 방법은 '입출력 관계'를 살펴보면 됩니다.  
- **Linear**  
**입력을 2배로 하면 출력도 2배**가 됩니다. 그래프로 보면, 완벽한 직선입니다.  
e.g., $y=2x$라는 관계는 Linear합니다.

- **Nonlinear**  
**입력을 2배로 하여도 출력이 2개가 되지 않습니다.** 그래프가 곡선 형태입니다.  
  
이제 항력 공식을 살펴보겠습니다.  
$$
F_{drag}=-C_d|\mathbf{v}|\mathbf{v}
$$
여기서 눈여겨 볼 점은 속도($\mathbf{v}$)가 두 번 곱해진다는 점입니다. 곡선이라는 것이죠.

Nonlinear를 해결하기 위하여, 오일러 방식으로 이산화하였습니다.  

---

## 2. 고전적 휴리스틱: 비례 항법 유도(PN)

이 시뮬레이션에서 $\mathbf{x}(t)$에 의존하여 제어 입력 $\mathbf{u}(t)$를 결정하는 `피드백 제어(Feedback Control)`입니다.  

**최적 제어 관점에서의 한계**
- **Sub-optimality**: PN은 요격을 보장하나, 연료 소모나 비행 시간 측면에서 최적이 아닐 가능성이 높습니다. 전체 비행 경로에 대한 고려 없이 현재의 오차에만 반응하기 때문입니다.  
- **Constraint Handling**: 제어 입력 크기($||\mathbf{u}(t)|| \leq u_{max}$) 나 상태 제약(e.g., 특정 경로 회피) 등을 명시적으로 다루기 어렵습니다.  

---

## 3. 최적 제어(Optimal Control) 문제로 재정의

이 유도 문제를 최적 제어 문제로 공식화해봅시다. 우리의 목표는 최소한의 제어 노력(연료 소모와 비례)으로 미사일을 목표물에 명중시키는 것입니다.  

**목적 함수 (Objective Function)**  
$$
\text{minimize} \ J=\int^{T}_{0} ||\mathbf{u}(t)||^2 dt
$$

**제약 조건 (Constraints)**  
- **시스템 동역학:** $\dot{\mathbf{x}}=f(\mathbf{x}(t), \mathbf{u}(t))$  
- **초기 조건:** $\mathbf{x}(0) = \mathbf{x}_{initial}$  
- **제어 입력 한계:** $||\mathbf{u}(t) \leq u_{max} \ \text{(연료가 있을 때)}$  
- **종단 조건 (Terminal Constraint):** $\mathbf{p}_{missile}(T)=\mathbf{p}_{target}(T) $  

최종 시간 $T$가 정해지지 않은 Nonlinear 최적 문제입니다. 동역학과 제약 조건의 비선형성으로 인해 이 문제는 Non-convex 문제입니다.  



$$
P_{target}(t + \Delta t) = P_{target}(t)  + v_{target} \cdot \Delta t
$$
