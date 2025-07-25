---
layout: default
title:  "유도 법칙에 대한 고찰: 최적 제어 관점에서 바라본 비례 항법 시뮬레이션"
date:   2025-07-20 20:06:59 +0900
categories: development
permalink: /20250721/missile-simulation.html
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

여기서 항력($F_{drag}$)은 속도 벡터 $\mathbf{v}$에 대한 비선형 함수 $-C_d \lVert \mathbf{v} \rVert \mathbf{v}$로, Nonlinear 합니다.  

### 왜 Nonlinear인가?

Linear와 Nonlinear를 구분하는 쉬운 방법은 '입출력 관계'를 살펴보면 됩니다.  
- **Linear**  
**입력을 2배로 하면 출력도 2배**가 됩니다. 그래프로 보면, 완벽한 직선입니다.  
e.g., $y=2x$라는 관계는 Linear합니다.

- **Nonlinear**  
**입력을 2배로 하여도 출력이 2개가 되지 않습니다.** 그래프가 곡선 형태입니다.  
  
이제 항력 공식을 살펴보겠습니다.  

$$
F_{drag}=-C_d \lVert \mathbf{v} \rVert \mathbf{v}
$$

여기서 눈여겨 볼 점은 속도($\mathbf{v}$)가 두 번 곱해진다는 점입니다. 곡선이라는 것이죠.

Nonlinear를 해결하기 위하여, 오일러 방식으로 이산화하였습니다.  

---

## 2. 고전적 휴리스틱: 비례 항법 유도(PN, Proportional Navigation)

[비례 항법 유도 설명](https://en.wikipedia.org/wiki/Proportional_navigation)

이 시뮬레이션에서 $\mathbf{x}(t)$에 의존하여 제어 입력 $\mathbf{u}(t)$를 결정하는 [**피드백 제어(Feedback Control)**](https://en.wikipedia.org/wiki/Closed-loop_controller)입니다.  

**최적 제어 관점에서의 한계**
- **Sub-optimality**: PN은 요격을 보장하나, 연료 소모나 비행 시간 측면에서 최적이 아닐 가능성이 높습니다. 전체 비행 경로에 대한 고려 없이 현재의 오차에만 반응하기 때문입니다.  
- **Constraint Handling**: 제어 입력 크기($\lVert \mathbf{u}(t) \rVert \leq u_{max}$) 나 상태 제약(e.g., 특정 경로 회피) 등을 명시적으로 다루기 어렵습니다.  

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
- **제어 입력 한계:** $\lVert \mathbf{u}(t) \rVert \leq u_{max} \ \text{(연료가 있을 때)}$  
- **종단 조건 (Terminal Constraint):** $\mathbf{p}\_{missile}(T) = \mathbf{p}\_{target}(T)$  

최종 시간 $T$가 정해지지 않은 Nonlinear 최적 문제입니다. 동역학과 제약 조건의 비선형성으로 인해 이 문제는 Non-convex 문제입니다.  

---

## 4. Convex Optimization 가능성: Successive Convexification

Non-convex 최적화 문제를 직접 푸는 것은 어렵습니다. 하지만 이 문제를 연속적인 **Convex-Subproblem**로 근사하여 푸는 접근법이 있습니다.  
바로 **Successive Convexification (SCP)** 또는 **Sequential Convex Programming** 입니다.  

**SCP 접근 방식:**  
1. **경로 추측:** 비행 전체 경로 $\mathbf{u}(\cdot)$ 에 대한 초기 추측값을 생성합니다. (e.g., 현재 PN 시뮬레이션 결과에 사용)  
2. **선형화 및 볼록화:** 추측된 경로 주변에서 비선형 동역학을 선형화하고, 비볼록 제약 조건들을 볼록하게 근사합니다.  

$$
\dot{\mathbf{x}}(t) \approx A(t)\mathbf{x}(t) + B(t)\mathbf{u}(t)
$$

이렇게 되면 문제는 우리가 쉽게 풀 수 있는 **볼록 최적화 문제** (e.g., Quadratic Progam, QP)로 변환됩니다.  

### 선형화 유도

**1. 기본 아이디어: 함수의 접선 근사**  

복잡한 비선형 함수 $f(x)$가 있다고 생각해봅시다. 우리는 이 함수 전체를 다루기 어렵지만, 특정 지점 $x_0$ 근처에서 이 함수를 $x_0$에서의 접선으로 유사하게 근사 가능합니다.  

이 접선의 방적식이 **1차 테일러 급수**입니다.  

$$
f(x) \approx f(x_0) + f'(x_0)(x - x_0)
$$

이 식은 복잡한 곡선 $f(x)$를 선형 함수로 바꿔줍니다.  

**2. 시스템 동역학으로 확장: 자코비안 행렬**  

이 아이디어를 다변수 비선형 동역학 함수 $\dot{\mathbf{x} = f(\mathbf{x}, \mathbf{u})}$에 적용해봅시다.  
여기서 미분, 즉 기울기는 편미분을 모아놓은 [**자코비안 행렬(Jacobian Matrix)**](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)가 대신합니다.  

우리는 미리 추측한 **기준 경로(Nominal Trajectory)** 인 ($\mathbf{x}_0(t), \mathbf{u}_0(t)$) 주변에서 테일러 급수 전개를 수행합니다.  

$$
f(\mathbf{x}, \mathbf{u}) \approx f(\mathbf{x}_0, \mathbf{u}_0) + \frac{\partial f}{\partial \mathbf{x}}\biggr|_{(\mathbf{x}_0, \mathbf{u}_0)} (\mathbf{x} - \mathbf{x}_0) + \frac{\partial f}{\partial \mathbf{u}}\biggr|_{(\mathbf{x}_0, \mathbf{u}_0)} (\mathbf{u} - \mathbf{u}_0)
$$

각 항을 정의합니다.  

$$
A(t) = \frac{\partial f}{\partial \mathbf{x}} \biggr|_{(\mathbf{x}_0(t), \mathbf{u}_0(t))}
$$

$$
B(t) = \frac{\partial f}{\partial \mathbf{u}}\biggr|_{(\mathbf{x}_0(t), \mathbf{u}_0(t))}
$$

이 행렬들을 위 테일러 급수 식에 대입하고, 기준 경로로부터의 **편자(deviation)** 를 $\delta \mathbf{x} = \mathbf{x} - \mathbf{x}_0$과 $\delta \mathbf{u} = \mathbf{u} - \mathbf{u}_0$로 정의 후 식을 정리하면, 다음과 같은 **선형화된 편자 동역학(Linearized Deviation Dynamics)** 를 얻게 됩니다.

$$
\delta\dot{\mathbf{x}}(t) \approx A(t)\delta\mathbf{x}(t) + B(t)\delta\mathbf(u)(t)
$$

$$
\begin{align*}
\dot{\mathbf{x}} & \approx f(\mathbf{x}_0, \mathbf{u}_0) + A(t)(\mathbf{x} - \mathbf{x}_0) + B(t)(\mathbf{u} - \mathbf{u}_0) \\
\delta\dot{\mathbf{x}} + \dot{\mathbf{x}}_0 & \approx \dot{\mathbf{x}}_0 + A(t)\delta\mathbf{x} + B(t)\delta\mathbf{u} \\
\dot{\delta\mathbf{x}}(t) & \approx A(t)\delta\mathbf{x}(t) + B(t)\delta\mathbf{u}(t)
\end{align*}
$$

**3. 최적화 문제로의 전환**

**이산화(Discretization):** 위 선형 미분방정식을 이산화하면 $\delta\mathbf{x}_{k+1} = A_k \delta\mathbf{x}_k + B_k \delta\mathbf{u}_k$ 와 같은 선형 대수 관계로 변환됩니다.  
**QP로의 변환:** 우리의 목적 함수는 $\sum\lVert\mathbf{u}_k\rVert^2$ 이므로 **이차 함수(Quadratic)** 입니다.  
제약 조건은 이제 **선형(Linear)** 입니다. 따라서 원래의 비볼록 문제가 [**QP(Quadratic Program)**](https://en.wikipedia.org/wiki/Quadratic_programming)라는 매우 잘 알려진 형태의 볼록 최적화 문제로 바뀌게 됩니다.

