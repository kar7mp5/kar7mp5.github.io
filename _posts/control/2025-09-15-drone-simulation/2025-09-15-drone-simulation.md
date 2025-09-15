---
layout: default
title: "6-DOF drone simulation with PID controller"
date: 2025-09-15 00:05:00 +0900
# image: ../assets/posts/2025-09-15-drone-simulation/drone-result.mp4
categories: control_system
permalink: /20250829/2025-09-15-drone-simulation.html
---

# 6-DOF drone simulation with PID controller

<video controls>
  <source src="../assets/posts/2025-09-15-drone-simulation/drone-result.mp4" type="video/mp4">
  브라우저가 비디오 태그를 지원하지 않습니다.
</video>

### **6-DOF 쿼드콥터의 자율 비행을 위한 계층적 모션 플래닝 및 제어 시스템 설계 및 시뮬레이션 검증**

#### **초록 (Abstract)**

본 논문은 다수의 경유지를 통과하는 6자유도(6-DOF) 쿼드콥터의 복잡한 임무 수행을 위한 계층적(hierarchical) 자율 비행 시스템의 설계와 시뮬레이션 기반 검증에 대해 다룬다. 제안하는 프레임워크는 상위 레벨의 임무 계획, 중간 레벨의 궤적 생성, 그리고 하위 레벨의 피드백 제어 계층으로 구성된다. 임무 계획 단계에서는 외판원 문제(TSP)의 휴리스틱 해법을 통해 최적의 방문 순서를 결정한다. 궤적 생성 단계에서는 전체 경로에 대한 전역적으로 부드러운(globally smooth) 궤적을 생성하기 위해 볼록 최적화(Convex Optimization)에 기반한 구간별(piecewise) 최소 스냅(Minimum Snap) 기법을 적용한다. 하위 레벨 제어기에는 강인한 비선형 제어 성능을 위해 SO(3) 상에서의 기하학적 제어기(Geometric Controller)를 설계 및 적용한다. 마지막으로, 시뮬레이션 기반의 수치 최적화 기법을 통해 제어기의 성능을 극대화하는 제어 파라미터를 자동으로 튜닝하는 프로세스를 제시한다. 시뮬레이션 결과는 제안된 계층적 시스템이 생성된 최적 궤적을 높은 정밀도로 추종함을 보임으로써 그 유효성을 입증한다.

---

#### **1. 시스템 동역학 모델: Newton-Euler Formulation**

쿼드콥터의 동역학은 강체(rigid body)의 운동을 기술하는 Newton-Euler 방정식에 의해 결정된다. 시스템의 상태 벡터 $x$는 위치 $p \\in \\mathbb{R}^3$, 속도 $v \\in \\mathbb{R}^3$, 자세 $R \\in SO(3)$, 그리고 각속도 $\\omega \\in \\mathbb{R}^3$로 정의된다. 제어 입력 벡터 $u$는 총 추력 $T \\in \\mathbb{R}$와 토크 $\\tau \\in \\mathbb{R}^3$로 구성된다.

- **병진 운동 방정식 (Translational Dynamics)**:
  $$m\ddot{p} = m\dot{v} = -mge_3 + R \cdot T e_3 + F_{drag}$$
  여기서 $m$은 기체의 질량, $g$는 중력 가속도, $e\_3 = [0, 0, 1]^T$, $R$은 월드 좌표계에 대한 기체 좌표계의 회전 행렬, $F\_{drag}$는 공기 저항력이다.

- **회전 운동 방정식 (Rotational Dynamics)**:
  $$I\dot{\omega} = \tau - \omega \times (I\omega)$$
  $I \\in \\mathbb{R}^{3 \\times 3}$는 관성 행렬(Inertia Matrix)이다. 자세 $R$의 변화율은 $\\dot{R} = R \\cdot \\hat{\\omega}$로 표현되며, 여기서 $\\hat{\\omega}$는 $\\omega$의 비대칭 행렬(skew-symmetric matrix) 표현이다.

**코드 구현 (`DroneDynamics.update`)**:
해당 클래스는 이산 시간(discrete-time) 환경에서 위 연속 시간(continuous-time) 방정식을 수치 적분하여 시뮬레이션을 수행한다. 자세 표현에는 짐벌 락을 회피하기 위해 쿼터니언(Quaternion)을 내부적으로 사용한다.

```python
# 발췌: DroneDynamics.update
ang_accel = self.invI @ (torques - np.cross(self.angular_velocity, self.I @ self.angular_velocity))
self.angular_velocity += ang_accel * dt
# ... (Quaternion update) ...
net_force = thrust_vector + drag_force + gravity_force
acceleration = net_force / self.mass
self.velocity += acceleration * dt
self.position += self.velocity * dt
```

---

#### **2. 계층적 모션 플래닝 및 제어**

##### **2.1. 상위 레벨: 임무 순서 계획**

`plan_tsp_mission` 함수는 주어진 경유지 집합에 대한 최적 방문 순서를 결정한다. 이 문제는 NP-Hard인 \*\*TSP(Traveling Salesperson Problem)\*\*로 정형화된다. 본 시스템에서는 전역 최적해를 보장하지는 않으나, 다항 시간 내에 합리적인 해를 제공하는 **최근접 이웃(Nearest Neighbor) 휴리스틱**을 채택하였다. 결정된 경로 순서와 사용자 지정 최대 속도($v\_{max}$)를 기반으로 각 구간의 소요 시간($T\_i = ||W\_{i+1} - W\_i|| / v\_{max}$)을 할당한다.

##### **2.2. 중간 레벨: Minimum Snap 궤적 생성**

`generate_optimal_trajectory` 함수는 전역적으로 $C^4$ 연속성을 갖는 부드러운 궤적을 생성한다. 이는 전체 임무를 $M$개의 구간으로 나누고, 각 구간을 $N$차 다항식으로 표현한 뒤, 모든 제약조건을 만족시키면서 아래의 목적 함수 $J(p)$를 최소화하는 계수 벡터 $p$를 찾는 문제로 귀결된다.

- **목적 함수 (Quadratic Program Objective)**:
  $$\min_{p} J(p) = \min_{p} \sum_{i=0}^{M-1} \int_0^{T_i} \left\| \frac{d^4 P_i(t)}{dt^4} \right\|^2 dt$$
  이 목적 함수는 계수 $p$에 대한 이차 형식(Quadratic Form) $\\frac{1}{2}p^T Q p$로 표현 가능하다. `cvxpy`는 이러한 QP(Quadratic Programming) 문제를 효율적으로 해결한다.

- **제약 조건 (Linear Constraints)**: 위치, 연속성, 경계 조건은 모두 계수 $p$에 대한 선형 제약 조건 $Ap=b$로 표현된다. 예를 들어, 구간 $i$와 $i+1$ 사이의 속도 연속성 조건은 다음과 같이 수식화된다.
  $$\frac{d P_i(T_i)}{dt} - \frac{d P_{i+1}(0)}{dt} = 0$$

**코드 구현 (`generate_optimal_trajectory`의 제약조건)**:

```python
# 발췌: 제약조건 행렬 A 구성
# 중간 경유지(i)에서의 k차 미분 연속성 제약
# P_i^(k)(T_i) - P_{i+1}^(k)(0) = 0
for i in range(n_segments - 1):
    for k in range(1, 5): # v, a, j, s
        A[row, i*n_coeffs:(i+1)*n_coeffs] = get_der_coeffs(k, durations[i], order)
        A[row, (i+1)*n_coeffs:(i+2)*n_coeffs] = -get_der_coeffs(k, 0, order)
        row += 1
```

##### **2.3. 하위 레벨: SO(3) 상의 기하학적 제어**

`GeometricController`는 생성된 참조 궤적($p\_d, v\_d, a\_d$)을 추종하기 위한 비선형 피드백 제어기이다. 이 제어기는 Taeyoung Lee 등의 연구에서 제시된 바와 같이, 회전 그룹 SO(3)의 기하학적 구조를 직접 활용하여 큰 자세 오차에 대해 강인한 성능을 보인다.

- **목표 힘 벡터 ($F\_{des}$)**: 위치 및 속도 오차($e\_p = p-p\_d, e\_v = v-v\_d$)를 보상하기 위한 목표 힘은 다음과 같다.
  $$F_{des} = -K_p e_p - K_d e_v + mge_3 + ma_d$$

- **목표 자세 ($R\_d$) 및 추력 ($T$)**: 목표 힘 벡터로부터 추력의 크기와 목표 자세를 분리한다.
  $$T = F_{des} \cdot R e_3$$ $$b_{3,d} = F_{des} / \|F_{des}\|$$
  목표 Yaw 방향 벡터 $a\_{\\psi}$를 고려하여 최종 목표 회전 행렬 $R\_d = [b\_{1,d}, b\_{2,d}, b\_{3,d}]$를 구성한다.

- **토크 명령 ($\\tau$)**: 자세 오차 $e\_R$와 각속도 오차 $e\_{\\omega}$를 이용해 최종 토크를 계산한다.
  $$e_R = \frac{1}{2}(R_d^T R - R^T R_d)^\vee \quad (\vee : \text{vee map from } so(3) \to \mathbb{R}^3)$$ $$\tau = -K_{R}e_R - K_{\omega}e_{\omega} + \omega \times (I\omega)$$

**코드 구현 (`GeometricController.update`)**:

```python
# 발췌: GeometricController.update
f_des = -self.kp_pos @ e_pos - self.kd_pos @ e_vel + ...
# ... R_des 계산 ...
e_R_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
e_R = np.array([e_R_matrix[2,1], e_R_matrix[0,2], e_R_matrix[1,0]])
torques_cmd = -self.kp_att @ e_R - self.kd_att @ e_omega + ...
```

##### **2.4. 제어 파라미터 최적화**

최적의 제어 게인 $g^\*$는 비용 함수 $J(g)$를 최소화하는 파라미터를 찾는 문제로 귀결된다.
$$g^* = \arg\min_{g} J(g) = \arg\min_{g} \left( \int_0^T \|e_p(t)\|^2 dt + \lambda \int_0^T \|\omega(t)\|^2 dt \right)$$
본 시스템에서는 `scipy.optimize.minimize`에 구현된 도함수-무관(derivative-free) 최적화 알고리즘 **Nelder-Mead**를 사용하여 이 문제를 수치적으로 해결한다.

---

#### **3. 결론**

본 문서는 쿼드콥터 자율 비행을 위한 계층적 시스템을 제시하고, 각 계층의 핵심적인 수학적 원리와 코드 구현을 상세히 설명했다. 제안된 프레임워크는 TSP 휴리스틱, QP 기반 궤적 생성, SO(3) 기하학적 제어, 그리고 자동 게인 튜닝을 통합하여, 복잡한 다중 경유지 임무에 대한 높은 수준의 자율성과 정밀한 추종 성능을 시뮬레이션을 통해 성공적으로 검증하였다.
