---
layout: default
title: "6-DOF drone simulation with PID controller"
date: 2025-09-15 00:05:00 +0900
categories: control_system
permalink: /20250829/2025-09-15-drone-simulation.html
---

# 6-DOF drone simulation with PID controller

<video controls>
<source src="../assets/posts/2025-09-15-drone-simulation/drone-result.mp4" type="video/mp4">
브라우저가 비디오 태그를 지원하지 않습니다.
</video>

## 1. Drone Dynamics

### 1.1. 드론의 운동 방정식

- **병진 운동 (Translational Dynamics)**  
   기체의 가속도($\dot{p}$)는 질량($m$)과 기체에 작용하는 힘의 합으로 결정된다.

  $$
  m\dot{p} = \begin{pmatrix} 0 \\ 0 \\ -mg \end{pmatrix} + R\begin{pmatrix} 0 \\ 0 \\ T \end{pmatrix} + F_{drag}
  $$

  이 식에서 우변의 항들은 각각 **중력**, 기체의 자세($R \in SO(3)$)에 따라 방향이 결정되는 **추력**, 그리고 속도에 비례하는 **공기 저항**을 의미한다.

- **회전 운동 (Rotational Dynamics)**: 기체의 각가속도($\dot{\omega}$)는 관성 행렬($I$)과 작용하는 토크($\tau$)로 결정된다.

  $$
  I\dot{\omega} = \tau - \omega \times (I\omega)
  $$

  두 번째 항 $\omega \times (I\omega)$은 회전 운동에서 발생하는 자이로스코픽 효과(gyroscopic effect)를 나타내는 비선형 항이다.

- **코드: `DroneDynamics.update`**
  위 연속 시간 방정식을 이산 시간에서 수치 적분하여 드론의 다음 상태를 계산하는 구현부이다.

```python
# 발췌: DroneDynamics.update
def update(self, total_thrust, torques, dt):
  # 회전 운동: 토크로부터 각가속도를 계산
  ang_accel = self.invI @ (torques - np.cross(self.angular_velocity, self.I @ self.angular_velocity))
  self.angular_velocity += ang_accel * dt

  # 병진 운동: 모든 힘을 합산하여 가속도를 계산
  R = self.orientation.as_matrix()
  thrust_vector = R @ np.array([0, 0, total_thrust])

  # 중력, 공기 저항 계산
  net_force = thrust_vector + drag_force + gravity_force
  acceleration = net_force / self.mass
  self.velocity += acceleration * dt
  self.position += self.velocity * dt
```

드론 동역학에서 자세를 다룰 때, 우리는 두 가지 다른 수학적 표현을 사용한다.

- [**쿼터니언(Quaternion)**](https://en.wikipedia.org/wiki/Quaternion): 자세를 저장하고 시간에 따라 업데이트(회전)할 때 사용한다.
- **회전 행렬**: 힘 벡터 등을 좌표계에 맞게 변환하는 실제 물리 계산에 사용한다.

드론의 움직임, 즉 동역학은 **병진 운동(Translational Motion)** 과 **회전 운동(Rotational Motion)** 의 두 가지로 나눌 수 있다.  
이를 [**6-DOF**](https://en.wikipedia.org/wiki/Six_degrees_of_freedom) 를 갖는다고 말한다.

---

### 1.2. 병진 운동: 드론은 어떻게 이동하는가?

병진 운동은 드론의 무게 중심이 3차원 공간(X, Y, Z)에서 어떻게 움직이는지를 설명하며, 이는 뉴턴의 운동 제2법칙 $F=ma$ 에 의해 지배됩니다. 드론의 가속도($\dot{p}$)를 결정하기 위해, 우리는 드론에 작용하는 모든 힘을 알아야 합니다.

$$
m\dot{p} = F_{total} = F_{gravity} + F_{thrust} + F_{drag}
$$

#### 중력 ($F_{gravity}$)

가장 간단한 힘으로, 항상 지구 중심 방향(월드 좌표계의 -Z축)으로 일정하게 작용합니다.

- **수식**

  $$
  F_{gravity} = \begin{pmatrix} 0 \\ 0 \\ -mg \end{pmatrix}
  $$

- **코드**

```python
gravity_force = np.array([0, 0, -self.mass * self.g])
```

#### 추력 ($F_{thrust}$)

**핵심은 추력이 항상 드론의 동체(Body Frame)를 기준으로 Z축 방향으로 작용한다는 것이다.**

- **수식**

  $$
  F_{thrust} = R \begin{pmatrix} 0 \\ 0 \\ T \end{pmatrix}
  $$

  여기서 $T$ 는 4개 프로펠러가 만들어내는 총 추력의 크기이며, $[0, 0, T]^T$는 드론 동체 기준의 추력 벡터이다.  
  회전 행렬 $R \in SO(3)$ 는 이 동체 기준의 힘을 좌표계의 힘으로 변환하는 역할이다.

- **코드**

```python
# R은 self.orientation (쿼터니언) 으로부터 계산된 회전 행렬
R = self.orientation.as_matrix()
thrust_vector = R @ np.array([0, 0, total_thrust])
```

#### [항력 ($F_{drag}$)](<https://en.wikipedia.org/wiki/Drag_(physics)>)

- **수식**

  $$
  F_{drag} = -k_d \cdot v \cdot \lVert v \rVert
  $$

  $k_d$ 는 항력 계수(drag coefficient)입니다.

- **코드**

```python
drag_force = -self.drag_coeff * self.velocity * np.linalg.norm(self.velocity)
```

이 세 가지 힘을 모두 합하여(`net_force`) 질량으로 나누면 최종적으로 드론의 가속도를 얻고, 이를 적분하여 속도와 위치를 업데이트한다.

---

### 1.3. 회전 운동: 드론은 어떻게 자세를 바꾸는가?

회전 운동은 드론의 자세, 즉 기울기가 어떻게 변하는지를 설명하며, 이는 오일러의 회전 운동 방정식에 의해 기술된다.

#### [오일러 방정식 (Euler's Equation)](https://en.wikipedia.org/wiki/Newton%E2%80%93Euler_equations)

토크($\tau$)가 관성 행렬($I$)과 각가속도($\dot{\omega}$)에 미치는 영향을 설명하겠다.

- **수식**
  $$
  I\dot{\omega} + \omega \times (I\omega) = \tau
  $$
- $\tau$: 제어 입력; 롤(roll), 피치(pitch), 요(yaw) 방향의 토크
- $I\dot{\omega}$: 관성에 의해 회전을 시작하거나 멈추는 데 필요한 토크
- $\omega \times (I\omega)$: [**자이로스코프 효과(Gyroscopic Effect)**](https://en.wikipedia.org/wiki/Gyroscope) 라 불리는 비선형 항. 회전하는 물체는 그 회전축을 유지하려는 성질이 있는데 그걸 제어해준다.

#### 자세 업데이트 (Attitude Update)

위 방정식을 통해 얻은 각가속도 $\dot{\omega}$ 를 적분하여 각속도 $\omega$ 를 구한다. 이 각속도를 이용해 드론의 최종 자세를 업데이트합니다. 이때 롤, 피치, 요 각도를 직접 사용하면 [**짐벌 락(Gimbal Lock)**](https://en.wikipedia.org/wiki/Gimbal_lock) 이라는 현상이 발생하여 특정 자세에서 회전 자유도를 잃는 문제가 생길 수 있다.

이를 피하기 위해, 본 시뮬레이션에서는 수학적으로 더 안정적인 [**쿼터니언(Quaternion)**](https://en.wikipedia.org/wiki/Quaternion) 을 사용하여 자세를 표현했다.

- **코드**

```python
# 발췌: DroneDynamics.update
# 각가속도 계산
ang_accel = self.invI @ (torques - np.cross(self.angular_velocity, self.I @ self.angular_velocity))
self.angular_velocity += ang_accel * dt

# 각속도를 이용해 쿼터니언 자세 업데이트
q_dot = self.orientation * Rotation.from_rotvec(self.angular_velocity * dt)
self.orientation = q_dot
```

---

### 1.4. 쿼터니언이란 무엇이며 왜 사용하는가?

단순히 롤(Roll), 피치(Pitch), 요(Yaw) 3개의 각도로 자세를 표현하면 **짐벌 락(Gimbal Lock)**이라는 심각한 문제가 발생할 수 있다.  
특정 자세(예: 피치 각도가 90도일 때)에서 회전 자유도 하나를 잃어버려 제어가 불가능해지는 현상입니다.

이 문제를 해결하기 위해 [**쿼터니언(Quaternion)**](https://en.wikipedia.org/wiki/Quaternion) 이라는 4차원 복소수 체계를 사용한다.

- **수식: 쿼터니언의 정의**  
  쿼터니언 $q$는 하나의 실수부($w$)와 세 개의 허수부($x, y, z$)로 구성된 4차원 벡터이다.
  $$
  q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}
  $$
  여기서 $w, x, y, z$는 실수이며, $\mathbf{i, j, k}$는 허수 단위이다.

자세를 표현할 때는 크기가 1인 **단위 쿼터니언(Unit Quaternion)**, 즉 $w^2 + x^2 + y^2 + z^2 = 1$을 사용합니다. 단위 쿼터니언은 특정 **회전축** ($\mathbf{u} = [u_x, u_y, u_z]$)을 중심으로 특정 **회전각** ($\theta$)만큼 회전하는 것을 매우 효율적이고 안정적으로 표현할 수 있다.

$$
w = \cos(\theta/2), \quad x = u_x \sin(\theta/2), \quad y = u_y \sin(\theta/2), \quad z = u_z \sin(\theta/2)
$$

#### 쿼터니언을 회전 행렬로 변환하기

- **수식: 쿼터니언-회전 행렬 변환 공식**  
  단위 쿼터니언 $q = (w, x, y, z)$는 다음과 같은 3x3 회전 행렬 $R$로 변환될 수 있다.

$$
R(q) = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}
$$

이 행렬은 쿼터니언 $q$ 가 나타내는 회전과 정확히 동일한 변환을 수행한다.

---

## 2. 궤적 생성: 볼록 최적화를 이용한 Minimum Snap 경로 계획

여러 경유지를 통과하는 비행에서는 단순한 경로 연결이 아닌, 동역학적으로 안정적이고 부드러운 전역 궤적(global trajectory) 생성이 필수적이다.

- **수식: 최소 스냅(Minimum Snap) 문제 정형화**  
  물리적으로 가장 부드러운 경로는 위치의 4차 미분인 **스냅(Snap)** 의 제곱 적분 값을 최소화하는 경로로 알려져 있다.

  $$
  \min \int_{0}^{T_{total}} \left\| \frac{d^4 p(t)}{dt^4} \right\|^2 dt
  $$

  이 문제는 모든 경유지를 지정된 시간에 통과해야 한다는 등식 제약 조건($Ap=b$) 하에서 목적 함수를 최소화하는 [**QP(Quadratic Programming)**](https://en.wikipedia.org/wiki/Quadratic_programming) 문제로 귀결된다. QP는 볼록 최적화 문제의 일종으로, 전역 최적해(global optimum)의 존재가 보장된다.

- **코드: `generate_optimal_trajectory`**  
  이 최적화 문제는 `cvxpy` 라이브러리를 통해 효율적으로 해결할 수 있다. 목적 함수는 `cp.quad_form`으로, 제약 조건은 `A @ p == b`의 형태로 명시하여 Solver에 전달한다.

```python
# 발췌: generate_optimal_trajectory
# p: 최적화할 다항식 계수 벡터
p = cp.Variable(n_segments * n_coeffs)

# Objective: p^T * Q * p 형태의 이차식을 최소화
objective = cp.Minimize(cp.quad_form(p, Q_total + 1e-6 * np.eye(n_segments * n_coeffs)))

# Constraints: Ap = b 형태의 등식 제약조건
constraints = [A @ p == b[i, :]]

# 문제 정의 및 풀이
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.OSQP)
```

## 3. 피드백 제어(PID Control)

계획된 궤적을 정확히 추종하기 위해 비선형 피드백 제어기인 기하학적 제어기를 사용한다. 이 제어기는 드론의 회전 그룹 SO(3)의 구조를 직접 활용하여 큰 자세 오차에 대해서도 안정적인 성능을 보인다.

- **수식: 목표 힘 벡터 및 자세 제어 법칙**  
  제어기의 핵심은 현재 상태 오차를 바탕으로 드론을 궤적으로 복귀시키기 위한 **목표 힘 벡터($F_{des}$)**를 계산하는 것이다.

  $$
  F_{des} = -K_p e_p - K_d e_v + mge_3 + ma_d
  $$

  여기서 $e_p, e_v$는 각각 위치와 속도 오차이며, $K_p, K_d$는 양의 정부호(Positive Definite) 게인 행렬이다. $ma_d$는 궤적의 목표 가속도를 따라가기 위한 피드포워드 항이다.

계산된 $F_{des}$로부터 목표 추력 $T$와 목표 자세 $R_d$를 분리할 수 있다.

$$
T = F_{des} \cdot R e_3, \quad b_{3,d} = F_{des} / \|F_{des}\| \implies R_d
$$

최종 토크 $\tau$는 현재 자세 $R$과 목표 자세 $R_d$ 간의 오차 $e_R$에 기반하여 결정된다.

$$
\tau = -K_{R}e_R - K_{\omega}e_{\omega} + \omega \times (I\omega)
$$

- **코드: `GeometricController.update`**  
  위 제어 법칙이 `update` 함수 내에 순차적으로 구현되어 있다.

```python
# 발췌: GeometricController.update
# 1. 목표 힘 벡터(f_des) 계산
e_pos = self.drone.position - pos_target
e_vel = self.drone.velocity - vel_target
f_des = -self.kp_pos @ e_pos - self.kd_pos @ e_vel + ...

# 2. 목표 추력(thrust_cmd) 및 목표 자세(R_des) 계산
thrust_cmd = np.dot(f_des, b3)
# ...
R_des = np.vstack([b1_des, b2_des, b3_des]).T

# 3. 자세 오차(e_R)를 이용해 최종 토크(torques_cmd) 계산
e_R_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
e_R = np.array([e_R_matrix[2,1], e_R_matrix[0,2], e_R_matrix[1,0]])
torques_cmd = -self.kp_att @ e_R - self.kd_att @ e_omega + ...
```

---

## 4. 제어 파라미터 최적화: 자동 튜닝

처음에는 게인 파라미터 $g = [K_p, K_d, ...]$ 는 수동 튜닝으로 했는데, 너무 어려워서 수치 최적화를 통해 자동으로 탐색했다.

- **수식: 비용 함수 최소화**  
  게인 벡터 $g$의 성능은 **비용 함수 $J(g)$** 를 통해 정량적으로 평가된다. 비용 함수는 궤적 추종 오차와 제어 노력(각속도 크기)의 가중합으로 정의된다.

  $$
  J(g) = \int_0^T \|p(t;g) - p_d(t)\|^2 dt + \lambda \int_0^T \|\omega(t;g)\|^2 dt
  $$

  최적의 게인 $g^*$는 위 비용 함수를 최소화하는 해이다.

  $$
  g^* = \arg\min_{g} J(g)
  $$

- **코드: `objective_function` 및 `minimize`**  
  `objective_function`은 주어진 `gains`로 전체 시뮬레이션을 수행하고 비용 $J(g)$를 반환한다. `scipy.optimize.minimize`는 이 함수의 출력을 최소화하는 `gains`를 **Nelder-Mead**와 같은 도함수-무관(derivative-free) 알고리즘을 사용하여 탐색한다.

```python
# 비용 함수: 주어진 gains로 시뮬레이션을 돌리고 성능 점수를 반환
def objective_function(gains):
  # ... 전체 시뮬레이션 실행 ...
  cost_pos = np.mean(error_pos**2)
  cost_rot = np.mean(np.array(angular_velocities)**2)
  total_cost = cost_pos + weight_rot * cost_rot
  return total_cost

# 최적화 실행: objective_function의 점수를 최소화하는 gains를 찾음
result = minimize(objective_function, initial_gains, method='Nelder-Mead', ...)
best_gains = result.x
```

## 전체코드

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from IPython.display import HTML
import cvxpy as cp
import math
from typing import List, Tuple

# --- 시뮬레이션 설정 ---
# 이륙 지점 및 방문할 경유지 목록
START_POINT: np.ndarray = np.array([0., 0., 2.])
WAYPOINTS_TO_VISIT: List[np.ndarray] = [
    np.array([8., -8., 8.]),
    np.array([8., 8., 4.]),
    np.array([-8., 8., 8.]),
    np.array([-8., -8., 5.]),
    np.array([0., 0., 15.])
]

# 비행 및 시뮬레이션 파라미터
MAX_SPEED: float = 3.0
SIMULATION_TIMESTEP: float = 0.01

# 최적화 및 제어 게인 설정
INITIAL_GAINS: np.ndarray = np.array([6.0, 8.0, 4.0, 5.0, 20.0, 10.0, 2.0, 1.0])
GAIN_BOUNDS: List[Tuple[float, float]] = [(0.1, 50.0)] * 8
OPTIMIZER_MAX_ITERATIONS: int = 100

# --- 클래스 정의 ---

class DroneDynamics:
    """드론의 물리적 역학을 시뮬레이션하는 클래스."""
    def __init__(self, start_pos: np.ndarray = np.zeros(3)):
        self.mass = 1.0
        self.g = 9.81
        self.I = np.diag([0.01, 0.01, 0.02])
        self.invI = np.linalg.inv(self.I)
        self.drag_coeff = 0.1
        self.position = start_pos.copy()
        self.velocity = np.zeros(3)
        self.orientation = Rotation.from_quat([0, 0, 0, 1])
        self.angular_velocity = np.zeros(3)

    def update(self, total_thrust: float, torques: np.ndarray, dt: float) -> None:
        """한 타임스텝 동안 드론의 상태를 업데이트합니다."""
        ang_accel = self.invI @ (torques - np.cross(self.angular_velocity, self.I @ self.angular_velocity))
        self.angular_velocity += ang_accel * dt
        q_dot = self.orientation * Rotation.from_rotvec(self.angular_velocity * dt)
        self.orientation = q_dot

        R = self.orientation.as_matrix()
        thrust_vector = R @ np.array([0, 0, total_thrust])
        drag_force = -self.drag_coeff * self.velocity * np.linalg.norm(self.velocity)
        gravity_force = np.array([0, 0, -self.mass * self.g])
        net_force = thrust_vector + drag_force + gravity_force

        acceleration = net_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        if self.position[2] < 0:
            self.position[2] = 0
            self.velocity[2] = 0

class Polynomial:
    """궤적의 한 조각을 나타내는 1차원 다항식 클래스."""
    def __init__(self, coeffs: np.ndarray):
        self.coeffs = coeffs

    def get_state(self, t: float) -> Tuple[float, float, float]:
        pos = np.polyval(self.coeffs, t)
        vel = np.polyval(np.polyder(self.coeffs, 1), t)
        acc = np.polyval(np.polyder(self.coeffs, 2), t)
        return pos, vel, acc

class PiecewisePolynomialTrajectory:
    """여러 다항식 조각으로 구성된 전체 비행 궤적 클래스."""
    def __init__(self, polys_xyz: List[List[Polynomial]], durations: List[float]):
        self.polys_xyz = polys_xyz
        self.durations = durations
        self.cumulative_durations = np.cumsum([0] + durations)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """특정 시간 t에서의 궤적 상태(위치, 속도, 가속도)를 반환합니다."""
        if t >= self.cumulative_durations[-1]:
            t = self.cumulative_durations[-1] - 1e-6

        segment_index = np.searchsorted(self.cumulative_durations, t, side='right') - 1
        time_in_segment = t - self.cumulative_durations[segment_index]

        pos = np.array([p_list[segment_index].get_state(time_in_segment)[0] for p_list in self.polys_xyz])
        vel = np.array([p_list[segment_index].get_state(time_in_segment)[1] for p_list in self.polys_xyz])
        acc = np.array([p_list[segment_index].get_state(time_in_segment)[2] for p_list in self.polys_xyz])
        return pos, vel, acc

class GeometricController:
    """기하학적 제어기(Geometric Controller) 클래스."""
    def __init__(self, drone: DroneDynamics, gains: np.ndarray):
        self.drone = drone
        self.kp_pos = np.diag([gains[0], gains[0], gains[1]])
        self.kd_pos = np.diag([gains[2], gains[2], gains[3]])
        self.kp_att = np.diag([gains[4], gains[4], gains[5]])
        self.kd_att = np.diag([gains[6], gains[6], gains[7]])

    def update(self, trajectory_target: Tuple[np.ndarray, np.ndarray, np.ndarray], yaw_target: float, dt: float) -> Tuple[float, np.ndarray]:
        """목표 궤적을 따라가기 위한 추력과 토크를 계산합니다."""
        pos_target, vel_target, acc_target = trajectory_target
        e_pos = self.drone.position - pos_target
        e_vel = self.drone.velocity - vel_target

        f_des = (-self.kp_pos @ e_pos - self.kd_pos @ e_vel +
                 self.drone.mass * self.drone.g * np.array([0, 0, 1]) +
                 self.drone.mass * acc_target)

        R = self.drone.orientation.as_matrix()
        b3 = R[:, 2]
        thrust_cmd = np.dot(f_des, b3)

        b3_des = f_des / np.linalg.norm(f_des) if np.linalg.norm(f_des) > 1e-6 else np.array([0, 0, 1])
        a_yaw = np.array([np.cos(yaw_target), np.sin(yaw_target), 0])
        b2_des_numerator = np.cross(b3_des, a_yaw)
        b2_des = b2_des_numerator / np.linalg.norm(b2_des_numerator) if np.linalg.norm(b2_des_numerator) > 1e-6 else np.cross(b3_des, a_yaw)

        b1_des = np.cross(b2_des, b3_des)
        R_des = np.vstack([b1_des, b2_des, b3_des]).T

        e_R_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])
        e_omega = self.drone.angular_velocity

        torques_cmd = -self.kp_att @ e_R - self.kd_att @ e_omega + np.cross(self.drone.angular_velocity, self.drone.I @ self.drone.angular_velocity)
        return thrust_cmd, torques_cmd

# --- 핵심 기능 함수 ---

def generate_optimal_trajectory(waypoints: List[np.ndarray], durations: List[float], order: int = 7) -> PiecewisePolynomialTrajectory:
    """경유지들을 통과하는 최적의(Minimum Snap) 부드러운 궤적을 생성합니다."""
    n_segments = len(waypoints) - 1
    n_coeffs = order + 1

    # QP 문제의 목적 함수 (p'Qp)를 위한 Q 행렬 구성
    Q_total = np.zeros((n_segments * n_coeffs, n_segments * n_coeffs))
    for i in range(n_segments):
        Q_i = np.zeros((n_coeffs, n_coeffs))
        T = durations[i]
        for r in range(4, n_coeffs):
            for c in range(4, n_coeffs):
                k1, k2 = order - r, order - c
                if k1 >= 4 and k2 >= 4:
                    d = (math.factorial(k1) / math.factorial(k1-4)) * (math.factorial(k2) / math.factorial(k2-4))
                    power = k1 + k2 - 7
                    if power > 0: Q_i[r, c] = 2 * d / power * (T**power)
        Q_total[i*n_coeffs:(i+1)*n_coeffs, i*n_coeffs:(i+1)*n_coeffs] = Q_i

    # 제약 조건 (Ap = b)을 위한 A, b 행렬 구성
    def get_der_coeffs(k, t, order):
        coeffs = np.zeros(order + 1)
        for i in range(order + 1):
            p = order - i
            if p >= k: coeffs[i] = math.factorial(p)/math.factorial(p-k) * (t**(p-k))
        return coeffs

    n_derivatives = 4
    n_constraints = 2 * n_derivatives + (n_segments - 1) * n_derivatives
    A = np.zeros((n_constraints, n_segments * n_coeffs))
    b = np.zeros((3, n_constraints))
    row = 0

    A[row, :n_coeffs] = get_der_coeffs(0, 0, order)
    for dim in range(3): b[dim, row] = waypoints[0][dim]
    row += 1
    for k in range(1, n_derivatives): A[row, :n_coeffs] = get_der_coeffs(k, 0, order); row += 1

    A[row, (n_segments-1)*n_coeffs:] = get_der_coeffs(0, durations[-1], order)
    for dim in range(3): b[dim, row] = waypoints[-1][dim]
    row += 1
    for k in range(1, n_derivatives): A[row, (n_segments-1)*n_coeffs:] = get_der_coeffs(k, durations[-1], order); row += 1

    for i in range(n_segments - 1):
        T_i = durations[i]
        A[row, i*n_coeffs:(i+1)*n_coeffs] = get_der_coeffs(0, T_i, order)
        for dim in range(3): b[dim, row] = waypoints[i+1][dim]
        row += 1
        for k in range(1, n_derivatives):
            A[row, i*n_coeffs:(i+1)*n_coeffs] = get_der_coeffs(k, T_i, order)
            A[row, (i+1)*n_coeffs:(i+2)*n_coeffs] = -get_der_coeffs(k, 0, order)
            row += 1

    # 각 축(x, y, z)에 대해 QP 문제 풀이
    polys_xyz = [[], [], []]
    for i in range(3):
        p = cp.Variable(n_segments * n_coeffs)
        objective = cp.Minimize(cp.quad_form(p, Q_total + 1e-6 * np.eye(n_segments * n_coeffs)))
        constraints = [A @ p == b[i, :]]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False, max_iter=50000)
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"최적화 실패 (축: {i}, 상태: {prob.status})")
        coeffs_all = p.value
        for j in range(n_segments):
            polys_xyz[i].append(Polynomial(coeffs_all[j*n_coeffs:(j+1)*n_coeffs]))
    return PiecewisePolynomialTrajectory(polys_xyz, durations)

def plan_tsp_mission(waypoints_to_visit: List[np.ndarray], start_end_point: np.ndarray, max_speed: float) -> Tuple[List[np.ndarray], List[float]]:
    """최근접 이웃 알고리즘으로 여러 경유지의 최적 방문 순서와 비행 시간을 계획합니다."""
    unvisited = list(waypoints_to_visit)
    ordered_path = [start_end_point]
    current_node = start_end_point
    while unvisited:
        distances = [np.linalg.norm(current_node - node) for node in unvisited]
        nearest_index = np.argmin(distances)
        current_node = unvisited.pop(nearest_index)
        ordered_path.append(current_node)
    ordered_path.append(start_end_point)

    durations = [max(np.linalg.norm(p2 - p1) / max_speed, 3.0) for p1, p2 in zip(ordered_path[:-1], ordered_path[1:])]
    return ordered_path, durations

def objective_function(gains: np.ndarray) -> float:
    """제어 게인 튜닝을 위한 목적 함수. 시뮬레이션 오차를 비용으로 반환합니다."""
    drone = DroneDynamics(start_pos=START_POINT)
    controller = GeometricController(drone, gains)
    positions, angular_velocities = [], []
    last_yaw_target = 0.0

    for i in range(timesteps):
        t = i * SIMULATION_TIMESTEP
        pos_target, vel_target, acc_target = trajectory.get_state(t)

        if np.linalg.norm(vel_target[:2]) > 0.1:
            current_yaw_target = np.arctan2(vel_target[1], vel_target[0])
            diff = current_yaw_target - last_yaw_target
            if diff > np.pi: last_yaw_target += 2 * np.pi
            elif diff < -np.pi: last_yaw_target -= 2 * np.pi
            target_yaw = 0.9 * last_yaw_target + 0.1 * current_yaw_target
            last_yaw_target = target_yaw
        else:
            target_yaw = last_yaw_target

        thrust, torques = controller.update((pos_target, vel_target, acc_target), target_yaw, SIMULATION_TIMESTEP)
        drone.update(thrust, torques, SIMULATION_TIMESTEP)
        positions.append(drone.position.copy())
        angular_velocities.append(drone.angular_velocity.copy())

    cost_pos = np.mean((np.array(positions) - ref_path)**2)
    cost_rot = np.mean(np.array(angular_velocities)**2)
    total_cost = cost_pos + 0.01 * cost_rot

    print(f"Cost: {total_cost:.6f} | Gains: {[f'{g:.2f}' for g in gains]}")
    return total_cost

# --- 메인 실행 로직 ---

# 1. 임무 계획 및 궤적 생성
print("1. 임무 계획 및 궤적 생성 중...")
waypoints, durations = plan_tsp_mission(WAYPOINTS_TO_VISIT, START_POINT, MAX_SPEED)
trajectory = generate_optimal_trajectory(waypoints, durations)
sim_time = sum(durations)
timesteps = int(sim_time / SIMULATION_TIMESTEP)
ref_path = np.array([trajectory.get_state(i * SIMULATION_TIMESTEP)[0] for i in range(timesteps)])

# 2. 제어 게인 자동 튜닝
print("\n2. 제어 게인 자동 튜닝 시작...")
result = minimize(objective_function, INITIAL_GAINS, method='Nelder-Mead', bounds=GAIN_BOUNDS,
                  options={'disp': True, 'maxiter': OPTIMIZER_MAX_ITERATIONS})
best_gains = result.x
print(f"\n튜닝 완료! 최적 게인: {[f'{g:.3f}' for g in best_gains]}")

# 3. 최적 게인으로 최종 시뮬레이션 실행
print("\n3. 최적 게인으로 최종 시뮬레이션 실행 중...")
drone = DroneDynamics(start_pos=START_POINT)
controller = GeometricController(drone, gains=best_gains)
history = {'time': [], 'pos': [], 'rot': []}
last_yaw_target = 0.0
for i in range(timesteps):
    t = i * SIMULATION_TIMESTEP
    pos_target, vel_target, acc_target = trajectory.get_state(t)
    if np.linalg.norm(vel_target[:2]) > 0.1:
        current_yaw_target = np.arctan2(vel_target[1], vel_target[0])
        diff = current_yaw_target - last_yaw_target
        if diff > np.pi: last_yaw_target += 2 * np.pi
        elif diff < -np.pi: last_yaw_target -= 2 * np.pi
        target_yaw = 0.9 * last_yaw_target + 0.1 * current_yaw_target; last_yaw_target = target_yaw
    else: target_yaw = last_yaw_target
    thrust, torques = controller.update((pos_target, vel_target, acc_target), target_yaw, SIMULATION_TIMESTEP)
    drone.update(thrust, torques, SIMULATION_TIMESTEP)
    history['time'].append(t)
    history['pos'].append(drone.position.copy())
    history['rot'].append(drone.orientation.as_matrix())

# 4. 애니메이션 생성
print("\n4. 최종 애니메이션 생성 중...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
body_size = 0.3
drone_body = np.array([[body_size, 0, 0], [-body_size, 0, 0], [0, body_size, 0], [0, -body_size, 0]]).T

all_points = np.array(waypoints)
padding = 3.0
ax_limits = [
    (all_points[:,0].min() - padding, all_points[:,0].max() + padding),
    (all_points[:,1].min() - padding, all_points[:,1].max() + padding),
    (0, all_points[:,2].max() + padding)
]

def update_animation(i: int):
    frame_idx = min(i * (timesteps // 500), len(history['pos']) - 1)
    ax.cla()
    pos_now, R_now = history['pos'][frame_idx], history['rot'][frame_idx]

    drone_body_world = (R_now @ drone_body).T + pos_now
    ax.plot(drone_body_world[:2,0], drone_body_world[:2,1], drone_body_world[:2,2], 'k-', lw=3)
    ax.plot(drone_body_world[2:,0], drone_body_world[2:,1], drone_body_world[2:,2], 'k-', lw=3)

    front_marker = (R_now @ np.array([body_size, 0, 0])).T + pos_now
    ax.scatter(front_marker[0], front_marker[1], front_marker[2], color='red', s=40, label='Front')

    ax.plot(np.array(history['pos'])[:frame_idx,0], np.array(history['pos'])[:frame_idx,1], np.array(history['pos'])[:frame_idx,2], 'b-', lw=2.5, label='실제 궤적')
    ax.plot(ref_path[:,0], ref_path[:,1], ref_path[:,2], 'g--', lw=2, label='목표 궤적')

    for j, wp in enumerate(waypoints):
        ax.scatter(wp[0], wp[1], wp[2], s=150, marker='s' if j in [0, len(waypoints)-1] else 'o', label=f'경유지 {j}', depthshade=False)

    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
           title=f"자동 튜닝된 드론 임무 | 시간: {history['time'][frame_idx]:.2f}s",
           xlim=ax_limits[0], ylim=ax_limits[1], zlim=ax_limits[2])
    ax.legend()
    ax.view_init(elev=30, azim=-110)

ani = FuncAnimation(fig, update_animation, frames=500, interval=40)
plt.close()
display(HTML(ani.to_html5_video()))
print("\n애니메이션 준비 완료.")
```
