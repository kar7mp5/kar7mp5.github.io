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

드론의 움직임은 [뉴턴-오일러 운동 방정식](https://en.wikipedia.org/wiki/Newton%E2%80%93Euler_equations)으로 기술된다.

### 드론의 운동 방정식

1.  **병진 운동 (Translational Dynamics)**  
    기체의 가속도($\dot{p}$)는 질량($m$)과 기체에 작용하는 힘의 합으로 결정된다.

    $$
    m\dot{p} = \begin{pmatrix} 0 \\ 0 \\ -mg \end{pmatrix} + R\begin{pmatrix} 0 \\ 0 \\ T \end{pmatrix} + F_{drag}
    $$

    이 식에서 우변의 항들은 각각 **중력**, 기체의 자세($R \in SO(3)$)에 따라 방향이 결정되는 **추력**, 그리고 속도에 비례하는 **공기 저항**을 의미한다.

2.  **회전 운동 (Rotational Dynamics)**: 기체의 각가속도($\dot{\omega}$)는 관성 행렬($I$)과 작용하는 토크($\tau$)로 결정된다.
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
  # ... 중력, 공기 저항 계산 ...
  net_force = thrust_vector + drag_force + gravity_force
  acceleration = net_force / self.mass
  self.velocity += acceleration * dt
  self.position += self.velocity * dt
```

드론 동역학에서 자세를 다룰 때, 우리는 두 가지 다른 수학적 표현을 사용합니다.

- **쿼터니언**: 자세를 저장하고 시간에 따라 업데이트(회전)할 때 사용합니다.
- **회전 행렬**: 힘 벡터 등을 좌표계에 맞게 변환하는 실제 물리 계산에 사용합니다.

왜 이렇게 두 가지를 혼용하는지, 그리고 쿼터니언이 어떻게 행렬로 변환되는지 자세히 알아보겠습니다.

네, 드론의 동역학 모델에 대해 수식과 물리적 의미를 중심으로 더 구체적으로 설명해 드리겠습니다.

드론의 움직임, 즉 동역학은 **병진 운동(Translational Motion)**과 **회전 운동(Rotational Motion)**의 두 가지로 나눌 수 있습니다. 이 둘을 합쳐 총 6개의 독립적인 움직임, 즉 **6자유도(6-DOF)**를 갖는다고 말합니다.

---

## 1. 병진 운동: 드론은 어떻게 이동하는가?

병진 운동은 드론의 무게 중심이 3차원 공간(X, Y, Z)에서 어떻게 움직이는지를 설명하며, 이는 뉴턴의 운동 제2법칙 $F=ma$에 의해 지배됩니다. 드론의 가속도($\dot{p}$)를 결정하기 위해, 우리는 드론에 작용하는 모든 힘을 알아야 합니다.

$$m\dot{p} = F_{total} = F_{gravity} + F_{thrust} + F_{drag}$$

### **1.1. 중력 ($F_{gravity}$)**

가장 간단한 힘으로, 항상 지구 중심 방향(월드 좌표계의 -Z축)으로 일정하게 작용합니다.

- **수식**:
  $$F_{gravity} = \begin{pmatrix} 0 \\ 0 \\ -mg \end{pmatrix}$$

- **코드**:

```python
gravity_force = np.array([0, 0, -self.mass * self.g])
```

#### **1.2. 추력 ($F_{thrust}$)**

드론을 움직이는 유일한 동력원입니다. 프로펠러가 회전하여 공기를 아래로 밀어내고, 그 반작용으로 위쪽으로 힘을 받습니다.

**핵심은 추력이 항상 드론의 동체(Body Frame)를 기준으로 Z축 방향으로 작용한다는 것입니다**  
드론이 수평일 때는 수직으로 위를 향하지만, 드론이 앞으로 기울어지면 추력 벡터도 함께 앞으로 기울어집니다. 이 기울어진 추력 벡터가 수평 방향의 힘을 만들어내어 드론을 전후좌우로 움직이게 합니다.

- **수식**:
  $$F_{thrust} = R \begin{pmatrix} 0 \\ 0 \\ T \end{pmatrix}$$
  여기서 $T$는 4개 프로펠러가 만들어내는 총 추력의 크기이며, $[0, 0, T]^T$는 드론 동체 기준의 추력 벡터입니다. 회전 행렬 $R \\in SO(3)$는 이 동체 기준의 힘을 우리가 보는 월드 좌표계의 힘으로 변환하는 역할을 합니다.

- **코드**:

```python
# R은 self.orientation (쿼터니언) 으로부터 계산된 회전 행렬
R = self.orientation.as_matrix()
thrust_vector = R @ np.array([0, 0, total_thrust])
```

#### **1.3. 항력 ($F_{drag}$)**

공기 저항을 모델링한 힘입니다. 드론의 속도($v$)에 반대 방향으로 작용하며, 속도가 빠를수록 커지는 특징이 있습니다. 일반적으로 속도의 제곱에 비례하는 모델을 사용합니다.

- **수식**:
  $$F_{drag} = -k_d \cdot v \cdot \|v\|$$
  $k\_d$는 항력 계수(drag coefficient)입니다.

- **코드**:

```python
drag_force = -self.drag_coeff * self.velocity * np.linalg.norm(self.velocity)
```

이 세 가지 힘을 모두 합하여(`net_force`) 질량으로 나누면 최종적으로 드론의 가속도를 얻고, 이를 적분하여 속도와 위치를 업데이트합니다.

---

### \#\# 2. 회전 운동: 드론은 어떻게 자세를 바꾸는가?

회전 운동은 드론의 자세, 즉 기울기가 어떻게 변하는지를 설명하며, 이는 오일러의 회전 운동 방정식에 의해 기술됩니다.

#### **2.1. 오일러 방정식 (Euler's Equation)**

토크($\tau$, 비트는 힘)가 관성 행렬($I$)과 각가속도($\dot{\omega}$)에 미치는 영향을 설명합니다.

- **수식**:
  $$I\dot{\omega} + \omega \times (I\omega) = \tau$$
- $\tau$: 제어 입력으로, 4개의 모터 회전 속도를 미세하게 조절하여 롤(roll), 피치(pitch), 요(yaw) 방향의 토크를 만들어냅니다. 이것이 드론의 자세를 바꾸는 원동력입니다.
- $I\dot{\omega}$: 관성에 의해 회전을 시작하거나 멈추는 데 필요한 토크를 의미합니다.
- $\omega \times (I\omega)$: **자이로스코픽 효과(Gyroscopic Effect)**라 불리는 비선형 항입니다. 회전하는 물체는 그 회전축을 유지하려는 성질이 있는데, 이 항이 바로 그 효과를 모델링합니다. 빠르게 도는 팽이가 잘 쓰러지지 않는 것과 같은 원리이며, 드론의 회전 안정성에 중요한 역할을 합니다.

#### **2.2. 자세 업데이트 (Attitude Update)**

위 방정식을 통해 얻은 각가속도 $\dot{\omega}$를 적분하여 각속도 $\omega$를 구합니다. 이 각속도를 이용해 드론의 최종 자세를 업데이트합니다. 이때 롤, 피치, 요 각도를 직접 사용하면 **짐벌 락(Gimbal Lock)**이라는 현상이 발생하여 특정 자세에서 회전 자유도를 잃는 문제가 생길 수 있습니다.

이를 피하기 위해, 본 시뮬레이션에서는 수학적으로 더 안정적인 **쿼터니언(Quaternion)**을 사용하여 자세를 표현하고 업데이트합니다.

- **코드**:
  `scipy.spatial.transform.Rotation` 라이브러리는 이러한 복잡한 쿼터니언 연산을 내부적으로 처리해줍니다. 현재 자세(`self.orientation`)에서 각속도(`self.angular_velocity`)로 `dt`초 만큼 회전했을 때의 다음 자세를 계산합니다.

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

### 1. 쿼터니언이란 무엇이며 왜 사용하는가?

단순히 롤(Roll), 피치(Pitch), 요(Yaw) 3개의 각도로 자세를 표현하면 **짐벌 락(Gimbal Lock)**이라는 심각한 문제가 발생할 수 있습니다. 특정 자세(예: 피치 각도가 90도일 때)에서 회전 자유도 하나를 잃어버려 제어가 불가능해지는 현상입니다.

이 문제를 해결하기 위해 로보틱스나 3D 그래픽스 분야에서는 **쿼터니언**이라는 4차원 복소수 체계를 사용합니다.

- **수식: 쿼터니언의 정의**
  쿼터니언 $q$는 하나의 실수부($w$)와 세 개의 허수부($x, y, z$)로 구성된 4차원 벡터입니다.
  $$q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$$
  여기서 $w, x, y, z$는 실수이며, $\mathbf{i, j, k}$는 허수 단위입니다.

자세를 표현할 때는 크기가 1인 **단위 쿼터니언(Unit Quaternion)**, 즉 $w^2 + x^2 + y^2 + z^2 = 1$을 사용합니다. 단위 쿼터니언은 특정 **회전축**($\mathbf{u} = [u_x, u_y, u_z]$)을 중심으로 특정 **회전각**($\theta$)만큼 회전하는 것을 매우 효율적이고 안정적으로 표현할 수 있습니다.

$$
w = \cos(\theta/2), \quad x = u_x \sin(\theta/2), \quad y = u_y \sin(\theta/2), \quad z = u_z \sin(\theta/2)
$$

짐벌 락 문제가 없고, 여러 회전을 연속적으로 적용(곱셈)하는 연산이 매우 효율적이어서 드론의 자세를 시간에 따라 업데이트하는 데 매우 적합합니다.

---

### 2. 쿼터니언을 회전 행렬로 변환하기

쿼터니언은 자세를 저장하고 업데이트하는 데는 훌륭하지만, 추력과 같은 **벡터(Vector)를 한 좌표계에서 다른 좌표계로 변환**하는 물리 계산에는 3x3 **회전 행렬(Rotation Matrix)**이 훨씬 직관적이고 편리합니다.

따라서 우리는 매 순간 쿼터니언으로 저장된 현재 자세를 물리 계산에 사용하기 위해 회전 행렬로 변환해야 합니다.

- **수식: 쿼터니언-회전 행렬 변환 공식**
  단위 쿼터니언 $q = (w, x, y, z)$는 다음과 같은 3x3 회전 행렬 $R$로 변환될 수 있습니다.

$$
R(q) = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}
$$

이 행렬은 쿼터니언 $q$가 나타내는 회전과 정확히 동일한 변환을 수행합니다. 예를 들어, 이 행렬 $R$에 드론 동체 기준의 추력 벡터 $[0, 0, T]^T$를 곱하면 월드 좌표계 기준의 추력 벡터가 계산됩니다.

---

### 3. 코드에서의 구현

다행히도 우리는 이 복잡한 변환 공식을 직접 코드로 작성할 필요가 없습니다. `scipy.spatial.transform.Rotation` 라이브러리가 모든 것을 처리해줍니다.

1.  **자세 저장 및 업데이트 (쿼터니언 사용)**:
    `DroneDynamics` 클래스는 드론의 자세를 `orientation`이라는 `Rotation` 객체로 저장합니다. 이 객체는 내부적으로 쿼터니언을 사용합니다. 자세 업데이트는 쿼터니언 곱셈으로 매우 간단하게 이루어집니다.

```python
# in DroneDynamics.__init__
# [x, y, z, w] 순서의 쿼터니언으로 초기 자세(회전 없음)를 설정
self.orientation = Rotation.from_quat([0, 0, 0, 1])

# in DroneDynamics.update
# 현재 각속도로 dt초 만큼 회전하는 작은 쿼터니언을 생성하여 곱함 (자세 업데이트)
q_dot = self.orientation * Rotation.from_rotvec(self.angular_velocity * dt)
self.orientation = q_dot
```

2.  **물리 계산을 위한 변환 (회전 행렬 사용)**:
    물리 계산이 필요한 시점에는 `.as_matrix()` 메소드를 호출하기만 하면 됩니다. 이 메소드 하나가 위에서 설명한 복잡한 변환 공식을 실행하여 3x3 회전 행렬을 반환합니다.

```python
# in DroneDynamics.update
# 쿼터니언으로 저장된 현재 자세를 회전 행렬 R로 변환
R = self.orientation.as_matrix()

# 변환된 행렬 R을 사용하여 물리량(추력 벡터)을 계산
thrust_vector = R @ np.array([0, 0, total_thrust])
```

결론적으로, 우리 시뮬레이션은 **자세의 누적과 업데이트는 쿼터니언의 장점**을, **벡터 변환 계산은 회전 행렬의 장점**을 모두 취하는 매우 효율적이고 안정적인 하이브리드 접근 방식을 사용하고 있습니다.

---

#### **2. 궤적 생성: 볼록 최적화를 이용한 Minimum Snap 경로 계획**

여러 경유지를 통과하는 비행에서는 단순한 경로 연결이 아닌, 동역학적으로 안정적이고 부드러운 전역 궤적(global trajectory) 생성이 필수적이다.

- **수식: 최소 스냅(Minimum Snap) 문제 정형화**
  물리적으로 가장 부드러운 경로는 위치의 4차 미분인 **스냅(Snap)**의 제곱 적분 값을 최소화하는 경로로 알려져 있다.

  $$
  \min \int_{0}^{T_{total}} \left\| \frac{d^4 p(t)}{dt^4} \right\|^2 dt
  $$

  이 문제는 모든 경유지를 지정된 시간에 통과해야 한다는 등식 제약 조건($Ap=b$) 하에서 목적 함수를 최소화하는 **QP(Quadratic Programming)** 문제로 귀결된다. QP는 볼록 최적화 문제의 일종으로, 전역 최적해(global optimum)의 존재가 보장된다.

- **코드: `generate_optimal_trajectory`**
  이 최적화 문제는 `cvxpy` 라이브러리를 통해 효율적으로 해결할 수 있다. 목적 함수는 `cp.quad_form`으로, 제약 조건은 `A @ p == b`의 형태로 명시하여 솔버에 전달한다.

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

---

#### **3. 피드백 제어: SO(3) 상의 기하학적 제어기**

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

#### **4. 제어 파라미터 최적화: 자동 튜닝**

제어기의 성능을 결정하는 게인 파라미터 $g = [K_p, K_d, ...]$는 수동 튜닝이 아닌, 수치 최적화를 통해 자동으로 탐색된다.

- **수식: 비용 함수 최소화**
  게인 벡터 $g$의 성능은 **비용 함수 $J(g)$**를 통해 정량적으로 평가된다. 비용 함수는 궤적 추종 오차와 제어 노력(각속도 크기)의 가중합으로 정의된다.

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
