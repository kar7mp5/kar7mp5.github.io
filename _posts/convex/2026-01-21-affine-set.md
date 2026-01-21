---
layout: default
title: "Affine set"
date: 2026-01-21 13:00:59 +0900
categories: convex_optimization
permalink: /20250121/affine-set.html
---

# Affine set

**Affine set**은 점(point), 직선(line), 평면(plane), 초평면(hyperplane)과 같이 선형적 특성이 있으면서 경계 없는 집합을 말한다.  
어떤 집합이 affine set이라 말할 수 있으려면 집합에 속한 임의의 두 점으로 직선을 만들어 그 직선이 집합에 포함되는지 보면 된다.  
즉, 어떤 공간이 경계가 있다면 affine set이 될 수 없다.  

### Affine set
집합 $C \subseteq \mathbb{R}^n$ 에 속한 두 점 $x_1, x_2 \in C$ 을 지나는 직선을 만들었을 때, 이 직선이 $C$ 에 포함되면 이 집합을 **affine set**이라 한다.

$$
\theta x_1 + (1 - \theta) x_2 \in C \quad \text{with} \quad \theta \in \mathbb{R}
$$

set $C$ 에 속한 두 점을 linear combination 하되 계수의 합을 $1$ 로 제한했다고 해석 가능하다.  

### Affine combination
여러 점들을 linear combination할 때 계수의 합을 $1$로 제한하게 되면 이를 **affine combination**이라 한다.

$$
\theta_1 x_1 + \theta_2 x_2 + \dots + \theta_k x_k \in C \quad \text{with} \quad \theta_1 + \theta_2 + \dots + \theta_k = 1
$$

affine set 정의를 affine combination 개념을 이용해 일반화 가능하다.  
즉, 어떤 집합에 속하는 점들을 affine combination 했을 때, 그 결과가 다시 그 집합에 속하면 affine set이라 말할 수 있다.

### Affine hull

![Convex hull image](https://media.geeksforgeeks.org/wp-content/uploads/20231218123325/Convex-Hull.jpg)

$C \subseteq \mathbb{R}^n$ 에 포함된 점들의 모든 affine combination의 집합을 $C$의 affine hull이라 하며 **aff $C$** 로 표기한다.  
Affine hull **aff $C$** 은 항상 affine set이며, 집합 $C$ 를 포함하는 가장 작은 affine set이다.  

$$
\text{aff}(C) = \{\theta_1 x_1 + \dots + \theta_k x_k \quad | \quad x_1, \dots, x_k \in C, \theta_1 + \dots + \theta_k = 1 \}
$$

### Affine set과 subspace 관계
Affine set $C$ 가 있을 때 $x_0 \in C$ 라면 set $V = C - x_0$ 는 subspace 이다.

$$
V = C - x_0 = \{x - x_0 \ | \ x \in C\}
$$

**"Affine set $C$ 은 linear subspace $V$ 를 $x_0$ 만큼 translation한 것이다"** 라 할 수 있으며, $x_0$ 는 집합 $C$ 에서 임의로 선택 가능하다.  
$C$ 의 차원은 $V$ 의 차원과 같다. ($C, V \subseteq \mathbb{R}^n$)

$$
C = V + x_0 = \{v + x_0 \ | \ v \in V\}
$$

#### [증명] $V$ 가 subspace임을 증명
$V$ 가 subspace임을 증명하려면 sum과 scalar multiplication에 닫혀있다는 것을 보이면 된다.  
즉, $v_1, v_2 \in V, \ \alpha, \beta \in \mathbb{R}$ 에 속한다는 것을 보이는 것이다. 이는 $V = C - x_0$ 에 의해 $\alpha v_1 + \beta v_2 \in V$ 가 되므로 결국 $V$ 가 subspace임을 의미한다.  
먼저, $v_1, v_2 \in V$ 이므로 $v_1 + x_0 \in C$ 이고 $v_2 + x_0 \in C$ 이다. $C$ 는 affine set이므로, affine set의 정의에 의해 다음이 성립한다.

$$
\alpha(v_1 + x_0) + \beta(v_2 + x_0) + (1 - \alpha - \beta)x_0 \in C
$$

왜냐하면 좌항 계수의 합이 $\alpha + \beta + (1 - \alpha - \beta) = 1$ 이기 때문이다.  또한,

$$
\alpha v_1 + \beta v_2 + x_0 = \alpha(v_1 + x_0) + \beta(v_2 + x_0) + (1 - \alpha - \beta)x_0
$$

이므로 $\alpha v_1 + \beta v_2 + x_0 \in C$ 이다. 따라서 $\alpha v_1 + \beta v_2 \in V$ 가 되어서 $V$ 는 sum과 scalar multiplication에 닫혀있는 subspace임을 알 수 있다.  
