---
layout: default
title: "[Gilbert Strang] Linear independence, Basis, and Dimension"
date: 2026-01-24 01:00:00 +0900
categories: linear_algebra
permalink: /20260124/linear-independence-basis-and-dimension.html
---

# Linear independence, Basis, and Dimension
> 출처: Gilbert Strang, Linear Algebra

**다룰 내용**  
1. Linear independence or dependence
2. Spanning a subspace
3. Basis for a subspace (a set of vectors)
4. Dimension of a subspace (a number)

## Linear Independent
주어진 벡터 $\nu_1, \dots, \nu_k$ 의 combination $c_1\nu_1, \dots, c_k\nu_k$ 를 보자.  
여기서 모든 가중치인 $c_i = 0$ 이면, $0\nu_1 + \dots + 0\nu_k = 0$ 가 된다.  

> 질문: 이것이 유일하게 0을 만드는 방법인가?  

**Linear dependence**는 쉽게 3차원 공간을 생각하면 된다.  
두 벡터가 dependent 하면 같은 직선(line)이다.  
세 벡터가 dependent 하면 같은 평면(plane)에 속한다.  

$$
\begin{align}
y &= 2x + 3z \tag{1} \\
2y &= 4x + 6z \tag{2}
\end{align}
$$

위 (1) 식에 2배를 하면, (2) 식이 된다. 이런 경우를 dependent 하다고 한다.  

**예제: Triangular matrix의 column들이 linearly independent할 경우**

$$
\text{No zeros on the diagonal}\quad A = \begin{bmatrix}
3 & 4 & 2 \\
0 & 1 & 5 \\
0 & 0 & 2
\end{bmatrix}
$$

Column들의 combination이 $0$ 이 되는 값을 찾아보자.  

$$
\text{Solve}\ Ac = 0\quad c_1\begin{bmatrix}3 \\ 0 \\ 0\end{bmatrix}
+ c_2\begin{bmatrix}4 \\ 1 \\ 0\end{bmatrix}
+ c_3\begin{bmatrix}2 \\ 5 \\ 2\end{bmatrix}
= \begin{bmatrix}0 \\ 0 \\ 0\end{bmatrix}
$$

여기서 $c_1, c_2, c_3$ 가 $0$ 이 되는지 살펴보자. 마지막 식에서 $2c_3 = 0$ 즉, $c_3 = 0$ 이 된다.  
다음 식에서 $c_2 = 0$ 이 되고, $c_1 = 0$ 이 된다.  
$A$ 의 nullspace에서 0 벡터는 오직 $c_1 = c_2 = c_3$ 가 된다.  

$$
\boxed{\text{The columns of }A \text{ are independent exactly when }N(A) = \text{\{zero vector\}}}
$$

$A$ 의 row들에 대해서도 성립한다.  

$$
c_1\left(3, 4, 2\right) + c_2\left(0, 1, 5\right) + c_3\left(0, 0, 2\right) = \left(0, 0, 0\right)
$$

non-zero row들을 가진 [echelon matrix](https://en.wikipedia.org/wiki/Row_echelon_form) $U$ 는 무조건 independent 하다.  

$$
U = \begin{bmatrix}
\boxed{1} & 3 & 3 & 2 \\
0 & 0 & \boxed{3} & 1 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

pivot column 1과 3은 independent 하다. 이제 세 column들이 independent 하는 경우는 없다.  
즉, pivot column들은 independent 가 보장되었다.  

## Spanning a Subspace
$A$ 의 column 공간은 column에 의해 **span** 되었다.  이 combination들은 전체 공간을 표현한다.  

> $w_1, \dots, w_l$ 의 linear combination들로 구성된 벡터 공간 $V$ 에서  
> 이 벡터들이 공간을 **span** 했다고 한다.  
> 여기서 $v$ 는 $v = c_1w_1 + \dots + c_lw_l$ 와 같다.  

**예제: $\mathbb{R}^3$ 공간에서 벡터들 $w_1 = \left(1, 0, 0\right),\ w_2 = \left(0, 1, 0\right),\ w_3 = \left(-2, 0, 0\right)$ 은 x-y 평면(plane)**  
첫 두 벡터들은 평면에 span 했고, $w_1$ 과 $w_3$ 은 한 직선에 있다.  

## Basis for a Vector Space
$Ax = b$ 를 해결해보자.  
그리고 만약 column들이 independent 하다면, 우리는 $Ax = 0$ 을 푸는 것이다.  
Span은 column 공간과 independence는 nullspace를 포함한다.  
벡터 $e_1, \dots, e_n$ 로 구성된 span $\mathbb{R}^n$ 에서 linearly independent 하다.  
쉽게 말해, 어떤 벡터도 낭비되지 않다는 것이다.  
이 아이디어가 **basis** 에서 가장 중요한 부분 중 하나다.  

공간에 있는 모든 벡터들은 basis 벡터들의 combination이다.  
때문에 다음과 같은 수식이 성립한다.  $v = a_1v_1 + \dots + a_kv_k$ 와 $v = b_1v_1 + \dots + b_kv_k$ 의 뺄셈이 가능하다. $0 = \Sigma{(a_i - b_i)v_i}$  
Independent가 성립하면 $a_i - b_i$ 는 무조건 $0$ 이다. 그러므로 $a_i = b_i$ 이고 유일한 표현식이기 때문에 basis 벡터의 combination이라 할 수 있다.  


## Dimension of a Vector Space
Basis 벡터의 수는 공간의 특성과 같다.  

> $\mathbb{R}^n$ 공간의 차원은 $n$ 이다.  

예를 들어 $\mathbb{R}^3$ 공간에서 2차원만 작동한다면 $\mathbb{R}^3$ 차원 공간에서 2차원 subspace라 표현할 수 있다.
