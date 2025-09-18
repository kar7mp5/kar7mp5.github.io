---
layout: default
title: "Linear Algebra: 행렬 기초 연산"
date: 2025-09-19 00:00:00 +0900
categories: linear_algebra
permalink: /20250808/lexer-parser-ast.html
---

# Linear Algebra: 행렬 기초 연산

## Block Vectors

$$
\mathbf{a} = \begin{bmatrix}
b \\
c \\
d
\end{bmatrix}
$$

---

## Zero, ones, and unit vectors

- `n-vector` 모든 값이 $0$ 이면, $0_n$, $0$라 표현한다.
- `n-vector` 모든 값이 $1$ 이면, $\mathbf{1}_n$, $\mathbf{1}$라 표현한다.
- `unit vector` 는 하나의 값이 $1$, 나머지는 $0$으로 채워짐.

e.g., 길이가 $3$인 단위벡터

$$
\mathbf{e}_1 = \begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}

\quad

\mathbf{e}_2 = \begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}

\quad

\mathbf{e}_3 = \begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix}
$$

---

## Sparsity

벡터에 $0$으로 채워진 경우가 많다. 이를 효율적으로 계산할 수 있음.  
$\mathbf{mnz}(x)$: non-zero인 값의 수

e.g., `zero vector`, `unit vector`

---

## Properties of vector addition

- `commutative`: $a + b = b + a$
- `associative`: $(a + b) + c = a + (b + c)$

---

## Scaler-vector multiplication

scalar $\beta$, n-vector $\mathbf{a}$

$$
\beta \mathbf{a} = (\beta \mathbf{a}_1, \dots, \beta \mathbf{a}_n)
$$

e.g.,

$$
(-2) \begin{bmatrix} 1 \\ 9 \\ 6 \end{bmatrix} = \begin{bmatrix} -2 \\ -18 \\ -12 \end{bmatrix}
$$

---

## Properties of scalar-vector multiplication

- `associative`: $(\beta \gamma)\mathbf{a} = \beta(\gamma \mathbf{a})$
- `left distributive`: $(\beta + \gamma) \mathbf{a} = \beta\mathbf{a} + \gamma\mathbf{a}$
- `right distributive`: $\beta(\mathbf{a} + \mathbf{b}) = \beta\mathbf{a} + \beta\mathbf{b}$

---

## Linear combinations

vectors $\mathbf{a}_1, \dots, \mathbf{a}_m$, scalars $\beta_1, \dots,\beta_m$ 의 linear combination은 $\beta_1\mathbf{a}_1 + \dots + \beta_m\mathbf{a}_m$ 이다.  

e.g., for any n-vector $\mathbf{b}$  

$$
\mathbf{b} = \mathbf{b}_1\mathbf{e}_1 + \dots + \mathbf{b}_n\mathbf{e}_n
$$

---

## Linear product

Inner product (or dot product) of n-vectors $\mathbf{a}$ and $\mathbf{b}$ is  

$$
\mathbf{a}^T\mathbf{b} = \mathbf{a}_1\mathbf{b}_1 + \dots + \mathbf{a}_n\mathbf{b}_n
$$

다른 notation: $<a, b>,\ <a \vert b>,\ (a, b),\ a \cdot b$  

e.g.,  

$$
\begin{bmatrix} -1 \\2 \\2 \end{bmatrix}^T
\begin{bmatrix} 1 \\0 \\-3 \end{bmatrix} =
(-1)(1) + (2)(0) + (2)(-3) = -7
$$

---

## Properties of inner product

- $\mathbf{a}^T\mathbf{b} = \mathbf{b}^T\mathbf{a}$
- $(\gamma\mathbf{a})^T\mathbf{b} = \gamma(\mathbf{a}^T\mathbf{b})$
- $(\mathbf{a}+\mathbf{b})^T\mathbf{c} = \mathbf{a}^T\mathbf{c}+\mathbf{b}^T\mathbf{c}$

e.g.,  

$$
(\mathbf{a}+\mathbf{b})^T (\mathbf{c}+\mathbf{d}) = \mathbf{a}^T\mathbf{c}+\mathbf{a}^T\mathbf{d}+\mathbf{b}^T\mathbf{c}+\mathbf{b}^T\mathbf{d}
$$

---

## 중요 예시

행렬에서 자주 사용하는 수식이다.

$$
\mathbf{e}^T_i\mathbf{a} = \mathbf{a}_i \quad \text{(picks out ith entry)}
$$

$$
\mathbf{1}^T\mathbf{a} = \mathbf{a}_1 + \dots + \mathbf{a}_n \quad \text{(sum of entries)}
$$

$$
\mathbf{a}^T\mathbf{a} = \mathbf{a}^2_i + \dots + \mathbf{a}^2_n \quad \text{(sum of squares of enties)}
$$