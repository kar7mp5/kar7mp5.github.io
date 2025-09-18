---
layout: default
title: "Linear Algebra"
date: 2025-08-08 09:00:00 +0900
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
e_1 = \begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}

\quad

e_2 = \begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}

\quad

e_3 = \begin{bmatrix}
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

vectors $\mathbf{a}_1, \dots \mathbf{a}_m$
