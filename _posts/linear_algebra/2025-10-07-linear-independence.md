---
layout: default
title: "Linear independence"
date: 2025-10-07 09:00:00 +0900
image: ../assets/posts/2025-09-21-clustering/cluster2.png
categories: linear_algebra
permalink: /20251007/linear-independence.html
---

# Linear independence

This blog is based on [Jong-han Kim's Linear Algebra](https://jonghank.github.io/ase2910.html)

## Linear dependence

set of $n$-vectors ${a_1, \dots, a_k} \ (\text{with } k \geq 1)$ is linearly dependent if

$$
\beta_1 a_1 + \dots + \beta_k a_k = 0
$$

holds for some $\beta_1, \dots, \beta_k$ that are not all zero

## Linear independence

set of $n$-vectors ${a_1, \dots, a_k} \ (\text{with } k \geq 1)$ is linearly independent if

$$
\beta_1 a_1 + \dots + \beta_k a_k = 0
$$

holds only when $\beta_1 = \dots = \beta_k = 0$

e.g., the unit $n$-vectors $e_1, \dots, e_n$ are linearly independent

### Linear combinations of linearly independent vectors

suppose $x$ is linear combination of linearly independent vectors $a_1, \dots, a_k$

$$
x = \beta_1 a_1 + \dots + \beta_k a_k
$$

the coefficients $\beta_1, \dots, \beta_k$ are unique, i.e. if

$$
x = \gamma_1a_1 + \dots + \gamma_ka_k
$$

then $\beta_i = \gamma_i$ for $i = 1, \dots, k$

this means that (in principle) we can deduce the coefficients from $x$  
to see why, note that

$$
(\beta_1 - \gamma_1)a_1 + \dots + (\beta_k - \gamma_k)a_k = 0
$$

and so (by linear independence) $\beta_1 - \gamma_1 = \dots = \beta_k - \gamma_k = 0$

### Independence-dimension inequality

- a linearly independent set of $n$-vectors can have at most $n$ elements
- put another way: any set of $n+1$ or more $n$-vectors is linearly dependent

---

## Basis

a set of $n$ linearly independent $n$-vectors $a_1, \dots, a_n$ is called a basis  
any $n$-vector $b$ can be expressed as a linear combination of them

$$
b = \beta_1a_1 + \dots + \beta_n a_n
$$

for some $\beta_1, \dots, \beta_n$

and these coefficients are unique  
formula above is called expansion of $b$ in the $a_1, \dots, a_n$ basis

e.g., $e_1, \dots, e_n$ is a basis, expansion of $b$ is

$$
b = b_1e_1 + \dots + b_ne_n
$$

---

## Orthonormal vectors

set of $n$-vectors $a_1, \dots, a_k$ are (mutually) orthogonal if $a_i \bot a_j$ for $i \neq j$  
they are normalized if $\Vert a_i\Vert = 1$ for $i = 1, \dots, k$  
they are orthonormal if both hold  
can be expressed using inner products as

$$
a^T_ia_j = \begin{cases}
1 & i = j \\
0 & i \neq j
\end{cases}
$$

orthonormal sets of vectors are linearly independent  
by independent-dimension inequality, must have $k \leq n$  
when $k = n, a_1, \dots, a_n$ are an orthonormal basis

### Orthonormal expansion

if $a_1, \dots, a_n$ is an orthonormal basis, we have for any $n$-vector $x$

$$
x = (a^T_1 x)a_1 + \dots + (a^T_nx)a_n
$$

called orthonormal expansion of $x$ (in the orthonormal basis)  
to verify formula, take inner product of both sides with $a_i$

### Gram-Schmidt (orthonormalization) algorithm

- an alorithm to check if $a_1, \dots, a_k$ are linearly independent
- we'll see later it has many other uses

$$
\begin{align*}
& \textbf{given } \textit{n}\text{-vectors } a_1, \dots, a_k \\
& \textbf{for } i = 1, \dots, k \\
& \quad \text{1. Orthogonalization: } \tilde{q}_i = a_i - (q_1^T a_i)q_1 - \cdots - (q_{i-1}^T a_i)q_{i-1} \\
& \quad \text{2. Test for linear dependence: if } \tilde{q}_i = 0, \text{ quit} \\
& \quad \text{3. Normalization: } q_i = \tilde{q}_i / \| \tilde{q}_i \|
\end{align*}
$$

if G-S does not stop early (in step 2), $a_1, \dots, a_k$ are linearly independent  
if G-S stops early in iteration $i=j$, then $a_j$ is a linear combination of $a_1, \dots, a_{j-1}$ (so $a_1, \dots, a_k$ are linearly dependent)

### Code

```python
import numpy as np

def gram_schmidt(vectors):
  A = np.array(vectors, dtype=float).T
  n, k = A.shape

  Q = np.zeros((n, k))

  for i in range(k):
    a_i = A[:, i]
    q_tilde = a_i

    for j in range(i):
      q_j = Q[:, j]
      projection = np.dot(q_j.T, a_i) * q_j
      q_tilde = q_tilde - projection

    norm_q_tilde = np.linalg.norm(q_tilde)

    if norm_q_tilde < 1e-10:
      print("linear dependent")
      return Q[:, :i]

    Q[:, i] = q_tilde / norm_q_tilde

  return Q

a1 = [1, 1, 0]
a2 = [2, 0, 1]
a3 = [0, 1, 2]
vectors_independent = [a1, a2, a3]

orthonormal_vectors = gram_schmidt(vectors_independent)
print(orthonormal_vectors.T)
print(np.dot(orthonormal_vectors.T, orthonormal_vectors))

print('='*30)

b1 = [1, 0, 1]
b2 = [0, 1, 1]
b3 = [2, 1, 3]
vectors_dependent = [b1, b2, b3]

orthonormal_vectors_dep = gram_schmidt(vectors_dependent)

print(orthonormal_vectors_dep.T)
```

```
[[ 0.70710678  0.70710678  0.        ]
 [ 0.57735027 -0.57735027  0.57735027]
 [-0.40824829  0.40824829  0.81649658]]
[[ 1.00000000e+00  2.50235355e-16  1.47380436e-17]
 [ 2.50235355e-16  1.00000000e+00 -3.33168241e-17]
 [ 1.47380436e-17 -3.33168241e-17  1.00000000e+00]]
==============================
linear dependent
[[ 0.70710678  0.          0.70710678]
 [-0.40824829  0.81649658  0.40824829]]
```
