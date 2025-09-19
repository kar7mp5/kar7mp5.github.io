---
layout: default
title: "Norm and distance"
date: 2025-09-19 00:00:00 +0900
image: ../assets/posts/2025-09-19-linear-algebra-base-w2/linear_verus_affine.png
categories: linear_algebra
permalink: /20250808/norm-distance.html
---

# Norm and distance

This blog is based on [Jong-han Kim's Linear Algebra](https://jonghank.github.io/ase2910.html)

## Norm

The [`Euclidean norm`]() (or just `norm`) of an $n$-vector $x$ is

$$
\lVert x \rVert = \sqrt{x^2_1 + x^2_2 + \dots + x^2_n} = \sqrt{x^T x}
$$

- used to measure the size of vector; vector distance

### Properties

for any $n$-vectors $x$ and $y$, and any scalar $\beta$

- **homogeneity**: $\lVert\beta x\rVert = \lvert\beta\rvert\lVert x\rVert$
- **triangle inequality**: $\lVert x + y \rVert \leq \lVert x \rVert + \lVert y \rVert$
- **nonnegativity**: $\lVert x \rVert \geq 0$
- **definiteness**: $\lVert x \rVert = 0 \quad \text{only if} \quad x = 0$

---

## RMS value

Mean-square value of $n$-vector $x$ is

$$
\frac{x^2_1 + \dots + x^2_n}{n} = \frac{\lVert x \rVert^2}{n}
$$

Root-mean-square value (RMS value) is

$$
\mathbf{rms}(x) = \sqrt{\frac{x^2_1 + \dots + x^2_n}{n}} = \frac{\lVert x \rVert}{\sqrt{n}}
$$

- $\mathbf{rms}(x)$ gives typical value of $\lvert x_i \rvert$
- e.g., $\mathbf{rms}(\mathbf{1}) = 1 \ (\text{independent of} \ n)$
- RMS value useful for comparing sizes of vectors of different lengths

---

## Norm of block vectors

suppose $a, b, c$ are vectors

$$
\lVert(a, b, c)\rVert^2 = a^Ta + b^Tb + c^Tc = \lVert a\rVert^2 + \lVert b\rVert^2 + \lVert c\rVert^2
$$

so we have

$$
\lVert(a, b, c)\rVert = \sqrt{\lVert a\rVert^2 + \lVert b\rVert^2 + \lVert c\rVert^2} = \lVert(\lVert a\rVert^2, \lVert b\rVert^2, \lVert c\rVert^2)\rVert
$$

---

## Chebyshev inequality

suppose that $k$ of the numbers $\lvert x_1\rvert, \dots, \lvert x_n\rvert$ are $\geq a$  
then $k$ of the numbers $x^2_1, \dots, x^2_n$ are $\geq a^2$  
so $\lVert x\rVert^2 = x^2_1 + \dots + x^2_n \geq k a^2$  
so we have $k \leq \lVert x\rVert^2 / a^2$
number of $x_i$ with $\lVert x_i\rVert \geq a$ is no more than $\lVert x\rVert^2 / a^2$

- In terms of RMS value:

fraction of entries with $\lvert x_i\rvert \geq a$ is no more than $\left(\frac{\mathbf{rms}(x)}{a}^2 \right)$

e.g., no more than 4% of entries can satisfy $\lvert x_i\rvert \geq 5 \mathbf{rms}(x)$

---

## Distance

Euclidean distance between $n$-vectors $a$ and $b$ is

$$
\mathbf{dist}(a, b) = \lVert a-b\rVert
$$

agrees with ordinary distance for $n = 1, 2, 3$  
$\mathbf{rms}(a - b)$ is the RMS deviation between $a$ and $b$

---

## Triangle inequality

Triangle with vertices at positions $a, b, c$  
edge lengths are $\lVert a - b\rVert, \lVert b - c\rVert, \lVert a - c\rVert$  
by triangle inequality

$$
\lVert a - c\rVert = \lVert(a-b) + (b-c)\rVert \leq \lVert a-b\rVert + \lVert b-c\rVert
$$

---

## Standard deviation

for $n$-vector $x$, $\mathbf{avg}(x) = \mathbf{1}^T x / n$  
de-meaned vector is $\tilde{x} = x - \mathbf{avg}(x)\mathbf{1} \ \left(\text{so} \ \mathbf{avg}(\tilde{x}) = 0 \right)$
standard deviation of $x$ is

$$
\mathbf{std}(x) = \mathbf{rms}(\tilde{x}) = \frac{\lVert x - (\mathbf{1}^T x/n)\mathbf{1}\rVert}{\sqrt{n}}
$$

$\mathbf{std}(x)$ gives typical amount $x_i$ vary from $\mathbf{avg}(x)$  
$\mathbf{std}(x) = 0$ only if $x = \alpha\mathbf{1}$ for some $\alpha$  
greek letters $\mu, \sigma$ commonly used for mean, standard deviation  
a basic formula:

$$
\mathbf{rms}(x)^2 = \mathbf{avg}(x)^2 + \mathbf{std}(x)^2
$$

### The Core Identity

The statistical identity relating the Root Mean Square (RMS), average (avg), and standard deviation (std) of a data vector $\mathbf{x}$ is given by:

$$
\text{rms}(\mathbf{x})^2 = \text{avg}(\mathbf{x})^2 + \text{std}(\mathbf{x})^2
$$

This identity is derived from the geometric decomposition of a vector in an n-dimensional space, based on the Pythagorean theorem.

### Vector Definitions

Let $\mathbf{x}$ be a data vector in $\mathbf{R}^n$ and $\mathbf{1}$ be the vector of ones in $\mathbf{R}^n$.

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}, \quad \mathbf{1} = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1
\end{bmatrix}
$$

We define the average (mean) of $\mathbf{x}$ as $\mu = \text{avg}(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^{n} x_i = \frac{1}{n}\mathbf{1}^T\mathbf{x}$.

The vector $\mathbf{x}$ can be decomposed into two fundamental components:

1.  **Average Component Vector**:  
    A vector where each element is the mean, $\mu$. This represents the constant part of the data.

$$
\mu\mathbf{1} = \begin{bmatrix} \mu \\ \mu \\ \vdots \\ \mu \end{bmatrix}
$$

2.  **Deviation (De-meaned) Vector**:
    The vector of deviations from the mean. This represents the fluctuating part of the data.

$$
\tilde{\mathbf{x}} = \mathbf{x} - \mu\mathbf{1} = \begin{bmatrix} x_1 - \mu \\ x_2 - \mu \\ \vdots \\ x_n - \mu \end{bmatrix}
$$

### Orthogonal Decomposition

The decomposition of $\mathbf{x}$ is written as $\mathbf{x} = \mu\mathbf{1} + \tilde{\mathbf{x}}$. The key geometric insight is that these two component vectors are **orthogonal**, meaning their dot product is zero.

**Proof of Orthogonality:**

$$
\begin{align*}
(\mu\mathbf{1})^T \tilde{\mathbf{x}} &= (\mu\mathbf{1})^T (\mathbf{x} - \mu\mathbf{1}) \\
&= \mu\mathbf{1}^T\mathbf{x} - \mu^2\mathbf{1}^T\mathbf{1}
\end{align*}
$$

By definition, $\mathbf{1}^T\mathbf{x} = n\mu$ and the dot product $\mathbf{1}^T\mathbf{1} = n$. Substituting these in:

$$
\begin{align*}
&= \mu(n\mu) - \mu^2(n) \\
&= n\mu^2 - n\mu^2 \\
&= 0
\end{align*}
$$

Since their dot product is zero, the vectors are orthogonal: $\mu\mathbf{1} \perp \tilde{\mathbf{x}}$.

### The Pythagorean Theorem & Final Derivation

Because the components are orthogonal, they form a right-angled triangle in $\mathbb{R}^n$. The Pythagorean theorem applies to their squared norms (lengths):

$$
\|\mathbf{x}\|^2 = \|\mu\mathbf{1}\|^2 + \|\tilde{\mathbf{x}}\|^2
$$

The statistical terms are the mean of these squared norms. By dividing the entire equation by $n$, we derive the final identity. We use the definitions:

- $\text{rms}(\mathbf{x})^2 = \frac{1}{n}\|\mathbf{x}\|^2$
- $\text{avg}(\mathbf{x})^2 = \mu^2 = \frac{1}{n}\|\mu\mathbf{1}\|^2$
- $\text{std}(\mathbf{x})^2 = \frac{1}{n}\|\tilde{\mathbf{x}}\|^2$

The final derivation is:

$$
\begin{align*}
\frac{\|\mathbf{x}\|^2}{n} &= \frac{\|\mu\mathbf{1}\|^2}{n} + \frac{\|\tilde{\mathbf{x}}\|^2}{n} \\[1em]
\text{rms}(\mathbf{x})^2 &= \text{avg}(\mathbf{x})^2 + \text{std}(\mathbf{x})^2
\end{align*}
$$

---

## Mean return and risk

- $\mathbf{avg}(x)$ is the mean return over the period, usually just called `return`.
- $\mathbf{std}(x)$ measures how variable the return is over the period, and is called the `risk`.

---

## Cheyshev inequality for standard deviation

For any two $n$-vectors $\mathbf{a}$ and $\mathbf{b}$, the absolute value of their dot product is less than or equal to the product of their norms.

$$
|\mathbf{a}^T\mathbf{b}| \leq \|\mathbf{a}\|\|\mathbf{b}\|
$$

This is true because the geometric definition of the dot product is $\|\mathbf{a}\|\|\mathbf{b}\|\cos(\theta)$, and the absolute value $|\cos(\theta)|$ cannot exceed 1. Written out in terms of their components, the inequality is:

$$
|a_1b_1 + \dots + a_nb_n| \leq (a^2_1 + \dots + a^2_n)^{1/2}(b^2_1 + \dots + b^2_n)^{1/2}
$$

### The Triangle Inequality

The norm of the sum of two vectors is less than or equal to the sum of their individual norms. Geometrically, this means the length of any side of a triangle is less than or equal to the sum of the lengths of the other two sides.

$$
\|\mathbf{a}+\mathbf{b}\| \leq \|\mathbf{a}\| + \|\mathbf{b}\|
$$

#### Proof

This inequality can be proven using the Cauchy-Schwarz inequality as follows.

$$
\begin{align*}
\|\mathbf{a}+\mathbf{b}\|^2 &= (\mathbf{a}+\mathbf{b})^T(\mathbf{a}+\mathbf{b}) \\
&= \|\mathbf{a}\|^2 + 2\mathbf{a}^T\mathbf{b} + \|\mathbf{b}\|^2 \\
&\leq \|\mathbf{a}\|^2 + 2|\mathbf{a}^T\mathbf{b}| + \|\mathbf{b}\|^2 \quad (\text{since } x \le |x|) \\
&\leq \|\mathbf{a}\|^2 + 2\|\mathbf{a}\|\|\mathbf{b}\| + \|\mathbf{b}\|^2 \quad (\text{by the Cauchy-Schwarz inequality}) \\
&= (\|\mathbf{a}\| + \|\mathbf{b}\|)^2
\end{align*}
$$

Taking the square root of both sides completes the proof of the triangle inequality.

$$
\|\mathbf{a}+\mathbf{b}\| \leq \|\mathbf{a}\| + \|\mathbf{b}\|
$$
