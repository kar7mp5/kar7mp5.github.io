---
layout: default
title: "Lipschitz and constant"
date: 2025-11-27 09:00:00 +0900
categories: linear_algebra
permalink: /20251127/lipschitz-function-and-constant.html
---

# Lipschitz function and constant

## Lipschitz Function

A function $f$ is called Lipschitz continous (or just Lipschitz) if there exists a constant $L \geq 0$ such that for all points $x$ and $y$ in the domain

$$
\|f(x) - f(y)\| \leq L \cdot \|x - y\|
$$

(or more generally, $\|\|f(x) - f(y)\|\| \leq L \cdot \|\|x - y\|\|$ in higher dimensions).

The output of the function can't change faster than a fixed multiple of how much the input changes. In other words, the function is not allowed to have infinitely steep slopes.

## Lipschitz Constant

The number $L$ in the inequality above is called the Lipschitz constant of the function.

-   The smallest possible $L$ that works for the whole domain is called the best Lipschitz constant or optimal Lipschitz constant.
-   If $L$ is small, the function changes more slowly/smoothly.
-   if $L$ is big, the function can change more quickly.

## Examples

| Function                         | Is it Lipschitz? | Lipschitz Constant (example)              |
| -------------------------------- | ---------------- | ----------------------------------------- |
| $f(x) = 5x + 2$                  | Yes              | $L = 5$                                   |
| $f(x) = \sin(x)$                 | Yes              | $L = 1$ (because the derivative ≤ 1)      |
| $f(x) = x^2$ on $[-1, 1]$        | Yes              | $L = 2$ (on this interval)                |
| $f(x) = x^2$ on all real numbers | No               | Not bounded (gets steeper as \|x\| grows) |
| $f(x) = \sqrt{x} $ on [0, ∞)     | No               | Slope becomes infinite near 0             |
