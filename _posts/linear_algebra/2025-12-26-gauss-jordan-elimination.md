---
layout: default
title: "[Gilbert Strang] Gauss-Jordan Elimination"
date: 2025-12-26 09:00:00 +0900
categories: linear_algebra
permalink: /20251226/gauss-jordan-elimination.html
---

# Gauss-Jordan Elimination

For example, let's solve the following system of linear equations.  

$$
\begin{cases}
x - 2y &= 2 \\
5x + 2y &= 11
\end{cases}
$$

We can rewrite this system in matrix form as follows.  

$$
\begin{bmatrix}
1 & -2 \\
5 & 2
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
2 \\
11
\end{bmatrix}
$$

This can be expressed more compactly using the augment matrix.  

$$
\begin{bmatrix}
1 & -2 & 2 \\
5 & 2 & 11
\end{bmatrix}
$$

This is called an augmented matrix.  
When performing Gaussian elimination, we use the following elementary row operations.
1. Multiply a row by a non-zero scalar.
2. Swap two rows.
3. Add a multiple of one row to another row.

### Implementation of Gaussian Elimination

Step 1: Start with the augmented matrix corresponding to the system.  

$$
\begin{cases}
x - 2y &= 2 \\
5x + 2y &= 11
\end{cases}
\ \rightarrow \
\begin{bmatrix}
1 & -2 & 2 \\
5 & 2 & 11
\end{bmatrix}
$$

Step 2: Multiply the first row by -5 to make the pivot for elimination easier when adding to the second row.    

$$
\begin{bmatrix}
-5 & 10 & -10 \\
5 & 2 & 11
\end{bmatrix}
$$

Step 3: Add the second row to the modified first row (Row 1 + Row 2).  
This gives a new first row of [0, 12, 1], while the second row remains unchanged.  
The resulting matrix is  

$$
\begin{cases}
-5x + &10y &= -10 \\
&12y &= 1
\end{cases}
\ \rightarrow \
\begin{bmatrix}
-5 & 10 & -10 \\
0 & 12 & 1
\end{bmatrix}
$$

Now, we can get solution in linear equation.