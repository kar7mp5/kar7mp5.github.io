---
layout: default
title: "[Gilbert Strang] Echelon Form"
date: 2026-01-22 09:00:00 +0900
categories: linear_algebra
permalink: /20260122/echelon-form.html
---

# Echelon Form
> 출처: Gilbert Strang, Linear Algebra
 
**Echelon form** 은 한국어로 사다리꼴 행렬이라고도 불린다.  
Lower triangular matrix 형태 중 하나이다.  

Echelon form이 아닌 예시를 살펴보자.  

$$
\text{Basic example} \quad A = 
\begin{bmatrix}
1 & 3 & 3 & 2 \\
2 & 6 & 9 & 7 \\
-1 & -3 & 3 & 4
\end{bmatrix}
$$

첫 번째 row의 $2$ 배 만큼 두 번째 row에 뺄셈하면 다음과 같다.  
세 번째 row에도 똑같이 계산해준다.

$$
A \rightarrow 
\begin{bmatrix}
1 & 3 & 3 & 2 \\
0 & 0 & 3 & 3 \\
0 & 0 & 6 & 6
\end{bmatrix}
$$

여기서 세 번째 row를 두 번째 row의 $2$ 배 만큼 빼주어 $0$ 으로 만들어준다.  

$$
U = 
\begin{bmatrix}
1 & 3 & 3 & 2 \\
0 & 0 & 3 & 3 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

두 번째 row의 pivot 값인 $3$ 을 나눠주고, 두 번째 row의 $3$ 배 만큼 첫 번째 row에 빼줘서 식을 정리한다.
이를 **Reduced row echelon form R** 이라 한다.

$$
\begin{bmatrix}
1 & 3 & 3 & 2 \\
0 & 0 & 3 & 3 \\
0 & 0 & 0 & 0
\end{bmatrix} \rightarrow
\begin{bmatrix}
1 & 3 & 3 & 2 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0
\end{bmatrix} \rightarrow
\begin{bmatrix}
1 & 3 & 0 & -1 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0
\end{bmatrix} = R
$$

MATLAB에서 다음과 같은 명령어로 표현한다.  `R = rref(A)`  
여기서 $R$ 이 identity matrix이고 full-pivot 하고 그 값이 $1$ 이고 그 위에 값들이 $0$ 이라면  
`rref(A) = I` 즉, $A$ 가 invertible하다는 점을 알 수 있다.  

### 일반해를 구해보자.

$$
x = \begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}
$$

**pivot column vs free column**  
RREF에서 pivot 위치가 중요하다.  

pivot columns  
- 1열 ($x_1$)  
- 3열 ($x_3$)

free columns  
- 2열 ($x_2$)
- 4열 ($x_4$)

#### 행렬을 방정식으로 해석

$$
x_1 + 3x_2 - x_4 = 0
$$

$$
x_3 + x_4 = 0
$$

#### pivot 변수를 자유변수로 해석

$$
x_2 = s,\quad x_4 = t
$$

그러면  

$$
\begin{align}
x_1 &= -3s + t \\
x_3 & = -t
\end{align}
$$

#### 일반해 (vector form)

$$
x = \begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix} = \begin{bmatrix}
-3s + t \\
s \\
-t \\
t
\end{bmatrix}
$$

이를 linear combination 형태로 정리한다.  

$$
x = s\begin{bmatrix}
-3 \\ 1 \\ 0 \\ 0
\end{bmatrix} + 
t \begin{bmatrix}
1 \\ 0 \\ -1 \\ 1
\end{bmatrix}
$$

- $s, t \in \mathbb{R}$ 
- 모든 해는 이 두 벡터의 linear combination
- 즉, null space의 basis가 2개
- nullity = 2

rank = 2  
nullity = 2  
(열 4개 = rank + nullity)

> Rank-Nullity Theorem에 맞는다.  
