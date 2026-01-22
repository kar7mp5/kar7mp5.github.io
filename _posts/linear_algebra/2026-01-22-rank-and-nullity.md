---
layout: default
title: "[Gilbert Strang] Rank and Nullity"
date: 2026-01-22 09:00:00 +0900
categories: linear_algebra
permalink: /20260122/rank-and-nullity.html
---

# Rank and Nullity
> rank는 정보가 실제로 전달되는 방향의 개수  
> nullity는 아무 정보도 안 주는 방향의 개수

$A$ 는 $3 \times 4$ 행렬에 $x \in \mathbb{R}^4$ 일 때 다음 식을 보자.  

$$
Ax = 0
$$

4차원 공간의 벡터를 받아서 3차원으로 보낸다.  

### rank란?  
pivot의 개수이고, 서로 독립적인 열의 개수를 의미한다.  

$$
\begin{bmatrix}
1 & 3 & 0 & -1 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

여기서 1행 1열, 3행 2열이 pivot이기에  
pivot는 2개이고,  
rank = 2이다.  

이것의 의미는 이 행렬을 아무리 잘 짜도 최대 2차원짜리 정보만 return할 수 있다는 것이다.  

### nullity란?  
$Ax = 0$ 을 만족하는 자유도의 개수를 의미한다.  
직관적으로 말하면, 입력했는데 결과가 전부 $0$ 이 되어버리는 방향의 개수를 의미한다.  

예를 들어,  
$x \neq 0$ 인데 $Ax = 0$ 이런 $x$ 들의 공간이 **null space**  
그 공간의 차원이 **nullity** 이다.  

### 일반해는 왜 null space의 basis가 될까?  

$$
x = s \begin{bmatrix}
-3 \\ 1 \\ 0 \\ 0
\end{bmatrix} + 
t \begin{bmatrix}
1 \\ 0 \\ -1 \\ 1
\end{bmatrix}
$$

여기서 $Ax = 0$ 을 만족하는 모든 해는 위 두 벡터의 linear combination으로 표현이 가능하다.  
- 두 벡터가 null space의 basis
- basis 개수 = 2
- nullity = 2

때문에 다음이 성립된다.  

$$
\text{Rank}\ +\ \text{Nullity} = \text{열의 개수}
$$

예를 들어, 카메라를 생각해보자.  
$x$ 는 4차원 물체이고,  
$A$ 는 카메라이고,  
사진은 $Ax$ 은 2차원이다.  

여기서 보여지는 방향 2개이고,  
보이지 않는 방향은 2개이다.  
