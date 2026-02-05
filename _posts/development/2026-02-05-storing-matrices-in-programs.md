---
layout: default
title: "Storing matrices in programs"
date: 2025-08-21 09:00:00 +0900
categories: development
permalink: /20250808/smart-pointer.html
---

# Storing matrices in programs

> [gsl 공식 문서](https://www.gnu.org/software/gsl/doc/html/vectors.html)

## 프로그래밍 언어에서 벡터, 행렬이 어떻게 저장되는지 알아보자
두 자료구조(벡터, 행렬) 모두 삽입(insert), 삭제(delete), 복사(copy), 슬라이싱(slice) 연산이 필요하다.   
간단히 생각해봤을 때, 삽입/삭제/복사까지는 자료구조를 공부해봤다면 쉽게 예상이 된다.  

그런데 슬라이싱(slice)은 조금 다르다.  
슬라이스는 “일부를 잘라 새로 만든다”는 느낌이지만, 실제 구현은 **대부분 ‘복사’가 아니라 ‘뷰(view)’** 다.

#### C언어 행렬 저장
```c
typedef struct
{
  size_t size;
  double * data;
} gsl_block;
```

`gsl_block` 구조체는 **실제 데이터(연속 메모리)** 를 들고 있는 컨테이너다.  
즉, `double* data`가 “진짜 숫자들이 있는 곳”이다.

```c
typedef struct
{
  size_t size1;
  size_t size2;
  size_t tda;
  double * data;
  gsl_block * block;
  int owner;
} gsl_matrix;
```

`gsl_matrix` 구조체는 “데이터가 어떻게 저장되는지” 자체라기보다, **어떤 block을 어떤 규칙으로 해석할지** 를 담는다.

예를 들면,  
- `block->data`는 길게 쭉 깔린 `double` 배열인데,
- `gsl_matrix`는 그걸 **(size1 × size2) 2차원처럼 보이게** 만들고,
- `tda`(leading dimension)는 “한 행에서 다음 행으로 넘어갈 때 몇 칸 점프해야 하는지”를 정한다.
#### 행렬은 보통 1차원 배열로 저장된다 (row-major + tda)
C에서 행렬은 보통 메모리에 이렇게 저장된다.  
- `data`가 `double data[ ... ]` 형태로 **연속 메모리**를 가진다
- `(i, j)` 원소는 보통 다음처럼 접근한다:

$$
A_{i,j} \leftrightarrow data[i \cdot tda + j]  
$$

여기서 `tda`는 “row stride” 같은 역할이다.
- 가장 단순한 **연속 행렬**이면 `tda == size2`
- 하지만 **서브행렬(view)** 을 만들면 `tda`가 `size2`와 달라질 수 있다  
(겉보기 열 개수는 줄었는데, 원래 큰 행렬의 행 간 간격은 그대로기 때문)

## 슬라이싱은 “복사”가 아니라 “view 생성”으로 구현된다
슬라이싱을 할 때 매번 새 메모리를 할당해서 복사하면:
- 슬라이스 한 번 할 때마다 `O(k)` 복사 비용이 들고
- 연쇄 슬라이싱(예: `A[10:20, 5:15]` 후 또 자르기)이 비싸지고
- 캐시/메모리도 더 많이 쓴다

그래서 GSL 같은 C 라이브러리들은 보통 이렇게 한다:
- **원본 block은 그대로 둔다**
- 대신, 새로운 `gsl_matrix`(혹은 view 구조체)를 만들어
    - `data` 포인터를 “시작 위치로 이동”
    - `size1`, `size2`를 “잘라낸 크기로 설정”
    - `tda`는 “원본의 행 간격 그대로 유지”
    - `owner = 0` (이 뷰는 메모리를 소유하지 않는다)

즉, 슬라이싱의 본질은:

> “데이터를 복사하지 않고, 같은 데이터에 대한 다른 해석(포인터 + stride + shape)만 만든다.”

#### 예시: (100×100) 행렬에서 (행 10~19, 열 30~39) 뷰 만들기
원본 행렬 `A`가 있다고 하자:
- `A.size1 = 100`, `A.size2 = 100`
- `A.tda = 100` (연속 행렬이라고 치자)
- `A.data == A.block->data`
이때 `B = A[10:20, 30:40]` 같은 슬라이스는 대략 이렇게 표현된다:
- `B.size1 = 10` (행 10개)
- `B.size2 = 10` (열 10개)
- `B.tda = A.tda` (여전히 100)
- `B.data = A.data + 10*A.tda + 30`
- `B.block = A.block`
- `B.owner = 0`

그리고 `B(i, j)`는 결국

$$  
B_{i,j} = B.data[i \cdot B.tda + j]  
$$

이지만, `B.data` 자체가 이미 “원본의 (10,30)에서 시작하도록” 이동해 있으니, 결과적으로 원본을 정확히 참조하게 된다.

#### owner가 중요한 이유
뷰는 원본 메모리를 “빌려 쓰는 것”이라서, 뷰를 free할 때 원본까지 같이 free하면 큰일난다.
그래서:
- 원본(진짜 할당한 객체): `owner = 1`
- 뷰(슬라이스로 만든 객체): `owner = 0`

같은 패턴이 많이 나온다.

## 결론: 슬라이싱 구현은 “(pointer, shape, stride) 조합”
벡터/행렬에서 슬라이싱을 구현하는 핵심은 결국 3가지다.
1. **시작 포인터(data)**
2. **크기(size1, size2 / 또는 size)**
3. **stride(tda 또는 stride)**

이 3개만 바꿔도 “같은 메모리”를 전혀 다른 부분처럼 바라볼 수 있다.  
그래서 슬라이싱은 보통 `O(1)`에 가깝게 구현되고, 복사는 선택사항(명시적 copy)로 분리된다.
