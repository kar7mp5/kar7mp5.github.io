---
layout: default
title: "[C++] Smart Pointer (auto_ptr, unique_ptr, shared_ptr, week_ptr)"
date: 2025-08-08 09:00:00 +0900
categories: development
permalink: /20250808/smart-pointer.html
---

# [C++] Smart Pointer (auto_ptr, unique_ptr, shared_ptr, week_ptr)

> **참고자료**  
> [C++ 공식문서](https://cplusplus.com/reference/memory/)  
> [GeeksforGeeks](https://www.geeksforgeeks.org/cpp/smart-pointers-cpp/)

## 일반 포인터 문제점

-   **메모리 누수 (Memory Leaks):** 프로그램이 메모리를 반복적으로 할당 후, 해제하지 않을 때 발생한다.
    과도한 메모리 사용으로 시스템 충돌이 발생할 수 있다.
-   **와일드 포인터 (Wild Pointers):** 유효한 객체나 주소로 초기화하지 않은 포인터를 의미한다.
-   **댕글링 포인터 (Dangling Pointers):** 이전에 할당 해제된 메모리 공간을 가리키고 있는 포인터를 의미한다.

**일반 포인터 vs 스마트 포인터**

|   특징    |                포인터 (Pointer)                |         스마트 포인터 (Smart Pointer)          |
| :-------: | :--------------------------------------------: | :--------------------------------------------: |
| 소멸 시점 |    범위를 벗어나도 자동으로 소멸되지 않음.     |        범위를 벗어나면 자동으로 소멸함.        |
|  효율성   | 다른 추가 기능이 없어 효율성이 떨어질 수 있음. |  메모리 관리 기능이 추가되어 있어 더 효율적.   |
| 관리방식  |       매우 수동적으로 직접 관리해야 함.        | 자동적이고 미리 프로그래밍 된 방식으로 동작함. |

---

## 스마트 포인터 종류

-   [`auto_ptr` (deprecated)](https://cplusplus.com/reference/memory/auto_ptr/)
-   `unique_ptr`
-   `shared_ptr`
-   `week_ptr`

---

## 1. `auto_ptr`

이름 그대로 다이나믹하게 할당하고, 스코프를 벗어나면 자동 할당 해제가 된다.

```cpp
auto_ptr <type> name;
```

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {

    // Pointer declaration
    auto_ptr<int> ptr1(new int(10));
    cout << *ptr1 << endl;

    // Transfer ownership to
    // pointer ptr2,
    auto_ptr<int> ptr2 = ptr1;
    cout << *ptr2;
    return 0;
}
```

```
10
10
```

---

## 2. `unique_ptr`

`unique_ptr`은 하나의 포인터가 한번만 사용된다. 그래서 `unique_ptr`를 복사할 수 없고, `move()` 메소드로 권한 이동이 가능하다.

```cpp
#include <iostream>
#include <memory>
using namespace std;

class Rectangle {
    int length;
    int breadth;

public:
    Rectangle(int l, int b) {
        length = l;
        breadth = b;
    }
    int area() { return length * breadth; }
};

int main() {

    unique_ptr<Rectangle> P1(new Rectangle(10, 5));
    cout << P1->area() << endl;

    unique_ptr<Rectangle> P2;

    // Copy the addres of P1 into p2
    P2 = move(P1);
    cout << P2->area();
    return 0;
}
```

```
50
50
```

위 예제에서 `P1`을 `P2`로 `move()`를 하였기 때문에, `P1` 값은 할당해제가 되고 삭제된다.

---

## 3. `shared_ptr`

`shared_ptr`는 같은 object에 여러 포인터를 할당할 수 있다. 참조한 개수는 `use_count()` 메소드를 통하여, 확인 가능하다.

```cpp
#include <iostream>
#include <memory>
using namespace std;

class Rectangle {
    int length;
    int breadth;

public:
    Rectangle(int l, int b) {
        length = l;
        breadth = b;
    }
    int area() { return length * breadth; }
};

int main() {

    shared_ptr<Rectangle> P1(new Rectangle(10, 5));
    cout << P1->area() << endl;

    shared_ptr<Rectangle> P2;

    // P1 and P2 are pointing
    // to same object
    P2 = P1;
    cout << P2->area() << endl;
    cout << P1->area() << endl;
    cout << P1.use_count();
    return 0;
}
```

```
50
50
50
2
```

위 예제를 보면, 한 object에 대해 여러 값을 할당할 수 있다. 설계를 잘못할 경우, 순환 참조가 될 수 있다.

---

## 4. `weak_ptr`

`weak_ptr`은 object에 대해서 소유권을 가지지 않고 참조한다. `shared_ptr`과 유사하지만 강한 참조를 하지 않는다. 위에서 언급되었듯, 순환참조를 방지하기 위해서 이다.

```cpp
#include <iostream>
#include <memory>
using namespace std;

class Rectangle {
    int length;
    int breadth;

public:
    Rectangle(int l, int b) {
        length = l;
        breadth = b;
    }

    int area() { return length * breadth; }
};

int main() {

    // Create shared_ptr Smart Pointer
    shared_ptr<Rectangle> P1(new Rectangle(10, 5));

    // Created a weak_ptr smart pointer
    weak_ptr<Rectangle> P2 (P1);
    cout << P1->area() << endl;

    // Returns the number of shared_ptr
    // objects that manage the object
    cout << P2.use_count();
    return 0;
}
```

```
50
1
```
