---
title:  "[Pytorch] 파이토치 텐서"
excerpt: "파이토치는 텐서플로우에 비해 조금 더 간편하고, 파이썬에 조금더 가까운 인터페이스로 인해 사용자가 점점 늘어나고 있습니다."

categories:
  - Pytorch
# tags:
#   - [Machine Learning, XAI,Surrogate Analysis ]

toc: true
toc_sticky: true
use_math: true

# date: 2022-04-02
# last_modified_at: 2022-03-29
---
# Pytorch - Tensor

# 1. 텐서의 특징

Tensor는 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료 구조입니다.

모델의 입력(input)과 출력(output), 그리고 매개변수들을 부호화(encode)합니다.

Numpy의 ndarray와 다른 점 - Pytorch의 텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있습니다.

Pytorch의 텐서, Numpy의 배열(ndarray)은 종종 동일한 내부(underly)메모리를 공유할 수 있어 데이터를 복사할 필요가 없습니다. 

텐서는 자동 미분(automatic differentiation)에 최적화되어 있습니다.

## torch.Tensor

torch.Tensor는 데이터를 Tensor 객체로 만들어주는 함수입니다.

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

# 2. 텐서의 초기화

텐서는 여러가지 방법으로 초기화할 수 있습니다

## 데이터로부터 직접(directly)생성하기

데이터로부터 직접 텐서를 생성할 수 있습니다.

데이터의 자료형(data type)은 자동으로 유추합니다

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

## NumPy 배열로부터 생성하기

텐서는 Numpy 배열로 생성할 수 있습니다

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_np
# tensor([[1, 2],
#         [3, 4]])
```

## 다른 텐서로부터 생성하기

명시적으로 재정의 (override)하지 않으면, 인자로 주어진 텐서의 속성(모양(shape), 자료형(datatype))을 유지합니다.

```python
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_zeros = torch.zeros_like(x_data)
print(f"Ones Tensor: \n {x_zeros} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")

# Ones Tensor: 
#  tensor([[1, 1],
#         [1, 1]]) 
# Zeros Tensor: 
#  tensor([[0, 0],
#         [0, 0]]) 
# Random Tensor: 
#  tensor([[0.8494, 0.6131],
#         [0.7235, 0.3144]])

x_ones.dtype
# torch.int64
x_zeros.dtype
# torch.int64
x_rand.dtype
# torch.float32
```

## 무작위(ramdom)또는 상수(constant)값 사용하기

shape는  텐서의 차원(dimension)을 나타내는 튜플(tuple)로, 아래 함수들에서는 출력 텐서의 차원을 결정합니다

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Random Tensor: 
#  tensor([[0.1509, 0.2249, 0.7626],
#         [0.3123, 0.8352, 0.9012]]) 
# Ones Tensor: 
#  tensor([[1., 1., 1.],
#         [1., 1., 1.]]) 
# Zeros Tensor: 
#  tensor([[0., 0., 0.],
#         [0., 0., 0.]])
```

 

# 3. 텐서의 속성(Attribute)

- `.data`: list나 Numpy의 ndarray 등 배열 데이터
    
    ```python
    x_data.data
    # tensor([[1, 2],
    #         [3, 4]])
    ```
    
- `.dtype`: 데이터의 타입(선언하지 않으면 보통 data에 맞춰서 적절하게 들어감)
    
    ```python
    x_data.dtype
    # torch.int64
    ```
    
- `.device`: default는 None이나 torch.set_default_tensor_type()에 맞게 들어감
    
    ```python
    x_data.device
    # device(type='cpu')
    
    # GPU가 존재하면 텐서를 이동합니다
    if torch.cuda.is_available():
      tensor = x_data.to('cuda')
      print(f"Device tensor is stored on: {x_data.device}")
    # Device tensor is stored on: cpu
    ```
    
- `.requires_grad`: default는 False이며, gradient 값 저장 유무
    
    ```python
    x_data.requires_grad
    # False
    ```
    
- `.pin_memory` : True시 pinned memory에 할당, CPU tensor에서 가능
    - 데이터를 CPU로 읽어들인 다음 GPU로 보내기 위해서는 GPU와 통신하기 위한 CPU의 메모리 공간이 필요하다. 이 때, 메모리를 할당시키는 기법을 memory pinning이라고 합니다.
    
    ```python
    x_data.pin_memory
    # <function Tensor.pin_memory>
    ```
    

# 4. 텐서 연산(Operation)

## NumPy식의 표준 인덱싱과 슬라이싱

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# First row: tensor([1., 1., 1., 1.])
# First column: tensor([1., 1., 1., 1.])
# Last column: tensor([1., 1., 1., 1.])
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])
```

## 텐서 합치기

`torch.cat()`을 사용하여 주어진 차원에 따라 일련의 텐서를 연결할 수 있습니다. 

```python
tensor1 = torch.ones(4, 4)
tensor2 = torch.zeros(4, 4)

t0 = torch.cat([tensor1, tensor2], dim=0)
print(t0)

# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])

# torch.Size([8, 4])

t1 = torch.cat([tensor1, tensor2], dim=1)
print(t1)

# tensor([[1., 1., 1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0., 0., 0.]])

# torch.Size([4, 8])

```

`dim=0`또는 설정 안함 : 1 번째 차원 또는 열(*row*)를 기준으로  연결

$$
cat\begin{pmatrix}\begin{bmatrix} 1 \ 1\ 1\ 1\\ 1 \ 1\ 1\ 1\\ 1 \ 1\ 1\ 1\\1 \ 1\ 1\ 1 \end{bmatrix}\\
\begin{bmatrix} 0 \ 0\ 0\ 0 \\ 0 \ 0\ 0\ 0 \\ 0 \ 0\ 0\ 0 \\0 \ 0\ 0\ 0  \end{bmatrix}\end{pmatrix}\to\begin{bmatrix}1 \ 1\ 1\ 1\\ 1 \ 1\ 1\ 1\\ 1 \ 1\ 1\ 1\\1 \ 1\ 1\ 1\\0 \ 0\ 0\ 0 \\ 0 \ 0\ 0\ 0 \\ 0 \ 0\ 0\ 0 \\0 \ 0\ 0\ 0\end{bmatrix}
$$

`dim`을 1로 설정히면 두 번째 차원 또는 행(*column*)을 기준으로 연결합니다. 

`dim=1` : 2 번째 차원 또는 행(*column*)을 기준으로 연결

$$
cat
\begin{pmatrix}
\begin{bmatrix} 1 \ 1\ 1\ 1\\ 1 \ 1\ 1\ 1\\ 1 \ 1\ 1\ 1\\1 \ 1\ 1\ 1\end{bmatrix}

\begin{bmatrix} 0 \ 0\ 0\ 0 \\ 0 \ 0\ 0\ 0 \\ 0 \ 0\ 0\ 0 \\0 \ 0\ 0\ 0  \end{bmatrix}
\end{pmatrix}
\to
\begin{bmatrix} 1\ 1\ 1\ 1 \ 0 \ 0\ 0\ 0 \\ 1\ 1\ 1\ 1 \ 0 \ 0\ 0\ 0 \\1\ 1\ 1\ 1 \  0 \ 0\ 0\ 0 \\1\ 1\ 1\ 1 \ 0 \ 0\ 0\ 0   \end{bmatrix}

$$

`dim=n`: *n-1* 번째 차원을 기준으로 연결

`torch.stack()`을 사용해 텐서를 연결하는 방법도 있습니다.

```python
matrix1 = torch.FloatTensor([1,2,3,4])
matrix2 = torch.FloatTensor([5,6,7,8])
matrix3 = torch.FloatTensor([9,10,11,12])
s1 = torch.stack([matrix1, matrix2, matrix3], dim = 0)
# tensor([[ 1.,  2.,  3.,  4.],
#         [ 5.,  6.,  7.,  8.],
#         [ 9., 10., 11., 12.]])
# torch.Size([3, 4])

s2 = torch.stack([matrix1, matrix2, matrix3], dim = 1)
# tensor([[ 1.,  5.,  9.],
#         [ 2.,  6., 10.],
#         [ 3.,  7., 11.],
#         [ 4.,  8., 12.]])
# torch.Size([4, 3])
```

`dim=0`또는 설정 안함 : 1 번째 차원 또는 열(*row*)를 기준으로  연결

$$
stack
\begin{pmatrix}
\begin{bmatrix} 1\ \ \  2\ \ \ 3\ \ \  4 \end{bmatrix}\\
\begin{bmatrix} 5\ \ \  6\ \ \ 7\ \ \ 8 \end{bmatrix}\\
\begin{bmatrix} 9\  10\ 11\ 12 \end{bmatrix}
\end{pmatrix}\to\begin{bmatrix}
1\ \ \  2\ \ \ 3\ \ \  4 \\
5\ \ \  6\ \ \ 7\ \ \  8 \\
9\  10\ 11\ 12 
\end{bmatrix}
$$

`dim=1` : 2 번째 차원 또는 행(*column*)을 기준으로 연결

$$
stack
\begin{pmatrix}
\begin{bmatrix} 1\\2\\ 3\\4 \end{bmatrix}
\begin{bmatrix} 5\\6\\ 7\\8 \end{bmatrix}
\begin{bmatrix} 9\\10\\ 11\\12 \end{bmatrix}
\end{pmatrix}\to\begin{bmatrix}
1 \ \ 5 \ \ \ 9 \\
\ 2 \ \ 6 \ \ 10 \\
\ 3 \ \ 7 \ \ 11 \\
\ 4 \ \ 8 \ \ 12
\end{bmatrix}
$$

`dim=n` : *n-1* 번째 차원을 기준으로 연결

## 산술 연산(Arithmetic operations)

```python
tensor = torch.ones(4, 4)

# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T)
# tensor([[4., 4., 4., 4.],
#         [4., 4., 4., 4.],
#         [4., 4., 4., 4.],
#         [4., 4., 4., 4.]])

# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
```

단일-요소(single-element) 텐서 텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 텐서의 경우, `item()` 을 사용하여 Python 숫자 값으로 변환할 수 있음

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
# 12.0 <class 'float'>
```

바꿔치기(in-place) 연산 연산 결과를 피연산자(operand)에 저장하는 연산을 바꿔치기 연산이라고 부르며, `_` 접미사를 갖습니다. 예를 들어: `x.copy_(y)` 나 `x.t_()` 는 `x` 를 변경함

```python
print(f"{tensor} \n")
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

tensor.add_(5)
print(tensor)
# tensor([[6., 5., 6., 6.],
#         [6., 5., 6., 6.],
#         [6., 5., 6., 6.],
#         [6., 5., 6., 6.]])
```

## NumPy의 변환 (Bridge)

CPU상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에, 텐서를 NumPy로 변환(Bridge) 할 수 있습니다.

즉, 하나를 변경하면 다른 하나도 변경됩니다.

### Tensor → Numpy 배열

```python
print('numpy:', x_np.numpy())
# numpy: [[1 2]
#         [3 4]]
print('numpy:', x_np.numpy().dtype)
# numpy: int64
```

```python
t = torch.tensor([[1, 2],[3, 4]])
n = t.numpy()
print(f"t(before add 1) : {t}")
print(f"n(before add 1) : {n}")
t.add_(1)
print(f"t(after add 1) : {t}")
print(f"n(after add 1) : {n}")

# t(before add 1) : tensor([[1, 2],
#                          [3, 4]])
# n(before add 1) : [[1 2]
#                    [3 4]]
# t(after add 1) : tensor([[2, 3],
#                          [4, 5]])
# n(after add 1) : [[2 3]
#                   [4 5]]
```

### NumPy 배열 → Tensor

```python
n = np.array([[1, 2],[3, 4]])
t = torch.from_numpy(n)
print(f"t(before add 1) : {t}")
print(f"n(before add 1) : {n}")
np.add(n, 1, out=n)
print(f"t(after add 1) : {t}")
print(f"n(after add 1) : {n}")

# n(before add 1) : [[1 2]
#                    [3 4]]
# t(before add 1) : tensor([[1, 2],
#                           [3, 4]])
# n(after add 1) : [[2 3]
#                   [4 5]]
# t(after add 1) : tensor([[2, 3],
#                          [4, 5]])
```