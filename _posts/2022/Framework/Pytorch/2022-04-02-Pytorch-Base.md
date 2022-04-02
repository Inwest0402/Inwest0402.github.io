---
title:  "[Pytorch] 파이토치 기초"
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

# 파이토치 기초

파이토치는 텐서플로우에 비해 조금 더 간편하고, 파이썬에 조금더 가까운 인터페이스로 인해 사용자가 점점 늘어나고 있습니다.

파이토치의 기본 구성요소는 다음과 같습니다.

1. torch
    
    텐서를 생성하는 라이브러리
    
2. torch.autograd
    
    자동미분을 제공하는 라이브러리
    
3. torch.nn
    
    신경망을 생성하는 라이브러리
    
4. torch.multiprocessing
    
    병렬처리 기능을 제공하는 라이브러리
    
5. torch.utils
    
    데이터 조작 등 유틸리티 기능 제공
    
6. torch.onnx
    
    서로 다른 프레임워크(Tensorflow 등등) 간의 모델을 공유할 때 사용 
    
7. torch.optim(/legacy)
    
    토치로부터 포팅해온 코드
    

## 텐서(Tensor)

텐서는 배열 및 행렬과 매우 유사한 특수 데이터 구조입니다. Pytorch에서는 텐서를 사용하여 모델의 입력과 출력, 모델의 매개변수를 인코딩합니다.

### Tensor의 장점:

1. Numpy의 ndarray와는 다르게 텐서는 GPU 또는 기타 하드웨어 가속기에서 실행할 수 있습니다. 
2. 텐서는 동일한 기본 메모리를 공유할 수 있으므로, 데이터를 복사할 필요가 없습니다. 
3. 턴서는 자동 미분에 최적화되어 있습니다. 

### Tensor의 시각화

텐서를 이해하기 앞서, 우선 벡터, 행렬에 대해 간략하게 이해해보겠습니다.

![Untitled.png](/assets/images/posts/Pytorch/2022-04-02-Pytorch-Base/Untitled.png)

위에 스칼라는 사실 차원이 없는 숫자입니다. 하지만, 이해를 쉽게하기 위해 그림에는 0차원이라고 표기했습니다. 벡터는 1차원으로 구성된 값입니다. 행렬은 2차원으로 구성된 값입니다. 저희가 흔히 알고 있는 DataFrame도 2차원이라고 할 수 있죠. 그리고 3차원은 저희가 현재 살고 있는 공간입니다. 그리고 3차원부터는 텐서라고 불립니다.  4차원부터는 사실 저희가 시각적으로 볼 수 없는 공간입니다. 하지만 이해를 돕기 위해 그림에는 6차원까지 표시했습니다. 

그리고 보통 차원을 떠나서 다 텐서라고 표현합니다. 예를 들어 벡터는 1차원 텐서, 행렬은 2차원 텐서 등등.

그럼 이제 딥러닝을 할 때 가장 많이 다루는 2차원 텐서와 3차원 텐서에 대해 알아보겠습니다.

### Pytorch Tensor Shape Convention

1. **2D Tensor(Typical simple Setting)**
    
    Tabular 데이터 분석을 할 때 주로 다루게 되는 데이터는 2차원 텐서입니다
    
    이때 텐서는 배치사이즈(Batch Size)와 차원(Dimension)으로 이루어져있습니다.
    

![Untitled.png](/assets/images/posts/Pytorch/2022-04-02-Pytorch-Base/Untitled1.png)

만약 훈련 데이터 하나의 크기가 32이라면, 길이가 32인 1D 텐서가 있다고 생각하시면 됩니다. 조금 더 쉽게 생각하면, 훈련 데이터 하나의 컬럼값이 32가 들어가 있다고 생각하면 됩니다. 근데 만약 이런 훈련 데이터의 개수가 1000개라면, 총 훈련 데이터의 크기는 1000$\times$32가 됩니다.

물론 모든 훈련 데이터를 하나씩 처리한 것도 가능하지만, 컨퓨터는 훈련 데이터를 하나씩 처리하는 것보다는 보통 덩어리로 처리합니다. 만약 1000개를 16개씩 꺼내서 처리한다면, 이 때 batch size를 16이라고 합니다. 이 때 컴퓨터가 한 번에 처리하는 2D 텐서의 크기는 Batch Size$\times$Dimension = 16$\times$32 = 512입니다.

1. **3D Tensor(Typical Computer Vision)**

3차원 텐서는 주로 비전쪽에서 많이 사용됩니다. 

![Untitled.png](/assets/images/posts/Pytorch/2022-04-02-Pytorch-Base/Untitled2.png)

2D텐서의 차원이 훈련 데이터 셋 이미지의 너비와 길이로 바뀐 것을 제외하면  같습니다.

이제 넘파이 라이브러리와 토치 라이브러리를 활용해 텐서를 만들어보겠습니다.

## 텐서 만들기

### 1 차원 텐서

파이토치로 1차원 텐서인 벡터를 만들어보겠습니다.

```python
array1D = torch.FloatTensor([1,2,3,4,5,6])
print(array1D)
```

[출력] :

```python
tensor([1., 2., 3., 4., 5., 6.])
```

`dim()`,`shape`, `size()`를 통해 텐서의 차원과 크기를 볼 수 있습니다.

```python
print(array1D.dim())
print(array1D.shape)
print(array1D.size())
```

[출력] :

```python
1
torch.Size([6])
torch.Size([6])
```

물론 인덱스로도 접근할 수 있습니다.

```python
print(array1D[0],array1D[-1])
print(array1D[2:-2])
print(array1D[:-3])
```

[출력] :

```python
tensor(1.) tensor(6.)
tensor([3., 4.])
tensor([1., 2., 3.])
```

### 2차원 텐서

```python
array2D = torch.FloatTensor([[1, 4, 7, 10],
                             [2, 5, 8, 11],
                             [3, 6, 9, 12]])

print(array2D)
```

[출력] :

```python
tensor([[ 1.,  4.,  7., 10.],
        [ 2.,  5.,  8., 11.],
        [ 3.,  6.,  9., 12.]])
```

1차원 텐서처럼 `dim()`,`shape`, `size()`를 통해 텐서의 차원과 크기를 볼 수 있습니다.

```python
print(array2D.dim())
print(array2D.shape)
print(array2D.size())
```

[출력] :

```python
2
torch.Size([3, 4])
torch.Size([3, 4])
```

슬라이싱도 1차원 텐서랑 같습니다.

```python
print(array2D[0],array2D[-1])
print(array2D[2:-2])
print(array2D[:-3])
```

[출력] :

```python
tensor([ 1.,  4.,  7., 10.]) tensor([ 3.,  6.,  9., 12.])
tensor([], size=(0, 4))
tensor([], size=(0, 4))
```

### 브로드캐스팅

두 2차원 텐서의 덧셈과 뺄셈을 하기 위해서는 두 2차원 텐서의 크기(Shape)가 같아야 됩니다. 

하지만 파이토치에서는 자동으로 크기를 맞춰서 연산을 수행하게 만드는 브로드캐스팅이라는 기능을 제공합니다. 

```python
matrix1 = torch.FloatTensor([[1, 2, 3]])
matrix2 = torch.FloatTensor([[1, 1, 1]])
print(matrix1 + matrix2)
```

[출력] :

```python
tensor([[2., 3., 4.]])
```

이번에는 크기가 서로 다른 텐서(벡터와 스칼라)를 더해보겠습니다.

```python
matrix1 = torch.FloatTensor([[1, 2, 3]])
matrix2 = torch.FloatTensor([1])
print(matrix1 + matrix2)
```

[출력] :

```python
tensor([[2., 3., 4.]])
```

이는 사실 수학적으로는 연산이 안 되는게 맞지만, 파이토치의 브로드캐스팅으로 통해 연산이 가능해졌습니다.

원래 matrix1의 크기는 (1,3)이며, matrix의 크기는 (1,)이지만, 파이토치는 이를 (1,2)로 변경하여 연산을 수행합니다. 이번에는 벡터 간 연산을 해보겠습니다.

```python
matrix1 = torch.FloatTensor([[1,3]])
matrix2 = torch.FloatTensor([[1],[4]])
print(matrix1 + matrix2)
```

[출력] :

```python
tensor([[2., 4.],
        [5., 7.]])
```

파이토치의 브로드캐스팅으로 통해 기존의 (1,2)와 (2,1) 크기의 두 벡터를 모두 (2,2)로 변경하여 덧셈을 수행했습니다. 

$$
[1,3]\to\begin{matrix} [[1 & 3]\  \\\ \  1 & 3]] \end{matrix}\\
\begin{matrix} \ [1]\  \\ [4] \end{matrix}\to\begin{matrix} [[1 & 1]\  \\\ \ [4 & 4]] \end{matrix}
$$

주의: 브로드캐스팅은 확실히 사용자에게 편리함을 가져다주지만, 만약 우리가 값의 크기를 잘못 입력했을 경우에도 브로드캐스팅으로 인해 연산이 수행될 수 있습니다. 그러면 실제로 사용자는 이 연산이 잘못되었음을 알기 어렵습니다.

### 자주 사용되는 기능들

1. `matmul()`과 `*`
    
    `matmul()`은  행렬곱 기능을 합니다. 행렬곱을 하기 위해서는 첫 번째 텐서의 행의 수와 두 번째 텐서의 열의 크기가 같아야 됩니다.
    
    즉, 첫 번째 텐서가 $m\times n$ 크기이고, 두 번째 텐서가  $n\times r$ 크기인 경우 곱은 $m \times r$ 크기의 행렬이 됩니다.
    
    ```python
    matrix1 = torch.FloatTensor([[1, 2, 3]
    														,[4, 5, 6]
    														,[7, 8, 9]])
    matrix2 = torch.FloatTensor([[1, 2]
    														,[4, 5]
                                ,[6, 7]])
    print('Shape of Matrix 1: ', matrix1.shape) 
    print('Shape of Matrix 2: ', matrix2.shape) 
    print(matrix1.matmul(matrix2))
    ```
    
    [출력] :
    
    ```python
    Shape of Matrix 1:  torch.Size([3, 3])
    Shape of Matrix 2:  torch.Size([3, 2])
    tensor([[ 27.,  33.],
            [ 60.,  75.],
            [ 93., 117.]])
    ```
    
    이번에는 `*` 를 사용해서 두 텐서를 곱해보겠습니다.
    
    ```python
    matrix1 = torch.FloatTensor([[1, 2, 3]
                                ,[4, 5, 6]
    														,[7, 8, 9]])
    matrix2 = torch.FloatTensor([[1]
    														,[4]
                                ,[6]])
    print('Shape of Matrix 1: ', matrix1.shape) 
    print('Shape of Matrix 2: ', matrix2.shape) 
    print(matrix1*matrix2)
    print(matrix1.matmul(matrix2))
    ```
    
    [출력] :
    
    ```python
    Shape of Matrix 1:  torch.Size([3, 3])
    Shape of Matrix 2:  torch.Size([3, 1])
    tensor([[ 1.,  2.,  3.],
            [16., 20., 24.],
            [42., 48., 54.]])
    tensor([[27.],
            [60.],
            [93.]])
    ```
    
    `*` 는 *Element wise product*(또는 *Hadamard product*) 라는 기능을 합니다. 이는 각 텐서의 원소끼리만 곱한다는 것을 의미하는데, 일반 행렬곱은  $m\times n$ 과 $n\times r$ 꼴의 두 행렬을 곱하지만, *Element wise product*은   $m\times n$ 과 $m\times n$ 꼴의 두 행렬을 곱합니다. 
    
    만약 위의 예시처럼 Matrix 1의 크기는 $3\times 3$, Matrix 2의 크기는 $3\times 1$ 인 경우, 즉 서로 다른 크기의 텐서일 떄는 브로드캐스팅이 된 후에 *Element wise product*을 수행합니다.
    
    브로드캐스팅은 다음과 같이 수행됩니다. 
    
    $$
    \begin{matrix} \ [1]\  \\ [4] \\ [6] \end{matrix}\to\begin{matrix} [[1 , 1, 1],\  \\\   [4 , 4 ,4],\\ \  [6, 6 ,6]] \end{matrix}
    $$
    
2. `mean()`
    
    mean은 텐서의 평균을 출력해줍니다. 
    
    ```python
    tensor = torch.FloatTensor([1, 2, 3, 4, 5])
    print(tensor.mean())
    ```
    
    [출력] :
    
    ```python
    tensor(3.)
    ```
    
    1차원 텐서에 `mean()`을 쓰니 텐서 안에 원소의 평균이 나왔습니다. 
    
    이번에는 2차원 텐서의 평균을 구해보겠습니다.
    
    ```python
    tensor = torch.FloatTensor([[1, 2, 3]
                              ,[4, 5, 6]])
    print(tensor.mean())
    ```
    
    [출력] :
    
    ```python
    tensor(3.5000)
    ```
    
    2차원 텐서의 모든 원소의 평균이 구해졌습니다.
    
    이번에는 `dim`이라는 인자를 설정해 평균을 구해보겠습니다.
    
    ```python
    tensor = torch.FloatTensor([[1, 2, 3]
                              ,[4, 5, 6]])
    print(tensor.mean(dim = 0))
    print(tensor.mean(dim = 1))
    ```
    
    [출력] :
    
    ```python
    tensor([2.5000, 3.5000, 4.5000])
    tensor([2., 5.])
    ```
    
    `dim`을 0으로 설정하면 열(*row*)을 기준으로 평균을 구하고, 1로 설정히면 행(*column*)을 기준으로 평균을 구합니다. 
    
    그럼 dim을 -1로 설정을 하면 어떻게 될까요?
    
    ```python
    tensor = torch.FloatTensor([[1, 2, 3]
                              ,[4, 5, 6]])
    print(tensor.mean(dim = -1))
    ```
    
    [출력] :
    
    ```python
    tensor([2., 5.])
    ```
    
    이는 마지막 차원을 제거한다는 의미이고, 결국은 행(*column*)을 기준으로 평균을 구한다는 것과 같습니다.
    
3. `sum()`
    
    덧셈은 평균을 구하는 방법과 인자를 설정하는 방법이 동일합니다. 
    
    ```python
    tensor = torch.FloatTensor([[1, 2, 3]
                              ,[4, 5, 6]])
    
    print(tensor.sum())
    print(tensor.sum(dim = 0))
    print(tensor.sum(dim = 1))
    print(tensor.sum(dim = -1))
    ```
    
    [출력] :
    
    ```python
    tensor(21.)
    tensor([5., 7., 9.])
    tensor([ 6., 15.])
    tensor([ 6., 15.])
    ```
    
    아무런 인자도 주지 않았을 경우는 모든 원소의 합을 구합니다.
    
    그리고 `dim`을 -1로 설정했을 경우 행(*column*)을 기준으로 합을 구합니다. 
    
4. `max()`
    
    max는 원소의 최댓값을 리턴하고, argmax는 최댓값을 가진 인덱스를 반환합니다.
    
    ```python
    tensor = torch.FloatTensor([[1, 2, 3]
                              ,[4, 5, 6]])
    
    print('tensor.max:',tensor.max())
    print('\ntensor.max(dim=0):\n',tensor.max(dim = 0))
    print('\ntensor.max(dim=1):\n',tensor.max(dim = 1))
    print('\ntensor.max(dim=-1):\n',tensor.max(dim = -1))
    ```
    
    [출력] :
    
    ```python
    tensor.max: tensor(6.)
    
    tensor.max(dim=0):
     torch.return_types.max(
    values=tensor([4., 5., 6.]),
    indices=tensor([1, 1, 1]))
    
    tensor.max(dim=1):
     torch.return_types.max(
    values=tensor([3., 6.]),
    indices=tensor([2, 2]))
    
    tensor.max(dim=-1):
     torch.return_types.max(
    values=tensor([3., 6.]),
    indices=tensor([2, 2]))
    ```
    
    아무런 인자도 쓰지 않고 `max()`을 사용하면 텐서 전체의 최댓값을 반환합니다 .
    
    `dim`을 0으로 설정하면 `values`와 `indices`를 반환합니다. 이때 values는 열(*row*)을 기준으로 각 열의 최대값을 반환합니다. 그리고 indices는 열을 기준으로 최댓값이 속한 인덱스 즉 argmax 를 반환합니다.
    
    `dim`을 1과 -1로 설정했을 경우에는`values`와 `indices`이  행(column)을 기준으로 각 행의 최댓값과 argmax 를 반환합니다.
    
    그럼 만약 최댓값 또는 argmax만 반환하고 싶을 때는 어떻게 할까요?
    
    ```python
    tensor = torch.FloatTensor([[1, 2, 3]
                              ,[4, 5, 6]])
    
    print('tensor.max:',tensor.max())
    print('\ntensor.max(dim=0):\n',tensor.max(dim = 0)[0])
    print('\ntensor.max(dim=0):\n',tensor.max(dim = 0)[1])
    print('\ntensor.max(dim=1):\n',tensor.max(dim = 1)[0])
    print('\ntensor.max(dim=1):\n',tensor.max(dim = 1)[1])
    ```
    
    [출력] :
    
    ```python
    tensor.max: tensor(6.)
    
    tensor.max(dim=0):
     tensor([4., 5., 6.])
    
    tensor.max(dim=0):
     tensor([1, 1, 1])
    
    tensor.max(dim=1):
     tensor([3., 6.])
    
    tensor.max(dim=1):
     tensor([2, 2])
    ```
    
    이처럼 반환 값에 인덱스를 부여하면 됩니다. 0번 인덱스를 부여하면 최대값만 받아올 수 있고, 1번 인덱스를 부여하면 argmax만 받아올 수 있습니다.  
    
5. `view()`
    
    파이토치 텐서의 뷰(view)는 넘파이의 `reshape`와 같은 역할을 합니다. 
    
    일단 2차원 텐서를 만듭니다.
    
    ```python
    tensor = torch.FloatTensor([[1, 2, 3]
                              ,[4, 5, 6]])
    
    print(tensor .shape)
    ```
    
    [출력] :
    
    ```python
    torch.Size([2, 3])
    ```
    
    현재 크기는 $2\times3$ 입니다.
    
    이번에는 이 텐서를 `view`를 사용해서 크기를 3차원으로 변경해보겠습니다.
    
    ```python
    tensor = torch.FloatTensor([[1, 2, 3]
                              ,[4, 5, 6]])
    print(tensor.view([-1, 2, 1])) 
    print(tensor.view([-1, 2, 1]).shape)
    ```
    
    [출력] :
    
    ```python
    tensor([[[1.],
             [2.]],
    
            [[3.],
             [4.]],
    
            [[5.],
             [6.]]])
    torch.Size([3, 2, 1])
    ```
    
    여기서 `view([-1, 2, 1])`의  -1은 첫 번째 차원은 사용자가 잘 모르겠으니 파이토치에 맡기겠다는 의미이고, 2는 두 번째 차원의 길이를 2로, 1을 세 번째 차원의 길이를 1로 하겠다는 뜻입니다.
    
    즉, 2차원의 텐서를 3차원으로 변경하되 나머지 두 차원은 지정해줄테니, 첫 번째 차원만 그에 맞게 채워달라는 뜻입니다. 
    
    `view`를 사용할 때는 다음과 같은 규칙이 있습니다. 1) view의 변환 전과 후의 텐서 원소 개수는 같아야 됩니다. 2) 사이즈에 -1를 설정하면 다른차원으로부터 해당 차원의 길이를 유추합니다.  
    
6. `squeeze()`
    
    스퀴즈는 1인 차원을 제거합니다. 그리고 차원이 1인 경우에는 해당 차원을 제거합니다.
    
    ```python
    matrix = torch.FloatTensor([[1]
                               ,[2] 
                               ,[3]])
    print(matrix.shape)
    print('squeeze후 matrix:',matrix.squeeze())
    print('squeeze후 크기:',matrix.squeeze().shape)
    ```
    
    [출력] :
    
    ```python
    torch.Size([3, 1])
    squeeze후 matrix: tensor([1., 2., 3.])
    squeeze후 크기: torch.Size([3])
    ```
    
    $3\times 1$ 크기였던 텐서는 `squeeze`로 변환 후 두 번째 차원의 길이가 1이기 때문에 3의 크기를 가진 텐서로 변경되었습니다.
    
7. `unsqueeze()`
    
    언스퀴즈는 스퀴즈와 반대로 특정 위치에 1인 차원을 추가할 수 있습니다. 
    
    ```python
    matrix = torch.FloatTensor([[1,2,3]
                               ,[4,5,6]])
    print(matrix.shape)
    print('\n첫 번째 차원에 unsqueeze후 matrix:\n',matrix.unsqueeze(0))
    print('\n첫 번째 차원에 unsqueeze후 크기:\n',matrix.unsqueeze(0).shape)
    print('\n두 번째 차원에 unsqueeze후 matrix:\n',matrix.unsqueeze(1))
    print('\n두 번째 차원에 unsqueeze후 크기:\n',matrix.unsqueeze(1).shape)
    print('\n마지막 차원에 unsqueeze후 matrix:\n',matrix.unsqueeze(-1))
    print('\n마지막 차원에 unsqueeze후 크기:\n',matrix.unsqueeze(-1).shape)
    ```
    
    [출력] :
    
    ```python
    torch.Size([2, 3])
    
    첫 번째 차원에 unsqueeze후 matrix:
     tensor([[[1., 2., 3.],
             [4., 5., 6.]]])
    
    첫 번째 차원에 unsqueeze후 크기:
     torch.Size([1, 2, 3])
    
    두 번째 차원에 unsqueeze후 matrix:
     tensor([[[1., 2., 3.]],
    
            [[4., 5., 6.]]])
    
    두 번째 차원에 unsqueeze후 크기:
     torch.Size([2, 1, 3])
    
    마지막 차원에 unsqueeze후 matrix:
     tensor([[[1.],
             [2.],
             [3.]],
    
            [[4.],
             [5.],
             [6.]]])
    
    마지막 차원에 unsqueeze후 크기:
     torch.Size([2, 3, 1])
    ```
    
    unsqueeze할 때, 숫자 0을 인자로 넣으면 첫 번째 차원에 1인 차원이 추가되고, 숫자 1을 인자로 넣으면 두 번째 차원에 1인 차원이 추가됩니다. 그리고 -1을 인자로 넣으면 마지막 차원에 1인 차원이 추가됩니다.
    
     `view()`, `squeeze()`, `unsqueeze()`는 텐서의 원소 수를 그대로 유지하면서 모양과 차원을 조절합니다. 
    
8. 타입 캐스팅 
    
    텐서에는 자료형이 있습니다. 
    
    우선 타입의 `LongTensor()`을 통해 텐서를 만들고 ,여기에 `.float()`로 형변환을 해주겠습니다.
    
    ```python
    long_tensor = torch.LongTensor([1, 2, 3, 4])
    print(long_tensor)
    print(long_tensor.float())
    ```
    
    [출력] :
    
    ```python
    tensor([1, 2, 3, 4])
    tensor([1., 2., 3., 4.])
    ```
    
    우선, `LongTensor`로 텐서를 생성하면, *int* 타입 텐서가 선언됩니다.그리고 `.float()`로 *float* 타입으로 변경됩니다. 
    
    이번에는 Byte 타입의 텐서를 만들고, `.long()`,과 `.float()`로 형변환을 해주겠습니다.
    
    ```python
    byte_tensor = torch.ByteTensor([True,True, True, False])
    print(byte_tensor)
    print(byte_tensor.long())
    print(byte_tensor.float())
    ```
    
    [출력] :
    
    ```python
    tensor([1, 1, 1, 0], dtype=torch.uint8)
    tensor([1, 1, 1, 0])
    tensor([1., 1., 1., 0.])
    ```
    
9. `cat()`
    
    이번에는 두 텐서를 연결(*Concatenate*)할 수 있는 방법에 대해 알아보겠습니다.
    
    우선 $3\times 1$의 텐서 두 개를 만들어줍니다. 그리고 여기에 `cat()`으로 두 텐서를 연결시켜줍니다.
    
    ```python
    matrix1 = torch.FloatTensor([[1]
                               ,[2] 
                               ,[3]])
    matrix2 = torch.FloatTensor([[4]
                               ,[5] 
                               ,[6]])
    print(torch.cat([matrix1, matrix2]))
    print(torch.cat([matrix1, matrix2], dim = 0))
    print(torch.cat([matrix1, matrix2], dim = 1))
    ```
    
    [출력] :
    
    ```python
    tensor([[1.],
            [2.],
            [3.],
            [4.],
            [5.],
            [6.]])
    tensor([[1.],
            [2.],
            [3.],
            [4.],
            [5.],
            [6.]])
    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    ```
    
    여기에 `dim`인자를 설정할 수 있습니다. 만약 `dim`을 설정 안하면 디폴트로 0이 설정됩니다.
    
    위에서 본 것처럼, `dim`을 0으로 설정하면 열(*row*)을 기준으로 연결하고, 1로 설정히면 행(*column*)을 기준으로 연결합니다. 
    
10. `stack()`
    
    텐서를 연결할 때는 *Concatenate*외에 *Stacking*하는 방법도 있습니다.
    
    ```python
    matrix1 = torch.FloatTensor([1,2,3,4])
    matrix2 = torch.FloatTensor([5,6,7,8])
    matrix3 = torch.FloatTensor([9,10,11,12])
    print(torch.stack([matrix1, matrix2, matrix3]))
    print(torch.stack([matrix1, matrix2, matrix3]).shape)
    print(torch.stack([matrix1, matrix2, matrix3], dim = 0))
    print(torch.stack([matrix1, matrix2, matrix3], dim = 0).shape)
    print(torch.stack([matrix1, matrix2, matrix3], dim = 1))
    print(torch.stack([matrix1, matrix2, matrix3], dim = 1).shape)
    ```
    
    [출력] :
    
    ```python
    tensor([[ 1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.]])
    torch.Size([3, 4])
    tensor([[ 1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.]])
    torch.Size([3, 4])
    tensor([[ 1.,  5.,  9.],
            [ 2.,  6., 10.],
            [ 3.,  7., 11.],
            [ 4.,  8., 12.]])
    torch.Size([4, 3])
    ```
    
    `dim`의 설정은 디폴트로 0이며, 0일 경우에는 첫 번째 차원이 증가하도록 쌓이는 것으로 해석하면 되고, 1일 경우 두 번째 차원이 증가하도록 쌓이는 것으로 해석하면 됩니다.
    
    stacking의 장점은 많은 연산을 한 번에 축약할 수 있다는 것입니다. 
    
    비교를 위해 동일한 텐서들을 *Concatenate* 방법으로 연결해보겠습니다.
    
    ```python
    matrix1 = torch.FloatTensor([1,2,3,4])
    matrix2 = torch.FloatTensor([5,6,7,8])
    matrix3 = torch.FloatTensor([9,10,11,12])
    print(torch.cat([matrix1.unsqueeze(0), matrix2.unsqueeze(0), matrix3.unsqueeze(0)]))
    
    ```
    
    [출력] :
    
    ```python
    tensor([[ 1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.]])
    ```
    
    `torch.stack([matrix1, matrix2, matrix3])`과 동일한 텐서로 연결하기 위해서는 각 텐서에 `unsqueeze(0)`를 추가해야 됩니다.
    
11. `ones_like()`, `zeros_like()`
    
    `ones_like()`을 통해 텐서의 모든 원소를 1로 변경할 수 있고, `zeros_like()`을 통해 텐서의 모든 원소를 0으로 변경할 수 있습니다.
    
    ```python
    matrix = torch.FloatTensor([[1, 2, 3]
                                ,[4, 5, 6]])
    print(matrix)
    print(torch.ones_like(matrix))
    print(torch.zeros_like(matrix))
    ```
    
    [출력] :
    
    ```python
    tensor([[1., 2., 3.],
            [4., 5., 6.]])
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    ```
    
12. `_연산()`
    
    텐서의 연산뒤에 `_`를 붙이면 기존의 값이 덮어씌어집니다.
    
    ```python
    matrix = torch.FloatTensor([[1, 2, 3]
                                ,[4, 5, 6]])
    print(matrix)
    print(matrix.mul_(2.)) 
    print(matrix)
    ```
    
    [출력] :
    
    ```python
    tensor([[1., 2., 3.],
            [4., 5., 6.]])
    tensor([[ 2.,  4.,  6.],
            [ 8., 10., 12.]])
    tensor([[ 2.,  4.,  6.],
            [ 8., 10., 12.]])
    ```
    

--------------------------

본 포스팅은 유원준님의 "**PyTorch로 시작하는 딥 러닝 입문**" 을 참고했습니다.

[점프 투 파이썬](https://wikidocs.net/book/2788)