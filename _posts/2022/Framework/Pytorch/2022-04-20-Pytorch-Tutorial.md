---
title:  "[Pytorch] 파이토치 튜토리얼"
excerpt: "파이토치로 간략한 딥러닝 모델 구축하기"

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

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

# **데이터 작업하기**

## Dataset과 DataLoader

`Dataset` → 샘플과 정답(*label*) 저장함

`DataLoader` → `Dataset`이 샘플에 쉽게 접근할 수 있도록 순회가능한(*iterable*) 객체로 감쌈

Pytorch의 `Dataset`과 `DataLoader`의 장점

1. 방대한 데이터를 미니배치 단위로 처리
2. 데이터를 무작위로 섞음으로써 학습의 효율성 향상

## Dataset

Dataset 클래스는 데이터 셋의 특징(*feature*)을 가져오고 하나의 샘플에 정답(*label*)을 지정하는 일을 한 번에 처리할 수 있습니다. 

pytorch는 도메인에 특화된 데이터 셋 제공 :`TorchText`, `TorchVision`, `TorchAudio`등등

```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data", # 학습/테스트 데이터가 저장되는 경로입니다.
    train=True, # True를 지정하면 훈련 데이터로 다운로드 
    download=True, # True일 경우 현재 root 위치를 기준으로 해당 데이터 셋 유무를 확인 후, 
									 # 자동으로 저장합니다.
    transform=ToTensor(), # 다양한 이미지 변환을 제공,ToTensor()는 이미지를 텐서로 변경
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # False를 지정하면 테스트 데이터로 다운로드
    download=True,
    transform=ToTensor(),
)
```

## DataLoader

파이토치의 `DataLoader`는 배치 관리를 담당

- `Dataset`을 `batch`기반의 딥러닝 모델 학습을 위해서 미니배치 형태로 변환,  전체 데이터가 `batch size`로 *slice*되어 공급
    
    → 즉, 우리가 실제로 학습할 때 이용할 수 있게 형태로 변환
    
- `Dataset`을 `Input`으로 넣어주면 여러 옵션(데이터 묶기, 섞기, 알아서 병렬처리)을 통해 `batch`생성
- `DataLoader`는 `Iterator`형식으로 데이터에 접근하도록 하며 `batch_size`나 `shuffle`유무를 설정할 수 있음

모델을 학습할 때, 일반적으로 샘플들을 “미니배치(*minibatch*)”로 전달하고, 매 에폭(*epoch*)마다 데이터를 다시 섞어서 과적합(*overfit*)을 막는다.

```python
batch_size = 64

# 데이터 로딩(공급) 객체 선언
train_dataloader = DataLoader(
															dataset    = training_data, 
															batch_size = batch_size,
															shuffle    = True
															)
test_dataloader = DataLoader(
															dataset    = test_data, 
															batch_size = batch_size,
															shuffle    = True
															)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

[Out]:

```
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

```python
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
```

[Out]:

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
```

# 모델 만들기

파이토치의 모델(nn.linear와 같은 내장된 모델도 포함)을 쓰기 위한 조건:

1. `torch.nn.Module`상속
    
    파이토치의 nn 라이브러리는 *Neural Network*의 모든 것을 포괄하는 모든 신경망 모델의 Base Class이다. 
    
    → 다른 말로, 모든 신경망 모델은 `nn.Module`의 *subclass*라고 할 수 있다.
    
2. `__init__()`과 `forward()` ***override***
    
    `__init__`  →신경망의 계층(*layer*)들을 정의
    
    `forward`    → 신경망에 데이터를 어떻게 전달할지, 즉 모델에서 실행되어야하는 계산을 정의
    
    예)  모델에 사용될 모듈(`nn.Linear`, `nn.Conv2d`), 활성함수(`nn.functional.relu`, `nn.functional.sigmoid`)등을 정의
    

문제) 왜 `backward()`는 정의하지 않았는가?

`nn.Module`은 자동미분을 지원하기 때문에 역전파가 어떻게 되는지 구현할 필요가 없다.

```python
# 가능한 경우 GPU로 신경망을 이동시켜 연산을 가속(*accelerate*)합니다.
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

[Out]:

```
Using cpu device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

문제) 여기서 상속 받을 때, `super().__init__()`은 왜 해주는 것일까?

`super()`로 기반 클래스(부모 클래스)를 초기화해줌으로써, 기반 클래스의 속성을 *subclass*가 받아오도록 한다. (초기화를 하지 않으면, 부모 클래스의 속성을 사용할 수 없음)

문제) 그럼 `super(NeuralNetwork, self).__init__()` 는?

위의 코드에 `NeuralNetwork`를 argument로 전달하지 않을 경우, `self.flatten`과 `self.linear_relu_stack`을 실행하면, 내부적으로 선언한 클래스의 `__setattr__`함수를 실행하게 된다(`__setattr__`는 `nn.Module`을 *extend* 하면서 상속받는 것 중에 하나).

```python
def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
    # [...]
    modules = self.__dict__.get('_modules')
    if isinstance(value, Module):
        if modules is None:
            raise AttributeError("cannot assign module before Module.__init__() call")
        remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
        modules[name] = value
```

위의 `__setattr__`함수의 정의에 따르면, `nn.Flatten과nn.Sequential`가 `nn.Module`의 인스턴스인 경우에는 클래스가 _modules 속성을 가지고 있지 않으면 AttributeError를 raise하도록 되어 있다.

즉, `_modules`가 초기화되어 있지 않으면 `AttributeError`가 발생한다.

`nn.Module`의 `__init__` ****함수를 보면, `self._modules`는 `nn.Module`의 `__init__`에서 선언 및 초기화됨을 알 수 있다.

# **모델 매개변수 최적화하기**

### Loss 클래스 초기화

### optimizer 클래스 초기화

`torch.optim`을 만들 때, 제일 중요한 매개변수는 신경망의 *parameter*이다. 

*Variable* 타입의 *parameter*들을 *iterable* 오브젝트로 넣어줘야한다. 

그 외에는 각 *optimizer* 타입에 따라 *learning rate*, *weight decay* 등을 넣어주면 된다.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

## Training Loop

## optimizer의 사용

옵티마이저 세팅 과정은 다음과 같습니다:

1. 옵티마이저 선택
2. 파라미터와 lr 설정
3. zero_grad()로 optimizer에 연결된 parameter들의 gradient를 0으로 설정
4. step()으로 optimizer는 argument로 전달받은 parameter를 업데이트  

`.backward()` 

파이토치의 모든 gradient를 계산할 수 있게 해주는 메서드.

## loss의 사용

역전파에서 *gradient*를 계산하는 첫 포인트가 되는 값이 *loss* 값이고, *loss*값으로 모든 가중치들(*parameters*; *weight and bias*)에 대해 미분을 계산한다.

그렇기 때문에 `.backward()` 메소드는 *gradient* 계산이 시작되는 지점인 *loss* 변수에 적용해주어야한다.

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
				# 이전 step에서 쌓아놓은 파라미터들의 변화량을 0으로 초기화하여
        optimizer.zero_grad()
				# 역전파 단계(backward pass), 파라미터들의 에러에 대한 변화도를 계산하여 누적함
        loss.backward()
				# optimizer에게 loss function를 효율적으로 최소화 할 수 있게 파라미터 수정 위탁
        optimizer.step()

        if batch % 100 == 0:
						# loss.item() 으로 손실이 갖고 있는 스칼라 값을 가져올 수 있음
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

각 학습 단계(training loop)에서 모델은 (배치(batch)로 제공되는) 학습 데이터셋에 대한 예측을 수행하고, 예측 오류를 역전파하여 모델의 매개변수 조정 후

모델이 학습하고 있는지를 확인하기 위해 테스트 데이터셋으로 모델의 성능을 확인

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

[Out]

```
Epoch 1
-------------------------------
loss: 2.292942  [    0/60000]
loss: 2.286783  [ 6400/60000]
loss: 2.275352  [12800/60000]
loss: 2.272381  [19200/60000]
loss: 2.267560  [25600/60000]
loss: 2.226463  [32000/60000]
loss: 2.240955  [38400/60000]
loss: 2.208499  [44800/60000]
loss: 2.201728  [51200/60000]
loss: 2.183573  [57600/60000]
Test Error: 
 Accuracy: 37.5%, Avg loss: 2.175330 

Epoch 2
-------------------------------
loss: 2.176534  [    0/60000]
loss: 2.171206  [ 6400/60000]
loss: 2.125134  [12800/60000]
loss: 2.140271  [19200/60000]
loss: 2.106725  [25600/60000]
loss: 2.033942  [32000/60000]
loss: 2.067958  [38400/60000]
loss: 1.999325  [44800/60000]
loss: 1.996967  [51200/60000]
loss: 1.934805  [57600/60000]
Test Error: 
 Accuracy: 51.5%, Avg loss: 1.933817 

Epoch 3
-------------------------------
loss: 1.954151  [    0/60000]
loss: 1.931603  [ 6400/60000]
loss: 1.829657  [12800/60000]
loss: 1.865057  [19200/60000]
loss: 1.766892  [25600/60000]
loss: 1.701264  [32000/60000]
loss: 1.729385  [38400/60000]
loss: 1.641068  [44800/60000]
loss: 1.655786  [51200/60000]
loss: 1.547911  [57600/60000]
Test Error: 
 Accuracy: 60.1%, Avg loss: 1.568292 

Epoch 4
-------------------------------
loss: 1.625453  [    0/60000]
loss: 1.592277  [ 6400/60000]
loss: 1.455264  [12800/60000]
loss: 1.516795  [19200/60000]
loss: 1.401989  [25600/60000]
loss: 1.386637  [32000/60000]
loss: 1.401594  [38400/60000]
loss: 1.340019  [44800/60000]
loss: 1.362031  [51200/60000]
loss: 1.259193  [57600/60000]
Test Error: 
 Accuracy: 62.9%, Avg loss: 1.284938 

Epoch 5
-------------------------------
loss: 1.357947  [    0/60000]
loss: 1.337290  [ 6400/60000]
loss: 1.183273  [12800/60000]
loss: 1.279450  [19200/60000]
loss: 1.158183  [25600/60000]
loss: 1.175247  [32000/60000]
loss: 1.195495  [38400/60000]
loss: 1.147743  [44800/60000]
loss: 1.175308  [51200/60000]
loss: 1.090713  [57600/60000]
Test Error: 
 Accuracy: 64.6%, Avg loss: 1.107701 

Done!
```

# **모델 저장 & 불러오기**

## 저장

모델을 저장하는 일반적인 방법은 (모델의 매개변수들을 포함하여) 내부 상태 사전(*internal state dictionary*)을 직렬화(*serialize*)하는 것.

`state_dict()`- 학습가능한 매개변수가 담겨있는 딕셔너리(Dictionary)

예) *weight, bias*

```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

[Out]

```
Saved PyTorch Model State to model.pth
```

## 불러오기

모델을 불러오는 과정에는 모델 구조를 다시 만들고 상태 사전을 모델에 불러오는 과정이 포함됨.

```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```

[Out]

```
<All keys matched successfully>
```

## 불러온 모델로 예측

```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

[Out]

```
Predicted: "Ankle boot", Actual: "Ankle boot"
```

`.eval()` -  *evaluation* 과정에서 사용하지 않아야 하는 *layer*들을 알아서 *off* 시키도록 하는 함수

*train time*과 *eval time*에서 다르게 동작해야 하는 대표적인 예: `Dropout`, `BatchNorm`

*evaluation/validation* 과정에선 보통 `model.eval()`과 `torch.no_grad()`를 함께 사용한다고 한다.

---

참고 사이트

[https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

[https://anweh.tistory.com/21](https://anweh.tistory.com/21)

[https://sanghyu.tistory.com/90](https://sanghyu.tistory.com/90)

[https://subinium.github.io/pytorch-dataloader/](https://subinium.github.io/pytorch-dataloader/)

[https://daebaq27.tistory.com/60](https://daebaq27.tistory.com/60)

[https://velog.io/@jkl133/pytorch의-autograd에-대해-알아보자](https://velog.io/@jkl133/pytorch%EC%9D%98-autograd%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90)

[https://bluehorn07.github.io/2021/02/27/model-eval-and-train.html](https://bluehorn07.github.io/2021/02/27/model-eval-and-train.html)