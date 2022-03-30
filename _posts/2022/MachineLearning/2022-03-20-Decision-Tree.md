---
title:  "[ML] 의사결정나무(Decision Tree)"
excerpt: "분류와 회귀 작업 그리도 다중 출력 또한 가능한 의사결정나무"

categories:
  - Machine-Learning
# tags:
#   - [Machine Learning, Classification, Decision Tree, CART]

toc: true
toc_sticky: true
use_math: true

# date: 2022-03-20
# last_modified_at: 2022-03-20
---
# Decision Tree(의사결정 나무)

# 1. 의사 결정 나무

회귀와 로지스틱 회귀 모델은 피쳐들 간에 관계가 존재하거나 결과값이 비선형일 때 좋은 성능을 낼 수 없습니다.

트리 알고리즘이 위의 단점들을 보완할 수 있는데, 이는 각 피쳐들의 어떠한 기준으로 데이터를 2개 혹은 그 이상의 하위 집합으로 나눕니다.

SVM(support vector machine)처럼 결정트리(Decision Tree)는 분류와 회귀 작업 그리고 다중 출력 작업도 가능한 다재다능한 머신러닝입니다. 

결정트리는 최근 자주 사용되는 가장 강력한 머신러닝 알고리즘 중 하나인 랜덤 포레스트의 기본 구성 요소이기도 합니다.

NOTE 결정 트리의 여러 장점 중 하나는 데이터 전처리가 거의 필요하지 않다는 것입니다.(사실 특성의 스케일을 맞추거나 평균을 원점에 맞추는 작업이 필요하지 않습니다)

의사결정 나무는 데이터를 2개 혹은 그 이상의 부분집합으로 분할하고, 데이터가 균일해지도록 분할 합니다.

![Untitled0.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled0.png)


`의사결정 나무는 데이터에 내재된 패턴을 변수의 조합으로 나타내는 예측 (목표변수 $y$가 수치형일 때) 또는 분류(목표변수 $y$가 범주형일 때) 모델을 나무의 형태로 만드는 것입니다.
![Untitled1.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree//Untitled1.png)

의사 결정 나무는 일반적으로 규칙을 가장 쉽게 표현하는 $if\ then\ else$  기반으로 나타냅니다.

그 과정은 각 관측치는 루트노드(*Root node*)를 시작으로 스무고개와 같은 일련의 기준(feature의 기준)을 판단하는 내부노드(Intermediate node)를 통해 하위 집합인 리프 노트(Leaf node)에 도달합니다.

![Untitled2.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled2.png)

즉, 일련의 범위과 관련된 문제들로 그 대상의 범위를 좁혀가면서 최종적으로 예측/분류하는 것입니다.

예를 들어, 타이타닉 생존 데이터로 봤을 때, 우리는 “이 탑승객은 생존할 수 있을까?” 라는 질문을 답할 수 있는 의사 결정 나무 모델을 만들어야됩니다. 

![Untitled3.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled3.png)

 

이진 분할로 생각했을 때, 일단 A집합이 있을 때, 이를 어떠한 기준으로 $B$와 $C$로 나누고, $B$를 또 $D$와 $E$로 나눴을 때, 아래처럼 표현할 수 있습니다.

![Untitled4.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled4.png)

트리를 키울 수 있는 다양한 알고리즘(트리의 가능한 구조(노드당 분할 수); 분할을 찾는 방법; 분할을 중지하는 시기 및 리프 노트 내의 단순 모델을 추정하는 방법)이 있습니다. 

이중 제일 자주 사용되는 알고리즘은 분류 및 회귀 트리 ***CART(***$Classification\ And\  Regression\ Tree$***)***입니다.

***CART***에서 가장 중요한 것은 이진분할입니다.

그리고 CART알고리즘 이름을 알 수 있 듯, 이 알고리즘으로 회귀와 분류를 할 수 있습니다.

# 예측나무 모델( *Regression Tree*)

여기서 우리가 구해야되는 $\hat{y}$ 는 수치형 변수입니다.

![Untitled5.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled5.png)

만약 새로운 관측치(빨간색 점)가  $R_5$에 속할 때, 이 관측치의 목표변수 $\hat{y}$은 같은 $R_5$에 속한 기타 $y$값들의 평균치로 예측합니다.

즉 빨간점의 목표변수  
$\hat{y}$을 해당 점이 속해 있는 작은 부분 집합에 있는 이웃 관측치들의 $y$값들의 평균으로 예측하겠다는 것입니다.

만약, $y_1=3$, $y_2=4$, $y_3=2$, $y_4=1$, $y_5=4$,  $y_6=3$일 때,

 

$$
\hat{y}=\frac{\sum^5_{i=1}{y_i}}{5}=\frac{3+4+2+1+4+3}{5}=\frac{17}{5}=3.4
$$

이를 함수의 형태로 표현하면 

$$
\hat{f}(x_1,x_2) =\sum^5_{m=1}{c_m}I\{(x_1,x_2)\in R_m\}
$$

여기서  

$$
I=\begin{cases}0, &  Fasle(값이\ 중괄호 \ 안에\ 있는게\ 거짓일\ 때)
\\ 1, & True(값이\ 중괄호\ 안에\ 있는게\ 참일 \ 때)
\end{cases}
$$

$$
c_m:회귀나무모델로부터\ 예측한\ R_m부분의\ 예측값
$$

그리고 $\{(x_1,x_2)\in R_m\}$는 해당 $x_1$값과 $x_2$값이 부분집합 $R_m$(*leaf node*)에 속해있는지 여부를 봅니다.

그래서, $\hat{f}(x)$는 아래와 같이 표현할 수 있습니다:

$$
\begin{matrix}
\hat{f}(x)
&=&c_1 I\{(x_1,x_2)\in R_1\}
\\&+&c_2 I\{(x_1,x_2)\in R_2\}
\\&+&c_3 I\{(x_1,x_2)\in R_3\}
\\&+&c_4 I\{(x_1,x_2)\in R_4\}
\\&+&c_5 I\{(x_1,x_2)\in R_5\}
\\&=&c_1 \cdot0+c_2 \cdot0+c_3 \cdot0+c_4 \cdot0+c_5 \cdot1
\\&=&c_5
\end{matrix}
$$

만약 데이터를 $M$개,로 분할 할 경우, 아래처럼 표현할 수 있습니다.


$$
f(x) =\sum^M_{m=1}{c_m}I\{x\in R_m\}
$$

## 분류 기준($R_m$부분의 예측값($C_m$), 분할변수($j$) 그리고 분할점($s$))

### $R_m$부분의 예측값($C_m$)

최상의 분할, 즉 최적의 $C_m$의 기준은 다음 비용함수(Cost function)를 최소로 할 때 얻어집니다. 

$$
\underset{C_m}{min}\sum^N_{i=1}(y_i-f(x_i))^2
\\=\underset{C_m}{min}\sum^N_{i=1}(y_i-\sum^M_{m=1}{c_m}I\{x\in R_m\})^2
$$


이 수식의 의미는 “ $y_i-f(x_i)$이 최소화되려면 $c_m$이 무슨 값이 되어야할까?” 와 같습니다.

$$
\hat{c}_m=ave(y_i\vert x_i \in R_m)
$$

이 수식의 의미는 “각 분할에 속해 있는 $y$값들의 평균으로 예측해라”와 같습니다. 여기서 $ave$는 *Average Variance Extracted*의 약자로, 쉽게 말해 평균 설명력입니다.

**각 분할에 속해 있는 y값들의 평균으로 예측했을 때 오류가 최소가 됩니다.**

### 분할변수($j$)와 분할점($s$)

![Untitled6.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled6.png)

위의 트리 모형의 루트노드를 봤을 때, 왜 $x_1$과 $x_2$중 $x_1$로 정했을까요? 그리고 수 많은 값들 중에 왜 $t_1$을 기준으로 나뉘어졌을까요?

$$
R_1(j,s)=\{x\vert x_j \leq s\}
\\R_2(j,s)=\{x\vert x_j > s\}
$$

여기서 
$x_1=t_1,t_3$, $x_2=t_2,t_4$일 때, $j$와 $s$를 다 바꿔가면서 총 고려해야될 점은 4개($t_1, t_2, t_3, t_4$)입니다.


$$
\underset{j,s}{argmin}[\underset{c_1}{min}\sum_{x_i\in R_1(j,s)}(y_i-c_1)^2+\underset{c_2}{min}\sum_{x_i\in R_2(j,s)}(y_i-c_2)^2]
\\=\underset{j,s}{argmin}[\sum_{x_i\in R_1(j,s)}(y_i-\hat{c}_1)^2+\sum_{x_i\in R_2(j,s)}(y_i-\hat{c}_2)^2]
\\ \hat{c}_1=ave(y_i\vert x_i \in R_1(j,s))\ and\ \hat{c}_2=ave(y_i\vert x_i \in R_2(j,s)
$$


$x_1\le t_1$ , $x_2\le t_2$,  $x_2\le t_2$,  $x_2\le t_4$를 하나씩 다 시도해보고, 어떤 기준일 때 값이 제일 작은지 비교하고($\underset{j,s}{argmin}$), 제일 작은 $j$와 $s$를 사용합니다 (Grid Search와 비슷한 느낌). 위의 트리모형을 보면, $x_1\le t_1$일 때 값이 제일 작아졌기 때문에 루트노드로 사용됐다는 것을 짐작할 수 있습니다.

# 분류나무 모델( *Decision Tree Classifier*)

여기서 우리가 구해야되는 $\hat{y}$ 는 범주형 변수입니다.

![Untitled7.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled7.png)

위의 그림처럼 데이터를 여러 부분 집합으로 나눠자고, 새로운 관측치(빨간색 점)이 $R_5$에 속할 때, 우리는 직감적으로 이 관측치는 노란색 점으로 범주예측, 즉 분류를 할 것 입니다. 그리고 또 새로운 또 다른 관측치(보라색 점)이 $R_2$에 속할 때, 우리는 이 또한 바로 노란색 점으로 분류를 할겁니다.

각 관측치 마다 반응변수 값 $y_i=1,2,3,..,k,$ 즉 $k$개의 클래스가 존재합니다.

$R_m$은 루트노드 $m$에 해당하며 $N_m$ 관측치 개수를 가지고 있습니다.

$\hat{p}_{mk}$는 루트노드 $m$에서 $k$ 클래스에 속해 있는 관측치의 비율

$$
\hat{p}_{mk}=\frac{1}{N_m}\underset{x_i \in R_m}\sum I(y_i=k)
$$

루트 노드 $m$으로 분류된 관측치는 $k(m)$ 클래스로 분류됩니다.

$$
k(m)=\underset{k}{argmax}\ \hat{p}_{mk}
$$

$$
\begin{matrix}
\hat{f}(x) 
&=&\sum^5_{m=1}{k(m)}I\{(x_1,x_2)\in R_m\}
\\&=& {k(1)}I\{(x_1,x_2)\in R_1\}
\\&+&{k(2)}I\{(x_1,x_2)\in R_2\}\\&+&{k(3)}I\{(x_1,x_2)\in R_3\}
\\&+&{k(4)}I\{(x_1,x_2)\in R_4\}
\\&+&{k(5)}I\{(x_1,x_2)\in R_5\}
\end{matrix}
$$

여기서 5는 루트노드의 개수입니다.

만약 $x_1, x_2$가 부분집합 $R_5$에 속해있다면, 아래와 같이 계산할 수 있습니다.

$$
\begin{matrix}
\hat{f}(x) 
&=&\sum^5_{m=1}{k(m)}I\{(x_1,x_2)\in R_m\}
\\&=& {k(1)}\cdot0+{k(2)}\cdot0+{k(3)}\cdot0+{k(4)}\cdot0+{k(5)}\cdot1
\\&=&K(5)
\end{matrix}
$$

### 비용함수(불순도 측정)

regression과는 다르게 분류모델에서의 비용함수는 3가지가 있습니다.

$$
Misclassification\ rate:\ \frac{1}{N_m}\underset{i \in R_m}\sum I(y_i \ne k(m))=1-\hat{p}_{(mk)m}
$$

실제에서 나온 범주(y_i)와 모델에서 나온 범주($k(m)$)가 얼마만큼 잘 매칭이 되었는지를 봅니다. 

$$
{GINI}\ \ Index:\ \underset{k \ne k^{'}}\sum\hat{p}_{mk}\hat{p}_{mk^{'}}=\sum^K_{k=1}\hat{p}_{mk}(1-\hat{p}_{mk}) 
$$

$$
Cross-entropy:\ -\sum^K_{k=1}\hat{p}_{mk}log\ \hat{p}_{mk}
$$

![Untitled8.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled8.png)

### 분할변수($j$)와 분할점($s$)

$$
R_1(j,s)=\{x\vert x_j \leq s\}
\\R_2(j,s)=\{x\vert x_j > s\}
$$

위의 예측모델과 같이, $x_1\le t_1$ , $x_2\le t_2$,  $x_2\le t_2$,  $x_2\le t_4$를 하나씩 다 시도해보면서 어떤 기준일 때 불순도가 가장 낮을 때의  $j$와 $s$를 사용합니다 (Grid Search와 비슷한 느낌). 위의 트리모형을 보면, $x_1\le t_1$일 때 값이 제일 작아졌기 때문에 루트노드로 사용됐다는 것을 짐작할 수 있습니다.

<!-- Misclassification rate을 비용함수로 사용했을 때,

$
 \underset{j,s}{argmin}[\frac{1}{N_{R_1(j,s)}}\underset{x_i \in R_1(j,s)}\sum I(y_i \ne k(m))+\frac{1}{N_{R_2(j,s)}}\underset{x_i \in R_2(j,s)}\sum I(y_i \ne k(m))]\\k(m)=\underset{k}{argmax}\ \hat{p}_{mk}\qquad  \hat{p}_{mk}=\frac{1}{N_m}\underset{x_i \in R_m}\sum I(y_i=k)
$ -->

### 분할법칙

- 분할 변수와 분할 기준은 목표변수($y$)의 분포를 가장 잘 구별해주는 쪽으로 정합니다.
    - 목표변수($y$)가 균일한 뱡향으로 갑니다.
- 목표변수($y$)의 분포를 잘 구별해주는 측도로 순수도(purity) 또는 불순도(impurity)를 정의합니다.
    - 예를들어 클래스 0과 클래스 1의 비율이 45%와 55%인 노드는 각 클래스의 비율이  90%와 10%인 노드에 비하여 순수도가 낮다(또는 불순도가 높다)라고 해석할 수 있습니다.
- 각 노드에서 분할변수와 분할점의 설정은 불순도의 감소가 최대가 되도록 설정합니다.

## 정보 이득 Information Gain

일반적으로 의사결정 트리는 정보 이득 수치를 계산해서 최적 목표를 달성하는 트리를 완성합니다.

정보이익은 정보의 가치를 의미하며, 그 값이 클수록 좋습니다.
- 즉, 사전 엔트로피(불확실성)$\ -\ $사후 엔트로피(불확실성)입니다.<br>
따라서 Inforamtion Gain은 불확실성이 얼마나 줄었는가? 라는 질문으로 해석할 수 있습니다.<br>
이 값이  크면 불확실성이 많이 감소했다고 볼 수 있습니다.

정보 이득은 엔트로피의 변화량으로 계산되기 때문에 엔트로피 계산법을 알고 있어야됩니다. 

엔트로피의 수식을 아래와 같습니다.

$$
Entropy(A)=-\sum_{k=1}^np_klog_2(p_k)
$$
여기서 $A$

는 의사 결정을 수행하는 전체 영역을 의미하고, n은 범주 개수, p_k는 A 영역에 속하는 레코드 가운데 k 범주에 속하는 레코드의 비율입니다.


![Untitled9.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled9.png)
예를 들어, 어떤 그룹 A가 있습니다. $Entropy(A)$는 다음과 같습니다.

$$
\begin{matrix}
Entropy(A)
&=& -\frac{5}{10}log_2{\frac{5}{10}}-\frac{5}{10}log_2{\frac{5}{10}}=1
\end{matrix}
$$

그리고 정보이익은 다음과 같습니다.

$
Information\ Gain(A)=1-1=0
$

이번에는 아래의 그림처럼,그룹 A를 임의로 실선을 하나 그어서, B,C 두 그룹으로 나누었을 때, $Entropy(A)$는 다음과 같습니다.
![Untitled10.png](/assets/images/posts/MachineLearning/2022-03-20-Decision-Tree/Untitled10.png)

$$
\begin{matrix}
Entropy(A)
&=&Entropy(파란색)+Entropy(노란색)
\\&=&(-\frac{5}{6}log_2{\frac{5}{6}}-\frac{1}{6}log_2{\frac{1}{6}})+(-\frac{0}{4}log_2{\frac{0}{4}}-\frac{4}{4}log_2{\frac{4}{4}})
\\&=&\frac{6}{10}(0.65)+\frac{4}{10}(0)
\\&=&0.39
\end{matrix}
$$

그리고 정보이익은 다음과 같습니다.

$$
Information\ Gain(A)=1-0.39=0.61
$$

이처럼, 아무런 분류를 하지 않았을 때의 $Entropy$는 1로, 최대입니다(지니지수는 0.5).<br>


# 모델의 해석력

의사결정 나무의 해석은 매우 간단합니다. 일단 루트 노드에서 시작해서 리프 노드까지 도달할 때까지 연결된 모든 노드들을 AND로 연결됩니다.

# 피처 중요도 (Feature Importance)

피처 중요도(Feature Importance, 또는 퍼뮤테이션 중요도[Permutation Importance])는 데이터의 피쳐가 알고리즘의 정확한 분류에 얼마나 큰 영향을 미치는지 분석하는 기법입니다.

### 피처 중요도의 주요 컨셉

특정 피처의 값을 임의의 값으로 치환했을 때 원래 데이터보다 예측 에러가 얼마나 더 커지는가를 측정하는 것입니다.

예를 들어, 한 피처 데이터를 변형했을 때 모델 예측 결과가 크게 달라졌다면 해당 모델은 이 피처에 의존해 판단을 내리고 있는 것입니다. 

일반적으로 어떤 피처가 모델 분류에 중요하지 않다면 그 피처는 모델 분류 성능에 영향을 미치지 않습니다.(왜냐면 특정 피처를 변경했을 때 모델 에러가 증가하는 경우, 머신러닝 모델은 해당 피처의 영향력이 아예없다는 것처럼 그것을 무시하기 때문입니다. )

# 장점

트리구조는 데이터의 특성 간 상호 작용을 고려하는데 적합합니다.

데이터는 선형 회귀 분석과 같이 다차원 초평면의 점보다 이해하기 쉬운 개별 그룹으로 나눕니다. 그렇기에 해석은 매우 간단해집니다.

트리구조는 시각화에 용이합니다.

# 단점

트리는 비선형 관계를 다루기에는 용이하지만, 선형 관계를 다루는데는 실패합니다. 

위의 타이타닉 의사결정나무 모델을 보면, 해당 루트노드에서 성별을 0.5보다 크거나 작은걸 기준으로 했습니다. 

트리는 안정적이지 않습니다, 즉 신뢰도(*reliability*)가 매우 낮습니다. 학습데이터 세트를 조금만 변경해도 완전히 다른 트리가 만들어질 수 있습니다(각 분할은 상위 분할에 따라 달라지기 때문입니다).

트리의 깊이가 깊어질 수로 해석이 어려워집니다.   
<!-- 
# 타이타닉 실습코드

[Google Colaboratory](https://colab.research.google.com/drive/1ZJz5gE4Kfk7FbUp4PTFvZPhlQeJlwmDO?usp=sharing) -->