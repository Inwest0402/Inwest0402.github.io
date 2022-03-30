---
title:  "[ML] 랜덤 포레스트(Random Forest)"
# excerpt: "분류와 회귀 작업 그리도 다중 출력 또한 가능한 의사결정나무"

categories:
  - Machine-Learning
tags:
  - [Machine Learning, Boostrap, Random Forest]

toc: true
toc_sticky: true
use_math: true

# date: 2022-03-28
# last_modified_at: 2022-03-28
---


머신 러닝에서 랜덤 포레스트는 분류, 회귀 분석등에 사용되는 앙상블 학습 방법의 일종으로, 훈련 과정에서 구성한 다수의 결정 트리로부터 부류(분류) 또는 평균치 예측치(회귀 분석)를 출력함으로써 동작합니다.

(출처 [위키백과](https://ko.wikipedia.org/wiki/%EB%9E%9C%EB%8D%A4_%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8))

랜덤 포레스트의 가장 큰 특징은 이름에서도 볼 수 있듯이 랜덤성이며, 이에 의해 트리들이 서로 조금씩 다른 특성을 갖습니다.

이 특성은 각 트리들의 예측들이 비상관(decorrelation) 되게하며, 결과적으로 일반화(generalization) 성능을 향상시킵니다. 

또한, 랜덤화는 포레스트가 노이즈가 포함된 데이터에 대해서도 강인하게 만들어줍니다. 

### 특징

- 랜덤 포레스트는 앙상블 알고리즘 중 비교적 빠른 수행 속도를 가지고 있으며, 다양한 영역에서 높은 예측 성능을 보이고 있습니다.
- 랜덤 포레스트의 기반 알고리즘은 결정 트리로서, 결정 트리의 쉽고 직관적인 장점은 그대로 가지고 있습니다.
- 랜덤 포레스트는 여러 개의 결정 트리 분류기가 전체 데이터에서 배깅 방식으로 각자의 데이터를 샘플링해 개별적으로 학습을 수행한 뒤 최종적으로 모든 분류기가 보팅을 통해 예측 결정을 하게 됩니다.

![Untitled.png](/assets/images/posts/MachineLearning/2022-03-28-Random-Forest/Untitled.png)

# 배깅(Bagging)

랜덤 포레스트 알고리즘은 대표적인 배깅 기법을 사용한 알고리즘입니다.

여기서 배깅은 Bootstrap을 여러개 뽑아 여러 개의 모델에 돌링 후 집계(Aggregation)하는 방법입니다. 즉, 여러 개의 모델의 결과들을 모두 살리며서 최적의 결과를 도출하는 방법입니다.

# 부트스트랩(Bootstap)

랜덤 포레스트는 개별적인 분류기의 기반 알고리즘은 결정트리이지만 개별 트리가 학습하는 데이터 세트는 전제 데이터에서 일부 중첩되게 샘플링된 데이터 세트입니다. 이렇게 여러 개의 데이터 세트를 중첩되게 분리하는 것을 부트스트래핑 분할 방식이라고 합니다. 

부트스트랩은 데이터 내에서 반복적으로 샘플을 사용하는 resampling 방법 중 하나이며,  이 방법을 사용하면 하나 밖에 없었든 샘플 데이터 셋을 $n$개의 데이터 셋과 같은 효과를 누릴 수 있습니다.

![bagging.webp](/assets/images/posts/MachineLearning/2022-03-28-Random-Forest/bagging.webp)

[https://corporatefinanceinstitute.com/resources/knowledge/other/bagging-bootstrap-aggregation/](https://corporatefinanceinstitute.com/resources/knowledge/other/bagging-bootstrap-aggregation/)

원래 부트스트랩은 통계학에서 여러 개의 작은 데이터 세트를 임의로 만들어 개별 평균의 분포도를 측정하는 등 목적을 위한 샘플링 방식을 지칭힙니다.

랜덤 포래스트의 서브세트(Subset) 데이터는 이러한 부트스트래핑으로 데이터가 임의로 만들어집니다. 

서브세트의 데이터 건수는 전체 데이터 건수와 동일하지만, 개별 데이터가 중첩되어 만들어집니다. 

예를 들어, 원본 데이터의 건수가 10개인 학습 데이터 세트에 랜덤 포레스트를 3개의 결정 트리 기반으로 학습하려고 n_estimators=3으로 하이퍼 파리미터를 부여하면 다음과 같이 데이터 서브세트가 만들어 집니다. 

![Untitled1.png](/assets/images/posts/MachineLearning/2022-03-28-Random-Forest/Untitled1.png)

위와 같이 데이터가 중첩된 개별 데이터 세트에 결정 트리 분류기를 각각 적용하는 것이 랜덤 포레스트입니다.

## 실습 코드

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# digit 데이터셋을 이용해 학습/테스트용 DataFrame 변환
dataset = load_iris()
X_data = dataset.data
y_data = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X_data,y_data)

# 랜덤 포레스트 학습 및 별도의 테스트 세트로 예측 성능 평가
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('Ramdom Forest 정확도:{0:.4f}'.format(accuracy))
```

[Output]

![Untitled2.png](/assets/images/posts/MachineLearning/2022-03-28-Random-Forest/Untitled2.png)

# 랜덤 포레스트 하이퍼 파라미터 및 튜닝

트리 기반의 앙상블 알고리즘의 단점은 하이퍼 파라미터가 너무 많고, 그로 인해 튜닝을 위한 시간이 많이 소모된다는 것입니다. 

- `n_estimator` : 랜덤 포레스트에서 결정 트리의 개수를 지정합니다. Default는 10개입니다.
    - 많이 설정할수록 좋은 성능을 기대할 수 있지만, 계속 증가시킨다고 성능이 무조건 향상되는 것은 아닙니다.
    - 많이 설정할수록 학습 수행 시간이 더 오래 걸립니다.
- `max_features` : 결정 트리에 사용된 `max_feature` 파라미터와 같습니다.
    - 랜덤 포레스트에서 이 파라미터의 Default 값은 ‘auto’입니다. 따라서  랜덤 포레스트의 트리를 분할하는 피처를 참조할 때 전체 피처가 아닌 sqrt(전체 피처 개수)만큼 참조합니다.
- `max_depth`/`min_samples_leaf` : 과적합을 개선하기 위해 사용되는 파라미터입니다.

```python
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators' : [100],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [8, 12, 18],
    'min_samples_split' : [8, 16, 20]
}
# 선 RandomForest 후 GridsearchCV
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf,
                       param_grid=params,
                       cv=2,
                       n_jobs=-1)
grid_cv.fit(X_train, y_train)
print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도정확도:{0:.4f}'.format(grid_cv.best_score_))
```

[Output]

![Untitled3.png](/assets/images/posts/MachineLearning/2022-03-28-Random-Forest/Untitled3.png)