---
title:  "[ML] 스태킹 앙상블(Stacking Ensemble)"
excerpt: "스태킹 앙상블이란 개별 알고리즘의 예측 결과 데이터 세트를 최종적인 메타 데이터 세트로 만들어 별도의 ML 알고리즘으로 최종 학습을 수행하고 테스트 데이터를 기반으로 다시 최종 예측을 수행하는 방식입니다."

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
# Stacking Ensemble

스태킹 앙상블은 개별 알고리즘으로 예측한 데이터를 기반으로 다시 예측을 수행합니다.

즉, 개별 알고리즘의 예측 결과 데이터 세트를 최종적인 메타 데이터 세트로 만들어 별도의 ML 알고리즘으로 최종 학습을 수행하고 테스트 데이터를 기반으로 다시 최종 예측을 수행하는 방식입니다.

스태킹 모델은 두 종류의 모델이 필요합니다. 

첫 번째로는 개별적인 기반 모델

두 번째로는 위의 개별적인 기반 모델의 예측 결과($\hat{y})$를 학습데이터로 만들어서 학습하는 최종 메타 모델입니다.

여기서, 메타 모델이란, 개별 모델의 예측된 데이터 세트를 다시 기반으로 하여 학습하고 예측하는 방식을 뜻합니다.

스태킹 모델의 핵심은 여러 개별 모델의 예측 데이터를 각각 스태킹 형태로 결합해 최종 메타 모델의 학습용 피처 테이터 세트와 테스트용 데이터 세트를 만드는 것입니다. 

![Untitled.png](/assets/images/posts/MachineLearning/2022-03-28-XGBoost/Untitled.png)

스태킹의 간단한 다이어그램은 다음과 같습니다.

일단 동일한 $r$ 개의 피처와 $M$개의 열이 있는 데이터 셋에 스태킹 앙상블을 적용한다고 가정하겠습니다. 이 학습에 사용할 머신러닝 모델은 모두 $n$개입니다. 

먼저 *Model 1*부터 *Model n* 까지 훈련 시켜서 각 모델 별로 예측을 수행하면 각각 길이가 $*M*$인 레이블 값을 도출합니다. 

모델별로 도출된 예측 레이블 값을 다시 합해서 새로운 $M \times n$데이터 세트를 만들고 이렇게 스태킹된 데이터 세트에 대해 최종 모델을 적용해 최종 예측을 합니다.

![Untitled.png](/assets/images/posts/MachineLearning/2022-03-28-XGBoost/Untitled1.png)

# CV 세트 기반의 스태킹

cv 세트 기반의 스태킹 모델은 과적합을 개선하기 위해 최종 메타 모델을 위한 데이터 세트를 만들 때 교차 검증 기반으로 예측된 결과 데이터 세트를 이용합니다.

### step1

각 모델별로 원본 학습/테스트 데이터를 예측한 결과 값을 기반으로 메타 모델을 위한 학습용/테스트용 데이터를 생성합니다. 

![Untitled.png](/assets/images/posts/MachineLearning/2022-03-28-XGBoost/Untitled2.png)

일단 스태킹에 적용할 모델은 3개이며, 각 모델별 교차검증 3번을 적용한다고 가정하겠습니다.

각 폴드마다 뽑아진 훈련 데이터로 모델을 훈련 한 다음, 검증 데이터를 활용해 예측 후 예측 값을 저장합니다. (*result* $\alpha$)

그리고 각 폴드마다 나온 *Model*을 기반으로 원본의 *Test*데이터 셋을 훈련하고 그 평균 값을 저장합니다. (*result* $\beta$)

### step2

![Untitled.png](/assets/images/posts/MachineLearning/2022-03-28-XGBoost/Untitled3.png)

step1에서 개별 모델들이 생성한 학습용 데이터를 모두 스태킹 형태로 합쳐서 메타 모델이 학습할 최종 학습용 데이터 세트를 생성합니다. 

그리고 result $\alpha$를 메타 모델의 학습 데이터로, result $\beta$를 메타 모델의 테스트 데이터로 사용해 메타 모델을 학습합니다.

다음 그림은 전체적인 CV기반의 스태킹 앙상블 전체적인 흐름입니다.

![Untitled.png](/assets/images/posts/MachineLearning/2022-03-28-XGBoost/Untitled4.png)