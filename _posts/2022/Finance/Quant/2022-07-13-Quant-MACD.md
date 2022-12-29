---
title:  "[Quant] MACD(Moving Average Convergence Divergence)"
excerpt: "MACD란 무엇인가?"

categories: 
  - Quant
# tags: 
#   - [머신러닝, sklearn, 핸즈온]
toc: true
toc_sticky: true
author_profile: false
use_math: true
# date: 2022-03-25
# last_modified_at: 2022-03-25
---

# 단기투자의 기술 - MACD

## 시계열 데이터와 이동평균

주가 차트를 보면 아래와 같이 여러 개의 선이 존재한다.

![APPL_shortcut](/assets/images/posts/Finance/quant_macd/APPL_chart.png)

이 차트를 보면 일자별 주가를 표기하는 캔들 이외에도 녹색, 빨간색, 오랜지색, 보라색 등 4가지 색의 선이 추가로 있다.
- 초록색: 5일선
- 빨간색: 20일선
- 주황색: 60일선
- 보라새: 120일선

이 선들은 각 다른 구간의 이동평균선이며, 각 날짜에 산출한 이동평균 주가를 연결한 선이다. 

이동평균(Moving Average)은 말 그대로 이동하면서 구한 평균을 말한다.
- 주가는 매 순간 변하기 때문에 지금 구하는 일주일 동안의 평균 주가와 어제 구한 일주일 간의 평균주가는 다르기 때문에 어제의 이동평균선과 오늘의 이동평균선은 다르다.
- 이동평균 처럼 시간의 흐름에 따라 달라지는 것을 시계열 데이터라고 한다.

이동평균을 구할 때는 몇 일을 단위로 이동평균을 구할지 정해야된다.



![MA_window](/assets/images/posts/Finance/quant_macd/MA_window.png)

*출처: <슬기로운 퀀트 투자>*

- 주식시장에서는 대개 공휴일을 제외하고 계산한다. 
- 그렇기 때문에 MA를 구할 때는 중간에 공휴일이 끼어 있으면 공휴일은 건너뛰고 영업일(주식시장이 열리는 날) 기준으로 5일 꽉 채워 평균을 구한다.


#### 이동평균선을 그리는 이유
1. 주가 추세를 한눈에 보기 위해서다.
    - 예를 들어, 5일 평균이 20일 평균보다 높다면 최근 추가는 상승중이라는 뜻이다.  
2. 노이즈 제거를 위해서이다.
    - 주가가 상당 기간 상승세에 있더라도 하루하루 들여다보면 하락한 날은 있기 마련이다.
    - 이동평균선의 기간이 짧을수록 차트는 매우 불안정해보이며, 기간이 길수록 안정적으로 보인다.
    - 이를 평활화(smoothing)라고 한다.

## 이동평균선을 이용한 추세 읽기, MACD(이동평균수렴확산지수)
MACD(Moving Average Convergence Divergence)는 기간이 다른 이동평균선 사이 관계에서 추세 변화를 찾는 지표다. 
- 예를 들어 기간이 다른 2개의 이동평균선을 그린 후 기간이 짧은 이동평균선이 위에 있을 때 상승 추세라고 판단한다.
### 지표 이해
이동평균선은 실제 추이보다 늦다.<br>
그래서 이동평균선을 구할 때 단순이동평균(MA)가 아닌 지수이동평균(EMA, Exponential Moving Average)을 사용한다. <br>
단순 이동평균은 누구나 아는 가장 기본적인 평균값이고, 지수이동평균은 최근 값에 더 높은 가중치를 주고 계산한 평균값이다. 즉, 주가 추이를 조금이라도 더 민감하게 감지하기 위해 지수이동평균을 이용한 것이다.
$$
EMA_{t} = \alpha p_{t}+(1-\alpha)EMA_{t-1}\\\\
\begin{pmatrix}
\alpha=\frac{2}{N+1}, \\
EMP_{t}: t일의 지수이동평균, \\
p_{t}: t일의 주가, \\
N: 기간
\end{pmatrix}
$$
위의 식은 쉽게 생각하면 오늘의 주가와 어제의 지수이동 평균값의 평균이다. 즉, 주가에 조금 더 가중치를 준 정도로 이해하면 된다. 
그리고 각 다른 기간별로 구한 EMA로 MACD를 구할 수 있다.
$$
MACD = EMA(period_1)-EMA(period_2),\\ period_1<period_2
$$

MACD는 단기평균에서 장기평균을 빼준 값으로 MACD가 양수면 주가가 상승추세라는 것을 직관적으로 이해할 수 있다.<br>
그리고 MACD signal이라는 것과 MACD Oscillator라는 것도 있는데, MACD signal은 MACD의 후행성을 극복하기 위해 만든 선으로 9일간 MACD 지수이동평균선을 나타내고, MACD Oscillator는 MACD를 더 쉽게 이해하기 위해 만든 보조지표로 MACD에서 MACD signal 값을 빼 히스토그램 형식으로 표현한다.
- 0을 중심으로 매수세와 매도세 간의 힘의 강도 혹은 변화 추이를 쉽게 판단할 수 있는데, MACD Oscillator가 0 이상인 경우 매수를 추천한다.


### 투자 전략
MACD를 이용한 투자 전략은 2가지로 MACD를 보고 매매하는 방법과 MACD Oscillator를 보고 매매하는 방법이 있다.
1. MACD를 보고 매매하는 방법: MACD가 +이면 매수, -이면 매도
2. MACD Oscillator를 보고 매매하는 방법: +이면 매수, -이면 매도
3. MACD signal을 보조지표로 보고 매매하는 방법: MACD가 MACD signal을 상향 돌파하면 매수, 하향 돌파하면 매도한다.


> 본 내용은 <슬기로운 퀀트투자> 책을 참고하여 정리한 글입니다. <br>
[https://www.hanbit.co.kr/store/books/look.php?p_code=B7110068665](https://www.hanbit.co.kr/store/books/look.php?p_code=B7110068665)
> 