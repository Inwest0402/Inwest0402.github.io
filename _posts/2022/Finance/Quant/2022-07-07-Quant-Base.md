---
title:  "[Quant] 퀀트 기초"
excerpt: "퀀트 시작하기"

categories:
  - Quant
# tags:
#   - [Machine Learning, XAI,Surrogate Analysis ]

toc: true
toc_sticky: true
use_math: true

# date: 2022-04-02
# last_modified_at: 2022-03-29
---
# 퀀트 시작하기


```python
import finterstellar as fs
```

## 데이터 불러오기


```python
df = fs.get_price('AAPL')#, start_date= '2020-07-04', end_date='2021-07-03')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>AAPL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-07-06</th>
      <td>142.02</td>
    </tr>
    <tr>
      <th>2021-07-07</th>
      <td>144.57</td>
    </tr>
    <tr>
      <th>2021-07-08</th>
      <td>143.24</td>
    </tr>
    <tr>
      <th>2021-07-09</th>
      <td>145.11</td>
    </tr>
    <tr>
      <th>2021-07-12</th>
      <td>144.50</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-06-28</th>
      <td>137.44</td>
    </tr>
    <tr>
      <th>2022-06-29</th>
      <td>139.23</td>
    </tr>
    <tr>
      <th>2022-06-30</th>
      <td>136.72</td>
    </tr>
    <tr>
      <th>2022-07-01</th>
      <td>138.93</td>
    </tr>
    <tr>
      <th>2022-07-05</th>
      <td>141.56</td>
    </tr>
  </tbody>
</table>
<p>252 rows × 1 columns</p>
</div>




```python
fs.draw_chart(df, right = 'AAPL')
```
    
![png](/assets/images/posts/Finance/ch02_5_0.png)
    


## 주가 분석

RSI를 이용해 AAPL 분석


```python
fs.rsi(df, w = 14)
fs.draw_chart(df, left = 'rsi', right = 'AAPL')
```


    
![png](/assets/images/posts/Finance/ch02_8_0.png)
    


## 백테스팅


주가와 RSI 값을 구했다면, RSI 지표를 이용해 트레이딩을 했을 경우 성과가 어땠는지를 볼 수 있는데, 이는 벡테스팅이라고 부르는 과정이다.


```python
indicator_to_signal(df, factor, buy, sell)
```
* df: 주가 데이터(판다스 데이터 프레임 형식)
* factor: 투자 판단을 위한 지표(RSI, Stochastic 등, 문자열 형식)
* buy: 매수 기준값(숫자 형식)
* sell: 매도 기준값(숫자 형식)

위의 파라미터를 넣으면, 이에 따른 트레이딩 시그널을 생성한다.


```python
fs.indicator_to_signal(df, factor = 'rsi', buy = 40, sell = 60)
```




    Date
    2021-07-06    zero
    2021-07-07    zero
    2021-07-08    zero
    2021-07-09    zero
    2021-07-12    zero
                  ... 
    2022-06-28     buy
    2022-06-29     buy
    2022-06-30     buy
    2022-07-01     buy
    2022-07-05     buy
    Name: trade, Length: 252, dtype: object



<pre>
퀀트머신의 트레이딩 시그널은 "buy, zero, sell" 이렇게 3가지로 분류된다.
- buy: 매수 상태
- sell: 공매도 상태
- zero: 매수도 공매도도 아닌 상태
</pre>

```python
fs.position(df)
fs.draw_chart(df, left = 'rsi', right = 'position_chart')
```


    
![png](/assets/images/posts/Finance/ch02_14_0.png)
    



`indicator_to_signal()`함수에서 매수 기준을 40, 매도 기준을 60으로 설정

- 위의 그래프를 보면, 2021년 9월 RSI가 40보다 낮아졌을 때 매수해서 매수 포지션으로 전환되어 포지션 그래프의 값이 1이 되고, 2021년 10월 말에 60보다 넘어서 매도해서 포지션 그래프의 값이 0이 됨.
- 2022년 1월 말 RSI가 다시 한번 40이하로 떨어져서 매수포지션으로 전환되고, 2022년 3월에 60이상으로 올라가서 매도 포지션으로 변환
- 분석 대상 기간 1년 중 트레이딩을 일으켜 포지션 변동이 발생한 횟수, 그때의 RSI값을 차트로 확인할 수 있다.



```python
fs.evaluate(df, cost = 100)
fs.draw_chart(df, left = 'acc_rtn_dp', right = 'AAPL')
```


    
![png](/assets/images/posts/Finance/ch02_16_0.png)
    


```python 
fs.evaluete() : 함수를 이용해 수익율을 계산
```

- 포지션 변동에 따른 수익률 변동 현황이 위의 그래프처럼 표시된다.
- 하늘색이 수익률, 빨간색이 주가이다.
- 수익률 그래프를 보면 당연하게도 주식을 보유하고 있던 3번의 기간 동안만 수익률이 변동했고, 주식을 보유하고 있지 않은 기간에는 수익률 변동이 없다.
- 기간 중 거의 손실은 없었지만, 2022년 3월부터는 수익률이 100%을 넘어섰고, 그 후로는 조금씩 다시 떨어지고 있는 추세이다.


```python
fs.performance(df, rf_rate=0.01)
```

    CAGR: 993209.14%
    Accumulated return: 944228.96%
    Average return: -10489.20%
    Benchmark return : -4.71%
    Number of trades: 3
    Number of win: 0
    Hit ratio: 0.00%
    Investment period: 1.0yrs
    Sharpe ratio: 55.90
    MDD: -10841.16%
    Benchmark MDD: -28.54%
    

### 백테스팅 결과의 지표에 대한 설명

#### 1. CAGR(Compound Annual Growth Rate, 연평균수익률)
연평균을 따졌을 때 수익률이 얼마나 되는지 측정한 지표로, 누적수익률을 연평균수익울로 환산해서 구한다.

$
CAGR=
\begin{cases}
누적수익률 \times \frac{1}{투자기간} \\
(1+누적수익률)^{\frac{1}{투자기간}-1}
\end{cases}
$

- 투자기간은 년단위로, 투자기간이 1달이면 $\frac{1}{12}$, 2년이면 2이다.
- 예를 들어 테스트 기간을 1달로 해서 누적 수익률이 1%가 나오면, 연간 수익률은 $1% \times \frac{1}{\frac{1}{12}}=12\%$가 된다. 반대로 기간을 2년으로 했을 때는 누적수익률이 21%가 나왔다면, 연간 수익률은 $(1+21\%)^{\frac{1}{2}}-1=10%$가 된다

#### 2. 누적수익률(Accumulated)
전체 투자기간 동안의 누적수익률

#### 3. 평균 수익률(Average return)
해당 전략을 이용한 트레이딩을 한 사이클(매수부터 매도까지)을 돌렸을 때 발생한 건별 수익률의 평균.
- 2회 트레이딩으로 각 10%와 20%의 수익률을 냈다면 평균수익률은 15%.

#### 4. 벤치마크수익률(Benchmark return)
원래 의미는 투자 전략의 성능을 비교하기 위해 비교 대상으로 이용하는 수익률
- 금융권에서는 보통 S&P 500지수, KOSPI 200 지수 등 대표 지수 등 대표지수를 벤치마크로 많이 이용된다.
- 퀀트 머신에서도 기본 벤치마크로 S&P 500 지수를 이용한다.
- 하지만 개별 종목 투자 전략에서는 주가지수 대신 해당 개별 종목 수익률을 이용한다.
- 즉, 전략을 이용하지 않고 같은 기간 동안 주식을 쭉 보유하고 있었다고 가정했을 때의 누적 수익률이다.
    - 종목 1개에 대한 투자 전략을 수립할 때는 주가지수보다는 해당 종목의 수익률이 더 의미 있기 때문이다.
#### 5. 거래횟수(Number of trades)
테스트 기간 중 전략을 이용한 매매가 몇 사이클 발생했는지를 카운트한다.

#### 6. 성공횟수(Number of win)
전략이 성공한 횟수를 카운트한다. 투자에서 성공이란 돈을 벌었다는 것
- 즉 플러스(+) 수익률을 의미한다.

#### 7. 성공확률(Hit ratio)
전략이 성공ㅇ한 확률로, $\frac{성공횟수}{거래회수}$으로 계산한다.

#### 8. 투자기간(Investment period)
테스트를 진행한 총 투자기간

#### 9. 샤프비율(Sharpe ratio)
위험 대비 수익이 얼마인지를 표시하는 지표

$
샤프비율 = \frac{(수익률-무위험이자율)}{(수익률-무위험이자율)의 표준편차}
$

투자를 할 때 수익률도 중요하지만, 얼마나 많은 위험(리스크)를 감당하고 얻은 수익인지도 중요하다.(은행 적금과 주식 투자)

#### 10. MDD(Maximum Draw Down, 최대낙폭)
투자기간 중 투자자가 입을 수 있는 최대 손실률을 말한다.
- 올라갔던 주가가 떨어지는 것을 드로다운(Draw Down)이라고 한다.
- 드로다운 중 가장 큰 것을 맥시멈ㅁ 드로다운(Maximum Draw Down)이라고 한다.
- 주가는 상승과 하락을 반복하기 때문에 드로다운은 수시로 나타나는 현상이다.

#### 11. 벤치마크 MDD(Benchmark MDD)
벤치마크 MDD는 전략을 사용하지 않고 단순히 주식을 보유하고 있었을 경우의 최대 손실을 말한다.
- 만약 만약 전략을 사용했을 때의 MDD가 벤치마크 MDD와 같다고 하면 그 투자 전략은 위험 방어를 전혀 못해준 것이다.
- 투자에 따르는 위험을 고려한다면 투자 전략을 선택할 때 MDD와 벤치마크MDD의 비교가 필요하다.



> 본 내용은 <슬기로운 퀀트투자> 책을 참고하여 정리한 글입니다. <br>
[https://www.hanbit.co.kr/store/books/look.php?p_code=B7110068665](https://www.hanbit.co.kr/store/books/look.php?p_code=B7110068665)
> 







