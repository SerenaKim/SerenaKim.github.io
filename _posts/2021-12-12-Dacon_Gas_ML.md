---
layout: single
title: "DACON 가스공급량 수요예측 모델개발 LGBM, XGBoost Ensemble"
categories: Competition
---

Uploaded on 2021.12.12<br/>
Analyzed by 김나래, 홍채원, 이지혜<br/>
NMAE: 0.1446<br/>

```python
from google.colab import drive

drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive
    

**LGBM**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import datetime
```


```python
total = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/대전_한국가스공사_시간별 공급량.csv', encoding='cp949')
total
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
      <th></th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>공급량</th>
      <th>기온</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>A</td>
      <td>2497.129</td>
      <td>-8.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>A</td>
      <td>2363.265</td>
      <td>-8.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>A</td>
      <td>2258.505</td>
      <td>-8.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>A</td>
      <td>2243.969</td>
      <td>-9.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>A</td>
      <td>2344.105</td>
      <td>-9.1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>368083</th>
      <td>2018-12-31</td>
      <td>20</td>
      <td>H</td>
      <td>681.033</td>
      <td>-2.8</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>2018-12-31</td>
      <td>21</td>
      <td>H</td>
      <td>669.961</td>
      <td>-3.5</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>2018-12-31</td>
      <td>22</td>
      <td>H</td>
      <td>657.941</td>
      <td>-4.0</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>2018-12-31</td>
      <td>23</td>
      <td>H</td>
      <td>610.953</td>
      <td>-4.6</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>2018-12-31</td>
      <td>24</td>
      <td>H</td>
      <td>560.896</td>
      <td>-5.2</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 5 columns</p>
</div>




```python
total['구분'].unique()
```




    array(['A', 'B', 'C', 'D', 'E', 'G', 'H'], dtype=object)



Training Data Preprocessing


```python
d_map = {}
for i, d in enumerate(total['구분'].unique()):
    d_map[d] = i
total['구분'] = total['구분'].map(d_map)
```


```python
total['연월일'] = pd.to_datetime(total['연월일'])
total['year'] = total['연월일'].dt.year
total['month'] = total['연월일'].dt.month
total['day'] = total['연월일'].dt.day
total['weekday'] = total['연월일'].dt.weekday
```


```python
#last_gas feature added
total['last_gas'] = total.groupby('구분')['공급량'].shift(8736)

total['구분'] = total['구분'].astype('category')
total
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
      <th></th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>공급량</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>last_gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>0</td>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>0</td>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>0</td>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>0</td>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>368083</th>
      <td>2018-12-31</td>
      <td>20</td>
      <td>6</td>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>556.857</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>2018-12-31</td>
      <td>21</td>
      <td>6</td>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>556.502</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>2018-12-31</td>
      <td>22</td>
      <td>6</td>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>545.146</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>2018-12-31</td>
      <td>23</td>
      <td>6</td>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>496.522</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>2018-12-31</td>
      <td>24</td>
      <td>6</td>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>457.441</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 10 columns</p>
</div>




```python
total.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 368088 entries, 0 to 368087
    Data columns (total 10 columns):
     #   Column    Non-Null Count   Dtype         
    ---  ------    --------------   -----         
     0   연월일       368088 non-null  datetime64[ns]
     1   시간        368088 non-null  int64         
     2   구분        368088 non-null  category      
     3   공급량       368088 non-null  float64       
     4   기온        368088 non-null  float64       
     5   year      368088 non-null  int64         
     6   month     368088 non-null  int64         
     7   day       368088 non-null  int64         
     8   weekday   368088 non-null  int64         
     9   last_gas  306936 non-null  float64       
    dtypes: category(1), datetime64[ns](1), float64(3), int64(5)
    memory usage: 25.6 MB
    


```python
train_years = [2013,2014,2015,2016,2017]
val_years = [2018]
```


```python
train = total[total['year'].isin(train_years)]
val = total[total['year'].isin(val_years)]
```


```python
train
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
      <th></th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>공급량</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>last_gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>0</td>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>0</td>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>0</td>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>0</td>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>306763</th>
      <td>2017-12-31</td>
      <td>20</td>
      <td>6</td>
      <td>517.264</td>
      <td>0.1</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
      <td>6</td>
      <td>492.916</td>
    </tr>
    <tr>
      <th>306764</th>
      <td>2017-12-31</td>
      <td>21</td>
      <td>6</td>
      <td>530.896</td>
      <td>-0.6</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
      <td>6</td>
      <td>492.724</td>
    </tr>
    <tr>
      <th>306765</th>
      <td>2017-12-31</td>
      <td>22</td>
      <td>6</td>
      <td>506.287</td>
      <td>-1.0</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
      <td>6</td>
      <td>459.372</td>
    </tr>
    <tr>
      <th>306766</th>
      <td>2017-12-31</td>
      <td>23</td>
      <td>6</td>
      <td>470.638</td>
      <td>-1.1</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
      <td>6</td>
      <td>417.828</td>
    </tr>
    <tr>
      <th>306767</th>
      <td>2017-12-31</td>
      <td>24</td>
      <td>6</td>
      <td>444.618</td>
      <td>-1.3</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
      <td>6</td>
      <td>383.548</td>
    </tr>
  </tbody>
</table>
<p>306768 rows × 10 columns</p>
</div>




```python
features = ['구분', 'month', 'day', 'weekday', '시간', 'last_gas', '기온'] 
train_x = train[features]
train_y = train['공급량']

val_x = val[features]
val_y = val['공급량']
```


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()   

train_scaled = scaler.fit(train_x)

train_x = scaler.fit_transform(train_x)
val_x = scaler.fit_transform(val_x)
```


```python
print(train_x.shape, len(train_y), val_x.shape, len(val_y))
```

    (306768, 7) 306768 (61320, 7) 61320
    

Model Training


```python
d_train = lgb.Dataset(train_x, train_y)
d_val = lgb.Dataset(val_x, val_y)

params = {
    'objective': 'regression',
    'metric':'mae',
    'seed':42
}

model_LGBM = lgb.train(params, d_train, 500, d_val, verbose_eval=20, early_stopping_rounds=10)
```

    Training until validation scores don't improve for 10 rounds.
    [20]	valid_0's l1: 193.553
    [40]	valid_0's l1: 140.205
    [60]	valid_0's l1: 132.359
    [80]	valid_0's l1: 129.074
    [100]	valid_0's l1: 127.957
    [120]	valid_0's l1: 126.919
    [140]	valid_0's l1: 126.062
    [160]	valid_0's l1: 125.951
    Early stopping, best iteration is:
    [152]	valid_0's l1: 125.733
    

Variable Importance Plot


```python
lgb.plot_importance(model_LGBM, max_num_features = len(features), importance_type='split') #'gain' was also experimented
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f08d7258890>



![output_19_1](https://user-images.githubusercontent.com/50707494/145716289-811fd248-a0f5-4ea6-9aa6-466709ba6999.png)



Test Data Preprocessing


```python
test = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/test.csv')
submission = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/sample_submission.csv')
```


```python
test['일자'] = test['일자|시간|구분'].str.split(' ').str[0]
test['시간'] = test['일자|시간|구분'].str.split(' ').str[1].astype(int)
test['구분'] = test['일자|시간|구분'].str.split(' ').str[2]
```


```python
test['일자'] = pd.to_datetime(test['일자'])
test['year'] = test['일자'].dt.year
test['month'] = test['일자'].dt.month
test['day'] = test['일자'].dt.day
test['weekday'] = test['일자'].dt.weekday
```


```python
test['구분'] = test['구분'].map(d_map)
```


```python
total_drop = total.drop('last_gas',axis=1)
total_drop
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
      <th></th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>공급량</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>0</td>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>0</td>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>0</td>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>0</td>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>368083</th>
      <td>2018-12-31</td>
      <td>20</td>
      <td>6</td>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>2018-12-31</td>
      <td>21</td>
      <td>6</td>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>2018-12-31</td>
      <td>22</td>
      <td>6</td>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>2018-12-31</td>
      <td>23</td>
      <td>6</td>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>2018-12-31</td>
      <td>24</td>
      <td>6</td>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 9 columns</p>
</div>




```python
total_drop['공급량_x'] = total_drop['공급량']
total_drop = total_drop.drop('공급량',axis=1)
total_drop.rename(columns = {'공급량_x' : '공급량'}, inplace = True)
total_drop
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
      <th></th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>공급량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>-8.8</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2497.129</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>0</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2363.265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>0</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2258.505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>0</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2243.969</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>0</td>
      <td>-9.1</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2344.105</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>368083</th>
      <td>2018-12-31</td>
      <td>20</td>
      <td>6</td>
      <td>-2.8</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>681.033</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>2018-12-31</td>
      <td>21</td>
      <td>6</td>
      <td>-3.5</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>669.961</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>2018-12-31</td>
      <td>22</td>
      <td>6</td>
      <td>-4.0</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>657.941</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>2018-12-31</td>
      <td>23</td>
      <td>6</td>
      <td>-4.6</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>610.953</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>2018-12-31</td>
      <td>24</td>
      <td>6</td>
      <td>-5.2</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>560.896</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 9 columns</p>
</div>




```python
test_lag = test.drop('일자|시간|구분', axis=1)
test_lag.rename(columns = {'일자' : '연월일'}, inplace = True)
test_lag['공급량'] = ''

test_lag
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
      <th></th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>공급량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-01</td>
      <td>2</td>
      <td>0</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-01</td>
      <td>3</td>
      <td>0</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-01</td>
      <td>4</td>
      <td>0</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-01</td>
      <td>5</td>
      <td>0</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>2019-03-31</td>
      <td>20</td>
      <td>6</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
    </tr>
    <tr>
      <th>15116</th>
      <td>2019-03-31</td>
      <td>21</td>
      <td>6</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
    </tr>
    <tr>
      <th>15117</th>
      <td>2019-03-31</td>
      <td>22</td>
      <td>6</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
    </tr>
    <tr>
      <th>15118</th>
      <td>2019-03-31</td>
      <td>23</td>
      <td>6</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
    </tr>
    <tr>
      <th>15119</th>
      <td>2019-03-31</td>
      <td>24</td>
      <td>6</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 8 columns</p>
</div>




```python
test_2 = pd.concat([total_drop, test_lag], axis=0)
```


```python
test_2['last_gas'] = test_2.groupby('구분')['공급량'].shift(8736)
test_2
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
      <th></th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>공급량</th>
      <th>last_gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>-8.8</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2497.13</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>0</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2363.26</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>0</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2258.51</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>0</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2243.97</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>0</td>
      <td>-9.1</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2344.11</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>2019-03-31</td>
      <td>20</td>
      <td>6</td>
      <td>NaN</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
      <td>242.861</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>2019-03-31</td>
      <td>21</td>
      <td>6</td>
      <td>NaN</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
      <td>242.898</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>2019-03-31</td>
      <td>22</td>
      <td>6</td>
      <td>NaN</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
      <td>219.412</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>2019-03-31</td>
      <td>23</td>
      <td>6</td>
      <td>NaN</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
      <td>185.367</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>2019-03-31</td>
      <td>24</td>
      <td>6</td>
      <td>NaN</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td></td>
      <td>169.691</td>
    </tr>
  </tbody>
</table>
<p>383208 rows × 10 columns</p>
</div>




```python
test_3 = test_2[len(total_drop):]
```


```python
test_3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15120 entries, 0 to 15119
    Data columns (total 10 columns):
     #   Column    Non-Null Count  Dtype         
    ---  ------    --------------  -----         
     0   연월일       15120 non-null  datetime64[ns]
     1   시간        15120 non-null  int64         
     2   구분        15120 non-null  int64         
     3   기온        0 non-null      float64       
     4   year      15120 non-null  int64         
     5   month     15120 non-null  int64         
     6   day       15120 non-null  int64         
     7   weekday   15120 non-null  int64         
     8   공급량       15120 non-null  object        
     9   last_gas  15120 non-null  object        
    dtypes: datetime64[ns](1), float64(1), int64(6), object(2)
    memory usage: 1.3+ MB
    


```python
test_3['last_gas'] = test_3['last_gas'].astype(float)
test_3['구분'] = test_3['구분'].astype('category')
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    


```python
test_x = test_3[features]
test_x = test_x.drop(['기온'], axis = 1)
test_x
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
      <th></th>
      <th>구분</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>시간</th>
      <th>last_gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1810.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1606.232</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1527.720</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1594.027</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1798.894</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>20</td>
      <td>242.861</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>21</td>
      <td>242.898</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>22</td>
      <td>219.412</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>23</td>
      <td>185.367</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>24</td>
      <td>169.691</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 6 columns</p>
</div>




```python
avg_temp = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/대전_평균기온_for2019.csv', encoding='cp949')
avg6 = avg_temp['avg6'] #6년 기온 산술 평균 값을 사용 
avg6
```




    0        -2.0
    1        -2.3
    2        -2.6
    3        -2.8
    4        -3.2
             ... 
    15115    14.2
    15116    13.1
    15117    12.2
    15118    11.5
    15119    10.8
    Name: avg6, Length: 15120, dtype: float64




```python
test = pd.concat([test_x, avg6], axis=1)
test
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
      <th></th>
      <th>구분</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>시간</th>
      <th>last_gas</th>
      <th>avg6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1810.393</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1606.232</td>
      <td>-2.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1527.720</td>
      <td>-2.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1594.027</td>
      <td>-2.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1798.894</td>
      <td>-3.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>20</td>
      <td>242.861</td>
      <td>14.2</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>21</td>
      <td>242.898</td>
      <td>13.1</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>22</td>
      <td>219.412</td>
      <td>12.2</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>23</td>
      <td>185.367</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>6</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>24</td>
      <td>169.691</td>
      <td>10.8</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 7 columns</p>
</div>




```python
LGBM_preds = model_LGBM.predict(test)
```

**XGBoost**

Training Data Preprocessing


```python
total = pd.get_dummies(total)
total
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
      <th></th>
      <th>연월일</th>
      <th>시간</th>
      <th>공급량</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>last_gas</th>
      <th>구분_0</th>
      <th>구분_1</th>
      <th>구분_2</th>
      <th>구분_3</th>
      <th>구분_4</th>
      <th>구분_5</th>
      <th>구분_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>368083</th>
      <td>2018-12-31</td>
      <td>20</td>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>556.857</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>2018-12-31</td>
      <td>21</td>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>556.502</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>2018-12-31</td>
      <td>22</td>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>545.146</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>2018-12-31</td>
      <td>23</td>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>496.522</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>2018-12-31</td>
      <td>24</td>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>457.441</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 16 columns</p>
</div>




```python
train_years = [2013,2014,2015,2016,2017]
val_years = [2018]
```


```python
train = total[total['year'].isin(train_years)]
val = total[total['year'].isin(val_years)]
```


```python
features = ['구분_0', '구분_1', '구분_2', '구분_3', '구분_4', '구분_5', '구분_6', 'month', 'day', 'weekday', '시간', 'last_gas', '기온'] 
train_x = train[features]
train_y = train['공급량']

val_x = val[features]
val_y = val['공급량']
```


```python
val_x
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
      <th></th>
      <th>구분_0</th>
      <th>구분_1</th>
      <th>구분_2</th>
      <th>구분_3</th>
      <th>구분_4</th>
      <th>구분_5</th>
      <th>구분_6</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>시간</th>
      <th>last_gas</th>
      <th>기온</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>306768</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1446.481</td>
      <td>-2.2</td>
    </tr>
    <tr>
      <th>306769</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1239.241</td>
      <td>-2.5</td>
    </tr>
    <tr>
      <th>306770</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1151.345</td>
      <td>-2.9</td>
    </tr>
    <tr>
      <th>306771</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1206.297</td>
      <td>-3.0</td>
    </tr>
    <tr>
      <th>306772</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1402.841</td>
      <td>-3.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>368083</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>20</td>
      <td>556.857</td>
      <td>-2.8</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>21</td>
      <td>556.502</td>
      <td>-3.5</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>22</td>
      <td>545.146</td>
      <td>-4.0</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>23</td>
      <td>496.522</td>
      <td>-4.6</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>24</td>
      <td>457.441</td>
      <td>-5.2</td>
    </tr>
  </tbody>
</table>
<p>61320 rows × 13 columns</p>
</div>




```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()   

train_scaled = scaler.fit(train_x)

train_x = scaler.fit_transform(train_x)
val_x = scaler.fit_transform(val_x)
```


```python
print(train_x.shape, len(train_y), val_x.shape, len(val_y))
```

    (306768, 13) 306768 (61320, 13) 61320
    

Model Training


```python
import xgboost as xgb
from xgboost import XGBRegressor

dtrain = xgb.DMatrix(data = train_x, label = train_y)
dval = xgb.DMatrix(data = val_x, label = val_y)
wlist = [(dtrain, 'train'), (dval,'eval')]

params = {
    'learning_rate': 0.05,
    'objective': 'reg:squarederror',
    'metric':'mae', 
    'seed':42
}

model_XGB = xgb.train(params, dtrain, 1000, evals = wlist, verbose_eval = 20, early_stopping_rounds = 100)
```

    [0]	train-rmse:1242.33	eval-rmse:1368.6
    Multiple eval metrics have been passed: 'eval-rmse' will be used for early stopping.
    
    Will train until eval-rmse hasn't improved in 100 rounds.
    [20]	train-rmse:509.509	eval-rmse:616.352
    [40]	train-rmse:266.766	eval-rmse:355.714
    [60]	train-rmse:187.031	eval-rmse:264.649
    [80]	train-rmse:161.417	eval-rmse:233.834
    [100]	train-rmse:149.162	eval-rmse:221.764
    [120]	train-rmse:142.003	eval-rmse:216.119
    [140]	train-rmse:136.718	eval-rmse:213.292
    [160]	train-rmse:132.8	eval-rmse:211.327
    [180]	train-rmse:129.682	eval-rmse:210.246
    [200]	train-rmse:126.676	eval-rmse:209.399
    [220]	train-rmse:123.806	eval-rmse:208.471
    [240]	train-rmse:121.079	eval-rmse:207.699
    [260]	train-rmse:118.615	eval-rmse:206.845
    [280]	train-rmse:116.699	eval-rmse:206.16
    [300]	train-rmse:114.853	eval-rmse:205.821
    [320]	train-rmse:113.204	eval-rmse:205.35
    [340]	train-rmse:111.402	eval-rmse:205.024
    [360]	train-rmse:109.785	eval-rmse:204.573
    [380]	train-rmse:108.298	eval-rmse:204.142
    [400]	train-rmse:107.108	eval-rmse:203.81
    [420]	train-rmse:105.81	eval-rmse:203.391
    [440]	train-rmse:104.744	eval-rmse:203.022
    [460]	train-rmse:103.789	eval-rmse:202.788
    [480]	train-rmse:102.802	eval-rmse:202.601
    [500]	train-rmse:101.954	eval-rmse:202.357
    [520]	train-rmse:101.097	eval-rmse:202.184
    [540]	train-rmse:100.344	eval-rmse:202.059
    [560]	train-rmse:99.5049	eval-rmse:202.008
    [580]	train-rmse:98.7725	eval-rmse:201.953
    [600]	train-rmse:98.0917	eval-rmse:201.874
    [620]	train-rmse:97.2724	eval-rmse:201.747
    [640]	train-rmse:96.4614	eval-rmse:201.48
    [660]	train-rmse:95.7151	eval-rmse:201.361
    [680]	train-rmse:95.1486	eval-rmse:201.212
    [700]	train-rmse:94.5088	eval-rmse:201.176
    [720]	train-rmse:93.8799	eval-rmse:200.99
    [740]	train-rmse:93.2594	eval-rmse:200.944
    [760]	train-rmse:92.6099	eval-rmse:200.873
    [780]	train-rmse:92.101	eval-rmse:200.831
    [800]	train-rmse:91.6029	eval-rmse:200.747
    [820]	train-rmse:91.1129	eval-rmse:200.766
    [840]	train-rmse:90.6206	eval-rmse:200.568
    [860]	train-rmse:90.1777	eval-rmse:200.446
    [880]	train-rmse:89.6935	eval-rmse:200.245
    [900]	train-rmse:89.2628	eval-rmse:200.172
    [920]	train-rmse:88.829	eval-rmse:200.14
    [940]	train-rmse:88.4288	eval-rmse:200.095
    [960]	train-rmse:87.9761	eval-rmse:200.066
    [980]	train-rmse:87.6245	eval-rmse:200.013
    [999]	train-rmse:87.2448	eval-rmse:199.932
    


```python
features = ['구분_0', '구분_1', '구분_2', '구분_3', '구분_4', '구분_5', '구분_6', 'month', 'day', 'weekday', '시간', 'last_gas', '기온'] 
```


```python
test = pd.get_dummies(test)
test.rename(columns = {'avg6':'기온'}, inplace = True)
test
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
      <th></th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>시간</th>
      <th>last_gas</th>
      <th>기온</th>
      <th>구분_0</th>
      <th>구분_1</th>
      <th>구분_2</th>
      <th>구분_3</th>
      <th>구분_4</th>
      <th>구분_5</th>
      <th>구분_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1810.393</td>
      <td>-2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1606.232</td>
      <td>-2.3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1527.720</td>
      <td>-2.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1594.027</td>
      <td>-2.8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1798.894</td>
      <td>-3.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>20</td>
      <td>242.861</td>
      <td>14.2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>21</td>
      <td>242.898</td>
      <td>13.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>22</td>
      <td>219.412</td>
      <td>12.2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>23</td>
      <td>185.367</td>
      <td>11.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>24</td>
      <td>169.691</td>
      <td>10.8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 13 columns</p>
</div>




```python
test = test[features].values
test
```




    array([[ 1.000000e+00,  0.000000e+00,  0.000000e+00, ...,  1.000000e+00,
             1.810393e+03, -2.000000e+00],
           [ 1.000000e+00,  0.000000e+00,  0.000000e+00, ...,  2.000000e+00,
             1.606232e+03, -2.300000e+00],
           [ 1.000000e+00,  0.000000e+00,  0.000000e+00, ...,  3.000000e+00,
             1.527720e+03, -2.600000e+00],
           ...,
           [ 0.000000e+00,  0.000000e+00,  0.000000e+00, ...,  2.200000e+01,
             2.194120e+02,  1.220000e+01],
           [ 0.000000e+00,  0.000000e+00,  0.000000e+00, ...,  2.300000e+01,
             1.853670e+02,  1.150000e+01],
           [ 0.000000e+00,  0.000000e+00,  0.000000e+00, ...,  2.400000e+01,
             1.696910e+02,  1.080000e+01]])




```python
test = xgb.DMatrix(test)
```


```python
XGB_preds = model_XGB.predict(test)
```

Ensemble


```python
LGBM_preds
```




    array([3015.05266295, 3018.57737932, 3018.57737932, ..., 2454.74132532,
           2454.74132532, 2454.74132532])




```python
XGB_preds
```




    array([2547.935  , 2319.089  , 2433.9653 , ...,  200.00774,  200.00774,
            200.00774], dtype=float32)




```python
model_LGBM.save_model('model_LGBM.json')
model_XGB.save_model('model_XGB.json')
```


```python
ensembled = (LGBM_preds + XGB_preds)/2
```


```python
submission = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/sample_submission.csv')
```


```python
submission['공급량'] = ensembled
```


```python
submission.to_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/submission.csv', index=False)
```
