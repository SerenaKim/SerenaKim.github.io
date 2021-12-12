```python
from google.colab import drive

drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive
    


```python
import os
import datetime

import IPython
import IPython.display
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import LSTM 
from keras.models import Sequential 
from keras.layers import Dense 
import keras.backend as K 
from keras.callbacks import EarlyStopping

import matplotlib as mpl
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc

mpl.rcParams['figure.figsize'] = (16, 9)
mpl.rcParams['axes.grid'] = False

import warnings
warnings.filterwarnings(action='ignore')

import lightgbm as lgb
from tqdm import tqdm 
import time
```


```python
data = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/대전_한국가스공사_시간별 공급량.csv', encoding='cp949')
data
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



Training Data Preprocessing


```python
data = data.fillna(method = 'ffill')
```


```python
pd.isnull(data).sum()
```




    연월일    0
    시간     0
    구분     0
    공급량    0
    기온     0
    dtype: int64




```python
data['공급량'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7efdfc9c9110>




    
![png](output_6_1.png)
    



```python
data['구분'].unique()
```




    array(['A', 'B', 'C', 'D', 'E', 'G', 'H'], dtype=object)




```python
data['연월일'] = pd.to_datetime(data['연월일'])
data['year'] = data['연월일'].dt.year
data['month'] = data['연월일'].dt.month
data['day'] = data['연월일'].dt.day
data['weekday'] = data['연월일'].dt.weekday
```


```python
data['연월일'] = data['연월일'].astype('str')
data['시간'] = data['시간'].astype('str')
data['월일시'] = data['연월일'].str[5:] + '_' + data['시간']
```


```python
data['구분'] = data['구분'].astype('category')
data
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
      <th>월일시</th>
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
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>A</td>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>A</td>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>A</td>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>A</td>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_5</td>
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
      <td>H</td>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_20</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>2018-12-31</td>
      <td>21</td>
      <td>H</td>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_21</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>2018-12-31</td>
      <td>22</td>
      <td>H</td>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_22</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>2018-12-31</td>
      <td>23</td>
      <td>H</td>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_23</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>2018-12-31</td>
      <td>24</td>
      <td>H</td>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_24</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 10 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 368088 entries, 0 to 368087
    Data columns (total 10 columns):
     #   Column   Non-Null Count   Dtype   
    ---  ------   --------------   -----   
     0   연월일      368088 non-null  object  
     1   시간       368088 non-null  object  
     2   구분       368088 non-null  category
     3   공급량      368088 non-null  float64 
     4   기온       368088 non-null  float64 
     5   year     368088 non-null  int64   
     6   month    368088 non-null  int64   
     7   day      368088 non-null  int64   
     8   weekday  368088 non-null  int64   
     9   월일시      368088 non-null  object  
    dtypes: category(1), float64(2), int64(4), object(3)
    memory usage: 25.6+ MB
    

KNN Imputation


```python
C = data[data['구분'] == 'C']
C['공급량'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7efde26d1990>




    
![png](output_13_1.png)
    



```python
data['공급량'] = data['공급량'].apply(lambda x: np.nan if x <= 20 else x)
data
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
      <th>월일시</th>
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
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>A</td>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>A</td>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>A</td>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>A</td>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_5</td>
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
      <td>H</td>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_20</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>2018-12-31</td>
      <td>21</td>
      <td>H</td>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_21</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>2018-12-31</td>
      <td>22</td>
      <td>H</td>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_22</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>2018-12-31</td>
      <td>23</td>
      <td>H</td>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_23</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>2018-12-31</td>
      <td>24</td>
      <td>H</td>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
      <td>12-31_24</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 10 columns</p>
</div>




```python
data['공급량'].isnull().sum()
```




    3283




```python
tmp_data = data.drop(['연월일', '시간', '구분', '월일시'], axis = 1)
tmp_data
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
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
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
    </tr>
    <tr>
      <th>368083</th>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018</td>
      <td>12</td>
      <td>31</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 6 columns</p>
</div>




```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors = 5, missing_values = np.nan)
tmp_data = imputer.fit_transform(tmp_data)
tmp_data
```




    array([[ 2.497129e+03, -8.800000e+00,  2.013000e+03,  1.000000e+00,
             1.000000e+00,  1.000000e+00],
           [ 2.363265e+03, -8.500000e+00,  2.013000e+03,  1.000000e+00,
             1.000000e+00,  1.000000e+00],
           [ 2.258505e+03, -8.500000e+00,  2.013000e+03,  1.000000e+00,
             1.000000e+00,  1.000000e+00],
           ...,
           [ 6.579410e+02, -4.000000e+00,  2.018000e+03,  1.200000e+01,
             3.100000e+01,  0.000000e+00],
           [ 6.109530e+02, -4.600000e+00,  2.018000e+03,  1.200000e+01,
             3.100000e+01,  0.000000e+00],
           [ 5.608960e+02, -5.200000e+00,  2.018000e+03,  1.200000e+01,
             3.100000e+01,  0.000000e+00]])




```python
tmp_data = pd.DataFrame(tmp_data)
tmp_data.columns = ['공급량', '기온', 'year', 'month', 'day', 'weekday']
tmp_data
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
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <th>368083</th>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 6 columns</p>
</div>




```python
tmp = data.iloc[:, [0, 1, 2, -1]]
tmp
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
      <th>월일시</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>A</td>
      <td>01-01_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-01</td>
      <td>2</td>
      <td>A</td>
      <td>01-01_2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-01</td>
      <td>3</td>
      <td>A</td>
      <td>01-01_3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-01</td>
      <td>4</td>
      <td>A</td>
      <td>01-01_4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-01</td>
      <td>5</td>
      <td>A</td>
      <td>01-01_5</td>
    </tr>
    <tr>
      <th>...</th>
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
      <td>12-31_20</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>2018-12-31</td>
      <td>21</td>
      <td>H</td>
      <td>12-31_21</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>2018-12-31</td>
      <td>22</td>
      <td>H</td>
      <td>12-31_22</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>2018-12-31</td>
      <td>23</td>
      <td>H</td>
      <td>12-31_23</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>2018-12-31</td>
      <td>24</td>
      <td>H</td>
      <td>12-31_24</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 4 columns</p>
</div>




```python
data = pd.concat([tmp_data, tmp], axis = 1)
data
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
      <th>공급량</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>월일시</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>A</td>
      <td>01-01_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>2</td>
      <td>A</td>
      <td>01-01_2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>3</td>
      <td>A</td>
      <td>01-01_3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>4</td>
      <td>A</td>
      <td>01-01_4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>5</td>
      <td>A</td>
      <td>01-01_5</td>
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
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>20</td>
      <td>H</td>
      <td>12-31_20</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>21</td>
      <td>H</td>
      <td>12-31_21</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>22</td>
      <td>H</td>
      <td>12-31_22</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>23</td>
      <td>H</td>
      <td>12-31_23</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>24</td>
      <td>H</td>
      <td>12-31_24</td>
    </tr>
  </tbody>
</table>
<p>368088 rows × 10 columns</p>
</div>




```python
idx_list = []
sup_rolling_df = pd.DataFrame()

for idx in tqdm(data['구분'].unique()) :
    temp_df = data[data['구분']==idx]
    temp_df['5D_moving_avg'] = temp_df['공급량'].rolling(window=5).mean()
    temp_df['5D_moving_avg'] = temp_df['5D_moving_avg'].fillna(method='backfill')
    sup_rolling_df = sup_rolling_df.append(temp_df)
```

    100%|██████████| 7/7 [00:00<00:00, 12.42it/s]
    


```python
sup_rolling_df[sup_rolling_df['구분'] == 'E']
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
      <th>공급량</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>월일시</th>
      <th>5D_moving_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35040</th>
      <td>3272.837</td>
      <td>-8.8</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>E</td>
      <td>01-01_1</td>
      <td>3047.2642</td>
    </tr>
    <tr>
      <th>35041</th>
      <td>3057.125</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>2</td>
      <td>E</td>
      <td>01-01_2</td>
      <td>3047.2642</td>
    </tr>
    <tr>
      <th>35042</th>
      <td>2907.765</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>3</td>
      <td>E</td>
      <td>01-01_3</td>
      <td>3047.2642</td>
    </tr>
    <tr>
      <th>35043</th>
      <td>2930.789</td>
      <td>-9.0</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>4</td>
      <td>E</td>
      <td>01-01_4</td>
      <td>3047.2642</td>
    </tr>
    <tr>
      <th>35044</th>
      <td>3067.805</td>
      <td>-9.1</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>5</td>
      <td>E</td>
      <td>01-01_5</td>
      <td>3047.2642</td>
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
    </tr>
    <tr>
      <th>350563</th>
      <td>4074.485</td>
      <td>-2.8</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>20</td>
      <td>E</td>
      <td>12-31_20</td>
      <td>3625.2616</td>
    </tr>
    <tr>
      <th>350564</th>
      <td>4037.720</td>
      <td>-3.5</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>21</td>
      <td>E</td>
      <td>12-31_21</td>
      <td>3799.5736</td>
    </tr>
    <tr>
      <th>350565</th>
      <td>3954.210</td>
      <td>-4.0</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>22</td>
      <td>E</td>
      <td>12-31_22</td>
      <td>3928.8994</td>
    </tr>
    <tr>
      <th>350566</th>
      <td>3745.844</td>
      <td>-4.6</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>23</td>
      <td>E</td>
      <td>12-31_23</td>
      <td>3958.3270</td>
    </tr>
    <tr>
      <th>350567</th>
      <td>3534.260</td>
      <td>-5.2</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>24</td>
      <td>E</td>
      <td>12-31_24</td>
      <td>3869.3038</td>
    </tr>
  </tbody>
</table>
<p>52584 rows × 11 columns</p>
</div>




```python
len(sup_rolling_df) == len(data)
```




    True




```python
sup_rolling_df.describe().transpose()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>공급량</th>
      <td>368088.0</td>
      <td>953.007769</td>
      <td>924.101466</td>
      <td>20.036</td>
      <td>230.9995</td>
      <td>642.8610</td>
      <td>1399.40475</td>
      <td>11593.6170</td>
    </tr>
    <tr>
      <th>기온</th>
      <td>368088.0</td>
      <td>13.620166</td>
      <td>10.878104</td>
      <td>-16.900</td>
      <td>4.4000</td>
      <td>14.5000</td>
      <td>22.80000</td>
      <td>39.3000</td>
    </tr>
    <tr>
      <th>year</th>
      <td>368088.0</td>
      <td>2015.500228</td>
      <td>1.707471</td>
      <td>2013.000</td>
      <td>2014.0000</td>
      <td>2016.0000</td>
      <td>2017.00000</td>
      <td>2018.0000</td>
    </tr>
    <tr>
      <th>month</th>
      <td>368088.0</td>
      <td>6.523962</td>
      <td>3.448424</td>
      <td>1.000</td>
      <td>4.0000</td>
      <td>7.0000</td>
      <td>10.00000</td>
      <td>12.0000</td>
    </tr>
    <tr>
      <th>day</th>
      <td>368088.0</td>
      <td>15.726609</td>
      <td>8.798824</td>
      <td>1.000</td>
      <td>8.0000</td>
      <td>16.0000</td>
      <td>23.00000</td>
      <td>31.0000</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>368088.0</td>
      <td>3.000000</td>
      <td>2.000003</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>3.0000</td>
      <td>5.00000</td>
      <td>6.0000</td>
    </tr>
    <tr>
      <th>5D_moving_avg</th>
      <td>368088.0</td>
      <td>952.991405</td>
      <td>911.436006</td>
      <td>27.618</td>
      <td>243.1611</td>
      <td>641.9988</td>
      <td>1400.47605</td>
      <td>6452.7474</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.get_dummies(sup_rolling_df, columns = ['구분'])
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
      <th></th>
      <th>공급량</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>연월일</th>
      <th>시간</th>
      <th>월일시</th>
      <th>5D_moving_avg</th>
      <th>구분_A</th>
      <th>구분_B</th>
      <th>구분_C</th>
      <th>구분_D</th>
      <th>구분_E</th>
      <th>구분_G</th>
      <th>구분_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>01-01_1</td>
      <td>2341.3946</td>
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
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>2</td>
      <td>01-01_2</td>
      <td>2341.3946</td>
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
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>3</td>
      <td>01-01_3</td>
      <td>2341.3946</td>
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
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>4</td>
      <td>01-01_4</td>
      <td>2341.3946</td>
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
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>5</td>
      <td>01-01_5</td>
      <td>2341.3946</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>368083</th>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>20</td>
      <td>12-31_20</td>
      <td>604.7030</td>
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
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>21</td>
      <td>12-31_21</td>
      <td>635.0934</td>
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
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>22</td>
      <td>12-31_22</td>
      <td>658.2096</td>
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
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>23</td>
      <td>12-31_23</td>
      <td>659.7726</td>
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
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>24</td>
      <td>12-31_24</td>
      <td>636.1568</td>
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
<p>368088 rows × 17 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 368088 entries, 0 to 368087
    Data columns (total 17 columns):
     #   Column         Non-Null Count   Dtype  
    ---  ------         --------------   -----  
     0   공급량            368088 non-null  float64
     1   기온             368088 non-null  float64
     2   year           368088 non-null  float64
     3   month          368088 non-null  float64
     4   day            368088 non-null  float64
     5   weekday        368088 non-null  float64
     6   연월일            368088 non-null  object 
     7   시간             368088 non-null  object 
     8   월일시            368088 non-null  object 
     9   5D_moving_avg  368088 non-null  float64
     10  구분_A           368088 non-null  uint8  
     11  구분_B           368088 non-null  uint8  
     12  구분_C           368088 non-null  uint8  
     13  구분_D           368088 non-null  uint8  
     14  구분_E           368088 non-null  uint8  
     15  구분_G           368088 non-null  uint8  
     16  구분_H           368088 non-null  uint8  
    dtypes: float64(7), object(3), uint8(7)
    memory usage: 33.3+ MB
    


```python
features = ['시간', 'year', 'month', 'day', 'weekday', '구분_A', '구분_B','구분_C', '구분_D', '구분_E', '구분_G', '구분_H', '공급량', '기온', '5D_moving_avg']
X = df[features]
```


```python
num = ['5D_moving_avg', '기온']
cat = ['시간', 'year', 'month', 'day', 'weekday', '구분_A', '구분_B','구분_C', '구분_D', '구분_E', '구분_G', '구분_H']
X[cat] = X[cat].astype(str)
```


```python
train_years = ['2013.0', '2014.0','2015.0','2016.0','2017.0']
val_years = ['2018.0']
train = X[X['year'].isin(train_years)]
val = X[X['year'].isin(val_years)]
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
      <th>시간</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>구분_A</th>
      <th>구분_B</th>
      <th>구분_C</th>
      <th>구분_D</th>
      <th>구분_E</th>
      <th>구분_G</th>
      <th>구분_H</th>
      <th>공급량</th>
      <th>기온</th>
      <th>5D_moving_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2497.129</td>
      <td>-8.8</td>
      <td>2341.3946</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2363.265</td>
      <td>-8.5</td>
      <td>2341.3946</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2258.505</td>
      <td>-8.5</td>
      <td>2341.3946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2243.969</td>
      <td>-9.0</td>
      <td>2341.3946</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2344.105</td>
      <td>-9.1</td>
      <td>2341.3946</td>
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
    </tr>
    <tr>
      <th>306763</th>
      <td>20</td>
      <td>2017.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>517.264</td>
      <td>0.1</td>
      <td>465.8186</td>
    </tr>
    <tr>
      <th>306764</th>
      <td>21</td>
      <td>2017.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>530.896</td>
      <td>-0.6</td>
      <td>492.7878</td>
    </tr>
    <tr>
      <th>306765</th>
      <td>22</td>
      <td>2017.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>506.287</td>
      <td>-1.0</td>
      <td>510.1782</td>
    </tr>
    <tr>
      <th>306766</th>
      <td>23</td>
      <td>2017.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>470.638</td>
      <td>-1.1</td>
      <td>509.1778</td>
    </tr>
    <tr>
      <th>306767</th>
      <td>24</td>
      <td>2017.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>444.618</td>
      <td>-1.3</td>
      <td>493.9406</td>
    </tr>
  </tbody>
</table>
<p>306768 rows × 15 columns</p>
</div>




```python
X_train = train.drop(['공급량', 'year'],axis=1)
y_train = train['공급량']

X_val = val.drop(['공급량', 'year'],axis=1)
y_val = val['공급량']
```


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()   

train_scaled = scaler.fit(X_train)

X_train_ = scaler.fit_transform(X_train)
X_val_ = scaler.fit_transform(X_val)
```


```python
print(X_train_.shape, len(y_train), X_val_.shape, len(y_val))
```

    (306768, 13) 306768 (61320, 13) 61320
    


```python
# input reshape 3 dim
X_train_t = X_train_.reshape(X_train_.shape[0], X_train_.shape[1], 1)
X_val_t = X_val_.reshape(X_val_.shape[0], X_val_.shape[1], 1)

print("최종 DATA")
print(X_train_t.shape, X_val_t.shape)
print(y_train)
```

    최종 DATA
    (306768, 13, 1) (61320, 13, 1)
    0         2497.129
    1         2363.265
    2         2258.505
    3         2243.969
    4         2344.105
                ...   
    306763     517.264
    306764     530.896
    306765     506.287
    306766     470.638
    306767     444.618
    Name: 공급량, Length: 306768, dtype: float64
    

**LSTM**


```python
K.clear_session()
    
model = Sequential() # Sequeatial Model 
model.add(LSTM(20, input_shape=(13, 1))) # (timestep, feature) 
model.add(Dense(1)) # output = 1 
model.compile(loss='mean_squared_error', optimizer='adam') 
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm (LSTM)                 (None, 20)                1760      
                                                                     
     dense (Dense)               (None, 1)                 21        
                                                                     
    =================================================================
    Total params: 1,781
    Trainable params: 1,781
    Non-trainable params: 0
    _________________________________________________________________
    


```python
EPOCH = 100
BATCH_SIZE = 32
```


```python
from keras.callbacks import ModelCheckpoint

filename = 'LSTM_checkpoint-trial-002.h5'
checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             monitor='val_loss', # val_loss 값이 개선되었을때 호출됩니다
                             patience=5,
                             verbose=1,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                             mode='auto',           # auto는 알아서 best를 찾습니다. min/max
                             restore_best_weights = True
                            )

history = model.fit(X_train_t, y_train, 
      validation_data=(X_val_t,y_val),
      epochs=100, 
      batch_size=32,
          verbose=1,
      callbacks=[checkpoint] # checkpoint 콜백
     )
```

    Epoch 1/100
    9587/9587 [==============================] - ETA: 0s - loss: 1519749.6250
    Epoch 00001: val_loss improved from inf to 1680573.62500, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 61s 6ms/step - loss: 1519749.6250 - val_loss: 1680573.6250
    Epoch 2/100
    9587/9587 [==============================] - ETA: 0s - loss: 1236409.7500
    Epoch 00002: val_loss improved from 1680573.62500 to 1405597.87500, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 55s 6ms/step - loss: 1236409.7500 - val_loss: 1405597.8750
    Epoch 3/100
    9585/9587 [============================>.] - ETA: 0s - loss: 1004742.6875
    Epoch 00003: val_loss improved from 1405597.87500 to 1151399.25000, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 1004673.5000 - val_loss: 1151399.2500
    Epoch 4/100
    9579/9587 [============================>.] - ETA: 0s - loss: 806808.4375
    Epoch 00004: val_loss improved from 1151399.25000 to 953864.81250, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 806617.7500 - val_loss: 953864.8125
    Epoch 5/100
    9579/9587 [============================>.] - ETA: 0s - loss: 650603.3125
    Epoch 00005: val_loss improved from 953864.81250 to 784279.31250, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 650560.9375 - val_loss: 784279.3125
    Epoch 6/100
    9586/9587 [============================>.] - ETA: 0s - loss: 519025.9688
    Epoch 00006: val_loss improved from 784279.31250 to 645890.62500, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 519023.6250 - val_loss: 645890.6250
    Epoch 7/100
    9580/9587 [============================>.] - ETA: 0s - loss: 414745.2812
    Epoch 00007: val_loss improved from 645890.62500 to 531672.75000, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 414656.6562 - val_loss: 531672.7500
    Epoch 8/100
    9586/9587 [============================>.] - ETA: 0s - loss: 330691.4688
    Epoch 00008: val_loss improved from 531672.75000 to 437395.28125, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 330699.9375 - val_loss: 437395.2812
    Epoch 9/100
    9583/9587 [============================>.] - ETA: 0s - loss: 262617.9375
    Epoch 00009: val_loss improved from 437395.28125 to 360491.81250, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 262586.7188 - val_loss: 360491.8125
    Epoch 10/100
    9586/9587 [============================>.] - ETA: 0s - loss: 209210.2031
    Epoch 00010: val_loss improved from 360491.81250 to 299714.46875, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 209210.6719 - val_loss: 299714.4688
    Epoch 11/100
    9581/9587 [============================>.] - ETA: 0s - loss: 167538.3906
    Epoch 00011: val_loss improved from 299714.46875 to 250368.89062, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 167490.2500 - val_loss: 250368.8906
    Epoch 12/100
    9585/9587 [============================>.] - ETA: 0s - loss: 135339.2188
    Epoch 00012: val_loss improved from 250368.89062 to 210826.70312, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 135341.2812 - val_loss: 210826.7031
    Epoch 13/100
    9582/9587 [============================>.] - ETA: 0s - loss: 110525.7656
    Epoch 00013: val_loss improved from 210826.70312 to 178603.01562, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 61s 6ms/step - loss: 110492.0234 - val_loss: 178603.0156
    Epoch 14/100
    9583/9587 [============================>.] - ETA: 0s - loss: 88522.1016
    Epoch 00014: val_loss improved from 178603.01562 to 148309.87500, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 61s 6ms/step - loss: 88558.5938 - val_loss: 148309.8750
    Epoch 15/100
    9581/9587 [============================>.] - ETA: 0s - loss: 70061.9922
    Epoch 00015: val_loss improved from 148309.87500 to 125266.96094, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 70058.4219 - val_loss: 125266.9609
    Epoch 16/100
    9581/9587 [============================>.] - ETA: 0s - loss: 57551.9062
    Epoch 00016: val_loss improved from 125266.96094 to 108270.10156, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 61s 6ms/step - loss: 57555.4297 - val_loss: 108270.1016
    Epoch 17/100
    9581/9587 [============================>.] - ETA: 0s - loss: 47949.0312
    Epoch 00017: val_loss improved from 108270.10156 to 94426.67188, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 61s 6ms/step - loss: 47928.6484 - val_loss: 94426.6719
    Epoch 18/100
    9585/9587 [============================>.] - ETA: 0s - loss: 39998.4258
    Epoch 00018: val_loss improved from 94426.67188 to 83409.39062, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 39993.6641 - val_loss: 83409.3906
    Epoch 19/100
    9586/9587 [============================>.] - ETA: 0s - loss: 33413.8945
    Epoch 00019: val_loss improved from 83409.39062 to 73949.04688, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 33412.3906 - val_loss: 73949.0469
    Epoch 20/100
    9585/9587 [============================>.] - ETA: 0s - loss: 28125.0449
    Epoch 00020: val_loss improved from 73949.04688 to 66104.28125, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 28134.4980 - val_loss: 66104.2812
    Epoch 21/100
    9579/9587 [============================>.] - ETA: 0s - loss: 24016.0293
    Epoch 00021: val_loss improved from 66104.28125 to 59915.57422, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 24009.1191 - val_loss: 59915.5742
    Epoch 22/100
    9579/9587 [============================>.] - ETA: 0s - loss: 20829.9648
    Epoch 00022: val_loss improved from 59915.57422 to 54804.23047, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 20834.6191 - val_loss: 54804.2305
    Epoch 23/100
    9584/9587 [============================>.] - ETA: 0s - loss: 18139.0898
    Epoch 00023: val_loss improved from 54804.23047 to 51098.08594, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 18140.7695 - val_loss: 51098.0859
    Epoch 24/100
    9587/9587 [==============================] - ETA: 0s - loss: 16039.4785
    Epoch 00024: val_loss improved from 51098.08594 to 46444.12500, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 16039.4785 - val_loss: 46444.1250
    Epoch 25/100
    9583/9587 [============================>.] - ETA: 0s - loss: 14321.2822
    Epoch 00025: val_loss improved from 46444.12500 to 43523.82812, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 14318.1934 - val_loss: 43523.8281
    Epoch 26/100
    9585/9587 [============================>.] - ETA: 0s - loss: 12834.6904
    Epoch 00026: val_loss improved from 43523.82812 to 39235.07812, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 12835.9082 - val_loss: 39235.0781
    Epoch 27/100
    9585/9587 [============================>.] - ETA: 0s - loss: 11630.3867
    Epoch 00027: val_loss improved from 39235.07812 to 38925.64062, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 11629.1084 - val_loss: 38925.6406
    Epoch 28/100
    9586/9587 [============================>.] - ETA: 0s - loss: 10653.1943
    Epoch 00028: val_loss improved from 38925.64062 to 35335.56641, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 10652.8486 - val_loss: 35335.5664
    Epoch 29/100
    9586/9587 [============================>.] - ETA: 0s - loss: 9810.1953
    Epoch 00029: val_loss improved from 35335.56641 to 34120.52344, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 9809.8047 - val_loss: 34120.5234
    Epoch 30/100
    9579/9587 [============================>.] - ETA: 0s - loss: 9031.7256
    Epoch 00030: val_loss improved from 34120.52344 to 33507.38672, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 9033.6943 - val_loss: 33507.3867
    Epoch 31/100
    9580/9587 [============================>.] - ETA: 0s - loss: 8447.9453
    Epoch 00031: val_loss improved from 33507.38672 to 32039.53516, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 8444.2227 - val_loss: 32039.5352
    Epoch 32/100
    9581/9587 [============================>.] - ETA: 0s - loss: 7899.3418
    Epoch 00032: val_loss improved from 32039.53516 to 30988.13281, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 7899.9570 - val_loss: 30988.1328
    Epoch 33/100
    9582/9587 [============================>.] - ETA: 0s - loss: 7443.9707
    Epoch 00033: val_loss improved from 30988.13281 to 30590.21875, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 7442.3906 - val_loss: 30590.2188
    Epoch 34/100
    9583/9587 [============================>.] - ETA: 0s - loss: 7034.0513
    Epoch 00034: val_loss improved from 30590.21875 to 29890.64648, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 7032.9434 - val_loss: 29890.6465
    Epoch 35/100
    9586/9587 [============================>.] - ETA: 0s - loss: 6703.4585
    Epoch 00035: val_loss improved from 29890.64648 to 28965.45312, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 6703.3428 - val_loss: 28965.4531
    Epoch 36/100
    9586/9587 [============================>.] - ETA: 0s - loss: 6382.2661
    Epoch 00036: val_loss did not improve from 28965.45312
    9587/9587 [==============================] - 56s 6ms/step - loss: 6382.0024 - val_loss: 29405.1445
    Epoch 37/100
    9581/9587 [============================>.] - ETA: 0s - loss: 6130.5811
    Epoch 00037: val_loss improved from 28965.45312 to 25998.05078, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 6128.6587 - val_loss: 25998.0508
    Epoch 38/100
    9587/9587 [==============================] - ETA: 0s - loss: 5893.2285
    Epoch 00038: val_loss did not improve from 25998.05078
    9587/9587 [==============================] - 56s 6ms/step - loss: 5893.2285 - val_loss: 26075.6621
    Epoch 39/100
    9584/9587 [============================>.] - ETA: 0s - loss: 5673.3091
    Epoch 00039: val_loss did not improve from 25998.05078
    9587/9587 [==============================] - 56s 6ms/step - loss: 5672.7817 - val_loss: 26573.2422
    Epoch 40/100
    9581/9587 [============================>.] - ETA: 0s - loss: 5484.4653
    Epoch 00040: val_loss improved from 25998.05078 to 25206.97070, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 5483.2988 - val_loss: 25206.9707
    Epoch 41/100
    9585/9587 [============================>.] - ETA: 0s - loss: 5325.0962
    Epoch 00041: val_loss improved from 25206.97070 to 24399.54297, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 5324.9556 - val_loss: 24399.5430
    Epoch 42/100
    9582/9587 [============================>.] - ETA: 0s - loss: 5177.4766
    Epoch 00042: val_loss did not improve from 24399.54297
    9587/9587 [==============================] - 56s 6ms/step - loss: 5176.2031 - val_loss: 26351.4883
    Epoch 43/100
    9586/9587 [============================>.] - ETA: 0s - loss: 5022.5396
    Epoch 00043: val_loss improved from 24399.54297 to 22714.05859, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 5022.6592 - val_loss: 22714.0586
    Epoch 44/100
    9581/9587 [============================>.] - ETA: 0s - loss: 4895.4976
    Epoch 00044: val_loss did not improve from 22714.05859
    9587/9587 [==============================] - 57s 6ms/step - loss: 4894.1113 - val_loss: 24632.9355
    Epoch 45/100
    9581/9587 [============================>.] - ETA: 0s - loss: 4788.2275
    Epoch 00045: val_loss did not improve from 22714.05859
    9587/9587 [==============================] - 56s 6ms/step - loss: 4793.1035 - val_loss: 23129.3086
    Epoch 46/100
    9586/9587 [============================>.] - ETA: 0s - loss: 4668.2300
    Epoch 00046: val_loss did not improve from 22714.05859
    9587/9587 [==============================] - 56s 6ms/step - loss: 4668.1177 - val_loss: 24495.5078
    Epoch 47/100
    9579/9587 [============================>.] - ETA: 0s - loss: 4567.7505
    Epoch 00047: val_loss did not improve from 22714.05859
    9587/9587 [==============================] - 56s 6ms/step - loss: 4569.3467 - val_loss: 24905.0684
    Epoch 48/100
    9581/9587 [============================>.] - ETA: 0s - loss: 4492.8242
    Epoch 00048: val_loss did not improve from 22714.05859
    9587/9587 [==============================] - 56s 6ms/step - loss: 4491.7329 - val_loss: 23945.8613
    Epoch 49/100
    9580/9587 [============================>.] - ETA: 0s - loss: 4410.2134
    Epoch 00049: val_loss did not improve from 22714.05859
    9587/9587 [==============================] - 56s 6ms/step - loss: 4411.4985 - val_loss: 24717.3242
    Epoch 50/100
    9582/9587 [============================>.] - ETA: 0s - loss: 4332.2769
    Epoch 00050: val_loss improved from 22714.05859 to 22340.17188, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 4335.4736 - val_loss: 22340.1719
    Epoch 51/100
    9582/9587 [============================>.] - ETA: 0s - loss: 4251.9521
    Epoch 00051: val_loss did not improve from 22340.17188
    9587/9587 [==============================] - 56s 6ms/step - loss: 4251.1953 - val_loss: 23544.6797
    Epoch 52/100
    9580/9587 [============================>.] - ETA: 0s - loss: 4196.9263
    Epoch 00052: val_loss did not improve from 22340.17188
    9587/9587 [==============================] - 56s 6ms/step - loss: 4195.5527 - val_loss: 22988.6504
    Epoch 53/100
    9586/9587 [============================>.] - ETA: 0s - loss: 4132.7070
    Epoch 00053: val_loss did not improve from 22340.17188
    9587/9587 [==============================] - 57s 6ms/step - loss: 4132.5952 - val_loss: 23269.0957
    Epoch 54/100
    9583/9587 [============================>.] - ETA: 0s - loss: 4111.8291
    Epoch 00054: val_loss did not improve from 22340.17188
    9587/9587 [==============================] - 57s 6ms/step - loss: 4111.2739 - val_loss: 22669.5430
    Epoch 55/100
    9585/9587 [============================>.] - ETA: 0s - loss: 4004.4285
    Epoch 00055: val_loss did not improve from 22340.17188
    9587/9587 [==============================] - 56s 6ms/step - loss: 4004.2798 - val_loss: 22991.5820
    Epoch 56/100
    9579/9587 [============================>.] - ETA: 0s - loss: 3988.9204
    Epoch 00056: val_loss improved from 22340.17188 to 21321.96680, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 57s 6ms/step - loss: 3987.3826 - val_loss: 21321.9668
    Epoch 57/100
    9581/9587 [============================>.] - ETA: 0s - loss: 3925.3484
    Epoch 00057: val_loss did not improve from 21321.96680
    9587/9587 [==============================] - 56s 6ms/step - loss: 3924.6506 - val_loss: 21665.1094
    Epoch 58/100
    9584/9587 [============================>.] - ETA: 0s - loss: 3891.1877
    Epoch 00058: val_loss did not improve from 21321.96680
    9587/9587 [==============================] - 57s 6ms/step - loss: 3890.8235 - val_loss: 21441.8320
    Epoch 59/100
    9585/9587 [============================>.] - ETA: 0s - loss: 3861.7974
    Epoch 00059: val_loss did not improve from 21321.96680
    9587/9587 [==============================] - 56s 6ms/step - loss: 3861.5610 - val_loss: 22074.0879
    Epoch 60/100
    9582/9587 [============================>.] - ETA: 0s - loss: 3809.5684
    Epoch 00060: val_loss did not improve from 21321.96680
    9587/9587 [==============================] - 57s 6ms/step - loss: 3808.7197 - val_loss: 21709.3809
    Epoch 61/100
    9580/9587 [============================>.] - ETA: 0s - loss: 3794.9854
    Epoch 00061: val_loss improved from 21321.96680 to 21246.72852, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 57s 6ms/step - loss: 3794.4104 - val_loss: 21246.7285
    Epoch 62/100
    9580/9587 [============================>.] - ETA: 0s - loss: 3754.7917
    Epoch 00062: val_loss improved from 21246.72852 to 20655.85156, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 56s 6ms/step - loss: 3753.4954 - val_loss: 20655.8516
    Epoch 63/100
    9580/9587 [============================>.] - ETA: 0s - loss: 3716.0923
    Epoch 00063: val_loss did not improve from 20655.85156
    9587/9587 [==============================] - 57s 6ms/step - loss: 3716.7493 - val_loss: 24069.9805
    Epoch 64/100
    9586/9587 [============================>.] - ETA: 0s - loss: 3692.6848
    Epoch 00064: val_loss did not improve from 20655.85156
    9587/9587 [==============================] - 57s 6ms/step - loss: 3692.5657 - val_loss: 20794.3008
    Epoch 65/100
    9581/9587 [============================>.] - ETA: 0s - loss: 3666.5840
    Epoch 00065: val_loss did not improve from 20655.85156
    9587/9587 [==============================] - 57s 6ms/step - loss: 3666.9341 - val_loss: 21261.3027
    Epoch 66/100
    9586/9587 [============================>.] - ETA: 0s - loss: 3642.6943
    Epoch 00066: val_loss did not improve from 20655.85156
    9587/9587 [==============================] - 57s 6ms/step - loss: 3642.5837 - val_loss: 23935.9121
    Epoch 67/100
    9587/9587 [==============================] - ETA: 0s - loss: 3613.9834
    Epoch 00067: val_loss improved from 20655.85156 to 20013.79492, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 57s 6ms/step - loss: 3613.9834 - val_loss: 20013.7949
    Epoch 68/100
    9582/9587 [============================>.] - ETA: 0s - loss: 3595.3779
    Epoch 00068: val_loss did not improve from 20013.79492
    9587/9587 [==============================] - 57s 6ms/step - loss: 3595.3042 - val_loss: 20042.8867
    Epoch 69/100
    9587/9587 [==============================] - ETA: 0s - loss: 3555.3618
    Epoch 00069: val_loss did not improve from 20013.79492
    9587/9587 [==============================] - 57s 6ms/step - loss: 3555.3618 - val_loss: 21728.9199
    Epoch 70/100
    9585/9587 [============================>.] - ETA: 0s - loss: 3549.0178
    Epoch 00070: val_loss improved from 20013.79492 to 18985.95117, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 57s 6ms/step - loss: 3548.7358 - val_loss: 18985.9512
    Epoch 71/100
    9584/9587 [============================>.] - ETA: 0s - loss: 3518.3105
    Epoch 00071: val_loss did not improve from 18985.95117
    9587/9587 [==============================] - 57s 6ms/step - loss: 3518.0146 - val_loss: 20626.4648
    Epoch 72/100
    9583/9587 [============================>.] - ETA: 0s - loss: 3488.8738
    Epoch 00072: val_loss did not improve from 18985.95117
    9587/9587 [==============================] - 57s 6ms/step - loss: 3491.5232 - val_loss: 21598.0156
    Epoch 73/100
    9586/9587 [============================>.] - ETA: 0s - loss: 3476.2300
    Epoch 00073: val_loss did not improve from 18985.95117
    9587/9587 [==============================] - 57s 6ms/step - loss: 3476.2559 - val_loss: 20173.1504
    Epoch 74/100
    9579/9587 [============================>.] - ETA: 0s - loss: 3461.8093
    Epoch 00074: val_loss did not improve from 18985.95117
    9587/9587 [==============================] - 57s 6ms/step - loss: 3461.0491 - val_loss: 19984.6152
    Epoch 75/100
    9585/9587 [============================>.] - ETA: 0s - loss: 3452.4875
    Epoch 00075: val_loss did not improve from 18985.95117
    9587/9587 [==============================] - 57s 6ms/step - loss: 3452.3291 - val_loss: 19915.4551
    Epoch 76/100
    9579/9587 [============================>.] - ETA: 0s - loss: 3409.8894
    Epoch 00076: val_loss did not improve from 18985.95117
    9587/9587 [==============================] - 57s 6ms/step - loss: 3411.1304 - val_loss: 21558.8477
    Epoch 77/100
    9579/9587 [============================>.] - ETA: 0s - loss: 3392.8237
    Epoch 00077: val_loss did not improve from 18985.95117
    9587/9587 [==============================] - 57s 6ms/step - loss: 3391.7202 - val_loss: 20962.8516
    Epoch 78/100
    9580/9587 [============================>.] - ETA: 0s - loss: 3397.5107
    Epoch 00078: val_loss did not improve from 18985.95117
    9587/9587 [==============================] - 57s 6ms/step - loss: 3396.9253 - val_loss: 20059.4336
    Epoch 79/100
    9583/9587 [============================>.] - ETA: 0s - loss: 3380.1257
    Epoch 00079: val_loss improved from 18985.95117 to 18218.00000, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 57s 6ms/step - loss: 3379.5393 - val_loss: 18218.0000
    Epoch 80/100
    9580/9587 [============================>.] - ETA: 0s - loss: 3366.0547
    Epoch 00080: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 57s 6ms/step - loss: 3364.8020 - val_loss: 20242.5820
    Epoch 81/100
    9586/9587 [============================>.] - ETA: 0s - loss: 3340.5796
    Epoch 00081: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 57s 6ms/step - loss: 3341.4004 - val_loss: 20945.5527
    Epoch 82/100
    9586/9587 [============================>.] - ETA: 0s - loss: 3336.1665
    Epoch 00082: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 57s 6ms/step - loss: 3336.2222 - val_loss: 21270.9707
    Epoch 83/100
    9587/9587 [==============================] - ETA: 0s - loss: 3317.3491
    Epoch 00083: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 58s 6ms/step - loss: 3317.3491 - val_loss: 20853.0840
    Epoch 84/100
    9583/9587 [============================>.] - ETA: 0s - loss: 3304.6118
    Epoch 00084: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 58s 6ms/step - loss: 3304.1086 - val_loss: 20063.0488
    Epoch 85/100
    9582/9587 [============================>.] - ETA: 0s - loss: 3290.3066
    Epoch 00085: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 58s 6ms/step - loss: 3289.8911 - val_loss: 20559.7520
    Epoch 86/100
    9579/9587 [============================>.] - ETA: 0s - loss: 3299.2407
    Epoch 00086: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 58s 6ms/step - loss: 3298.2905 - val_loss: 22290.5449
    Epoch 87/100
    9583/9587 [============================>.] - ETA: 0s - loss: 3263.4446
    Epoch 00087: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 58s 6ms/step - loss: 3262.9719 - val_loss: 19738.8574
    Epoch 88/100
    9586/9587 [============================>.] - ETA: 0s - loss: 3260.0859
    Epoch 00088: val_loss did not improve from 18218.00000
    9587/9587 [==============================] - 58s 6ms/step - loss: 3259.9578 - val_loss: 19521.3281
    Epoch 89/100
    9586/9587 [============================>.] - ETA: 0s - loss: 3251.4138
    Epoch 00089: val_loss improved from 18218.00000 to 15410.90527, saving model to LSTM_checkpoint-trial-002.h5
    9587/9587 [==============================] - 58s 6ms/step - loss: 3251.3735 - val_loss: 15410.9053
    Epoch 90/100
    9585/9587 [============================>.] - ETA: 0s - loss: 3232.1782
    Epoch 00090: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 58s 6ms/step - loss: 3233.0234 - val_loss: 21497.7051
    Epoch 91/100
    9580/9587 [============================>.] - ETA: 0s - loss: 3234.0498
    Epoch 00091: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 59s 6ms/step - loss: 3232.8406 - val_loss: 20225.1035
    Epoch 92/100
    9586/9587 [============================>.] - ETA: 0s - loss: 3213.2058
    Epoch 00092: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 58s 6ms/step - loss: 3213.1079 - val_loss: 20446.8418
    Epoch 93/100
    9583/9587 [============================>.] - ETA: 0s - loss: 3197.4075
    Epoch 00093: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 58s 6ms/step - loss: 3196.7568 - val_loss: 19125.7637
    Epoch 94/100
    9584/9587 [============================>.] - ETA: 0s - loss: 3199.1919
    Epoch 00094: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 58s 6ms/step - loss: 3198.9512 - val_loss: 20297.9219
    Epoch 95/100
    9584/9587 [============================>.] - ETA: 0s - loss: 3174.4500
    Epoch 00095: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 58s 6ms/step - loss: 3174.5908 - val_loss: 21598.9473
    Epoch 96/100
    9583/9587 [============================>.] - ETA: 0s - loss: 3180.9241
    Epoch 00096: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 59s 6ms/step - loss: 3180.6731 - val_loss: 20007.7910
    Epoch 97/100
    9583/9587 [============================>.] - ETA: 0s - loss: 3179.5820
    Epoch 00097: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 59s 6ms/step - loss: 3179.1116 - val_loss: 19714.3867
    Epoch 98/100
    9584/9587 [============================>.] - ETA: 0s - loss: 3165.2510
    Epoch 00098: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 59s 6ms/step - loss: 3165.1216 - val_loss: 19943.8984
    Epoch 99/100
    9584/9587 [============================>.] - ETA: 0s - loss: 3149.3623
    Epoch 00099: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 64s 7ms/step - loss: 3149.1067 - val_loss: 23984.8457
    Epoch 100/100
    9584/9587 [============================>.] - ETA: 0s - loss: 3158.6492
    Epoch 00100: val_loss did not improve from 15410.90527
    9587/9587 [==============================] - 59s 6ms/step - loss: 3158.0696 - val_loss: 20575.5117
    


```python
# loss check
def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')

plot_loss (history, 'LSTM')
```


    
![png](output_40_0.png)
    



```python
model.save('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/LSTM_checkpoint-trial-002.h5')
```


```python
from tensorflow.python.keras.models import load_model

model_LSTM = load_model("LSTM_checkpoint-trial-002.h5")
```


```python
# validatation set prediction
y_pred_ = model.predict(X_val_t)
```


```python
plt.scatter(y_val, y_pred_)
plt.xlabel("TRUE: $Y_i$")
plt.ylabel("Predict: $\hat{Y}_i$")
```




    Text(0, 0.5, 'Predict: $\\hat{Y}_i$')




    
![png](output_44_1.png)
    


METRIC


```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_val, y_pred_)
rmse = mse**0.5
r_2 = r2_score(y_val, y_pred_)

print(mse)
print(rmse)
print(r_2)
```

    20575.496025546145
    143.44161190375038
    0.9796009829105287
    

**1D CNN**


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers

# CNN 
K.clear_session()

model = Sequential()
model.add(layers.Conv1D(32, 2, activation='relu', input_shape=(X_train_t.shape[1],X_train_t.shape[2]))) 
model.add(layers.Conv1D(32, 2, activation='relu'))
model.add(layers.Conv1D(32, 2, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(32, 2, activation='relu'))
model.add(layers.Conv1D(32, 2, activation='relu'))
# model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(16, 1, activation='relu'))
model.add(layers.GlobalMaxPooling1D()) 
model.add(layers.Dense(1))

adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae']) 
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv1d (Conv1D)             (None, 12, 32)            96        
                                                                     
     conv1d_1 (Conv1D)           (None, 11, 32)            2080      
                                                                     
     conv1d_2 (Conv1D)           (None, 10, 32)            2080      
                                                                     
     max_pooling1d (MaxPooling1D  (None, 5, 32)            0         
     )                                                               
                                                                     
     conv1d_3 (Conv1D)           (None, 4, 32)             2080      
                                                                     
     conv1d_4 (Conv1D)           (None, 3, 32)             2080      
                                                                     
     conv1d_5 (Conv1D)           (None, 3, 16)             528       
                                                                     
     global_max_pooling1d (Globa  (None, 16)               0         
     lMaxPooling1D)                                                  
                                                                     
     dense (Dense)               (None, 1)                 17        
                                                                     
    =================================================================
    Total params: 8,961
    Trainable params: 8,961
    Non-trainable params: 0
    _________________________________________________________________
    


```python
EPOCH = 100
BATCH_SIZE = 128
```


```python
from keras.callbacks import ModelCheckpoint

filename = 'CNN1D_checkpoint-trial-001.h5'
checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             monitor='val_loss', # val_loss 값이 개선되었을때 호출됩니다
                             patience=5,
                             verbose=1,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                             mode='min',           # auto는 알아서 best를 찾습니다. min/max
                             restore_best_weights = True
                            )

history = model.fit(X_train_t, y_train, 
      validation_data=(X_val_t,y_val),
      epochs=EPOCH, 
      batch_size=BATCH_SIZE,
          verbose=1,
      callbacks=[checkpoint], # checkpoint 콜백
     )
```

    Epoch 1/100
    2397/2397 [==============================] - ETA: 0s - loss: 188147.2969 - mae: 262.9436
    Epoch 00001: val_loss improved from inf to 51784.17578, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 27s 8ms/step - loss: 188147.2969 - mae: 262.9436 - val_loss: 51784.1758 - val_mae: 149.1424
    Epoch 2/100
    2394/2397 [============================>.] - ETA: 0s - loss: 36270.4805 - mae: 129.2903
    Epoch 00002: val_loss did not improve from 51784.17578
    2397/2397 [==============================] - 20s 8ms/step - loss: 36266.3945 - mae: 129.2850 - val_loss: 65608.2500 - val_mae: 165.9681
    Epoch 3/100
    2393/2397 [============================>.] - ETA: 0s - loss: 30014.4395 - mae: 117.9232
    Epoch 00003: val_loss improved from 51784.17578 to 45767.62500, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 21s 9ms/step - loss: 30004.1074 - mae: 117.9103 - val_loss: 45767.6250 - val_mae: 137.1059
    Epoch 4/100
    2393/2397 [============================>.] - ETA: 0s - loss: 26651.4258 - mae: 110.4234
    Epoch 00004: val_loss improved from 45767.62500 to 38783.95703, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 20s 8ms/step - loss: 26652.2910 - mae: 110.4195 - val_loss: 38783.9570 - val_mae: 125.7338
    Epoch 5/100
    2396/2397 [============================>.] - ETA: 0s - loss: 23713.3027 - mae: 103.9515
    Epoch 00005: val_loss improved from 38783.95703 to 38395.39062, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 19s 8ms/step - loss: 23710.3438 - mae: 103.9459 - val_loss: 38395.3906 - val_mae: 127.4267
    Epoch 6/100
    2395/2397 [============================>.] - ETA: 0s - loss: 20763.1816 - mae: 97.4207
    Epoch 00006: val_loss improved from 38395.39062 to 29848.17969, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 21s 9ms/step - loss: 20766.4043 - mae: 97.4301 - val_loss: 29848.1797 - val_mae: 112.5357
    Epoch 7/100
    2395/2397 [============================>.] - ETA: 0s - loss: 18308.2051 - mae: 91.0685
    Epoch 00007: val_loss did not improve from 29848.17969
    2397/2397 [==============================] - 21s 9ms/step - loss: 18305.3047 - mae: 91.0618 - val_loss: 32184.1172 - val_mae: 117.4321
    Epoch 8/100
    2395/2397 [============================>.] - ETA: 0s - loss: 16076.8955 - mae: 84.8844
    Epoch 00008: val_loss improved from 29848.17969 to 21981.68750, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 20s 8ms/step - loss: 16077.2070 - mae: 84.8857 - val_loss: 21981.6875 - val_mae: 100.2051
    Epoch 9/100
    2392/2397 [============================>.] - ETA: 0s - loss: 13792.2393 - mae: 78.8656
    Epoch 00009: val_loss did not improve from 21981.68750
    2397/2397 [==============================] - 20s 8ms/step - loss: 13787.9697 - mae: 78.8565 - val_loss: 26785.0156 - val_mae: 107.2080
    Epoch 10/100
    2393/2397 [============================>.] - ETA: 0s - loss: 11834.1650 - mae: 73.1940
    Epoch 00010: val_loss did not improve from 21981.68750
    2397/2397 [==============================] - 20s 8ms/step - loss: 11829.6367 - mae: 73.1800 - val_loss: 30182.4941 - val_mae: 116.8527
    Epoch 11/100
    2395/2397 [============================>.] - ETA: 0s - loss: 10162.6230 - mae: 67.6122
    Epoch 00011: val_loss improved from 21981.68750 to 19511.18750, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 20s 8ms/step - loss: 10162.1768 - mae: 67.6131 - val_loss: 19511.1875 - val_mae: 94.4681
    Epoch 12/100
    2396/2397 [============================>.] - ETA: 0s - loss: 8894.7637 - mae: 62.6406
    Epoch 00012: val_loss did not improve from 19511.18750
    2397/2397 [==============================] - 21s 9ms/step - loss: 8894.6260 - mae: 62.6431 - val_loss: 27825.5098 - val_mae: 111.2787
    Epoch 13/100
    2393/2397 [============================>.] - ETA: 0s - loss: 8183.4858 - mae: 59.4684
    Epoch 00013: val_loss improved from 19511.18750 to 18993.81250, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 20s 8ms/step - loss: 8179.7012 - mae: 59.4598 - val_loss: 18993.8125 - val_mae: 92.5075
    Epoch 14/100
    2393/2397 [============================>.] - ETA: 0s - loss: 7593.5479 - mae: 56.7841
    Epoch 00014: val_loss did not improve from 18993.81250
    2397/2397 [==============================] - 20s 8ms/step - loss: 7591.6572 - mae: 56.7846 - val_loss: 34112.3438 - val_mae: 126.5419
    Epoch 15/100
    2397/2397 [==============================] - ETA: 0s - loss: 7149.5073 - mae: 54.5885
    Epoch 00015: val_loss did not improve from 18993.81250
    2397/2397 [==============================] - 20s 8ms/step - loss: 7149.5073 - mae: 54.5885 - val_loss: 19647.2695 - val_mae: 90.4077
    Epoch 16/100
    2395/2397 [============================>.] - ETA: 0s - loss: 6838.2705 - mae: 52.9317
    Epoch 00016: val_loss did not improve from 18993.81250
    2397/2397 [==============================] - 21s 9ms/step - loss: 6837.4761 - mae: 52.9320 - val_loss: 23136.5703 - val_mae: 101.6038
    Epoch 17/100
    2395/2397 [============================>.] - ETA: 0s - loss: 6499.0942 - mae: 51.1566
    Epoch 00017: val_loss improved from 18993.81250 to 17076.65820, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 21s 9ms/step - loss: 6498.3281 - mae: 51.1572 - val_loss: 17076.6582 - val_mae: 85.4076
    Epoch 18/100
    2394/2397 [============================>.] - ETA: 0s - loss: 6253.6050 - mae: 49.8656
    Epoch 00018: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 21s 9ms/step - loss: 6251.3389 - mae: 49.8627 - val_loss: 17604.8750 - val_mae: 87.5394
    Epoch 19/100
    2395/2397 [============================>.] - ETA: 0s - loss: 6111.1753 - mae: 49.0764
    Epoch 00019: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 21s 9ms/step - loss: 6110.3579 - mae: 49.0762 - val_loss: 25900.6523 - val_mae: 105.2921
    Epoch 20/100
    2396/2397 [============================>.] - ETA: 0s - loss: 5867.0591 - mae: 47.7034
    Epoch 00020: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 21s 9ms/step - loss: 5866.3447 - mae: 47.7019 - val_loss: 27444.1035 - val_mae: 107.9658
    Epoch 21/100
    2393/2397 [============================>.] - ETA: 0s - loss: 5744.3306 - mae: 47.0508
    Epoch 00021: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 21s 9ms/step - loss: 5742.4697 - mae: 47.0480 - val_loss: 27408.7891 - val_mae: 111.0046
    Epoch 22/100
    2396/2397 [============================>.] - ETA: 0s - loss: 5613.7661 - mae: 46.1854
    Epoch 00022: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 20s 8ms/step - loss: 5613.5698 - mae: 46.1861 - val_loss: 24128.7344 - val_mae: 102.5730
    Epoch 23/100
    2393/2397 [============================>.] - ETA: 0s - loss: 5515.7397 - mae: 45.7049
    Epoch 00023: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 20s 8ms/step - loss: 5513.1240 - mae: 45.7009 - val_loss: 20905.0449 - val_mae: 98.4764
    Epoch 24/100
    2393/2397 [============================>.] - ETA: 0s - loss: 5416.4805 - mae: 45.0913
    Epoch 00024: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 20s 8ms/step - loss: 5413.0659 - mae: 45.0849 - val_loss: 19771.3887 - val_mae: 91.9351
    Epoch 25/100
    2393/2397 [============================>.] - ETA: 0s - loss: 5318.3691 - mae: 44.5324
    Epoch 00025: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 20s 8ms/step - loss: 5318.4487 - mae: 44.5418 - val_loss: 23135.2734 - val_mae: 101.1632
    Epoch 26/100
    2397/2397 [==============================] - ETA: 0s - loss: 5184.8618 - mae: 43.8319
    Epoch 00026: val_loss did not improve from 17076.65820
    2397/2397 [==============================] - 20s 8ms/step - loss: 5184.8618 - mae: 43.8319 - val_loss: 31698.6348 - val_mae: 120.1319
    Epoch 27/100
    2396/2397 [============================>.] - ETA: 0s - loss: 5016.8784 - mae: 43.4043
    Epoch 00027: val_loss improved from 17076.65820 to 7619.15869, saving model to CNN1D_checkpoint-trial-001.h5
    2397/2397 [==============================] - 21s 9ms/step - loss: 5114.6172 - mae: 43.4220 - val_loss: 7619.1587 - val_mae: 60.6986
    Epoch 28/100
    2397/2397 [==============================] - ETA: 0s - loss: 5073.7510 - mae: 43.0957
    Epoch 00028: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 5073.7510 - mae: 43.0957 - val_loss: 16027.0811 - val_mae: 83.9120
    Epoch 29/100
    2394/2397 [============================>.] - ETA: 0s - loss: 5025.1123 - mae: 42.8477
    Epoch 00029: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 5022.6992 - mae: 42.8405 - val_loss: 15706.7383 - val_mae: 83.3770
    Epoch 30/100
    2393/2397 [============================>.] - ETA: 0s - loss: 4957.9663 - mae: 42.4904
    Epoch 00030: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 4956.1836 - mae: 42.4926 - val_loss: 22502.3984 - val_mae: 104.0005
    Epoch 31/100
    2397/2397 [==============================] - ETA: 0s - loss: 4841.9023 - mae: 41.8614
    Epoch 00031: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4841.9023 - mae: 41.8614 - val_loss: 21055.1250 - val_mae: 94.7186
    Epoch 32/100
    2395/2397 [============================>.] - ETA: 0s - loss: 4788.5146 - mae: 41.5895
    Epoch 00032: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4787.1748 - mae: 41.5850 - val_loss: 24332.2246 - val_mae: 100.8155
    Epoch 33/100
    2394/2397 [============================>.] - ETA: 0s - loss: 4697.1899 - mae: 41.0212
    Epoch 00033: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4695.2456 - mae: 41.0175 - val_loss: 14876.4229 - val_mae: 80.1063
    Epoch 34/100
    2397/2397 [==============================] - ETA: 0s - loss: 4647.7378 - mae: 40.6258
    Epoch 00034: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 19s 8ms/step - loss: 4647.7378 - mae: 40.6258 - val_loss: 20558.0215 - val_mae: 96.0635
    Epoch 35/100
    2394/2397 [============================>.] - ETA: 0s - loss: 4618.3711 - mae: 40.4727
    Epoch 00035: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4616.6719 - mae: 40.4711 - val_loss: 19154.0840 - val_mae: 91.5727
    Epoch 36/100
    2393/2397 [============================>.] - ETA: 0s - loss: 4562.2437 - mae: 40.1505
    Epoch 00036: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 4559.5654 - mae: 40.1440 - val_loss: 19633.4219 - val_mae: 92.9066
    Epoch 37/100
    2392/2397 [============================>.] - ETA: 0s - loss: 4474.2808 - mae: 39.6459
    Epoch 00037: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4472.0986 - mae: 39.6509 - val_loss: 19858.7520 - val_mae: 92.1077
    Epoch 38/100
    2396/2397 [============================>.] - ETA: 0s - loss: 4467.6616 - mae: 39.7224
    Epoch 00038: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4467.1787 - mae: 39.7212 - val_loss: 23584.2598 - val_mae: 104.1187
    Epoch 39/100
    2395/2397 [============================>.] - ETA: 0s - loss: 4448.8071 - mae: 39.4918
    Epoch 00039: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4447.6450 - mae: 39.4899 - val_loss: 20765.1660 - val_mae: 91.4972
    Epoch 40/100
    2396/2397 [============================>.] - ETA: 0s - loss: 4350.4004 - mae: 38.8594
    Epoch 00040: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4350.0225 - mae: 38.8597 - val_loss: 18169.4531 - val_mae: 88.1037
    Epoch 41/100
    2396/2397 [============================>.] - ETA: 0s - loss: 4325.9199 - mae: 38.6897
    Epoch 00041: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4325.6650 - mae: 38.6902 - val_loss: 18071.4980 - val_mae: 86.3377
    Epoch 42/100
    2395/2397 [============================>.] - ETA: 0s - loss: 4342.1621 - mae: 38.8315
    Epoch 00042: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 4341.7031 - mae: 38.8310 - val_loss: 25379.0449 - val_mae: 108.9421
    Epoch 43/100
    2397/2397 [==============================] - ETA: 0s - loss: 4272.8901 - mae: 38.3839
    Epoch 00043: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 4272.8901 - mae: 38.3839 - val_loss: 17517.1895 - val_mae: 85.8550
    Epoch 44/100
    2397/2397 [==============================] - ETA: 0s - loss: 4214.4697 - mae: 37.9152
    Epoch 00044: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 4214.4697 - mae: 37.9152 - val_loss: 15429.5498 - val_mae: 81.3351
    Epoch 45/100
    2395/2397 [============================>.] - ETA: 0s - loss: 4203.0605 - mae: 38.0113
    Epoch 00045: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4202.6079 - mae: 38.0121 - val_loss: 20019.9180 - val_mae: 92.4071
    Epoch 46/100
    2391/2397 [============================>.] - ETA: 0s - loss: 4162.7622 - mae: 37.6749
    Epoch 00046: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4158.7734 - mae: 37.6664 - val_loss: 17362.3828 - val_mae: 87.7112
    Epoch 47/100
    2395/2397 [============================>.] - ETA: 0s - loss: 4136.2480 - mae: 37.4970
    Epoch 00047: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 19s 8ms/step - loss: 4134.9819 - mae: 37.4941 - val_loss: 17665.1875 - val_mae: 87.4111
    Epoch 48/100
    2392/2397 [============================>.] - ETA: 0s - loss: 4151.3169 - mae: 37.6184
    Epoch 00048: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 19s 8ms/step - loss: 4154.4966 - mae: 37.6202 - val_loss: 22694.3594 - val_mae: 93.7356
    Epoch 49/100
    2397/2397 [==============================] - ETA: 0s - loss: 4064.1169 - mae: 36.9726
    Epoch 00049: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 19s 8ms/step - loss: 4064.1169 - mae: 36.9726 - val_loss: 19071.7188 - val_mae: 89.6364
    Epoch 50/100
    2396/2397 [============================>.] - ETA: 0s - loss: 4039.8813 - mae: 36.7735
    Epoch 00050: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4039.8447 - mae: 36.7726 - val_loss: 22397.9141 - val_mae: 97.0632
    Epoch 51/100
    2393/2397 [============================>.] - ETA: 0s - loss: 4006.5732 - mae: 36.6364
    Epoch 00051: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 4003.7456 - mae: 36.6274 - val_loss: 21233.3535 - val_mae: 99.7115
    Epoch 52/100
    2394/2397 [============================>.] - ETA: 0s - loss: 3979.7493 - mae: 36.3502
    Epoch 00052: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3978.5063 - mae: 36.3514 - val_loss: 18066.2871 - val_mae: 86.7488
    Epoch 53/100
    2391/2397 [============================>.] - ETA: 0s - loss: 3962.4175 - mae: 36.4111
    Epoch 00053: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3958.2410 - mae: 36.4042 - val_loss: 16665.3867 - val_mae: 86.5433
    Epoch 54/100
    2391/2397 [============================>.] - ETA: 0s - loss: 3995.7896 - mae: 36.5411
    Epoch 00054: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3992.1665 - mae: 36.5349 - val_loss: 20489.3828 - val_mae: 96.2497
    Epoch 55/100
    2397/2397 [==============================] - ETA: 0s - loss: 3942.3486 - mae: 36.2450
    Epoch 00055: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3942.3486 - mae: 36.2450 - val_loss: 19526.4844 - val_mae: 92.2552
    Epoch 56/100
    2397/2397 [==============================] - ETA: 0s - loss: 3906.9670 - mae: 35.9410
    Epoch 00056: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3906.9670 - mae: 35.9410 - val_loss: 14360.3672 - val_mae: 80.8813
    Epoch 57/100
    2396/2397 [============================>.] - ETA: 0s - loss: 3889.7993 - mae: 35.8218
    Epoch 00057: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3889.4270 - mae: 35.8213 - val_loss: 21348.0000 - val_mae: 97.0086
    Epoch 58/100
    2392/2397 [============================>.] - ETA: 0s - loss: 3876.7390 - mae: 35.7073
    Epoch 00058: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3872.5354 - mae: 35.6942 - val_loss: 16895.1289 - val_mae: 87.5857
    Epoch 59/100
    2397/2397 [==============================] - ETA: 0s - loss: 3880.1272 - mae: 35.7894
    Epoch 00059: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3880.1272 - mae: 35.7894 - val_loss: 16334.2529 - val_mae: 83.5163
    Epoch 60/100
    2394/2397 [============================>.] - ETA: 0s - loss: 3802.8384 - mae: 35.2782
    Epoch 00060: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3801.4680 - mae: 35.2771 - val_loss: 20695.1328 - val_mae: 93.4124
    Epoch 61/100
    2397/2397 [==============================] - ETA: 0s - loss: 3830.3328 - mae: 35.4988
    Epoch 00061: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3830.3328 - mae: 35.4988 - val_loss: 19321.1309 - val_mae: 91.0686
    Epoch 62/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3749.6885 - mae: 34.9580
    Epoch 00062: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3749.5015 - mae: 34.9594 - val_loss: 17729.0000 - val_mae: 85.7895
    Epoch 63/100
    2392/2397 [============================>.] - ETA: 0s - loss: 3788.5815 - mae: 35.2851
    Epoch 00063: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3785.9424 - mae: 35.2826 - val_loss: 19236.6934 - val_mae: 92.7020
    Epoch 64/100
    2392/2397 [============================>.] - ETA: 0s - loss: 3735.1162 - mae: 34.8709
    Epoch 00064: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3733.3501 - mae: 34.8708 - val_loss: 21558.5098 - val_mae: 99.1869
    Epoch 65/100
    2397/2397 [==============================] - ETA: 0s - loss: 3745.3652 - mae: 34.9510
    Epoch 00065: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3745.3652 - mae: 34.9510 - val_loss: 18338.7832 - val_mae: 89.5082
    Epoch 66/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3704.7073 - mae: 34.5670
    Epoch 00066: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3703.6199 - mae: 34.5656 - val_loss: 19670.9844 - val_mae: 88.2477
    Epoch 67/100
    2393/2397 [============================>.] - ETA: 0s - loss: 3704.6465 - mae: 34.6365
    Epoch 00067: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3709.0190 - mae: 34.6392 - val_loss: 21071.9688 - val_mae: 95.9938
    Epoch 68/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3645.3552 - mae: 34.3189
    Epoch 00068: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3643.9751 - mae: 34.3151 - val_loss: 20404.1016 - val_mae: 92.5235
    Epoch 69/100
    2396/2397 [============================>.] - ETA: 0s - loss: 3679.2405 - mae: 34.5261
    Epoch 00069: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3678.7200 - mae: 34.5251 - val_loss: 17954.0508 - val_mae: 89.0136
    Epoch 70/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3654.4832 - mae: 34.4450
    Epoch 00070: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3653.8135 - mae: 34.4456 - val_loss: 19995.4824 - val_mae: 92.1203
    Epoch 71/100
    2392/2397 [============================>.] - ETA: 0s - loss: 3624.5732 - mae: 34.1043
    Epoch 00071: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3622.5659 - mae: 34.1049 - val_loss: 15441.4922 - val_mae: 80.1024
    Epoch 72/100
    2391/2397 [============================>.] - ETA: 0s - loss: 3603.3254 - mae: 33.9495
    Epoch 00072: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3600.2473 - mae: 33.9442 - val_loss: 17126.4941 - val_mae: 84.0963
    Epoch 73/100
    2392/2397 [============================>.] - ETA: 0s - loss: 3608.9819 - mae: 34.0216
    Epoch 00073: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3605.4189 - mae: 34.0113 - val_loss: 19226.5020 - val_mae: 90.6872
    Epoch 74/100
    2396/2397 [============================>.] - ETA: 0s - loss: 3598.2705 - mae: 34.0442
    Epoch 00074: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3597.7310 - mae: 34.0427 - val_loss: 20900.6660 - val_mae: 93.3583
    Epoch 75/100
    2391/2397 [============================>.] - ETA: 0s - loss: 3594.2300 - mae: 33.8629
    Epoch 00075: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3590.8838 - mae: 33.8601 - val_loss: 22189.9434 - val_mae: 98.7513
    Epoch 76/100
    2394/2397 [============================>.] - ETA: 0s - loss: 3548.3484 - mae: 33.5563
    Epoch 00076: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3546.9500 - mae: 33.5548 - val_loss: 20340.0195 - val_mae: 91.1810
    Epoch 77/100
    2392/2397 [============================>.] - ETA: 0s - loss: 3525.7048 - mae: 33.6311
    Epoch 00077: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3522.9092 - mae: 33.6249 - val_loss: 16799.0781 - val_mae: 86.3860
    Epoch 78/100
    2397/2397 [==============================] - ETA: 0s - loss: 3527.9390 - mae: 33.5043
    Epoch 00078: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3527.9390 - mae: 33.5043 - val_loss: 20345.0371 - val_mae: 96.3921
    Epoch 79/100
    2397/2397 [==============================] - ETA: 0s - loss: 3514.8169 - mae: 33.4349
    Epoch 00079: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3514.8169 - mae: 33.4349 - val_loss: 19520.3633 - val_mae: 94.9939
    Epoch 80/100
    2393/2397 [============================>.] - ETA: 0s - loss: 3503.2964 - mae: 33.3605
    Epoch 00080: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3501.2407 - mae: 33.3596 - val_loss: 16433.5156 - val_mae: 82.7234
    Epoch 81/100
    2391/2397 [============================>.] - ETA: 0s - loss: 3503.1448 - mae: 33.3876
    Epoch 00081: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3500.1343 - mae: 33.3851 - val_loss: 17878.3125 - val_mae: 89.5524
    Epoch 82/100
    2392/2397 [============================>.] - ETA: 0s - loss: 3480.7175 - mae: 33.1525
    Epoch 00082: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3477.4863 - mae: 33.1476 - val_loss: 18538.8223 - val_mae: 87.9640
    Epoch 83/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3479.6313 - mae: 33.2711
    Epoch 00083: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3478.3064 - mae: 33.2676 - val_loss: 21565.4531 - val_mae: 99.5980
    Epoch 84/100
    2394/2397 [============================>.] - ETA: 0s - loss: 3464.7417 - mae: 33.1021
    Epoch 00084: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3462.8909 - mae: 33.0969 - val_loss: 19551.7578 - val_mae: 94.6972
    Epoch 85/100
    2394/2397 [============================>.] - ETA: 0s - loss: 3437.1121 - mae: 32.8807
    Epoch 00085: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3437.0989 - mae: 32.8867 - val_loss: 21225.9609 - val_mae: 95.5878
    Epoch 86/100
    2396/2397 [============================>.] - ETA: 0s - loss: 3410.5586 - mae: 32.7766
    Epoch 00086: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3416.6675 - mae: 32.7812 - val_loss: 23085.9785 - val_mae: 98.9691
    Epoch 87/100
    2394/2397 [============================>.] - ETA: 0s - loss: 3424.7966 - mae: 32.7808
    Epoch 00087: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3423.6936 - mae: 32.7810 - val_loss: 16339.2402 - val_mae: 86.1068
    Epoch 88/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3383.7471 - mae: 32.4718
    Epoch 00088: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3383.1643 - mae: 32.4713 - val_loss: 16959.2559 - val_mae: 85.8159
    Epoch 89/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3380.5957 - mae: 32.5653
    Epoch 00089: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3380.2991 - mae: 32.5716 - val_loss: 25586.9102 - val_mae: 109.8902
    Epoch 90/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3441.9431 - mae: 32.9966
    Epoch 00090: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3440.9106 - mae: 32.9959 - val_loss: 21750.4609 - val_mae: 100.0006
    Epoch 91/100
    2392/2397 [============================>.] - ETA: 0s - loss: 3371.2637 - mae: 32.4547
    Epoch 00091: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3368.5413 - mae: 32.4522 - val_loss: 16095.8076 - val_mae: 83.9329
    Epoch 92/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3420.0623 - mae: 32.7833
    Epoch 00092: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3419.2000 - mae: 32.7820 - val_loss: 20091.4785 - val_mae: 92.2565
    Epoch 93/100
    2394/2397 [============================>.] - ETA: 0s - loss: 3366.0627 - mae: 32.4548
    Epoch 00093: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3364.1733 - mae: 32.4504 - val_loss: 20796.8320 - val_mae: 96.5870
    Epoch 94/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3418.1553 - mae: 32.7885
    Epoch 00094: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3417.3123 - mae: 32.7881 - val_loss: 21289.4023 - val_mae: 99.1552
    Epoch 95/100
    2396/2397 [============================>.] - ETA: 0s - loss: 3356.6099 - mae: 32.3963
    Epoch 00095: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3356.3665 - mae: 32.3967 - val_loss: 17515.8613 - val_mae: 85.8274
    Epoch 96/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3373.3992 - mae: 32.4427
    Epoch 00096: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3372.8972 - mae: 32.4469 - val_loss: 19252.3301 - val_mae: 94.2289
    Epoch 97/100
    2393/2397 [============================>.] - ETA: 0s - loss: 3336.5410 - mae: 32.3148
    Epoch 00097: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3340.5291 - mae: 32.3163 - val_loss: 17755.5625 - val_mae: 87.2857
    Epoch 98/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3323.8889 - mae: 32.2282
    Epoch 00098: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3322.6523 - mae: 32.2251 - val_loss: 20415.9668 - val_mae: 93.6937
    Epoch 99/100
    2395/2397 [============================>.] - ETA: 0s - loss: 3330.3877 - mae: 32.1499
    Epoch 00099: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 20s 8ms/step - loss: 3329.1423 - mae: 32.1461 - val_loss: 18766.6445 - val_mae: 89.7315
    Epoch 100/100
    2393/2397 [============================>.] - ETA: 0s - loss: 3306.7590 - mae: 32.0285
    Epoch 00100: val_loss did not improve from 7619.15869
    2397/2397 [==============================] - 21s 9ms/step - loss: 3305.4934 - mae: 32.0298 - val_loss: 19734.6660 - val_mae: 96.1380
    


```python
# loss check
def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')

plot_loss (history, 'CNN')
```


    
![png](output_51_0.png)
    



```python
model.save('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/CNN1D_checkpoint-trial-001.h5')
```


```python
from tensorflow.python.keras.models import load_model

model2 = load_model('CNN1D_checkpoint-trial-001.h5')
```


```python
# validatation set prediction
y_pred_ = model2.predict(X_val_t)
```


```python
x = y_val
y = y_pred_

min_val = np.min([x.min(), y.min()])
max_val = np.max([x.max(), y.max()])


plt.scatter(x, y)
plt.xlabel("TRUE: $Y_i$")
plt.ylabel("Predict: $\hat{Y}_i$")

plt.plot([min_val, max_val], [min_val, max_val], color='r')
```




    [<matplotlib.lines.Line2D at 0x7efe0620ff90>]




    
![png](output_55_1.png)
    



```python
plt.plot(np.arange(y_val.shape[0]), y_val)
plt.plot(np.arange(y_val.shape[0]), y_pred_)
```




    [<matplotlib.lines.Line2D at 0x7efe067f6410>]




    
![png](output_56_1.png)
    



```python
plt.plot(np.arange(y_val.shape[0]), y_pred_)
plt.plot(np.arange(y_val.shape[0]), y_val)
```




    [<matplotlib.lines.Line2D at 0x7efde2c1d450>]




    
![png](output_57_1.png)
    


METRIC


```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_val, y_pred_)
rmse = mse**0.5
r_2 = r2_score(y_val, y_pred_)

print(mse)
print(rmse)
print(r_2)
```

    7619.1607875817335
    87.28780434620711
    0.9924461898308387
    

Test Data Preprocessing


```python
# test => inference dataset
test = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/test.csv')
submission = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/sample_submission.csv')
```


```python
test['일자'] = test['일자|시간|구분'].str.split(' ').str[0]
test['시간'] = test['일자|시간|구분'].str.split(' ').str[1].astype(int)
test['구분'] = test['일자|시간|구분'].str.split(' ').str[2]

test['일자'] = pd.to_datetime(test['일자'])
test['year'] = test['일자'].dt.year
test['month'] = test['일자'].dt.month
test['day'] = test['일자'].dt.day
test['weekday'] = test['일자'].dt.weekday
```


```python
data['연월일'] = data['연월일'].astype('str')
data['시간'] = data['시간'].astype('str')

test['일자'] = test['일자'].astype('str')
test['시간'] = test['시간'].astype('str')

data['월일시'] = data['연월일'].str[5:] + '_' + data['시간']
test['월일시'] = test['일자'].str[5:] + '_' + test['시간']
```


```python
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
      <th>일자|시간|구분</th>
      <th>일자</th>
      <th>시간</th>
      <th>구분</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>월일시</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01 01 A</td>
      <td>2019-01-01</td>
      <td>1</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-01 02 A</td>
      <td>2019-01-01</td>
      <td>2</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-01 03 A</td>
      <td>2019-01-01</td>
      <td>3</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-01 04 A</td>
      <td>2019-01-01</td>
      <td>4</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-01 05 A</td>
      <td>2019-01-01</td>
      <td>5</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_5</td>
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
      <th>15115</th>
      <td>2019-03-31 20 H</td>
      <td>2019-03-31</td>
      <td>20</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_20</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>2019-03-31 21 H</td>
      <td>2019-03-31</td>
      <td>21</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_21</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>2019-03-31 22 H</td>
      <td>2019-03-31</td>
      <td>22</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_22</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>2019-03-31 23 H</td>
      <td>2019-03-31</td>
      <td>23</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_23</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>2019-03-31 24 H</td>
      <td>2019-03-31</td>
      <td>24</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_24</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 9 columns</p>
</div>




```python
test['월일시'].unique()
```




    array(['01-01_1', '01-01_2', '01-01_3', ..., '03-31_22', '03-31_23',
           '03-31_24'], dtype=object)




```python
date = []
supplier = []
mv_avg = []
for i in tqdm(test['월일시'].unique()):
    df_mv_avg = sup_rolling_df[sup_rolling_df['월일시']==i]
    
    for j in df_mv_avg['구분'].unique():
        supplier_mean_df = df_mv_avg[df_mv_avg['구분']==j]
        mean_list = round(np.mean(supplier_mean_df['5D_moving_avg']),4)
     
        date.append(supplier_mean_df['월일시'].iloc[0])
        supplier.append(supplier_mean_df['구분'].iloc[0])
        mv_avg.append(mean_list)
```

    100%|██████████| 2160/2160 [01:05<00:00, 32.92it/s]
    


```python
temp_df
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
      <th>공급량</th>
      <th>기온</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>연월일</th>
      <th>시간</th>
      <th>구분</th>
      <th>월일시</th>
      <th>5D_moving_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52560</th>
      <td>562.964</td>
      <td>-8.8</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>H</td>
      <td>01-01_1</td>
      <td>523.0568</td>
    </tr>
    <tr>
      <th>52561</th>
      <td>531.228</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>2</td>
      <td>H</td>
      <td>01-01_2</td>
      <td>523.0568</td>
    </tr>
    <tr>
      <th>52562</th>
      <td>496.276</td>
      <td>-8.5</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>3</td>
      <td>H</td>
      <td>01-01_3</td>
      <td>523.0568</td>
    </tr>
    <tr>
      <th>52563</th>
      <td>489.396</td>
      <td>-9.0</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>4</td>
      <td>H</td>
      <td>01-01_4</td>
      <td>523.0568</td>
    </tr>
    <tr>
      <th>52564</th>
      <td>535.420</td>
      <td>-9.1</td>
      <td>2013.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2013-01-01</td>
      <td>5</td>
      <td>H</td>
      <td>01-01_5</td>
      <td>523.0568</td>
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
    </tr>
    <tr>
      <th>368083</th>
      <td>681.033</td>
      <td>-2.8</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>20</td>
      <td>H</td>
      <td>12-31_20</td>
      <td>604.7030</td>
    </tr>
    <tr>
      <th>368084</th>
      <td>669.961</td>
      <td>-3.5</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>21</td>
      <td>H</td>
      <td>12-31_21</td>
      <td>635.0934</td>
    </tr>
    <tr>
      <th>368085</th>
      <td>657.941</td>
      <td>-4.0</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>22</td>
      <td>H</td>
      <td>12-31_22</td>
      <td>658.2096</td>
    </tr>
    <tr>
      <th>368086</th>
      <td>610.953</td>
      <td>-4.6</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>23</td>
      <td>H</td>
      <td>12-31_23</td>
      <td>659.7726</td>
    </tr>
    <tr>
      <th>368087</th>
      <td>560.896</td>
      <td>-5.2</td>
      <td>2018.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2018-12-31</td>
      <td>24</td>
      <td>H</td>
      <td>12-31_24</td>
      <td>636.1568</td>
    </tr>
  </tbody>
</table>
<p>52584 rows × 11 columns</p>
</div>




```python
temp_df = pd.DataFrame()
```


```python
temp_df['월일시'] = date
temp_df['구분'] = supplier
temp_df['5D_moving_avg'] = mv_avg
```


```python
temp_df[temp_df['월일시'] == '01-01_4']
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
      <th>월일시</th>
      <th>구분</th>
      <th>5D_moving_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>01-01_4</td>
      <td>A</td>
      <td>1825.0762</td>
    </tr>
    <tr>
      <th>22</th>
      <td>01-01_4</td>
      <td>B</td>
      <td>1637.6443</td>
    </tr>
    <tr>
      <th>23</th>
      <td>01-01_4</td>
      <td>C</td>
      <td>190.3049</td>
    </tr>
    <tr>
      <th>24</th>
      <td>01-01_4</td>
      <td>D</td>
      <td>1110.6390</td>
    </tr>
    <tr>
      <th>25</th>
      <td>01-01_4</td>
      <td>E</td>
      <td>2406.0732</td>
    </tr>
    <tr>
      <th>26</th>
      <td>01-01_4</td>
      <td>G</td>
      <td>2743.7452</td>
    </tr>
    <tr>
      <th>27</th>
      <td>01-01_4</td>
      <td>H</td>
      <td>422.1951</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 월일시, 구분, 평균
print(supplier_mean_df['월일시'].iloc[0],
supplier_mean_df['구분'].iloc[0],
round(np.mean(supplier_mean_df['5D_moving_avg']),4))
```

    03-31_24 H 298.3528
    


```python
test_set = pd.merge(test, temp_df, on = ['구분','월일시'], how = 'left')
```


```python
test_set
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
      <th>일자|시간|구분</th>
      <th>일자</th>
      <th>시간</th>
      <th>구분</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>월일시</th>
      <th>5D_moving_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01 01 A</td>
      <td>2019-01-01</td>
      <td>1</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_1</td>
      <td>2095.5065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-01 02 A</td>
      <td>2019-01-01</td>
      <td>2</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_2</td>
      <td>2002.3103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-01 03 A</td>
      <td>2019-01-01</td>
      <td>3</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_3</td>
      <td>1907.0166</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-01 04 A</td>
      <td>2019-01-01</td>
      <td>4</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_4</td>
      <td>1825.0762</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-01 05 A</td>
      <td>2019-01-01</td>
      <td>5</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_5</td>
      <td>1779.5306</td>
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
      <td>2019-03-31 20 H</td>
      <td>2019-03-31</td>
      <td>20</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_20</td>
      <td>239.8929</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>2019-03-31 21 H</td>
      <td>2019-03-31</td>
      <td>21</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_21</td>
      <td>268.8691</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>2019-03-31 22 H</td>
      <td>2019-03-31</td>
      <td>22</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_22</td>
      <td>291.2746</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>2019-03-31 23 H</td>
      <td>2019-03-31</td>
      <td>23</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_23</td>
      <td>302.2607</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>2019-03-31 24 H</td>
      <td>2019-03-31</td>
      <td>24</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_24</td>
      <td>298.3528</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 10 columns</p>
</div>




```python
avg_temp = pd.read_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/대전_평균기온_for2019.csv', encoding='cp949')
avg = avg_temp['avg6']
avg
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
test = pd.concat([test_set, avg], axis=1)
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
      <th>일자|시간|구분</th>
      <th>일자</th>
      <th>시간</th>
      <th>구분</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>월일시</th>
      <th>5D_moving_avg</th>
      <th>avg6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01 01 A</td>
      <td>2019-01-01</td>
      <td>1</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_1</td>
      <td>2095.5065</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-01 02 A</td>
      <td>2019-01-01</td>
      <td>2</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_2</td>
      <td>2002.3103</td>
      <td>-2.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-01 03 A</td>
      <td>2019-01-01</td>
      <td>3</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_3</td>
      <td>1907.0166</td>
      <td>-2.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-01 04 A</td>
      <td>2019-01-01</td>
      <td>4</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_4</td>
      <td>1825.0762</td>
      <td>-2.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-01 05 A</td>
      <td>2019-01-01</td>
      <td>5</td>
      <td>A</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_5</td>
      <td>1779.5306</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>2019-03-31 20 H</td>
      <td>2019-03-31</td>
      <td>20</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_20</td>
      <td>239.8929</td>
      <td>14.2</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>2019-03-31 21 H</td>
      <td>2019-03-31</td>
      <td>21</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_21</td>
      <td>268.8691</td>
      <td>13.1</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>2019-03-31 22 H</td>
      <td>2019-03-31</td>
      <td>22</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_22</td>
      <td>291.2746</td>
      <td>12.2</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>2019-03-31 23 H</td>
      <td>2019-03-31</td>
      <td>23</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_23</td>
      <td>302.2607</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>2019-03-31 24 H</td>
      <td>2019-03-31</td>
      <td>24</td>
      <td>H</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_24</td>
      <td>298.3528</td>
      <td>10.8</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 11 columns</p>
</div>




```python
test_dummies = pd.get_dummies(test, columns = ['구분'])
```


```python
test_dummies
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
      <th>일자|시간|구분</th>
      <th>일자</th>
      <th>시간</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>월일시</th>
      <th>5D_moving_avg</th>
      <th>avg6</th>
      <th>구분_A</th>
      <th>구분_B</th>
      <th>구분_C</th>
      <th>구분_D</th>
      <th>구분_E</th>
      <th>구분_G</th>
      <th>구분_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01 01 A</td>
      <td>2019-01-01</td>
      <td>1</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_1</td>
      <td>2095.5065</td>
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
      <td>2019-01-01 02 A</td>
      <td>2019-01-01</td>
      <td>2</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_2</td>
      <td>2002.3103</td>
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
      <td>2019-01-01 03 A</td>
      <td>2019-01-01</td>
      <td>3</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_3</td>
      <td>1907.0166</td>
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
      <td>2019-01-01 04 A</td>
      <td>2019-01-01</td>
      <td>4</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_4</td>
      <td>1825.0762</td>
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
      <td>2019-01-01 05 A</td>
      <td>2019-01-01</td>
      <td>5</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>01-01_5</td>
      <td>1779.5306</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>2019-03-31 20 H</td>
      <td>2019-03-31</td>
      <td>20</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_20</td>
      <td>239.8929</td>
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
      <td>2019-03-31 21 H</td>
      <td>2019-03-31</td>
      <td>21</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_21</td>
      <td>268.8691</td>
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
      <td>2019-03-31 22 H</td>
      <td>2019-03-31</td>
      <td>22</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_22</td>
      <td>291.2746</td>
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
      <td>2019-03-31 23 H</td>
      <td>2019-03-31</td>
      <td>23</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_23</td>
      <td>302.2607</td>
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
      <td>2019-03-31 24 H</td>
      <td>2019-03-31</td>
      <td>24</td>
      <td>2019</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>03-31_24</td>
      <td>298.3528</td>
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
<p>15120 rows × 17 columns</p>
</div>




```python
features_test = ['시간', 'month', 'day', 'weekday', '구분_A', '구분_B','구분_C', '구분_D', '구분_E', '구분_G', '구분_H', 'avg6', '5D_moving_avg']

test_x = test_dummies[features_test]

test_x['시간'] = test_x['시간'].astype(int)
```


```python
test_x = scaler.transform(test_x)
```


```python
test_t_ = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
```


```python
pred_LSTM = model.predict(test_t_)
```


```python
pred_LSTM
```




    array([[1670.3248 ],
           [1517.8932 ],
           [1449.4465 ],
           ...,
           [ 279.64273],
           [ 240.18819],
           [ 232.65659]], dtype=float32)




```python
pred_CNN = model2.predict(test_t_)
```


```python
pred_CNN
```




    array([[1668.871  ],
           [1582.562  ],
           [1592.4841 ],
           ...,
           [ 265.34375],
           [ 255.18242],
           [ 236.21263]], dtype=float32)



LSTM과 1D CNN 모델 예측 비교 시각화


```python
plt.plot(pred_LSTM, color = 'b')
plt.plot(pred_CNN, color = 'c')
```




    [<matplotlib.lines.Line2D at 0x7efde3ebef50>]




    
![png](output_86_1.png)
    



```python
plt.plot(pred_CNN, color = 'c')
plt.plot(pred_LSTM, color = 'b')
```




    [<matplotlib.lines.Line2D at 0x7efe060c2ad0>]




    
![png](output_87_1.png)
    


Ensemble


```python
preds = (pred_LSTM + pred_CNN)/2
preds
```




    array([[1669.5979 ],
           [1550.2275 ],
           [1520.9653 ],
           ...,
           [ 272.49323],
           [ 247.6853 ],
           [ 234.4346 ]], dtype=float32)




```python
submission['공급량'] = preds
```


```python
submission
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
      <th>일자|시간|구분</th>
      <th>공급량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01 01 A</td>
      <td>1669.597900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-01 02 A</td>
      <td>1550.227539</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-01 03 A</td>
      <td>1520.965332</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-01 04 A</td>
      <td>1631.760010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-01 05 A</td>
      <td>1817.551514</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>2019-03-31 20 H</td>
      <td>280.106049</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>2019-03-31 21 H</td>
      <td>276.584656</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>2019-03-31 22 H</td>
      <td>272.493225</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>2019-03-31 23 H</td>
      <td>247.685303</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>2019-03-31 24 H</td>
      <td>234.434601</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 2 columns</p>
</div>




```python
submission.to_csv('/content/gdrive/My Drive/Data Analysis/가스공급량 수요예측 모델개발/submission.csv', index=False)
```
