---
layout: single
title: "DACON 물류 유통량 예측 경진대회 LGBM"
---

```python
from google.colab import drive

drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
    


```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tabgan.sampler import OriginalGenerator, GANGenerator

train = pd.read_csv('/content/gdrive/My Drive/Data Analysis/물류 유통량 예측 경진대회/train_df.csv', encoding = 'cp949')
test = pd.read_csv('/content/gdrive/My Drive/Data Analysis/물류 유통량 예측 경진대회/test_df.csv', encoding = 'cp949')

submission = pd.read_csv('/content/gdrive/My Drive/Data Analysis/물류 유통량 예측 경진대회/sample_submission.csv')
```

    /usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


```python
pip install --upgrade tabgan
```

    Collecting tabgan
      Downloading tabgan-1.1.2-py2.py3-none-any.whl (44 kB)
    [K     |████████████████████████████████| 44 kB 2.3 MB/s 
    [?25hRequirement already satisfied: torch in /usr/local/lib/python3.7/dist-packages (from tabgan) (1.10.0+cu111)
    Requirement already satisfied: lightgbm in /usr/local/lib/python3.7/dist-packages (from tabgan) (2.2.3)
    Collecting category-encoders
      Downloading category_encoders-2.3.0-py2.py3-none-any.whl (82 kB)
    [K     |████████████████████████████████| 82 kB 358 kB/s 
    [?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from tabgan) (4.62.3)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from tabgan) (2.8.2)
    Collecting scikit-learn==0.23.2
      Downloading scikit_learn-0.23.2-cp37-cp37m-manylinux1_x86_64.whl (6.8 MB)
    [K     |████████████████████████████████| 6.8 MB 19.2 MB/s 
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from tabgan) (1.1.5)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.7/dist-packages (from tabgan) (0.11.1+cu111)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from tabgan) (1.19.5)
    Requirement already satisfied: scipy>=0.19.1 in /usr/local/lib/python3.7/dist-packages (from scikit-learn==0.23.2->tabgan) (1.4.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn==0.23.2->tabgan) (1.1.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn==0.23.2->tabgan) (3.0.0)
    Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.7/dist-packages (from category-encoders->tabgan) (0.5.2)
    Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.7/dist-packages (from category-encoders->tabgan) (0.10.2)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->tabgan) (2018.9)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from patsy>=0.5.1->category-encoders->tabgan) (1.15.0)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch->tabgan) (3.10.0.2)
    Requirement already satisfied: pillow!=8.3.0,>=5.3.0 in /usr/local/lib/python3.7/dist-packages (from torchvision->tabgan) (7.1.2)
    Installing collected packages: scikit-learn, category-encoders, tabgan
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 1.0.1
        Uninstalling scikit-learn-1.0.1:
          Successfully uninstalled scikit-learn-1.0.1
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    imbalanced-learn 0.8.1 requires scikit-learn>=0.24, but you have scikit-learn 0.23.2 which is incompatible.[0m
    Successfully installed category-encoders-2.3.0 scikit-learn-0.23.2 tabgan-1.1.2
    



Data Preprocessing


```python
train = train.drop(columns = ['index', 'DL_GD_LCLS_NM', 'DL_GD_MCLS_NM'], axis = 1)
train
```





  <div id="df-ce85f02f-01db-40dd-9b4c-dda3c3da547a">
    <div class="colab-df-container">
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
      <th>SEND_SPG_INNB</th>
      <th>REC_SPG_INNB</th>
      <th>INVC_CONT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1129000014045300</td>
      <td>5011000220046300</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1135000009051200</td>
      <td>5011000178037300</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1135000030093100</td>
      <td>5011000265091400</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1154500002014200</td>
      <td>5011000315087400</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1165000021008300</td>
      <td>5011000177051200</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31995</th>
      <td>5011001060063300</td>
      <td>2635000026053400</td>
      <td>6</td>
    </tr>
    <tr>
      <th>31996</th>
      <td>5011001095042400</td>
      <td>1168000017002200</td>
      <td>5</td>
    </tr>
    <tr>
      <th>31997</th>
      <td>5011001108036200</td>
      <td>4119700008012100</td>
      <td>9</td>
    </tr>
    <tr>
      <th>31998</th>
      <td>5011001115011400</td>
      <td>1132000015085100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>31999</th>
      <td>5011001116066400</td>
      <td>4719000594022400</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>32000 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ce85f02f-01db-40dd-9b4c-dda3c3da547a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ce85f02f-01db-40dd-9b4c-dda3c3da547a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ce85f02f-01db-40dd-9b4c-dda3c3da547a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
test = test.drop(columns = ['index', 'DL_GD_LCLS_NM', 'DL_GD_MCLS_NM'], axis = 1)
test
```





  <div id="df-493d7c12-ed50-4b19-9f62-708373981d93">
    <div class="colab-df-container">
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
      <th>SEND_SPG_INNB</th>
      <th>REC_SPG_INNB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5013000043028400</td>
      <td>1165000021097200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5013000044016100</td>
      <td>1154500002066400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5013000205030200</td>
      <td>4139000102013200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5013000205030200</td>
      <td>4221000040093400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5013000268011400</td>
      <td>2726000004017100</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4635</th>
      <td>5013000858004400</td>
      <td>4725000719072200</td>
    </tr>
    <tr>
      <th>4636</th>
      <td>5013000870018300</td>
      <td>2826000106075300</td>
    </tr>
    <tr>
      <th>4637</th>
      <td>5013000897086300</td>
      <td>4311100034004300</td>
    </tr>
    <tr>
      <th>4638</th>
      <td>5013000902065100</td>
      <td>4145000013011200</td>
    </tr>
    <tr>
      <th>4639</th>
      <td>5013000926078200</td>
      <td>2714000174062400</td>
    </tr>
  </tbody>
</table>
<p>4640 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-493d7c12-ed50-4b19-9f62-708373981d93')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-493d7c12-ed50-4b19-9f62-708373981d93 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-493d7c12-ed50-4b19-9f62-708373981d93');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
INVC_CONT = train['INVC_CONT']
INVC_CONT = pd.DataFrame(INVC_CONT)
INVC_CONT
```





  <div id="df-defe39d8-f234-405e-8fa4-97e3810958f7">
    <div class="colab-df-container">
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
      <th>INVC_CONT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>31995</th>
      <td>6</td>
    </tr>
    <tr>
      <th>31996</th>
      <td>5</td>
    </tr>
    <tr>
      <th>31997</th>
      <td>9</td>
    </tr>
    <tr>
      <th>31998</th>
      <td>3</td>
    </tr>
    <tr>
      <th>31999</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>32000 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-defe39d8-f234-405e-8fa4-97e3810958f7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-defe39d8-f234-405e-8fa4-97e3810958f7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-defe39d8-f234-405e-8fa4-97e3810958f7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#격자고유번호를 도시, 동별로 나눠 파생변수를 생성	

def split1(x):
  return int(str(x['SEND_SPG_INNB'])[:7])
def split2(x):
  return int(str(x['SEND_SPG_INNB'])[7:11])
def split3(x):
  return int(str(x['SEND_SPG_INNB'])[11:])
def split4(x):
  return int(str(x['REC_SPG_INNB'])[:7])
def split5(x):
  return int(str(x['REC_SPG_INNB'])[7:11])
def split6(x):
  return int(str(x['REC_SPG_INNB'])[11:])

train['send_1'] = train.apply(split1,axis=1)
train['send_2'] = train.apply(split2,axis=1)
train['send_3'] = train.apply(split3,axis=1)
train['rec_1'] = train.apply(split4,axis=1)
train['rec_2'] = train.apply(split5,axis=1)
train['rec_3'] = train.apply(split6,axis=1)

train = train.drop(columns = ['SEND_SPG_INNB', 'REC_SPG_INNB', 'INVC_CONT'], axis = 1)
train
```





  <div id="df-b3c23fce-4010-431c-b844-5f9328f77c3b">
    <div class="colab-df-container">
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
      <th>send_1</th>
      <th>send_2</th>
      <th>send_3</th>
      <th>rec_1</th>
      <th>rec_2</th>
      <th>rec_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1129000</td>
      <td>140</td>
      <td>45300</td>
      <td>5011000</td>
      <td>2200</td>
      <td>46300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1135000</td>
      <td>90</td>
      <td>51200</td>
      <td>5011000</td>
      <td>1780</td>
      <td>37300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1135000</td>
      <td>300</td>
      <td>93100</td>
      <td>5011000</td>
      <td>2650</td>
      <td>91400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1154500</td>
      <td>20</td>
      <td>14200</td>
      <td>5011000</td>
      <td>3150</td>
      <td>87400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1165000</td>
      <td>210</td>
      <td>8300</td>
      <td>5011000</td>
      <td>1770</td>
      <td>51200</td>
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
      <th>31995</th>
      <td>5011001</td>
      <td>600</td>
      <td>63300</td>
      <td>2635000</td>
      <td>260</td>
      <td>53400</td>
    </tr>
    <tr>
      <th>31996</th>
      <td>5011001</td>
      <td>950</td>
      <td>42400</td>
      <td>1168000</td>
      <td>170</td>
      <td>2200</td>
    </tr>
    <tr>
      <th>31997</th>
      <td>5011001</td>
      <td>1080</td>
      <td>36200</td>
      <td>4119700</td>
      <td>80</td>
      <td>12100</td>
    </tr>
    <tr>
      <th>31998</th>
      <td>5011001</td>
      <td>1150</td>
      <td>11400</td>
      <td>1132000</td>
      <td>150</td>
      <td>85100</td>
    </tr>
    <tr>
      <th>31999</th>
      <td>5011001</td>
      <td>1160</td>
      <td>66400</td>
      <td>4719000</td>
      <td>5940</td>
      <td>22400</td>
    </tr>
  </tbody>
</table>
<p>32000 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b3c23fce-4010-431c-b844-5f9328f77c3b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b3c23fce-4010-431c-b844-5f9328f77c3b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b3c23fce-4010-431c-b844-5f9328f77c3b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
test['send_1'] = test.apply(split1,axis=1)
test['send_2'] = test.apply(split2,axis=1)
test['send_3'] = test.apply(split3,axis=1)
test['rec_1'] = test.apply(split4,axis=1)
test['rec_2'] = test.apply(split5,axis=1)
test['rec_3'] = test.apply(split6,axis=1)

test = test.drop(columns = ['SEND_SPG_INNB', 'REC_SPG_INNB'], axis = 1)
test
```





  <div id="df-ec0bb9e1-b133-4b10-9554-41ab50e9e6db">
    <div class="colab-df-container">
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
      <th>send_1</th>
      <th>send_2</th>
      <th>send_3</th>
      <th>rec_1</th>
      <th>rec_2</th>
      <th>rec_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5013000</td>
      <td>430</td>
      <td>28400</td>
      <td>1165000</td>
      <td>210</td>
      <td>97200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5013000</td>
      <td>440</td>
      <td>16100</td>
      <td>1154500</td>
      <td>20</td>
      <td>66400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5013000</td>
      <td>2050</td>
      <td>30200</td>
      <td>4139000</td>
      <td>1020</td>
      <td>13200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5013000</td>
      <td>2050</td>
      <td>30200</td>
      <td>4221000</td>
      <td>400</td>
      <td>93400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5013000</td>
      <td>2680</td>
      <td>11400</td>
      <td>2726000</td>
      <td>40</td>
      <td>17100</td>
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
      <th>4635</th>
      <td>5013000</td>
      <td>8580</td>
      <td>4400</td>
      <td>4725000</td>
      <td>7190</td>
      <td>72200</td>
    </tr>
    <tr>
      <th>4636</th>
      <td>5013000</td>
      <td>8700</td>
      <td>18300</td>
      <td>2826000</td>
      <td>1060</td>
      <td>75300</td>
    </tr>
    <tr>
      <th>4637</th>
      <td>5013000</td>
      <td>8970</td>
      <td>86300</td>
      <td>4311100</td>
      <td>340</td>
      <td>4300</td>
    </tr>
    <tr>
      <th>4638</th>
      <td>5013000</td>
      <td>9020</td>
      <td>65100</td>
      <td>4145000</td>
      <td>130</td>
      <td>11200</td>
    </tr>
    <tr>
      <th>4639</th>
      <td>5013000</td>
      <td>9260</td>
      <td>78200</td>
      <td>2714000</td>
      <td>1740</td>
      <td>62400</td>
    </tr>
  </tbody>
</table>
<p>4640 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ec0bb9e1-b133-4b10-9554-41ab50e9e6db')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ec0bb9e1-b133-4b10-9554-41ab50e9e6db button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ec0bb9e1-b133-4b10-9554-41ab50e9e6db');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4640 entries, 0 to 4639
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   send_1  4640 non-null   int64
     1   send_2  4640 non-null   int64
     2   send_3  4640 non-null   int64
     3   rec_1   4640 non-null   int64
     4   rec_2   4640 non-null   int64
     5   rec_3   4640 non-null   int64
    dtypes: int64(6)
    memory usage: 217.6 KB
    


```python
X_train, X_val, y_train, y_val = train_test_split(train, INVC_CONT, shuffle = True)
```


```python
X_train
```





  <div id="df-77f341b8-7602-4f28-b2d9-2ff99eff9b85">
    <div class="colab-df-container">
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
      <th>send_1</th>
      <th>send_2</th>
      <th>send_3</th>
      <th>rec_1</th>
      <th>rec_2</th>
      <th>rec_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7452</th>
      <td>5011001</td>
      <td>70</td>
      <td>11200</td>
      <td>4812100</td>
      <td>1740</td>
      <td>31100</td>
    </tr>
    <tr>
      <th>8798</th>
      <td>4148000</td>
      <td>6260</td>
      <td>30400</td>
      <td>5013000</td>
      <td>8230</td>
      <td>99200</td>
    </tr>
    <tr>
      <th>24761</th>
      <td>5013000</td>
      <td>7350</td>
      <td>22300</td>
      <td>4117100</td>
      <td>200</td>
      <td>5300</td>
    </tr>
    <tr>
      <th>23357</th>
      <td>5011000</td>
      <td>1370</td>
      <td>30100</td>
      <td>4111100</td>
      <td>240</td>
      <td>58200</td>
    </tr>
    <tr>
      <th>22791</th>
      <td>5011000</td>
      <td>4240</td>
      <td>78200</td>
      <td>1174000</td>
      <td>210</td>
      <td>72300</td>
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
      <th>18356</th>
      <td>4127100</td>
      <td>480</td>
      <td>6400</td>
      <td>5011000</td>
      <td>2140</td>
      <td>25400</td>
    </tr>
    <tr>
      <th>10322</th>
      <td>5013000</td>
      <td>8030</td>
      <td>23100</td>
      <td>4824000</td>
      <td>4530</td>
      <td>78300</td>
    </tr>
    <tr>
      <th>13619</th>
      <td>5013000</td>
      <td>6220</td>
      <td>57100</td>
      <td>4139000</td>
      <td>840</td>
      <td>62400</td>
    </tr>
    <tr>
      <th>31008</th>
      <td>5011000</td>
      <td>3840</td>
      <td>15100</td>
      <td>1165000</td>
      <td>200</td>
      <td>20400</td>
    </tr>
    <tr>
      <th>21047</th>
      <td>5013000</td>
      <td>7710</td>
      <td>32200</td>
      <td>4221000</td>
      <td>260</td>
      <td>29300</td>
    </tr>
  </tbody>
</table>
<p>24000 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-77f341b8-7602-4f28-b2d9-2ff99eff9b85')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-77f341b8-7602-4f28-b2d9-2ff99eff9b85 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-77f341b8-7602-4f28-b2d9-2ff99eff9b85');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




GAN Generator를 이용한 데이터 추가 확보


```python
#generate data
new_train1, new_target1 = OriginalGenerator().generate_data_pipe(X_train, pd.DataFrame(y_train), X_val, )
new_train2, new_target2 = GANGenerator().generate_data_pipe(X_train, pd.DataFrame(y_train), X_val, )

new_train3, new_target3 = GANGenerator(gen_x_times=1.1, cat_cols=None, bot_filter_quantile=0.001,
                                       top_filter_quantile=0.999,
                                       is_post_process=True,
                                       pregeneration_frac=2, 
                                       only_generated_data=False).generate_data_pipe(X_train, pd.DataFrame(y_train),
                                                                      X_val, deep_copy=True,
                                                                      only_adversarial=False,
                                                                      use_adversarial=True)
```


```python
new_train1
```





  <div id="df-a2169f0a-e470-4d0d-b31a-9edc0ce56391">
    <div class="colab-df-container">
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
      <th>send_1</th>
      <th>send_2</th>
      <th>send_3</th>
      <th>rec_1</th>
      <th>rec_2</th>
      <th>rec_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5013000</td>
      <td>7310</td>
      <td>55200</td>
      <td>1111000</td>
      <td>210</td>
      <td>5400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5013000</td>
      <td>7310</td>
      <td>55200</td>
      <td>1111000</td>
      <td>170</td>
      <td>2400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5013000</td>
      <td>7310</td>
      <td>55200</td>
      <td>1111000</td>
      <td>170</td>
      <td>2400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5013000</td>
      <td>7310</td>
      <td>55200</td>
      <td>1111000</td>
      <td>210</td>
      <td>5400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5013000</td>
      <td>8520</td>
      <td>45200</td>
      <td>1114000</td>
      <td>20</td>
      <td>2400</td>
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
      <th>71696</th>
      <td>5013000</td>
      <td>7350</td>
      <td>22300</td>
      <td>4115000</td>
      <td>360</td>
      <td>12200</td>
    </tr>
    <tr>
      <th>71697</th>
      <td>5013000</td>
      <td>7350</td>
      <td>22300</td>
      <td>4117300</td>
      <td>220</td>
      <td>43100</td>
    </tr>
    <tr>
      <th>71698</th>
      <td>5013000</td>
      <td>8180</td>
      <td>21200</td>
      <td>4117100</td>
      <td>320</td>
      <td>16300</td>
    </tr>
    <tr>
      <th>71699</th>
      <td>5013000</td>
      <td>8180</td>
      <td>21200</td>
      <td>4117100</td>
      <td>320</td>
      <td>19200</td>
    </tr>
    <tr>
      <th>71700</th>
      <td>5013000</td>
      <td>8180</td>
      <td>21200</td>
      <td>4117100</td>
      <td>320</td>
      <td>19200</td>
    </tr>
  </tbody>
</table>
<p>71701 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a2169f0a-e470-4d0d-b31a-9edc0ce56391')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a2169f0a-e470-4d0d-b31a-9edc0ce56391 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a2169f0a-e470-4d0d-b31a-9edc0ce56391');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
new_target1.mean()
```




    4.770156622641246




```python
new_train2
```





  <div id="df-f711947b-adaa-4268-85e9-2556c8835236">
    <div class="colab-df-container">
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
      <th>send_1</th>
      <th>send_2</th>
      <th>send_3</th>
      <th>rec_1</th>
      <th>rec_2</th>
      <th>rec_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3533130</td>
      <td>1204</td>
      <td>54683</td>
      <td>3458552</td>
      <td>434</td>
      <td>76104</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4008187</td>
      <td>1886</td>
      <td>34956</td>
      <td>3450662</td>
      <td>3796</td>
      <td>60613</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4452543</td>
      <td>4723</td>
      <td>57614</td>
      <td>4658787</td>
      <td>8968</td>
      <td>2099</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1245916</td>
      <td>2019</td>
      <td>67895</td>
      <td>2612794</td>
      <td>7677</td>
      <td>30848</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4988850</td>
      <td>4166</td>
      <td>29218</td>
      <td>3367996</td>
      <td>3590</td>
      <td>34707</td>
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
      <th>43825</th>
      <td>5011000</td>
      <td>780</td>
      <td>68400</td>
      <td>1130500</td>
      <td>140</td>
      <td>96400</td>
    </tr>
    <tr>
      <th>43826</th>
      <td>5011000</td>
      <td>780</td>
      <td>68400</td>
      <td>1121500</td>
      <td>140</td>
      <td>63300</td>
    </tr>
    <tr>
      <th>43827</th>
      <td>5011000</td>
      <td>2130</td>
      <td>90100</td>
      <td>1165000</td>
      <td>110</td>
      <td>41200</td>
    </tr>
    <tr>
      <th>43828</th>
      <td>5011000</td>
      <td>2260</td>
      <td>90200</td>
      <td>1138000</td>
      <td>250</td>
      <td>96400</td>
    </tr>
    <tr>
      <th>43829</th>
      <td>5011001</td>
      <td>700</td>
      <td>65300</td>
      <td>1156000</td>
      <td>230</td>
      <td>23100</td>
    </tr>
  </tbody>
</table>
<p>43830 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f711947b-adaa-4268-85e9-2556c8835236')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f711947b-adaa-4268-85e9-2556c8835236 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f711947b-adaa-4268-85e9-2556c8835236');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
new_target2.mean()
```




    7.6397444672598676




```python
new_train3
```





  <div id="df-9ff03579-c416-42bd-9ed1-140bdccd0874">
    <div class="colab-df-container">
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
      <th>send_1</th>
      <th>send_2</th>
      <th>send_3</th>
      <th>rec_1</th>
      <th>rec_2</th>
      <th>rec_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3972203</td>
      <td>817</td>
      <td>27774</td>
      <td>4251749</td>
      <td>8251</td>
      <td>18267</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4412065</td>
      <td>2898</td>
      <td>68709</td>
      <td>4055151</td>
      <td>2628</td>
      <td>33154</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4999196</td>
      <td>5346</td>
      <td>81321</td>
      <td>4213887</td>
      <td>181</td>
      <td>19383</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3236699</td>
      <td>470</td>
      <td>84296</td>
      <td>4853502</td>
      <td>9144</td>
      <td>75495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2146672</td>
      <td>1095</td>
      <td>80022</td>
      <td>2595826</td>
      <td>3572</td>
      <td>26934</td>
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
      <th>41517</th>
      <td>3114000</td>
      <td>260</td>
      <td>13300</td>
      <td>5011000</td>
      <td>3130</td>
      <td>78300</td>
    </tr>
    <tr>
      <th>41518</th>
      <td>4113500</td>
      <td>90</td>
      <td>78300</td>
      <td>5011001</td>
      <td>60</td>
      <td>25200</td>
    </tr>
    <tr>
      <th>41519</th>
      <td>2650000</td>
      <td>110</td>
      <td>87300</td>
      <td>5013000</td>
      <td>2360</td>
      <td>31300</td>
    </tr>
    <tr>
      <th>41520</th>
      <td>4148000</td>
      <td>6900</td>
      <td>15300</td>
      <td>5011000</td>
      <td>2630</td>
      <td>60400</td>
    </tr>
    <tr>
      <th>41521</th>
      <td>2638000</td>
      <td>360</td>
      <td>9100</td>
      <td>5011000</td>
      <td>3140</td>
      <td>97300</td>
    </tr>
  </tbody>
</table>
<p>41522 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9ff03579-c416-42bd-9ed1-140bdccd0874')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9ff03579-c416-42bd-9ed1-140bdccd0874 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9ff03579-c416-42bd-9ed1-140bdccd0874');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
new_target3.mean()
```




    6.781344829247146




```python
#OriginalGenerator를 사용한 new_train1과 new_target1이 기존 데이터와 가장 유사하므로 채택해 기존 데이터와 병합

train = pd.concat([train, new_train1], axis = 0)
INVC_CONT = pd.concat([INVC_CONT, new_target1], axis = 0)
```

**LGBM**


```python
from lightgbm import LGBMRegressor

model = LGBMRegressor()
model.fit(train, INVC_CONT)
```




    LGBMRegressor()




```python
pred = model.predict(test)
pred
```




    array([7.09578575, 5.98793529, 7.03175209, ..., 4.43203345, 4.59217849,
           4.79222297])




```python
submission['INVC_CONT'] = pred
```


```python
submission['INVC_CONT'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd5dfec6550>




    
![png](output_25_1.png)
    



```python
#음수 값이 있으면 대치

submission['INVC_CONT'] = submission['INVC_CONT'].apply(lambda x: 0 if x < 0 else x)
```


```python
submission[submission['INVC_CONT'] < 3]
```





  <div id="df-29b3512c-04e4-4420-b91f-6d59072ec060">
    <div class="colab-df-container">
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
      <th>index</th>
      <th>INVC_CONT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>123</th>
      <td>32123</td>
      <td>0.437781</td>
    </tr>
    <tr>
      <th>124</th>
      <td>32124</td>
      <td>1.141195</td>
    </tr>
    <tr>
      <th>755</th>
      <td>32755</td>
      <td>2.839438</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>33007</td>
      <td>2.959767</td>
    </tr>
    <tr>
      <th>2503</th>
      <td>34503</td>
      <td>0.421015</td>
    </tr>
    <tr>
      <th>2801</th>
      <td>34801</td>
      <td>2.794173</td>
    </tr>
    <tr>
      <th>3085</th>
      <td>35085</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>3709</th>
      <td>35709</td>
      <td>1.076666</td>
    </tr>
    <tr>
      <th>4333</th>
      <td>36333</td>
      <td>1.620776</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-29b3512c-04e4-4420-b91f-6d59072ec060')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-29b3512c-04e4-4420-b91f-6d59072ec060 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-29b3512c-04e4-4420-b91f-6d59072ec060');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
submission['INVC_CONT'].mean()
```




    4.786530414644545




```python
submission
```





  <div id="df-7dea47fc-a678-4673-9cf0-f2f50627bdb2">
    <div class="colab-df-container">
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
      <th>index</th>
      <th>INVC_CONT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32000</td>
      <td>7.095786</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32001</td>
      <td>5.987935</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32002</td>
      <td>7.031752</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32003</td>
      <td>5.065661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32004</td>
      <td>5.636205</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4635</th>
      <td>36635</td>
      <td>4.389590</td>
    </tr>
    <tr>
      <th>4636</th>
      <td>36636</td>
      <td>4.658016</td>
    </tr>
    <tr>
      <th>4637</th>
      <td>36637</td>
      <td>4.432033</td>
    </tr>
    <tr>
      <th>4638</th>
      <td>36638</td>
      <td>4.592178</td>
    </tr>
    <tr>
      <th>4639</th>
      <td>36639</td>
      <td>4.792223</td>
    </tr>
  </tbody>
</table>
<p>4640 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7dea47fc-a678-4673-9cf0-f2f50627bdb2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7dea47fc-a678-4673-9cf0-f2f50627bdb2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7dea47fc-a678-4673-9cf0-f2f50627bdb2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
submission.to_csv('/content/gdrive/My Drive/Data Analysis/물류 유통량 예측 경진대회/submission.csv',index = False)
```
