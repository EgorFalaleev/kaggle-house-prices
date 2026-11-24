# House Prices Prediction

В данном репозитории представлено решение соревнования [Kaggle — House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).

## Что реализовано

- **EDA и предобработка данных:** анализ датасета, выявление пропусков и корреляций признаков, обоснования для feature engineering (`notebooks/01_eda.ipynb`).
- **Classic ML:** Linear Regression (baseline), Lasso, Ridge, ElasticNet, SGD, Decision Tree, Random Forest, XGBoost и CatBoost (`notebooks/02_models.ipynb`).
- **Оценка и подбор:** Кросс-валидация с помощью `KFold`, подбор гиперпараметров через `GridSearchCV` (`notebooks/02_models.ipynb`).
- **Deep Learning:** MLP на PyTorch, эксперименты с BatchNorm и Dropout, Early Stopping (`notebooks/02_models.ipynb`).
- **Kaggle submission:** инференс на `test.csv` и сохранение файла для валидации решения на Kaggle (`main.py`, `main_dl.py`).

## Установка и запуск

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Запуск инференса с формированием файла решения:

Для Classic ML моделей:
```bash
python main.py
```

Модель выбирается в зависимости от поля `general.model` в файле `config.py`

Для Deep Learning моделей:
```bash
python main_dl.py
```

Файл с решением сохраняется в папку `submissions/`

## Результаты

Сравнение моделей проводится по RMSE из результатов кросс-валидации.

| Модель | CV Score | Kaggle Score |
|---|---|---|
| Linear Regression (baseline) | 0.14027 | 0.14821 |
| Ridge | 0.13976 | 0.14697 |
| ElasticNet | 0.13991 | 0.14699 |
| CatBoost | 0.13891 | 0.14794 |
| Lasso | 0.13951 | 0.14825 |
| XGBoost | 0.14047 | 0.14939 |
| SGD | 0.13964 | 0.15236 |
| Random Forest | 0.15088 | 0.15865 |
| Decition Tree | 0.19912 | 0.20792 |
| MLP (PyTorch) | — | 0.56571 |

