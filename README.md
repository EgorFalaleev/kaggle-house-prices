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

Для Deep Learning моделей:
```bash
python main_dl.py
```

Файл с решением сохраняется в папку `submissions/`

## Результаты

Сравнение моделей проводится по RMSE из результатов кросс-валидации.

| Модель | CV Accuracy (mean) | Kaggle Accuracy |
|---|---|---|
| Logistic Regression (baseline) | 0.14135 | 0.20552 |
| CatBoost | 0.13987 | 0.16004 |
| XGBoost | 0.14157 | 0.16700 |
| Random Forest | 0.15114 | 0.17425 |
| Lasso | 0.14026 | 0.19936 |
| Ridge | 0.13968 | 0.20581 |
| ElasticNet | 0.13974 | 0.20625 |
| Decition Tree | 0.19912 | 0.21502 |
| SGD | 0.14056 | 0.22288 |
| MLP (PyTorch) | — | 0.56571 |

