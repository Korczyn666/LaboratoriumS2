import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# wczytaj zbiór danych
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
data = pd.read_csv(url, delimiter=';')
#data = pd.read_csv('winequality-white.csv', delimiter=';')

# wybierz kolumny z wartościami numerycznymi
numeric_cols = data.select_dtypes(include=np.number).columns

# przed normalizacją
stats_before = pd.DataFrame({'średnia': data[numeric_cols].mean(),
                             'odchylenie standardowe': data[numeric_cols].std(),
                             'min': data[numeric_cols].min(),
                             'max': data[numeric_cols].max(),
                             'mediana': data[numeric_cols].median(),
                             'Q1': data[numeric_cols].quantile(0.25),
                             'Q3': data[numeric_cols].quantile(0.75)})
print('Statystyki przed normalizacją:\n', stats_before)

# normalizuj dane
data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()

# po normalizacji
stats_after = pd.DataFrame({'średnia': data[numeric_cols].mean(),
                            'odchylenie standardowe': data[numeric_cols].std(),
                            'min': data[numeric_cols].min(),
                            'max': data[numeric_cols].max(),
                            'mediana': data[numeric_cols].median(),
                            'Q1': data[numeric_cols].quantile(0.25),
                            'Q3': data[numeric_cols].quantile(0.75)})
print('Statystyki po normalizacji:\n', stats_after)

# wizualizacja rozkładu
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 6, i % 10 + 1)
    sns.histplot(data[col], kde=True)
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# korelacja
corr_before = data[numeric_cols].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_before, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Macierz korelacji przed normalizacją')
plt.show()

corr_after = data[numeric_cols].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_after, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Macierz korelacji po normalizacji')
plt.show()