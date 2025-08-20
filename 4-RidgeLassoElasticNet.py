import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV


df = pd.read_csv("4-Algerian_forest_fires_dataset.csv")

print(df.columns) # Sütun isimlerini gösterir

print(df.head()) # İlk 5 satırı gösterir

print(df.info()) # DataFrame hakkında bilgi verir

print(df.isnull().sum()) # Her sütundaki eksik değerlerin sayısını gösterir

print(df[df.isnull().any(axis=1)]) # Eksik değer içeren satırları gösterir

print(df.shape) # satır ve sütun sayısını gösterir

df.drop(122, inplace=True)  # 122 numaralı satırı kaldırır

df.loc[:123, "Region"] = 0 # 0-122 kadar 0 olarak atandı
df.loc[123:, "Region"] = 1 # 123 ten sonrakadar 1 olarak atandı

df = df.dropna().reset_index(drop=True) # Eksik değerleri kaldır ve indexi sıfırla
 
print(df.isnull().sum())

df.columns = df.columns.str.strip() # boşlukları kaldırma

print(df["day"].unique()) # Gün sütunundaki benzersiz değerleri gösterir

df[df["day"] == "day"]

df.drop(122, inplace=True)

df[["day", "month", "year", "Temperature", "RH", "Ws"]] = df[["day", "month", "year", "Temperature", "RH", "Ws"]].astype(int)  # Dtype dönüşümü

print(df.info())

df[['Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']] = df[['Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']].astype(float) # Dtype dönüşümü

print(df.info())

print(df.describe()) #istatistiksel özet

df['Classes'].value_counts() # Sınıfların sayısını gösterir

df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 0, 1) # Sınıfları 0 ve 1 olarak değiştirir

df['Classes'].value_counts(normalize=True)*100 # Sınıfların yüzdelik dağılımını gösterir

print(df.corr()) # Korelasyon matrisini gösterir

sns.heatmap(df.corr(), annot=True, cmap='coolwarm') # Korelasyon matrisini görselleştirir
plt.title("Correlation Heatmap")
plt.show() # Grafiği gösterir

#depend ve independent değişkenleri ayırma
X=df.drop("FWI", axis=1) # FWI bağımlı değişkeni
y=df["FWI"] # FWI bağımlı değişkeni

print(X.head()) 

# Bağımlı ve bağımsız değişkenleri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15) # Eğitim ve test setlerini ayırma
print(X_train.shape, X_test.shape) # Eğitim ve test setlerinin boyutlarını gösterir
print(X_train.corr()) # Eğitim setinin korelasyon matrisini gösterir

#redundancy ve multicollinearity kontrolü ,overfitting riskini azaltma

print(X_train.corr().iloc[0,2])

def correlation_for_dropping(df, threshold):
    columns_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                columns_to_drop.add(corr.columns[i])
    return columns_to_drop

columns_dropping = correlation_for_dropping(X_train, 0.85) #0.85 eşik değerini kullanarak korelasyon matrisinden sütunları kaldırma
print("Columns to drop based on correlation threshold of 0.85:", columns_dropping)

X_train.drop(columns_dropping, axis= 1, inplace = True) #x_train'den sütunları kaldırma
X_test.drop(columns_dropping, axis= 1, inplace = True) # x_test'ten sütunları kaldırma

print("X_train shape after dropping columns:", X_train.shape)
print("X_test shape after dropping columns:", X_test.shape)

#scaling
scaler = StandardScaler() # Standartlaştırma için scaler oluşturma
X_train_scaled = scaler.fit_transform(X_train) # Eğitim setini standartlaştırma
X_test_scaled = scaler.transform(X_test) # Test setini standartlaştırma

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

plt.subplots(figsize=(15,5)) # subplot boyutlarını ayarlama
plt.subplot(1,2,1) # İlk subplot
sns.boxplot(data=X_train) # X_train için boxplot oluşturma
plt.title("X_train") # Boxplot başlığı
plt.subplot(1,2,2) 
sns.boxplot(data=X_train_scaled) 
plt.title("X_train_scaled") 
plt.show()

#farklı regresyon modelleri kıyaslama

linear = LinearRegression() # Lineer regresyon modeli oluşturma
linear.fit(X_train_scaled, y_train) # Modeli eğitim verisi ile eğitme
print("Linear Regression Coefficients:", linear.coef_) # Modelin katsayılarını gösterme
y_pred = linear.predict(X_test_scaled) # Test verisi üzerinde tahmin yapma
print("Predictions:", y_pred[:5]) # İlk 5 tahmini gösterme
mae = mean_absolute_error(y_test, y_pred) # Ortalama mutlak hata hesaplama
mse = mean_squared_error(y_test, y_pred) # Ortalama kare hata hesaplama
score = r2_score(y_test, y_pred)

print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test, y_pred)
plt.show()

#lasso regresyon modeli
lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test, y_pred)
plt.show()

# Ridge regresyon modeli
ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test, y_pred)
plt.show()

# ElasticNet regresyon modeli

elastic = ElasticNet()
elastic.fit(X_train_scaled, y_train)
y_pred = elastic.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test, y_pred)
plt.show()

#sonuçları karşılaştırma
results = pd.DataFrame({
    "Model": ["Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression"],
    "MAE": [mean_absolute_error(y_test, linear.predict(X_test_scaled)),
            mean_absolute_error(y_test, lasso.predict(X_test_scaled)),
            mean_absolute_error(y_test, ridge.predict(X_test_scaled)),
            mean_absolute_error(y_test, elastic.predict(X_test_scaled))],
    "MSE": [mean_squared_error(y_test, linear.predict(X_test_scaled)),
            mean_squared_error(y_test, lasso.predict(X_test_scaled)),
            mean_squared_error(y_test, ridge.predict(X_test_scaled)),
            mean_squared_error(y_test, elastic.predict(X_test_scaled))],
    "R2 Score": [r2_score(y_test, linear.predict(X_test_scaled)),
                 r2_score(y_test, lasso.predict(X_test_scaled)),
                 r2_score(y_test, ridge.predict(X_test_scaled)),
                 r2_score(y_test, elastic.predict(X_test_scaled))]
})
print(results)

# lasso cross validation

lassocv = LassoCV(cv=5) #cross-validation=5 için LassoCV modeli oluşturma
lassocv.fit(X_train_scaled, y_train)
y_pred = lassocv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test, y_pred)
plt.show()

print(lassocv.alpha_) # En iyi alpha değerini gösterir
print(lassocv.alphas_) # LassoCV modelinin alpha değerlerini gösterir

# ridge cross validation

from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(cv=5)
ridgecv.fit(X_train_scaled, y_train)
y_pred = ridgecv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test, y_pred)
plt.show()

# elastic net cross validation
from sklearn.linear_model import ElasticNetCV
elasticnetcv = ElasticNetCV(cv=5)
elasticnetcv.fit(X_train_scaled, y_train)
y_pred = elasticnetcv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test, y_pred)
plt.show()

print(elasticnetcv.alpha_) 
print(elasticnetcv.alphas_)