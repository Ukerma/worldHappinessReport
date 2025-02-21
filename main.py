"""
Proje Başlığı: Dünya Mutluluk Raporu Analizi
Yazar: Umut Kerim ACAR ve Tuna DURUKAN
Tarih: Ocak 2025
Kullanılan Data Seti: World Happiness Report (https://www.kaggle.com/datasets/unsdsn/world-happiness)
Kullanılan Programlama Dili: Python
Kütüphaneler:
- pandas: Veri işleme ve analizi.
- numpy: Sayısal işlemler.
- matplotlib & seaborn: Veri görselleştirme.
- scipy: İstatistiksel analizler.
- scikit-learn: Veri ön işleme ve modelleme.
Açıklama:
Bu proje, Dünya Mutluluk Raporu veri seti kullanılarak ülkelerin yaşam kalitesi, ekonomik refah ve sosyal destek gibi faktörlerin mutluluk seviyesiyle ilişkisini analiz etmeyi amaçlamaktadır. 
Projede istatistiksel özetleme, korelasyon analizi, regresyon modeli ve normalizasyon gibi yöntemler kullanılmıştır. 
Analiz sonuçları görselleştirme yöntemleriyle detaylandırılmıştır.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, ttest_ind, pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score

# ════ Veri Setini Yükleme ════
filePath = "World Happiness Report.csv"
happinessData = pd.read_csv(filePath)

# ════ Yıl Kolonunun Veri Tipini Düzenleme ════
happinessData['Year'] = happinessData['Year'].astype(int)

# ════ Eksik Değerleri Silme ════
happinessData.dropna(inplace=True)

# ════ Analizde Kullanılacak Kolonlar ════
columnsToAnalyze = [
    'Life Ladder',
    'Log GDP Per Capita',
    'Social Support',
    'Healthy Life Expectancy At Birth',
    'Freedom To Make Life Choices',
    'Generosity',
    'Perceptions Of Corruption'
]

# ════ Temel İstatistiksel Analizler ════
statisticsSummary = happinessData[columnsToAnalyze].agg(['mean', 'median', 'var', 'std']).T
statisticsSummary['skewness'] = happinessData[columnsToAnalyze].skew()
statisticsSummary['kurtosis'] = happinessData[columnsToAnalyze].kurtosis()
modeValues = happinessData[columnsToAnalyze].mode().iloc[0]
statisticsSummary['mode'] = modeValues

print("\n═══════════════════════════════════════════════════════════════════════╣ Temel İstatistiksel Özellikler ╠══════════════════════════════════════════════════════════════════════")
print(statisticsSummary.to_string())

# ════ Kovaryans Matrisi ════
print("\n═════════════════════════════════════════════════════════════════════════════╣ Kovaryans Matrisi ╠═════════════════════════════════════════════════════════════════════════════")
print(covarianceMatrix := happinessData[columnsToAnalyze].cov().round(2))

# ════ Normalizasyon ════
scaler = MinMaxScaler()
normalizedData = pd.DataFrame(
    scaler.fit_transform(happinessData[columnsToAnalyze]),
    columns=columnsToAnalyze
)

# ════ Küme Sayısını Belirleme (Değişken Sayısı) ════
optimal_clusters = len(columnsToAnalyze)
print(f"Optimal Küme Sayısı (Değişken Sayısı): {optimal_clusters}")

# ════ K-means Clustering ════
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
happinessData['Cluster'] = kmeans.fit_predict(normalizedData)

# ════ Kümeleme Sonuçları ════
print("\n══════════════════════════════════════════════════════════════════════╣ K-means Kümeleme Sonuçları ╠══════════════════════════════════════════════════════════════════════")
print(happinessData[['Regional Indicator', 'Cluster']].head())

# ════ PCA ile veriyi 2 boyuta indir ════
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalizedData)
happinessData['PCA1'] = pca_result[:, 0]
happinessData['PCA2'] = pca_result[:, 1]

# ════ Kümelere değişken adlarını atıyoruz ════
cluster_names = {
    0: "Life Ladder",
    1: "Log GDP Per Capita",
    2: "Social Support",
    3: "Healthy Life Expectancy",
    4: "Freedom To Make Life Choices",
    5: "Generosity",
    6: "Perceptions Of Corruption"
}

# ════ Sayısal kümeleri isimlerle eşleştir ════
happinessData['Cluster_Name'] = happinessData['Cluster'].map(cluster_names)

# ════ PCA sonuçlarını görselleştirme ════
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=happinessData['PCA1'],
    y=happinessData['PCA2'],
    hue=happinessData['Cluster_Name'],  # Kümelerin isimlerini kullan
    palette='tab10',
    alpha=0.7
)
plt.title('K-means Kümeleme: PCA ile Görselleştirme (Küme İsimleri)', fontsize=16)
plt.xlabel('PCA1', fontsize=14)
plt.ylabel('PCA2', fontsize=14)
plt.legend(title='Cluster', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# ════ Kişi Başına Düşen GSYİH ile Mutluluk Düzeyi İlişkisi ════
plt.figure(figsize=(15, 6))
sns.scatterplot(
    data=happinessData,
    x='Log GDP Per Capita',
    y='Life Ladder',
    hue='Regional Indicator',
    alpha=0.7
)
plt.title('Kişi Başına Düşen GSYİH ve Mutluluk Seviyesi', fontsize=16)
plt.xlabel('Log GDP Per Capita', fontsize=14)
plt.ylabel('Life Ladder', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ════ Relative Uncertainty Index (RUI) Hesaplama ════
# def calculateRui(df, columns):
#     ruiResults = {}
#     for col in columns:
#         mean = df[col].mean()
#         std = df[col].std()
#         rui = std / mean if mean != 0 else 0
#         ruiResults[col] = rui
#     return ruiResults

def calculateRui(df, columns):
    ruiResults = {}
    num_data_points = len(df)
    num_clusters = len(columns)

    for col in columns:
        membership_values = df[col]
        if membership_values.min() > 0:
            numerator = np.sum(membership_values * np.log(membership_values))
            denominator = num_data_points * np.log(num_clusters)
            rui = numerator / denominator
        else:
            rui = 0
        ruiResults[col] = rui
    return ruiResults


ruiValues = calculateRui(happinessData, columnsToAnalyze)

print("\n══════════════════════════════════════════════════════════════════════╣ Relative Uncertainty Index (RUI) ╠══════════════════════════════════════════════════════════════════════")
for col, rui in ruiValues.items():
    print(f"» {col}: {rui:.4f}")

plt.figure(figsize=(10, 6))
bars = plt.bar(ruiValues.keys(), ruiValues.values(), color='skyblue', edgecolor='black')
plt.title('Relative Uncertainty Index (RUI)', fontsize=16)
plt.xlabel('Değişkenler', fontsize=14)
plt.ylabel('RUI Değeri', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

plt.show()

# ════ Korelasyon Matrisi ════
plt.figure(figsize=(10, 8))
sns.heatmap(happinessData[columnsToAnalyze].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title('Korelasyon Matrisi', fontsize=16)
plt.show()

# ════ Zaman Serisi Analizi: Yıllık Ortalama Mutluluk Seviyesi ════
plt.figure(figsize=(12, 6))
yearlyTrend = happinessData.groupby('Year')['Life Ladder'].mean()
plt.plot(yearlyTrend, marker='o', linestyle='-', color='coral')
plt.title('Yıllık Ortalama Mutluluk Seviyesi', fontsize=16)
plt.xlabel('Yıl', fontsize=14)
plt.ylabel('Ortalama Mutluluk Seviyesi (Life Ladder)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ════ En Mutlu ve En Mutsuz Bölgelerin Karşılaştırması ════
happiestRegion = happinessData.groupby('Regional Indicator')['Life Ladder'].mean().idxmax()
leastHappyRegion = happinessData.groupby('Regional Indicator')['Life Ladder'].mean().idxmin()

print(f"\n» En Mutlu Bölge: {happiestRegion}")
print(f"» En Mutsuz Bölge: {leastHappyRegion}")

regionalSummary = happinessData.groupby('Regional Indicator')[columnsToAnalyze].mean()
regionalSummary.loc[[happiestRegion, leastHappyRegion]].plot(kind='bar', figsize=(12, 6))
plt.title('En Mutlu ve En Mutsuz Bölgelerin Karşılaştırması', fontsize=16)
plt.ylabel('Değerler', fontsize=14)
plt.xticks(rotation=0)
plt.legend(loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ════ Bölgesel Bazda Mutluluk Karşılaştırması ════
plt.figure(figsize=(12, 6))
regionalMeans = happinessData.groupby('Regional Indicator')['Life Ladder'].mean().sort_values()
sns.barplot(x=regionalMeans.values, y=regionalMeans.index, palette="viridis", hue=regionalMeans.index)
plt.title('Bölgelere Göre Ortalama Mutluluk Seviyesi', fontsize=16)
plt.xlabel('Ortalama Life Ladder', fontsize=14)
plt.ylabel('Bölgeler', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# ════ Uç Değer Analizi ════
plt.figure(figsize=(12, 8))
sns.violinplot(data=happinessData[columnsToAnalyze], orient="h", palette="muted", density_norm="width", inner="quartile")
plt.title('Uç Değerlerin Görselleştirilmesi (Violin Plot)', fontsize=16)
plt.xlabel('Değerler', fontsize=14)
plt.ylabel('Değişkenler', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

outlierData = happinessData[columnsToAnalyze].apply(zscore)
outliers = (outlierData.abs() > 3).any(axis=1)
print(f"\n» Toplam {outliers.sum()} adet uç değer tespit edildi.")

# ════ Normalizasyon ════
scaler = MinMaxScaler()
normalizedData = pd.DataFrame(
    scaler.fit_transform(happinessData[columnsToAnalyze]),
    columns=columnsToAnalyze
)

print("\n══════════════════════════════════════════════════════════════════════╣ Normalizasyon Sonrası İlk 5 Satır ╠══════════════════════════════════════════════════════════════════════")
print(normalizedData.head().to_string())

plt.figure(figsize=(15, 6))
sns.scatterplot(
    data=normalizedData,
    x='Log GDP Per Capita',
    y='Life Ladder',
    alpha=0.7
)
plt.title('Kişi Başına Düşen GSYİH ve Mutluluk Seviyesi (Normalize Edilmiş)', fontsize=16)
plt.xlabel('Log GDP Per Capita (Normalized)', fontsize=14)
plt.ylabel('Life Ladder (Normalized)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ════ Regresyon Analizi ════
X = happinessData[['Log GDP Per Capita', 'Social Support', 'Healthy Life Expectancy At Birth']]
y = happinessData['Life Ladder']

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(XTrain, yTrain)

yPred = model.predict(XTest)
mse = mean_squared_error(yTest, yPred)
r2 = r2_score(yTest, yPred)

print("\n══════════════════════════════════════════════════════════════════════════════╣ Regresyon Analizi ╠══════════════════════════════════════════════════════════════════════════════")
print(f"» R-Kare Skoru: {r2:.4f}")
print(f"» Ortalama Kare Hata (MSE): {mse:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(yTest, yPred, alpha=0.7, label='Tahminler')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Doğru Çizgi')
plt.title('Gerçek vs Tahmin (Regresyon)', fontsize=16)
plt.xlabel('Gerçek Değerler', fontsize=14)
plt.ylabel('Tahmin Edilen Değerler', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()