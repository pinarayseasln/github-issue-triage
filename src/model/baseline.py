import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Temizlenmiş veriyi oku
df = pd.read_csv("data/processed/processed_issues.csv")

# Boş (NaN) metinleri tekrar kontrol edip boş string'e çeviriyoruz
df['body'] = df['body'].fillna("")

# 2. Veriyi Eğitim (Train) ve Test olarak ayır (Sınıf dengesizliği olduğu için stratify=y kullanıyoruz)
X_train, X_test, y_train, y_test = train_test_split(
    df['body'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# 3. Metinleri sayısal vektörlere dönüştür (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Lojistik Regresyon modelini eğit
# Sınıf dengesizliğini hafifletmek için class_weight='balanced' ekliyoruz
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test_tfidf)

# 6. Sonuçları (F1 skoru ve Accuracy) raporla
print("--- Baseline Model Sonuçları ---")
print(f"Genel Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.3f}\n")
print("Her Sınıf İçin Detaylı Rapor (Per-class F1):")
print(classification_report(y_test, y_pred))