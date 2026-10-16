import pandas as pd
import re
import os

# 1. Ham veriyi oku
df = pd.read_csv("data/raw/github_issues.csv")

# 2. Metin temizleme fonksiyonu (Kod bloklarını ve URL'leri kaldır)
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # URL'leri temizle
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Markdown kod bloklarını temizle
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    return text.strip()

# 3. Karmaşık etiketleri 4 ana sınıfa haritalama (Mapping)
def map_label(label_str):
    if pd.isna(label_str):
        return None
        
    label_str = str(label_str).lower()
    found_classes = set()
    
    # Ekranda gördüğümüz etiket isimlerine göre eşleştirme yapıyoruz
    if 'bug' in label_str or 'defect' in label_str:
        found_classes.add('Bug')
    if 'feature' in label_str or 'enhancement' in label_str:
        found_classes.add('Feature Request')
    if 'question' in label_str or 'help' in label_str or 'info-needed' in label_str:
        found_classes.add('Question')
    if 'doc' in label_str:
        found_classes.add('Documentation')
        
    # Sadece 1 kategoriyle eşleşenleri al, çakışanları veya hiç eşleşmeyenleri (None) yoksay
    if len(found_classes) == 1:
        return list(found_classes)[0]
    return None

# Fonksiyonları uygula
df['cleaned_body'] = df['body'].apply(clean_text)
df['mapped_label'] = df['labels'].apply(map_label)

# Boş metinleri ve eşleşmeyen etiketleri sil (Drop issues)
df = df.dropna(subset=['mapped_label'])
df = df[df['cleaned_body'] != ""]

# Sadece gerekli sütunları seç (Örn: metin ve etiket)
df_final = df[['title', 'cleaned_body', 'mapped_label']].copy()
df_final.columns = ['title', 'body', 'label']

# 4. Sonucu data/processed/ klasörüne kaydet
os.makedirs("data/processed", exist_ok=True)
df_final.to_csv("data/processed/processed_issues.csv", index=False)

print("Veri temizleme tamamlandı!")
print(f"Kalan temiz veri sayısı: {len(df_final)}")
print("\nSınıf Dengesi (Class Balance):")
print(df_final['label'].value_counts())