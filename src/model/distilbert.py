import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Veriyi Oku ve Hazırla
df = pd.read_csv("data/processed/processed_issues.csv")
df['body'] = df['body'].fillna("")

# Modelin anlayabilmesi için metin etiketlerini (Bug, vs.) sayılara çeviriyoruz
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Baseline ile tam olarak aynı verilerle test etmek için aynı random_state'i kullanıyoruz
X_train, X_test, y_train, y_test = train_test_split(
    df['body'], 
    df['label_encoded'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label_encoded']
)

# Hugging Face kütüphanesinin veri formatına çeviriyoruz
train_ds = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
test_ds = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

# 2. DistilBERT Tokenizer ve Modelini Yükle
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Metinleri aynı uzunlukta kesip/dolduruyoruz
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_tokenized = train_ds.map(tokenize_function, batched=True)
test_tokenized = test_ds.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))

# 3. Eğitim Ayarları (Parametreleri)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,              # Veri seti üzerinde 3 tam tur atacak
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_dir='./logs',
    disable_tqdm=False
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics
)

# 4. Eğitimi Başlat
print("DistilBERT modeli eğitiliyor... (Derin öğrenme modellerinin eğitimi bilgisayarınızın işlemci hızına göre 3-5 dakika sürebilir)")
trainer.train()

# 5. Test verisi üzerinde tahmin yap ve Baseline ile karşılaştır
predictions = trainer.predict(test_tokenized)
preds = predictions.predictions.argmax(-1)

print("\n--- DistilBERT Model Sonuçları ---")
print(f"Genel Doğruluk (Accuracy): {accuracy_score(y_test, preds):.3f}\n")
print("Her Sınıf İçin Detaylı Rapor:")
print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))