# duygu-analizi-yemek-oneri
Türkçe duygu analizi ile çalışan yemek öneri sistemi


# 🍽️ Duygu Analizi ile Yemek Tarifi Öneri Sistemi

Bu proje, kullanıcıların Türkçe olarak ifade ettiği duygu durumuna göre onlara en uygun yemek tariflerini sunmayı amaçlayan bir NLP tabanlı öneri sistemidir.

## 📌 Proje Özeti

Kullanıcılar sisteme nasıl hissettiklerini yazar (örneğin: *"çok yorgunum, hafif bir şey istiyorum"*), sistem hem duygu analizi yaparak hem de anlamsal benzerlik ölçerek en uygun tarifleri önerir.

Proje, **BERT tabanlı duygu analizi**, **TF-IDF vektörleştirme**, ve **kosinüs benzerliği** gibi doğal dil işleme tekniklerini bir araya getirerek çalışır.

---

## ⚙️ Kullanılan Teknolojiler

- 🧠 **HuggingFace Transformers** – Türkçe BERT modeli (`savasy/bert-base-turkish-sentiment-cased`)
- 📊 **scikit-learn** – TF-IDF ve cosine similarity
- 🌐 **Streamlit** – Kullanıcı arayüzü
- 🐼 **Pandas** – Veri işleme
- 🔤 **JSON/CSV** – Veri seti yapısı

---

## 🚀 Nasıl Çalışır?

1. Kullanıcıdan gelen duygu metni analiz edilir.
2. Duyguya uygun etiket (`mutlu eder`, `rahatlatır`, `enerji verir`, vb.) belirlenir.
3. Uygun duyguya sahip tarifler filtrelenir.
4. Kullanıcı cümlesine anlamca en yakın tarifler TF-IDF + cosine similarity ile sıralanır.
5. Kullanıcıya tarif adı, malzemeler ve hazırlanış adımı görsel olarak sunulur.

---

## 🧪 Örnek Kullanım

- **Girdi:** `Tatlı bir şey yemek istiyorum, çok mutluyum.`
- **Çıktı:** `Mutlu eder` etiketli tarifler → *Milföy pasta*, *Vogue tatlısı* vb.

---

📂 Veri Seti Kaynağı
Bu projede kullanılan yemek tarifleri ve besin değerleri veri seti şu kaynaktan alınmıştır:
🔗 Kaggle: Food Recipe and Dataset with Nutritional Values
Veri seti Mehmet Uslu tarafından sağlanmıştır. Bu proje kapsamında sadece Türkçe içeriklere odaklanılmış, duygu analizi etiketleri tarafımızdan oluşturulmuştur.

