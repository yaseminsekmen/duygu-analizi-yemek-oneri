import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BERT modeli yükleniyor
sentiment_analyzer = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

# Tarif veri setini yükle
df = pd.read_csv("recipes_hybrid_labeled.csv")

# TF-IDF vektörleştirici hazırla
vectorizer = TfidfVectorizer()

# Duyguya göre filtrelenen veri için TF-IDF matrisini dinamik oluşturacağız

# Geliştirilmiş duygu analiz fonksiyonu (hibrit)
def analyze_user_input(text):
    text = text.lower()
    if any(word in text for word in ["mutluyum", "keyfim yerinde", "tatlı bir şey", "cheesecake", "çikolata"]):
        return "mutlu eder"
    elif any(word in text for word in ["stres", "yorgun", "sakinleşmek", "hafif", "rahatlatıcı", "bitkin", "yoğun gün"]):
        return "rahatlatır"
    elif any(word in text for word in ["enerji", "güçlü", "tok", "doyurucu", "acı", "etli", "kızartma", "baharatlı"]):
        return "enerji verir"
    elif any(word in text for word in ["moral", "motivasyon", "canım sıkkın", "biraz iyi hissetmek"]):
        return "moral yükseltir"
    elif any(word in text for word in ["huzurlu", "anne yemeği", "nostalji", "ev yemeği"]):
        return "huzur verir"

    # Eğer eşleşme yoksa BERT ile tahmin et
    result = sentiment_analyzer(text)[0]
    label = result['label']
    if label == "Positive":
        return "mutlu eder"
    elif label == "Negative":
        return "rahatlatır"
    return "nötr"

# TF-IDF benzerliğine göre tarif bulma (aynı duygu etiketine sahipler arasından)
def benzer_tarifleri_bul(user_text, duygu_label):
    alt_df = df[df["duygu_etiketi"] == duygu_label]
    if alt_df.empty:
        return pd.DataFrame()
    tfidf_matrix = vectorizer.fit_transform(alt_df["aciklama"].fillna(""))
    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    en_benzer_indexler = similarities.argsort()[-5:][::-1]
    return alt_df.iloc[en_benzer_indexler]

# Streamlit arayüzü
st.set_page_config(page_title="Yemek Öneri Sistemi", layout="wide")
st.title("🍽️ NLP Destekli Yemek Tarifi Öneri Sistemi")

user_input = st.text_input("Nasıl hissediyorsun? Birkaç kelimeyle anlat...")

if user_input:
    duygu = analyze_user_input(user_input)
    st.markdown(f"### Tespit edilen duygu: `{duygu}`")

    # 1. Duyguya göre tarif önerileri
    uygun_tarifler = df[df["duygu_etiketi"] == duygu]

    if not uygun_tarifler.empty:
        st.success(f"{len(uygun_tarifler)} tarif bulundu.")
        for i, row in uygun_tarifler.sample(3).iterrows():  # RANDOM 3 tarif
            st.markdown(f"---\n###  {row['ad']}")
            st.markdown(f"**Malzemeler:** {row.get('malzemeler', 'Belirtilmemiş')}")
            st.markdown(f"**Hazırlanış:** {row.get('hazirlanis', 'Belirtilmemiş')}")
    else:
        st.warning("Bu duyguya uygun tarif bulunamadı.")

    # 2. Anlamca benzer tarifler
    st.markdown("### 🔍 Anlamca Benzer Tarifler")
    benzer_tarifler = benzer_tarifleri_bul(user_input, duygu)
    if not benzer_tarifler.empty:
        for i, row in benzer_tarifler.iterrows():
            st.markdown(f"---\n### 🍴 {row['ad']}")
            st.markdown(f"**Malzemeler:** {row.get('malzemeler', 'Belirtilmemiş')}")
            st.markdown(f"**Hazırlanış:** {row.get('hazirlanis', 'Belirtilmemiş')}")
    else:
        st.info("Anlamca benzer tarif bulunamadı.")
