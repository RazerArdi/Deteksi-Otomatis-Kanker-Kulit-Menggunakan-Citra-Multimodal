import streamlit as st
import pandas as pd
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.express as px  # Library baru untuk grafik interaktif

# ==============================================================================
# 1. KONFIGURASI PATH & SISTEM
# ==============================================================================

# Mendapatkan lokasi absolut file app.py ini berada
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# KONFIGURASI JALUR FOLDER
IMAGE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "notebook", "image"))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Models"))
CSV_FILE = os.path.join(IMAGE_DIR, "poster_grand_final_results.csv")

# Label Kelas (Wajib sama dengan urutan training)
CLASS_NAMES = [
    'AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 
    'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. DEFINISI ARSITEKTUR MODEL (WAJIB DI-COPY DARI NOTEBOOK!)
# ==============================================================================
def get_model_structure(model_name, num_classes=11):
    """
    Membangun arsitektur model kosong.
    ⚠️ PASTE DEFINISI CLASS MODEL ASLI ANDA DI ATAS FUNGSI INI ⚠️
    """
    try:
        model = None
        # --- PLACEHOLDER GENERIC (GANTI DENGAN KODE ASLI) ---
        if "ResNet50" in model_name:
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "VGG16" in model_name:
            model = models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif "EffNet" in model_name:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(1280, num_classes)
        elif "Swin" in model_name:
            # Pastikan library timm atau torchvision mendukung ini
            model = models.swin_t(weights=None)
            model.head = nn.Linear(model.head.in_features, num_classes)
        elif "ViT" in model_name or "DeiT" in model_name:
            model = models.vit_b_16(weights=None)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            
        return model
    except Exception as e:
        return None

# ==============================================================================
# 3. FUNGSI UTILITAS (LOAD DATA & CLEANING)
# ==============================================================================

def load_and_clean_data():
    """
    Membaca data CSV dan membersihkan nilai NaN/Kosong.
    """
    if not os.path.exists(CSV_FILE):
        return None
    
    try:
        df = pd.read_csv(CSV_FILE)
        
        # 1. Pastikan kolom ada
        required_cols = ['Model', 'Accuracy (%)', 'Kategori']
        for col in required_cols:
            if col not in df.columns:
                return None # Struktur CSV salah
        
        # 2. Handle NaN pada kolom numerik penting
        if 'Weighted Avg F1-Score' in df.columns:
            # Ganti NaN dengan 0 untuk keperluan sorting, tapi simpan statusnya
            df['Weighted Avg F1-Score'] = df['Weighted Avg F1-Score'].fillna(0)
            
        # 3. Rapikan nama Kategori jika kosong
        if 'Kategori' in df.columns:
            df['Kategori'] = df['Kategori'].fillna('Other')
            
        return df
    except Exception:
        return None

def load_model_weights(model_name):
    """Memuat bobot model dari file .pth."""
    pth_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    
    if not os.path.exists(pth_path):
        return None, f"File {model_name}.pth tidak ditemukan."
    
    model = get_model_structure(model_name)
    if model is None:
        return None, "Struktur Class Model belum didefinisikan di app.py"
        
    try:
        state_dict = torch.load(pth_path, map_location=DEVICE)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False) 
        model.to(DEVICE)
        model.eval()
        return model, "Success"
    except Exception as e:
        return None, f"Error load state_dict: {str(e)}"

def predict_image(model, image):
    """Inferensi gambar tunggal."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    return probs.cpu().numpy()

# ==============================================================================
# 4. USER INTERFACE
# ==============================================================================

def main():
    st.set_page_config(page_title="MILKFusionNet Dashboard", layout="wide")
    
    # CSS Minimalis & Adaptif (Support Dark/Light Mode)
    st.markdown("""
        <style>
            /* Hapus background hardcoded agar ikut tema Streamlit */
            .main {
                padding-top: 2rem;
            }
            /* Styling container metric agar terlihat jelas di kedua tema */
            div[data-testid="stMetric"] {
                background-color: rgba(128, 128, 128, 0.1); /* Transparan Abu */
                border-radius: 8px;
                padding: 15px;
                border: 1px solid rgba(128, 128, 128, 0.2);
            }
            /* Judul yang lebih elegan */
            h1 {
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-weight: 700;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    st.sidebar.header("🔬 MILKFusionNet")
    st.sidebar.caption("v1.0.0 - Research Release")
    
    # Debug Info (Bisa di-collapse)
    with st.sidebar.expander("📂 System Paths"):
        st.write(f"CSV: `{ '✅ Found' if os.path.exists(CSV_FILE) else '❌ Missing'}`")
        st.write(f"Models: `{MODEL_DIR}`")

    menu = st.sidebar.radio("Main Menu", ["📊 Comparative Analysis", "🩺 Live Diagnosis"])
    
    # Load Data
    df = load_and_clean_data()
    
    if df is None:
        st.error("❌ **Data Error:** File `poster_grand_final_results.csv` tidak ditemukan atau rusak.")
        st.info("Jalankan notebook bagian visualisasi terakhir untuk menghasilkan file ini.")
        return

    # --- HALAMAN 1: LEADERBOARD INTERAKTIF ---
    if menu == "📊 Comparative Analysis":
        st.title("Comparative Performance Benchmark")
        st.markdown("Analisis komprehensif performa 22 model arsitektur berbeda.")
        
        col_chart, col_table = st.columns([1.5, 1])
        
        with col_chart:
            st.subheader("Interactive Leaderboard")
            
            # SORT DATA UNTUK GRAFIK
            df_chart = df.sort_values(by="Accuracy (%)", ascending=True) # Ascending agar bar chart horizontal urut dari atas
            
            # MEMBUAT PLOTLY CHART (Support Dark/Light Mode Otomatis)
            fig = px.bar(
                df_chart, 
                x="Accuracy (%)", 
                y="Model", 
                color="Kategori",
                orientation='h', # Horizontal
                text_auto='.1f', # Tampilkan angka di bar
                color_discrete_sequence=px.colors.qualitative.Prism,
                height=600
            )
            
            fig.update_layout(
                xaxis_title="Validation Accuracy (%)",
                yaxis_title=None,
                legend_title_text="Architecture Family",
                # Agar background transparan mengikuti tema streamlit
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col_table:
            st.subheader("Detailed Metrics")
            # Tampilkan tabel data
            display_cols = ['Model', 'Accuracy (%)', 'Kategori']
            if 'Weighted Avg F1-Score' in df.columns:
                display_cols.append('Weighted Avg F1-Score')
                
            st.dataframe(
                df[display_cols].sort_values(by="Accuracy (%)", ascending=False),
                use_container_width=True,
                height=600,
                hide_index=True
            )

    # --- HALAMAN 2: DIAGNOSTIC TEST ---
    elif menu == "🩺 Live Diagnosis":
        st.title("Diagnostic Inference Interface")
        st.markdown("Uji coba model secara *real-time* menggunakan citra dermoskopi.")
        
        # Pilihan Model
        model_list = df['Model'].tolist()
        default_idx = model_list.index('Swin_FullFT') if 'Swin_FullFT' in model_list else 0
        selected_model = st.selectbox("Select Model Architecture", model_list, index=default_idx)
        
        # Ambil info model terpilih
        row = df[df['Model'] == selected_model].iloc[0]
        
        # --- METRIC DISPLAY (Handling NaN) ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", row['Kategori'])
        c2.metric("Accuracy", f"{row['Accuracy (%)']:.2f}%")
        
        # Logic Handling NaN F1-Score
        f1_val = row.get('Weighted Avg F1-Score', 0)
        if f1_val == 0 or pd.isna(f1_val):
            f1_display = "N/A" # Tampilkan N/A jika kosong/0
        else:
            f1_display = f"{f1_val:.4f}"
            
        c3.metric("F1-Score (Weighted)", f1_display)

        st.markdown("---")
        
        # Area Upload
        col_left, col_right = st.columns([1, 1.2])
        
        with col_left:
            st.write("#### 1. Input Image")
            uploaded_file = st.file_uploader("Upload Dermoscopy Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Specimen", use_column_width=True)
        
        with col_right:
            st.write("#### 2. Analysis Result")
            
            if uploaded_file:
                if st.button("Run Diagnostics", type="primary"):
                    with st.spinner(f"Running inference with {selected_model}..."):
                        # Load & Predict
                        model, msg = load_model_weights(selected_model)
                        
                        if model:
                            probs = predict_image(model, image)
                            
                            # Result Processing
                            top_idx = np.argmax(probs)
                            top_class = CLASS_NAMES[top_idx]
                            top_conf = probs[top_idx] * 100
                            
                            # Alert System
                            if top_conf > 85:
                                st.success(f"### Prediction: {top_class}")
                            elif top_conf > 50:
                                st.warning(f"### Prediction: {top_class} (Uncertain)")
                            else:
                                st.error(f"### Prediction: {top_class} (Low Confidence)")
                                
                            st.write(f"**Confidence Score:** {top_conf:.2f}%")
                            
                            # Probability Chart (Interactive Plotly)
                            prob_df = pd.DataFrame({
                                "Diagnosis": CLASS_NAMES,
                                "Probability (%)": probs * 100
                            }).sort_values("Probability (%)", ascending=True).tail(5) # Top 5
                            
                            fig_prob = px.bar(
                                prob_df,
                                x="Probability (%)",
                                y="Diagnosis",
                                orientation='h',
                                title="Top 5 Probabilities",
                                text_auto='.1f',
                                color="Probability (%)",
                                color_continuous_scale="Blues"
                            )
                            fig_prob.update_layout(
                                xaxis_range=[0, 100],
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=300
                            )
                            st.plotly_chart(fig_prob, use_container_width=True)
                            
                        else:
                            st.error("Model Error")
                            st.code(msg)
            else:
                st.info("Waiting for image upload...")

if __name__ == "__main__":
    main()