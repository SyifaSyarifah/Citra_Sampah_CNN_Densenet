import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import plotly.graph_objects as go

# Coba import TensorFlow Lite
try:
    import tensorflow as tf
    USE_TFLITE = True
    st.sidebar.success("✅ TensorFlow Lite Ready")
except ImportError:
    st.sidebar.error("❌ TensorFlow tidak tersedia")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🗂️ Klasifikasi Sampah AI",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
    .confidence-high { border-color: #28a745; background: #d4edda; }
    .confidence-medium { border-color: #ffc107; background: #fff3cd; }
    .confidence-low { border-color: #dc3545; background: #f8d7da; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_tflite_model():
    """Load TensorFlow Lite model"""
    model_path = "model.tflite"
    
    if not os.path.exists(model_path):
        return None, False, f"❌ File {model_path} tidak ditemukan di folder ini"
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details
        }, True, f"✅ Model TFLite berhasil dimuat dari {model_path}"
        
    except Exception as e:
        return None, False, f"❌ Error loading TFLite model: {str(e)}"

def load_class_names():
    """Load class names"""
    # Default classes sesuai dengan training Anda
    default_classes = ['battery', 'brown-glass','paper']
    
    # Coba baca dari file jika ada
    try:
        if os.path.exists("class_names.txt"):
            with open("class_names.txt", "r") as f:
                classes = [line.strip() for line in f.readlines()]
                return classes
        else:
            return default_classes
    except:
        return default_classes

def preprocess_image_for_tflite(image, target_size=(224, 224)):
    """Preprocess image untuk TFLite model"""
    # Convert to RGB jika perlu
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize ke ukuran yang diharapkan model
    image = image.resize(target_size)
    
    # Convert ke numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize ke range [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_with_tflite(image, model_data, class_names):
    """Make prediction using TFLite model"""
    try:
        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']
        
        # Preprocess image
        processed_image = preprocess_image_for_tflite(image)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Process results
        predictions = predictions[0]  # Remove batch dimension
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Get all predictions
        all_predictions = {}
        for i, class_name in enumerate(class_names):
            all_predictions[class_name] = float(predictions[i])
        
        return predicted_class, confidence, all_predictions
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, 0.0, {}

def create_prediction_chart(predictions):
    """Create prediction probability chart"""
    if not predictions:
        return None
        
    classes = list(predictions.keys())
    probs = list(predictions.values())
    
    # Sort by probability
    sorted_data = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    classes, probs = zip(*sorted_data)
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=probs,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            text=[f'{p:.1%}' for p in probs],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Probabilitas Prediksi untuk Setiap Kelas',
        xaxis_title='Confidence Score',
        yaxis_title='Kelas Sampah',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed")  # Highest probability on top
    )
    
    return fig

def get_waste_info(class_name):
    """Get information about waste type"""
    waste_info = {
        'paper': {
            'emoji': '📄',
            'description': 'Sampah kertas yang dapat didaur ulang',
            'examples': 'Koran, majalah, kardus, kertas HVS',
            'disposal': 'Dapat didaur ulang. Pisahkan dari sampah basah.',
            'color': '#4CAF50'
        },
        'battery': {
            'emoji': '🔋',
            'description': 'Sampah elektronik berbahaya',
            'examples': 'Baterai AA, AAA, baterai ponsel, baterai laptop',
            'disposal': 'Sampah B3 - perlu penanganan khusus di tempat pengumpulan baterai bekas.',
            'color': '#FF9800'
        },
        'brown-glass': {
            'emoji': '🍺',
            'description': 'Sampah kaca berwarna coklat',
            'examples': 'Botol bir, botol sirup, botol kecap',
            'disposal': 'Dapat didaur ulang. Bersihkan sebelum membuang.',
            'color': '#8D6E63'
        }
    }
    
    return waste_info.get(class_name.lower(), {
        'emoji': '🗑️',
        'description': 'Jenis sampah tidak dikenali',
        'examples': '-',
        'disposal': 'Ikuti panduan pembuangan sampah setempat',
        'color': '#9E9E9E'
    })

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🗂️ AI Klasifikasi Sampah</h1>
        <p>Sistem cerdas untuk mengklasifikasikan jenis sampah menggunakan TensorFlow Lite</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and classes
    model_data, model_loaded, model_message = load_tflite_model()
    class_names = load_class_names()
    
    # Sidebar
    st.sidebar.title("ℹ️ Informasi Aplikasi")
    st.sidebar.markdown(f"""
    **Model:** TensorFlow Lite
    
    **Status:** {model_message}
    
    **Kelas yang dapat diklasifikasi:**
    - 📄 Paper (Kertas)
    - 🔋 Battery (Baterai)  
    - 🍺 Brown Glass (Kaca Coklat)
    
    **Cara penggunaan:**
    1. Pastikan file `model.tflite` ada di folder
    2. Upload gambar sampah  
    3. Klik tombol klasifikasi
    4. Lihat hasil dan rekomendasi
    """)
    
    # Check model status
    if not model_loaded:
        st.error(model_message)
        st.info("""
        **Untuk menjalankan aplikasi ini:**
        1. Pastikan file `model.tflite` ada di folder yang sama dengan aplikasi ini
        2. File dapat diperoleh dari hasil training atau konversi model Keras
        
        **Struktur folder yang diharapkan:**
        ```
        project_folder/
        ├── app.py
        ├── model.tflite  ← File ini harus ada!
        └── class_names.txt (opsional)
        ```
        """)
        return
    
    # Main content
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("📤 Upload Gambar")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar sampah untuk diklasifikasi",
            type=['png', 'jpg', 'jpeg'],
            help="Format yang didukung: PNG, JPG, JPEG (maksimal 224x224 pixel optimal)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diupload", use_container_width=True)

            
            # Show image info
            st.info(f"📏 Ukuran gambar: {image.size[0]} x {image.size[1]} pixel")
            
            # Predict button
            if st.button("🔍 Klasifikasi Sampah", type="primary", use_container_width=True):
                with st.spinner("🧠 Menganalisis gambar..."):
                    predicted_class, confidence, all_predictions = predict_with_tflite(
                        image, model_data, class_names
                    )
                    
                    if predicted_class:
                        # Store results in session state
                        st.session_state.prediction_results = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'all_predictions': all_predictions
                        }
                        st.success("✅ Prediksi selesai!")
                    else:
                        st.error("❌ Gagal melakukan prediksi")
    
    with col2:
        st.header("📊 Hasil Prediksi")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            predicted_class = results['predicted_class']
            confidence = results['confidence']
            all_predictions = results['all_predictions']
            
            # Get waste info
            waste_info = get_waste_info(predicted_class)
            
            # Determine confidence level
            if confidence >= 0.8:
                conf_class = "confidence-high"
                conf_text = "Tinggi"
                conf_emoji = "🎯"
            elif confidence >= 0.5:
                conf_class = "confidence-medium" 
                conf_text = "Sedang"
                conf_emoji = "⚡"
            else:
                conf_class = "confidence-low"
                conf_text = "Rendah"
                conf_emoji = "⚠️"
            
            # Main prediction result
            st.markdown(f"""
            <div class="prediction-card {conf_class}">
                <h2>{waste_info['emoji']} {predicted_class.title()}</h2>
                <h3>{conf_emoji} Confidence: {confidence:.1%} ({conf_text})</h3>
                <p><strong>📝 Deskripsi:</strong> {waste_info['description']}</p>
                <p><strong>📋 Contoh:</strong> {waste_info['examples']}</p>
                <p><strong>♻️ Cara Pembuangan:</strong> {waste_info['disposal']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Prediction chart
            chart = create_prediction_chart(all_predictions)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Detailed predictions table
            st.subheader("📈 Detail Semua Prediksi")
            
            for class_name, prob in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
                info = get_waste_info(class_name)
                
                col_icon, col_name, col_prob = st.columns([1, 4, 3])
                with col_icon:
                    st.markdown(f"<h3 style='text-align: center;'>{info['emoji']}</h3>", unsafe_allow_html=True)
                with col_name:
                    st.write(f"**{class_name.title()}**")
                    st.caption(info['description'])
                with col_prob:
                    st.metric(label="Probabilitas", value=f"{prob:.1%}")
                    
            # Clear results button
            if st.button("🗑️ Hapus Hasil", help="Hapus hasil prediksi untuk memulai yang baru"):
                del st.session_state.prediction_results
                st.rerun()
        
        else:
            st.info("👆 Upload gambar di sebelah kiri dan klik 'Klasifikasi Sampah' untuk melihat hasil prediksi.")
            
            # Show sample prediction format
            with st.expander("👁️ Lihat Contoh Hasil"):
                st.markdown("""
                **Contoh hasil prediksi yang akan ditampilkan:**
                - Kelas prediksi dengan emoji
                - Tingkat confidence (kepercayaan)
                - Informasi detail tentang jenis sampah
                - Chart probabilitas untuk semua kelas
                - Rekomendasi cara pembuangan
                """)

    # Additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Model TFLite</h3>
            <p>Menggunakan TensorFlow Lite untuk prediksi yang cepat dan efisien</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Prediksi Real-time</h3>
            <p>Klasifikasi instan dalam hitungan detik</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌱 Ramah Lingkungan</h3>
            <p>Membantu pengelolaan sampah yang lebih baik untuk bumi</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚀 Aplikasi Klasifikasi Sampah AI menggunakan TensorFlow Lite</p>
        <p>Dibuat untuk membantu pengelolaan sampah yang lebih cerdas</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()