import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

# Sayfa Yapılandırması
st.set_page_config(
    page_title="BEKO Endüstriyel Hata Tespiti",
    page_icon="🔍",
    layout="wide"
)

# Cache mekanizması: Modeli sadece bir kere yükle, her resim yüklemesinde tekrar yüklemesin
@st.cache_resource
def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

# Tespiti gerçekleştiren fonksiyon (GPU Batch Inference & NMS Optimize Edilmiş)
def detect_defects_on_image(model, img_array, tile_size=128, stride=64, conf_threshold=0.5, batch_size=32):
    img_h, img_w = img_array.shape[:2]
    output_img = img_array.copy()
    
    # Batch İşleme için Geçici Listeler
    batch_tiles = []
    batch_coords = []
    
    # Tüm Tespit Edilen Kutuları (BBox), Skorları ve Sınıfları Tutacağımız Listeler
    # NMS (Non-Maximum Suppression) için gerekli
    boxes = []
    confidences = []
    class_ids = []
    
    # Sınıf İsimleri ile ID'leri Eşleştirme (NMS için ID gerekli)
    classes_map = {"eksik_vida": 0, "aparatsiz": 1}
    
    # Döngü İle Resmin Parçalanması ve Batch'e Eklenmesi
    for y in range(0, img_h - tile_size + 1, stride):
        for x in range(0, img_w - tile_size + 1, stride):
            tile = img_array[y:y+tile_size, x:x+tile_size]
            
            # Tile boyutu hatası (kenar payları)
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue
                
            batch_tiles.append(tile)
            batch_coords.append((x, y))
            
            # Batch Boyutu Ulaştığında Modeli Çalıştır
            if len(batch_tiles) == batch_size:
                results = model.predict(batch_tiles, verbose=False, device=0)
                
                for idx, result in enumerate(results):
                    top1_index = result.probs.top1
                    predicted_class = result.names[top1_index]
                    confidence = float(result.probs.top1conf)
                    
                    if confidence >= conf_threshold and predicted_class in classes_map:
                        x_coord, y_coord = batch_coords[idx]
                        
                        # OpenCV NMS formatı için Bounding Box listesi: [x, y, genişlik, yükseklik]
                        boxes.append([x_coord, y_coord, tile_size, tile_size])
                        confidences.append(confidence)
                        class_ids.append(classes_map[predicted_class])
                
                # Listeyi Sıfırla
                batch_tiles = []
                batch_coords = []
    
    # Kalan son batch parçaları için işlemler:
    if len(batch_tiles) > 0:
        results = model.predict(batch_tiles, verbose=False, device=0)
        for idx, result in enumerate(results):
            top1_index = result.probs.top1
            predicted_class = result.names[top1_index]
            confidence = float(result.probs.top1conf)
            
            if confidence >= conf_threshold and predicted_class in classes_map:
                x_coord, y_coord = batch_coords[idx]
                boxes.append([x_coord, y_coord, tile_size, tile_size])
                confidences.append(confidence)
                class_ids.append(classes_map[predicted_class])

    # ---------------- NMS (NON-MAXIMUM SUPPRESSION) UYGULAMASI ----------------
    missing_screw_count = 0
    missing_part_count = 0
    
    if len(boxes) > 0:
        # cv2.dnn.NMSBoxes(kutular, skorlar, eşik_skoru, NMS_eşiği)
        # NMS Eşiği (IoU) 0.1 diyoruz. Sadece bir parçanın 0.1 (Yani %10) kadar çakışıyorsa bile
        # zayıf ihtimalleri temizle. Ne kadar küçükse kutular o kadar ayrık olmak zorundadır.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.2)
        
        # indices bazen farklı OpenCV sürümlerine göre iç içe Tuple/List dönebilir
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                conf = confidences[i]
                c_id = class_ids[i]
                
                if c_id == 0:  # "eksik_vida"
                    missing_screw_count += 1
                    color = (255, 0, 0) # Kırmızı
                    label = f"Eksik Vida ({conf:.2f})"
                else:          # "aparatsiz"
                    missing_part_count += 1
                    color = (255, 165, 0) # Turuncu
                    label = f"Aparatsiz ({conf:.2f})"
                
                # Sadece Filtrelenmiş / Gerçek Kutu ve Etiketi Çiz
                cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 3)
                
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_img, (x, y), (x + text_w, y + text_h + baseline + 5), color, -1)
                cv2.putText(output_img, label, (x, y + text_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    return output_img, missing_screw_count, missing_part_count


# Arayüz Tasarımı
st.title("🌟 Beko Kalite Kontrol Dashboard")
st.markdown("Yapay zeka (YOLO) ile üretim bantlarındaki geniş buzdolabı görsellerinden saniyeler içinde anlık hata ve eksik tespiti.")

# Modeli Yükle - Yeni 50-epoch GPU modeli
MODEL_PATH = r"runs\classify\runs\classify\beko_final_model2\weights\best.pt"
model = load_yolo_model(MODEL_PATH)

if model is None:
    # Eski CPU modellerine fallback
    MODEL_PATH = r"runs\classify\beko_vida_modeli\train_sonuclari2\weights\best.pt"
    model = load_yolo_model(MODEL_PATH)
    if model is None:
        st.error(f"⚠️ Model dosyası bulunamadı! Lütfen modelin varlığından emin olun.")
        st.stop()


# Sol panel ayarları
with st.sidebar:
    st.header("⚙️ Hassasiyet & Kontrol Ayarları")
    confidence_slider = st.slider(
        "Güven Eşiği (Confidence)", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05, 
        help="Sistemin şüpheli bir noktayı 'HATA' olarak işaretlemesi için gereken eminlik oranı."
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Analiz Sınıfları")
    st.markdown("🟢 **OK**: Sorunsuz yüzey")
    st.markdown("🔴 **Eksik Vida**: Vida yuvası boş")
    st.markdown("🟠 **Aparatsız**: Parça takılmamış")
    st.markdown("🔵 **Diğer**: Boş metal/plastik vb.")

# Resim Yükleme Alanı
uploaded_file = st.file_uploader("📥 İncelemek için büyük bir buzdolabı / üretim görseli yükleyin (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi oku ve Numpy dizisine çevir (RGB)
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    st.markdown("---")
    
    # Butonla analizi başlat
    if st.button("🚀 Akıllı Analizi Başlat", use_container_width=True, type="primary"):
        with st.spinner("Yapay zeka mikroskobik düzeyde parçaları tarıyor..."):
            
            # Analiz fonksiyonunu çağır (128x128 ile tarama)
            result_img, screw_err, part_err = detect_defects_on_image(model, image_np, tile_size=128, conf_threshold=confidence_slider)
            
            # Sonuçları Yan Yana Göster (Görselleştirme)
            col_orig, col_res = st.columns(2)
            
            with col_orig:
                st.subheader("Orijinal Görüntü")
                st.image(image_np, use_container_width=True)
                
            with col_res:
                st.subheader("Tespit Edilen Hatalar")
                st.image(result_img, use_container_width=True)
                
            # Raporlama: Kart / Metrik Dizilimi
            st.markdown("---")
            st.markdown("### 📊 Kalite Kontrol Algoritması Tespit Özeti")
            
            # 3'lü kolon ile estetik kartlar (Metrik)
            m1, m2, m3 = st.columns(3)
            
            total_defect = screw_err + part_err
            
            m1.metric(label="🟥 Eksik Vida Tespiti", value=f"{screw_err} Adet")
            m2.metric(label="🟧 Aparat Eksikliği", value=f"{part_err} Adet")
            
            if total_defect == 0:
                m3.metric(label="✅ Ürün Durumu", value="KUSURSUZ")
                st.balloons()
            else:
                m3.metric(label="🚨 Ürün Durumu", value="HATALI", delta=f"-{total_defect} Hata", delta_color="inverse")
                st.error("Bant operatörünün müdahalesi gerekiyor! Orijinal resim üzerindeki işaretli koordinatları inceleyiniz.")
