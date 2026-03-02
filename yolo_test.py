from ultralytics import YOLO
import os
import glob
import random
import cv2

def test_yolo_model(model_path, image_source):
    """
    Eğitilmiş YOLO sınıflandırma modelini yükler ve belirtilen resim üzerinde test eder.
    
    Argümanlar:
        model_path (str): Eğitilmiş modelin ağırlık dosyasının (.pt) yolu.
        image_source (str): Test edilecek resmin yolu veya içinde resimler bulunan bir klasör.
    """
    print(f"Model yükleniyor: {model_path}")
    if not os.path.exists(model_path):
        print(f"HATA: '{model_path}' bulunamadı!")
        print("Lütfen eğitim işleminin tamamlandığından ve dosyanın var olduğundan emin olun.")
        return

    try:
        # Modeli Yükle
        model = YOLO(model_path)
        
        test_image_path = None
        
        # Eğer kaynak bir klasörse içinden rastgele bir resim seç
        if os.path.isdir(image_source):
            print(f"'{image_source}' klasöründen rastgele bir resim seçiliyor...")
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            files = []
            for ext in valid_extensions:
                files.extend(glob.glob(os.path.join(image_source, f'*{ext}')))
                files.extend(glob.glob(os.path.join(image_source, f'*{ext.upper()}')))
                
            if not files:
                print(f"HATA: '{image_source}' klasöründe geçerli resim bulunamadı.")
                return
                
            test_image_path = random.choice(files)
        # Eğer kaynak doğrudan bir dosyaysa onu kullan 
        elif os.path.isfile(image_source):
            test_image_path = image_source
        else:
            print(f"HATA: '{image_source}' geçerli bir dosya veya klasör değil!")
            return
            
        print(f"\nSeçilen Test Resmi: {test_image_path}")
        
        # Tahmin Et (Predict)
        # Sınıflandırma modelinde results değişkeni tek bir elemanlı liste döner
        results = model.predict(test_image_path, verbose=False)
        result = results[0]
        
        # En yüksek olasılığa sahip sınıfın indeksini al (Top-1)
        top1_index = result.probs.top1
        
        # Sınıf adını sözlükten (names) al
        predicted_class = result.names[top1_index]
        
        # Doğruluk oranını al (0 ile 1 arasındaki değeri yüzdeye çevir)
        confidence = float(result.probs.top1conf) * 100
        
        # Çıktıları Formatla
        print("-" * 40)
        print(" TAHMİN SONUCU ")
        print("-" * 40)
        print(f" Sınıf        : {predicted_class.upper()}")
        print(f" Doğruluk(%)  : {confidence:.2f}%")
        print("-" * 40)
        
        # Özel Durum Bildirimleri (Opsiyonel)
        if predicted_class == "eksik_vida":
            print("\n🚨 UYARI: Üründe EKSİK VİDA tespit edildi! 🚨")
            renk = (0, 0, 255) # Kırmızı
        elif predicted_class == "aparatsiz":
            print("\n🚨 UYARI: Üründe APARAT EKSİK! 🚨")
            renk = (0, 165, 255) # Turuncu
        elif predicted_class == "diger":
            print("\nℹ️ BİLGİ: Boş metal yüzey tespit edildi (Vidalık alan değil).")
            renk = (255, 255, 0) # Mavi
        else:
            print("\n✅ Ürün KUSURSUZ (OK).")
            renk = (0, 255, 0) # Yeşil

            
        # Resmi OpenCV ile oku ve üzerine çizim yap
        img = cv2.imread(test_image_path)
        if img is not None:
            h, w = img.shape[:2]
            
            # Sınıflandırma problemi olduğu için resmin tamamını temsil eden bir dış çerçeve çizelim
            # cv2.rectangle(resim, (sol_ust_x, sol_ust_y), (sag_alt_x, sag_alt_y), renk, kalinlik)
            cv2.rectangle(img, (5, 5), (w-5, h-5), renk, 3)
            
            # Metin oluştur: SINIF_ADI: %95.2
            text = f"{predicted_class.upper()}: {confidence:.1f}%"
            
            # Metni yazmak için arkaplanı belirginleştirecek bir dikdörtgen kutu çiz
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (5, 5), (5 + text_w, 5 + text_h + baseline + 10), renk, -1)
            
            # Metni yaz
            cv2.putText(img, text, (10, 5 + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Test sonucunu kaydet
            kayit_yolu = "test_sonucu.jpg"
            cv2.imwrite(kayit_yolu, img)
            print(f"\nGörsel sonucu '{kayit_yolu}' olarak kaydedildi!")
        else:
            print("\nUyarı: Sonuç resmi üzerine çizim yapılamadı (Resim okunamadı).")
            
    except Exception as e:
        print(f"\nTest sırasında bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    # Eğittiğimiz özel modelin gerçek yolu (30 Epoch - 4 Sınıf)
    model_yolu = r"runs\classify\beko_vida_modeli\train_sonuclari2\weights\best.pt"
    
    # Alternatif olarak eğer kendi özel dizinimizi kullandıysak:
    # model_yolu = "beko_vida_modeli/train_sonuclari/weights/best.pt"
    
    # Test edilecek resim veya rastgele seçim yapılacak klasör
    # Örneğin: "orijinal_vida.jpg" veya "ham_parcalar"
    test_kaynagi = "ham_parcalar"
    
    # Eğer öncelikli bir tekil dosya varsa onu deneyebiliriz
    # Daha önce eski modelde "Eksik Vida" sanılan boş metal alanı (y256_x768 koordinatları) özellikle test ediyoruz:
    tekil_test_resmi = r"ham_parcalar\eksik_vida_3_vidasizlar_y256_x768.jpg"
    
    if os.path.exists(tekil_test_resmi):
        test_yolo_model(model_path=model_yolu, image_source=tekil_test_resmi)
    else:
        test_yolo_model(model_path=model_yolu, image_source=test_kaynagi)
