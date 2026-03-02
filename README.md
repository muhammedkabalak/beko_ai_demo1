# 🚀 Beko AI - Buzdolabı Montaj Hata Tespit Sistemi

Bu proje, Beko üretim hattındaki montaj süreçlerini yapay zeka ile denetlemek ve hatalı üretimleri minimize etmek amacıyla geliştirilmiş bir mezuniyet tez çalışmasıdır. [cite: 2025-11-24] Sistem, yüksek çözünürlüklü sanayi tipi görseller üzerinde gerçek zamanlı hata analizi gerçekleştirmektedir. [cite: 2026-02-09]



## ✨ Öne Çıkan Özellikler
* **YOLOv11 Entegrasyonu:** En güncel nesne algılama mimarisiyle %100'e varan eğitim doğruluğu.
* **Sliding Window Inference:** 128x128 pencerelerle görüntü parçalama tekniği sayesinde en küçük detayları kaçırmaz.


## 🛠️ Teknik Özellikler
| Bileşen | Detay |
| :--- | :--- |
| **Dil / Sürüm** | Python 3.12  |
| **Algoritma** | YOLOv11 - Image Classification |
| **Sınıflar** | `ok`, `eksik_vida`, `aparatsiz`, `diger` |
| **Kütüphaneler** | Ultralytics, Streamlit, OpenCV, Torch (CUDA)  |

## ⚙️ Hızlı Kurulum ve Çalıştırma

Sistemi herhangi bir Windows bilgisayarda tek tıkla çalıştırmak için:

1.  **Projeyi İndirin:** Klasörü bilgisayarınıza kopyalayın.
2.  **`baslat.bat` Dosyasını Çalıştırın:** Klasördeki bu dosyaya çift tıklayın. Script; sanal ortamı (`beko_env`) kuracak, gerekli kütüphaneleri yükleyecek ve sistemi otomatik başlatacaktır.
3.  **Analiz:** Açılan tarayıcı ekranından `test_resmi.jpg` dosyasını yükleyerek sistemi test edin.



## 📁 Proje Klasör Yapısı
* `beko_arayuz.py`: Ana arayüz ve analiz kodları.
* `runs/`: Eğitilmiş model ağırlıklarını (`best.pt`) barındıran klasör.
* `requirements.txt`: Gerekli kütüphane listesi.

---
*Bu çalışma, Beko kalite kontrol standartları gözetilerek bir mezuniyet tezi kapsamında hazırlanmıştır.* 
