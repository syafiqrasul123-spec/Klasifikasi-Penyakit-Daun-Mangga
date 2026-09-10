<div align="center">

  <h1>🥭 Klasifikasi Penyakit Die Back pada Daun Mangga</h1>
  <p><b>Project Computer Vision & Deep Learning untuk Deteksi Dini Kesehatan Daun Mangga</b></p>

  [![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

</div>

---

## 📌 Deskripsi Proyek

Proyek ini berfokus pada pengolahan citra (*Computer Vision*) dan pembelajaran mesin (*Machine & Deep Learning*) untuk mengidentifikasi dan mengklasifikasikan kondisi kesehatan daun mangga secara otomatis. Sistem dirancang untuk membedakan antara **daun sehat (Healthy)** dan daun yang terinfeksi **penyakit Die Back**.

---

## 🔬 Metode & Algoritma

Penelitian dan eksperimen model membandingkan pendekatan *Classical Machine Learning* dengan arsitektur *Deep Learning*:

| Kategori | Algoritma / Arsitektur |
| :--- | :--- |
| **Machine Learning** | • Support Vector Machine (SVM)<br>• Random Forest<br>• Naive Bayes<br>• Decision Tree |
| **Deep Learning** | • Convolutional Neural Network (CNN)<br>• MobileNetV2<br>• ResNet50 |

---

## 🛠️ Teknologi & Tools

* **Bahasa Pemrograman:** Python 3.x
* **Computer Vision & Image Processing:** OpenCV
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning Framework:** Scikit-Learn
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Visualization:** Matplotlib

---

## 📊 Dataset & Preprocessing

> **Catatan:** Dataset tidak dimuat langsung ke dalam repositori GitHub ini karena keterbatasan ukuran file.

### Sumber Dataset
* **Dataset Asli:** [Mango Leaf Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)
* **Dataset Augmented:** [Mango Leaf Disease Dataset Augmented (Kaggle)](https://www.kaggle.com/datasets/shajedur0/mango-leaf-disease-dataset)

### Alur Pengolahan Data
1. **Preprocessing Citra:** Resizing, Normalisasi, dan Data Augmentation.
2. **Ekstraksi Fitur:** Pengambilan karakteristik citra berdasarkan Fitur Warna, Tekstur, dan Bentuk (untuk Machine Learning).
3. **Pelatihan Model:** Evaluasi performa model Machine Learning vs Transfer Learning pada Deep Learning.

### Klasifikasi Kelas Target
* `Healthy` *(Daun Sehat)*
* `Die Back` *(Daun Terinfeksi Penyakit Die Back)*

---

## ⚙️ Panduan Penggunaan Lokal

```bash
# 1. Clone repositori ini
git clone [https://github.com/syafiqrasul123-spec/Klasifikasi-Penyakit-Daun-Mangga.git](https://github.com/syafiqrasul123-spec/Klasifikasi-Penyakit-Daun-Mangga.git)

# 2. Masuk ke direktori proyek
cd Klasifikasi-Penyakit-Daun-Mangga

# 3. Buat dan aktifkan virtual environment (Opsional)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependency library yang dibutuhkan
pip install -r requirements.txt