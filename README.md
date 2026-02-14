1. Import Library
google.colab: Digunakan untuk mengakses file di Google Drive jika Anda menggunakan Google Colab.
warnings.filterwarnings('ignore'): Mengabaikan peringatan agar output lebih bersih.
numpy, pandas, seaborn, matplotlib: Library untuk analisis data dan visualisasi.
imblearn.over_sampling.SMOTE: Digunakan untuk menangani data yang tidak seimbang dengan membuat sampel sintetis.
sklearn: Library untuk machine learning, digunakan untuk preprocessing data, membagi dataset, membuat model, dan evaluasi.
2. Load Dataset
pd.read_csv: Membaca dataset dari file CSV.
df.head(5): Menampilkan 5 baris pertama dari dataset.
df.info(): Memberikan informasi tentang dataset, seperti jumlah baris, kolom, tipe data, dan jumlah nilai yang hilang.
3. Data Cleaning
df.drop('customerID', axis=1): Menghapus kolom customerID karena tidak relevan untuk analisis.
pd.to_numeric: Mengubah kolom TotalCharges menjadi tipe numerik. Nilai yang tidak bisa dikonversi akan diubah menjadi NaN.
df.isnull().sum(): Mengecek jumlah nilai yang hilang di setiap kolom.
df.dropna(): Menghapus baris yang memiliki nilai kosong.
pd.get_dummies: Mengubah data kategorikal menjadi data numerik dengan teknik one-hot encoding.
4. Split Dataset
X: Variabel independen (fitur) yang digunakan untuk memprediksi.
y: Variabel target (Churn_Yes), yaitu apakah pelanggan berhenti berlangganan atau tidak.
train_test_split: Membagi dataset menjadi data latih (80%) dan data uji (20%). Parameter stratify=y memastikan distribusi kelas target tetap seimbang di data latih dan uji.
5. Exploratory Data Analysis (EDA)
Distribusi Churn: Menggunakan sns.countplot untuk melihat distribusi pelanggan yang berhenti (Churn) dan tidak berhenti.
Histogram: Menampilkan distribusi data numerik seperti tenure, MonthlyCharges, dan TotalCharges.
Boxplot: Membandingkan distribusi data numerik berdasarkan kategori Churn.
6. Data Preprocessing
StandardScaler: Digunakan untuk menstandarisasi data agar memiliki mean = 0 dan standar deviasi = 1. Ini penting untuk algoritma yang sensitif terhadap skala data, seperti regresi logistik.
7. Modeling
Logistic Regression
LogisticRegression: Model machine learning untuk klasifikasi biner.
fit: Melatih model menggunakan data latih.
predict: Memprediksi kelas untuk data uji.
predict_proba: Menghasilkan probabilitas untuk setiap kelas.
classification_report: Menampilkan metrik evaluasi seperti precision, recall, F1-score.
roc_auc_score: Menghitung skor AUC (Area Under Curve) untuk mengevaluasi performa model.
Random Forest
RandomForestClassifier: Model ensemble yang menggunakan banyak decision tree untuk meningkatkan akurasi prediksi.
n_estimators=200: Menggunakan 200 pohon keputusan dalam ensemble.
fit: Melatih model menggunakan data latih.
predict: Memprediksi kelas untuk data uji.
predict_proba: Menghasilkan probabilitas untuk setiap kelas.
classification_report dan roc_auc_score: Sama seperti pada regresi logistik.
8. Model Evaluation
evaluate_model: Fungsi untuk mengevaluasi model menggunakan metrik seperti:
Accuracy: Persentase prediksi yang benar.
Precision: Proporsi prediksi positif yang benar.
Recall: Proporsi data positif yang berhasil diprediksi dengan benar.
F1 Score: Rata-rata harmonis antara precision dan recall.
AUC: Mengukur kemampuan model membedakan antara kelas positif dan negatif.
9. Menangani Data Tidak Seimbang dengan SMOTE
SMOTE (Synthetic Minority Oversampling Technique): Teknik untuk menangani data yang tidak seimbang dengan membuat sampel sintetis untuk kelas minoritas.
10. Evaluasi Model Setelah SMOTE
Setelah data latih diseimbangkan menggunakan SMOTE, model regresi logistik dilatih ulang.
Confusion Matrix: Menampilkan jumlah prediksi benar dan salah untuk setiap kelas.
Classification Report: Menampilkan metrik evaluasi seperti precision, recall, dan F1-score.
AUC Score: Mengukur performa model setelah data diseimbangkan.
Kesimpulan
Data Cleaning: Anda membersihkan data dengan menghapus kolom yang tidak relevan, menangani nilai kosong, dan mengubah data kategorikal menjadi numerik.
EDA: Anda menganalisis data untuk memahami distribusi dan pola.
Modeling: Anda menggunakan dua model (Logistic Regression dan Random Forest) untuk memprediksi churn pelanggan.
Evaluasi: Anda mengevaluasi model menggunakan metrik seperti accuracy, precision, recall, F1-score, dan AUC.