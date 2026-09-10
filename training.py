import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


print("===== MEMULAI PROSES TRAINING MODEL =====")


# =========================
# 1. LOAD DATA
# =========================

print("\n----- MEMBACA DATA DARI EXCEL -----")

df_nama = pd.read_excel("fitur_dataset.xlsx", sheet_name="nama")
df_warna = pd.read_excel("fitur_dataset.xlsx", sheet_name="warna")
df_tekstur = pd.read_excel("fitur_dataset.xlsx", sheet_name="tekstur")
df_bentuk = pd.read_excel("fitur_dataset.xlsx", sheet_name="bentuk")

print("Data berhasil dibaca")
print("Jumlah data :", len(df_nama))


# =========================
# 2. GABUNGKAN FITUR
# =========================

print("\n----- MENGGABUNGKAN SEMUA FITUR -----")

fitur = df_warna.merge(df_tekstur, on="file").merge(df_bentuk, on="file")

print("Jumlah fitur :", fitur.shape)


# =========================
# 3. SPLIT X dan Y
# =========================

print("\n----- MEMISAHKAN FITUR DAN LABEL -----")

X = fitur.drop("file", axis=1)
y = df_nama["class"]

print("Jumlah fitur input :", X.shape)
print("Jumlah label :", y.shape)


# =========================
# 4. SCALING
# =========================

print("\n----- MELAKUKAN SCALING DATA -----")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling selesai")
print("Contoh data setelah scaling:")
print(X_scaled[:3])


# =========================
# 5. TRAIN TEST SPLIT
# =========================

print("\n----- MEMBAGI DATA TRAINING DAN TESTING -----")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Data training :", len(X_train))
print("Data testing :", len(X_test))


# =========================
# 6. SVM
# =========================

print("\n----- TRAINING SVM -----")

svm_model = SVC(kernel="rbf")
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print("SVM selesai")


# =========================
# 7. RANDOM FOREST
# =========================

print("\n----- TRAINING RANDOM FOREST -----")

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest selesai")


# =========================
# 8. NAIVE BAYES
# =========================

print("\n----- TRAINING NAIVE BAYES -----")

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

print("Naive Bayes selesai")


# =========================
# 9. DECISION TREE
# =========================

print("\n----- TRAINING DECISION TREE -----")

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("Decision Tree selesai")


# =========================
# 10. HASIL
# =========================

print("\n===== HASIL PERBANDINGAN MODEL =====")

print("Accuracy SVM            :", svm_acc)
print("Accuracy Random Forest  :", rf_acc)
print("Accuracy Naive Bayes    :", nb_acc)
print("Accuracy Decision Tree  :", dt_acc)

print("\n----- VISUALISASI AKURASI -----")

model_names = ["SVM", "Random Forest", "Naive Bayes", "Decision Tree"]
accuracies = [svm_acc, rf_acc, nb_acc, dt_acc]

plt.figure()
plt.bar(model_names, accuracies)

plt.title("Perbandingan Akurasi Model")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0.9, 1.0)

plt.show()

print("\n----- VISUALISASI t-SNE -----")

# Convert label ke numpy array
y_array = y.values

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()

# Ambil label unik
unique_labels = set(y_array)

for label in unique_labels:
    idx = (y_array == label)
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label)

plt.legend()
plt.title("Visualisasi t-SNE")
plt.xlabel("Dimensi 1")
plt.ylabel("Dimensi 2")

plt.show()

print("\n===== SELESAI =====")
