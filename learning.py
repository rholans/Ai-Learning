import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Dataset cuaca untuk pelatihan
data = {
    'Cuaca': ['Cerah', 'Cerah', 'Mendung', 'Hujan', 'Hujan', 'Hujan', 'Mendung', 'Cerah', 'Cerah', 'Hujan'],
    'Suhu': ['Panas', 'Panas', 'Panas', 'Sedang', 'Dingin', 'Dingin', 'Dingin', 'Sedang', 'Dingin', 'Sedang'],
    'Kelembaban': ['Tinggi', 'Tinggi', 'Tinggi', 'Tinggi', 'Normal', 'Normal', 'Normal', 'Tinggi', 'Normal', 'Tinggi'],
    'Berangin': ['Tidak', 'Ya', 'Tidak', 'Tidak', 'Tidak', 'Ya', 'Ya', 'Tidak', 'Tidak', 'Ya'],
    'Main': ['Tidak', 'Tidak', 'Ya', 'Ya', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak']
}

# Konversi ke DataFrame
df = pd.DataFrame(data)

# Label encoding semua kolom
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Simpan encoder untuk nanti

# Pisahkan fitur dan label
X = df.drop('Main', axis=1)
y = df['Main']

# Latih model Decision Tree dengan kriteria 'entropy' (ID3)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Input user (bisa dimodifikasi)
user_input = {
    'Cuaca': 'Mendung',
    'Suhu': 'Dingin',
    'Kelembaban': 'Normal',
    'Berangin': 'Ya'
}

# Siapkan DataFrame input user
input_df = pd.DataFrame([user_input])

# Encode input user dengan encoder yang sama
for col in input_df.columns:
    le = label_encoders[col]
    combined = pd.concat([pd.Series(le.classes_), input_df[col]], ignore_index=True)
    le.fit(combined)
    input_df[col] = le.transform(input_df[col])

# Prediksi
prediction = model.predict(input_df)[0]
hasil = 'YA' if prediction == 1 else 'TIDAK'

print("=== HASIL PREDIKSI ===")
print("Apakah boleh bermain di luar?", hasil)

# Visualisasi pohon keputusan
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=X.columns, class_names=label_encoders['Main'].classes_, filled=True)
plt.title("Pohon Keputusan (Decision Tree)")
plt.show()
