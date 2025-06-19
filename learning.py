import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data = {
    'Cuaca': ['Cerah', 'Cerah', 'Mendung', 'Hujan', 'Hujan', 'Hujan', 'Mendung', 'Cerah', 'Cerah', 'Hujan'],
    'Suhu': ['Panas', 'Panas', 'Panas', 'Sedang', 'Dingin', 'Dingin', 'Dingin', 'Sedang', 'Dingin', 'Sedang'],
    'Kelembaban': ['Tinggi', 'Tinggi', 'Tinggi', 'Tinggi', 'Normal', 'Normal', 'Normal', 'Tinggi', 'Normal', 'Tinggi'],
    'Berangin': ['Tidak', 'Ya', 'Tidak', 'Tidak', 'Tidak', 'Ya', 'Ya', 'Tidak', 'Tidak', 'Ya'],
    'Main': ['Tidak', 'Tidak', 'Ya', 'Ya', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak']
}

df = pd.DataFrame(data)

label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 

X = df.drop('Main', axis=1)
y = df['Main']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

user_input = {
    'Cuaca': 'Mendung',
    'Suhu': 'Dingin',
    'Kelembaban': 'Normal',
    'Berangin': 'Ya'
}

input_df = pd.DataFrame([user_input])

for col in input_df.columns:
    le = label_encoders[col]
    combined = pd.concat([pd.Series(le.classes_), input_df[col]], ignore_index=True)
    le.fit(combined)
    input_df[col] = le.transform(input_df[col])

prediction = model.predict(input_df)[0]
hasil = 'YA' if prediction == 1 else 'TIDAK'

print("=== HASIL PREDIKSI ===")
print("Apakah boleh bermain di luar?", hasil)

plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=X.columns, class_names=label_encoders['Main'].classes_, filled=True)
plt.title("Pohon Keputusan (Decision Tree)")
plt.show()
