import pandas as pd

# Load dataset
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# Filter data dengan sleep disorder
disorder_data = df[df['Sleep Disorder'] != 'None']

print('=== INPUT YANG PALING BERPOTENSI UNTUK PREDIKSI ===')
print()

# Ambil beberapa contoh untuk setiap kategori
print('CONTOH INPUT UNTUK SLEEP APNEA (3 contoh):')
apnea_samples = disorder_data[disorder_data['Sleep Disorder'] == 'Sleep Apnea'].head(3)
for i, (_, sample) in enumerate(apnea_samples.iterrows(), 1):
    bp_parts = sample['Blood Pressure'].split('/')
    print(f'{i}. Age: {sample["Age"]}, Gender: {sample["Gender"]}, Occupation: {sample["Occupation"]}')
    print(f'   Sleep Duration: {sample["Sleep Duration"]}, Quality: {sample["Quality of Sleep"]}, BMI: {sample["BMI Category"]}')
    print(f'   Activity: {sample["Physical Activity Level"]}, Heart Rate: {sample["Heart Rate"]}, Steps: {sample["Daily Steps"]}, BP: {bp_parts[0]}/{bp_parts[1]}')
    print()

print('CONTOH INPUT UNTUK INSOMNIA (3 contoh):')
insomnia_samples = disorder_data[disorder_data['Sleep Disorder'] == 'Insomnia'].head(3)
for i, (_, sample) in enumerate(insomnia_samples.iterrows(), 1):
    bp_parts = sample['Blood Pressure'].split('/')
    print(f'{i}. Age: {sample["Age"]}, Gender: {sample["Gender"]}, Occupation: {sample["Occupation"]}')
    print(f'   Sleep Duration: {sample["Sleep Duration"]}, Quality: {sample["Quality of Sleep"]}, BMI: {sample["BMI Category"]}')
    print(f'   Activity: {sample["Physical Activity Level"]}, Heart Rate: {sample["Heart Rate"]}, Steps: {sample["Daily Steps"]}, BP: {bp_parts[0]}/{bp_parts[1]}')
    print()

print('=== FAKTOR KUNCI YANG MEMBEDAKAN ===')
print()

# Hitung rata-rata untuk setiap disorder
summary = disorder_data.groupby('Sleep Disorder').agg({
    'Age': 'mean',
    'Sleep Duration': 'mean',
    'Quality of Sleep': 'mean',
    'Physical Activity Level': 'mean',
    'Heart Rate': 'mean',
    'Daily Steps': 'mean'
}).round(1)

print('RATA-RATA NILAI NUMERIK:')
print(summary)
print()

print('KESIMPULAN POLA UTAMA:')
print('• SLEEP APNEA: Usia 49+, Nurse/Sales Rep, BMI Overweight/Obese, Heart Rate 80+, Steps 3000-4000')
print('• INSOMNIA: Usia 43+, Teacher/Software Engineer, BMI Overweight, Activity 30-40, Quality Sleep 4-6')