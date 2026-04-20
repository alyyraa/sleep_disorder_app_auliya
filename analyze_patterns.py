import pandas as pd

# Load dataset
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# Filter data dengan sleep disorder
disorder_data = df[df['Sleep Disorder'] != 'None']

print('=== ANALISIS POLA INPUT UNTUK INSOMNIA vs SLEEP APNEA ===')
print()

# Group by Sleep Disorder dan hitung rata-rata
summary = disorder_data.groupby('Sleep Disorder').agg({
    'Age': 'mean',
    'Sleep Duration': 'mean',
    'Quality of Sleep': 'mean',
    'Physical Activity Level': 'mean',
    'Stress Level': 'mean',
    'Heart Rate': 'mean',
    'Daily Steps': 'mean',
    'BMI Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
    'Gender': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
    'Occupation': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
}).round(2)

print('RATA-RATA KARAKTERISTIK:')
print(summary)
print()

# Analisis khusus untuk BMI dan Occupation
print('DISTRIBUSI BMI CATEGORY:')
bmi_dist = pd.crosstab(disorder_data['Sleep Disorder'], disorder_data['BMI Category'])
print(bmi_dist)
print()

print('DISTRIBUSI OCCUPATION:')
occ_dist = pd.crosstab(disorder_data['Sleep Disorder'], disorder_data['Occupation'])
print(occ_dist)
print()

# Contoh input yang tipikal
print('=== CONTOH INPUT TIPIKAL ===')
print()
print('UNTUK SLEEP APNEA:')
apnea_data = disorder_data[disorder_data['Sleep Disorder'] == 'Sleep Apnea']
if len(apnea_data) > 0:
    sample = apnea_data.iloc[0]
    print(f'  Gender: {sample["Gender"]}')
    print(f'  Age: {sample["Age"]}')
    print(f'  Occupation: {sample["Occupation"]}')
    print(f'  Sleep Duration: {sample["Sleep Duration"]}')
    print(f'  Quality of Sleep: {sample["Quality of Sleep"]}')
    print(f'  Physical Activity Level: {sample["Physical Activity Level"]}')
    print(f'  BMI Category: {sample["BMI Category"]}')
    print(f'  Heart Rate: {sample["Heart Rate"]}')
    print(f'  Daily Steps: {sample["Daily Steps"]}')
    bp_parts = sample["Blood Pressure"].split("/")
    print(f'  Systolic BP: {bp_parts[0]}')
    print(f'  Diastolic BP: {bp_parts[1]}')

print()
print('UNTUK INSOMNIA:')
insomnia_data = disorder_data[disorder_data['Sleep Disorder'] == 'Insomnia']
if len(insomnia_data) > 0:
    sample = insomnia_data.iloc[0]
    print(f'  Gender: {sample["Gender"]}')
    print(f'  Age: {sample["Age"]}')
    print(f'  Occupation: {sample["Occupation"]}')
    print(f'  Sleep Duration: {sample["Sleep Duration"]}')
    print(f'  Quality of Sleep: {sample["Quality of Sleep"]}')
    print(f'  Physical Activity Level: {sample["Physical Activity Level"]}')
    print(f'  BMI Category: {sample["BMI Category"]}')
    print(f'  Heart Rate: {sample["Heart Rate"]}')
    print(f'  Daily Steps: {sample["Daily Steps"]}')
    bp_parts = sample["Blood Pressure"].split("/")
    print(f'  Systolic BP: {bp_parts[0]}')
    print(f'  Diastolic BP: {bp_parts[1]}')