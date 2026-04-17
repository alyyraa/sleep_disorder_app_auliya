# Menggunakan image Python resmi versi ringan
FROM python:3.10-slim

# Mengatur working directory di dalam container
WORKDIR /app

# Menginstal dependency sistem yang mungkin dibutuhkan oleh pandas/numpy/xgboost
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Menyalin file requirements terlebih dahulu
# Ini memanfaatkan cache Docker layer sehingga tidak perlu install ulang library jika kode berubah
COPY requirements.txt .

# Menginstal library Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Menyalin seluruh kode aplikasi ke dalam container
COPY . .

# Mengekspos port 5000 untuk Flask
EXPOSE 5000

# Menyiapkan environment variable
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Menjalankan aplikasi menggunakan Gunicorn untuk lingkungan production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
