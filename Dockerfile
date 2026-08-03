# Menggunakan image Python resmi versi ringan
FROM python:3.10-slim

ENV TZ=Asia/Jakarta \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Mengatur working directory di dalam container
WORKDIR /app

# Menginstal dependency sistem yang mungkin dibutuhkan oleh pandas/numpy/xgboost
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    libgomp1 \
    tzdata \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo "${TZ}" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Menyalin file requirements terlebih dahulu
# Ini memanfaatkan cache Docker layer sehingga tidak perlu install ulang library jika kode berubah
COPY requirements.txt .

# Menginstal library Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Menyalin seluruh kode aplikasi ke dalam container
COPY . .

# Preserve an immutable copy of the deployment state inside the image. At
# runtime, SQLite and model artifacts are initialized together in one volume.
# /app/models is then a stable link to the persistent runtime model directory.
RUN mkdir -p /opt/sleep-stress-seed \
    && cp /app/sleep_disorder.db /opt/sleep-stress-seed/sleep_disorder.db \
    && cp -a /app/models /opt/sleep-stress-seed/models \
    && cd /opt/sleep-stress-seed \
    && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d ' ' -f 1 > /opt/sleep-stress-seed.sha256 \
    && rm -rf /app/models \
    && ln -s /data/models /app/models

RUN chmod +x /app/docker-entrypoint.sh

# Mengekspos port 5000 untuk Flask
EXPOSE 5000

# Menyiapkan environment variable
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Menjalankan aplikasi menggunakan Gunicorn untuk lingkungan production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
