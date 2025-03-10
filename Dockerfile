# Usa una imagen base optimizada de Python 3.10
FROM python:3.13

# Define variables de entorno
ENV HOST=0.0.0.0
ENV LISTEN_PORT=8000

# Expone el puerto 8000 para que se pueda acceder desde fuera del contenedor
EXPOSE 8000

# Actualiza el sistema e instala dependencias necesarias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    gcc \
    g++ \
    make \
    wget \
    libsqlite3-dev \
    sqlite3 \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Descargar y compilar una versión compatible de SQLite
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3450100.tar.gz && \
    tar -xzf sqlite-autoconf-3450100.tar.gz && \
    cd sqlite-autoconf-3450100 && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3450100 sqlite-autoconf-3450100.tar.gz

# Verificar la versión de SQLite instalada
RUN sqlite3 --version

# Recompilar Python para que use la nueva versión de SQLite
RUN wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz && \
    tar -xzf Python-3.10.13.tgz && \
    cd Python-3.10.13 && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make && \
    make install && \
    cd .. && \
    rm -rf Python-3.10.13 Python-3.10.13.tgz

# Verificar que Python usa la nueva versión de SQLite
RUN python3 -c "import sqlite3; print('SQLite Version:', sqlite3.sqlite_version)"

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias y las instala sin caché para optimizar el tamaño del contenedor
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copia el código fuente de la aplicación al contenedor
COPY . .

# Copia el archivo .env al contenedor
COPY .env .env

# Comando por defecto al iniciar el contenedor: ejecuta la aplicación con FastAPI y Uvicorn
CMD ["uvicorn", "thebestchatbotwithfastapi:app", "--host", "0.0.0.0", "--port", "8000"]
