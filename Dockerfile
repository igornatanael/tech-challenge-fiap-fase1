FROM python:3.11-slim

WORKDIR /app

# Dependencias do sistema (build de algumas libs cientificas)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Codigo do projeto
COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
