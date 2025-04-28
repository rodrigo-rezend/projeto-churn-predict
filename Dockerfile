# Imagem base oficial do Python
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o conteúdo do projeto para o container
COPY . .

# Atualiza o pip
RUN pip install --upgrade pip

# Instala as dependências
RUN pip install -r requirements.txt

# Expor a porta 8000
EXPOSE 8000

# Comando para rodar o app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
