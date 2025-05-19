# Usa l'immagine base di Python
FROM python:3.10



# Imposta la cartella di lavoro all'interno del container
WORKDIR /workspace

# Copia tutti i file del progetto nella cartella di lavoro del container
COPY . .


# Installa le dipendenze necessarie 
RUN pip install --no-cache-dir -r requirements.txt


# Comando per eseguire il codice Python (main.py)
CMD ["python", "main.py"]
