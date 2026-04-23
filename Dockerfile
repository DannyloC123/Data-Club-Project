FROM python:3.10-slim

WORKDIR /app

# Only install dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    pandas \
    scikit-learn \
    openpyxl \
    nltk \
    matplotlib \
    sentence-transformers

CMD ["python", "main.py"]