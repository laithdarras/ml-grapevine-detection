FROM continuumio/anaconda3:main

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]