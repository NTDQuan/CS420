FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

ENV PIP_DEFAULT_TIMEOUT=10000

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "FaceRecognition", "/bin/bash", "-c"]

RUN conda install -c conda-forge uvicorn

COPY . .

EXPOSE 8000

CMD ["conda", "run", "-n", "FaceRecognition", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]