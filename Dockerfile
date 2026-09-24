FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e .
ENV ROCKMAP_DATA_DIR=/data
VOLUME /data
EXPOSE 5000
CMD ["rockmap", "serve", "--host", "0.0.0.0", "--port", "5000"]
