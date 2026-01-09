# LaRS-segmentation-train-and-inference

Установка

```bash
docker build -t lars_inference .
```

Запуск

```bash
docker run --gpus all -p 8000:8000 lars_inference
```

Использование

```bash
curl -X POST http://localhost:8000/predict -F "file=@D:\path\to\image.jpg" --output mask.png
```