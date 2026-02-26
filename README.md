---
title: Face Expression Detection
emoji: 😊
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 5000
pinned: false
---

# Face Expression Detection in Group Photos

A deep learning web application that detects and classifies facial expressions in images. Built with PyTorch and Flask, featuring a ResNet-18 model trained on the RAF-DB dataset achieving **80% accuracy**.

## Features

- **7 Emotion Classes**: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
- **Multi-face Detection**: Detects and analyzes multiple faces in a single image
- **MTCNN Face Detection**: High-accuracy face detection with Haar Cascade fallback
- **Real-time Visualization**: Annotated images with bounding boxes and emotion labels
- **Confidence Scores**: Probability distribution across all emotion classes


## Tech Stack

- **Model**: ResNet-18 (transfer learning from ImageNet)
- **Face Detection**: MTCNN + Haar Cascade fallback
- **Backend**: Flask + Gunicorn
- **Dataset**: RAF-DB (Real-world Affective Faces Database)

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Azohajutt/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition

# Install dependencies
pip install -r requirements-docker.txt

# Run the app
python app.py
```

Open http://localhost:5000 in your browser.

### Docker

```bash
docker build -t face-expression .
docker run -p 5000:5000 face-expression
```

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 80% |
| Dataset | RAF-DB |
| Architecture | ResNet-18 |
| Input Size | 100x100 |

### Per-Class Performance

| Emotion | Description |
|---------|-------------|
| Happiness | Highest accuracy |
| Neutral | High accuracy |
| Surprise | Good accuracy |
| Sadness | Moderate accuracy |
| Anger | Moderate accuracy |
| Fear | Lower accuracy (limited samples) |
| Disgust | Lower accuracy (limited samples) |

## Project Structure

```
├── app.py                 # Flask application
├── config.py              # Configuration settings
├── Dockerfile             # Docker configuration
├── requirements-docker.txt # Python dependencies
├── models/
│   └── best_resnet_rafdb.pth  # Trained model weights
├── src/
│   └── models/
│       └── emotion_resnet.py  # Model architecture
├── static/
│   └── css/
│       └── style.css      # Styling
└── templates/
    └── index.html         # Frontend
```


## License

MIT License

## Acknowledgments

- [RAF-DB Dataset](http://www.whdeng.cn/RAF/model1.html)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for MTCNN
- [PyTorch](https://pytorch.org/) for the deep learning framework
