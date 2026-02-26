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

🚀 **[Live Demo](https://huggingface.co/spaces/TayyabManan/face-expression-detection)**

![Demo](https://img.shields.io/badge/Demo-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **7 Emotion Classes**: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
- **Multi-face Detection**: Detects and analyzes multiple faces in a single image
- **MTCNN Face Detection**: High-accuracy face detection with Haar Cascade fallback
- **Real-time Visualization**: Annotated images with bounding boxes and emotion labels
- **Confidence Scores**: Probability distribution across all emotion classes
- **Dark/Light Mode**: Toggle between themes
- **Responsive Design**: Works on desktop and mobile

## Demo

Upload any image with faces and get instant emotion predictions:

| Input | Output |
|-------|--------|
| Group photo | Annotated with emotion labels |
| Single portrait | Detailed probability breakdown |

## Tech Stack

- **Model**: ResNet-18 (transfer learning from ImageNet)
- **Face Detection**: MTCNN + Haar Cascade fallback
- **Backend**: Flask + Gunicorn
- **Frontend**: Vanilla JS with CSS animations
- **Dataset**: RAF-DB (Real-world Affective Faces Database)

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/TayyabManan/face-expression-detection.git
cd face-expression-detection

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

## Authors

- **[Muhammad Tayyab](https://github.com/TayyabManan)**
- **[Syed Measum](https://github.com/Syedmeasum14)**
- **Mustafa Rahim**

## Course

Machine Learning for Engineering Design

## License

MIT License

## Acknowledgments

- [RAF-DB Dataset](http://www.whdeng.cn/RAF/model1.html)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for MTCNN
- [PyTorch](https://pytorch.org/) for the deep learning framework
