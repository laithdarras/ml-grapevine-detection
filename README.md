# Grapevine Disease Detection

An educational project exploring image classification of grapevine diseases using **TensorFlow** and **Docker**. This repository demonstrates setting up a containerized training and evaluation workflow for a convolutional neural network. **Results are not production-grade**; the project is for learning purposes.

---

## Getting Started

### Requirements
- Docker installed and running  
- No local Python, TensorFlow, or GPU required  

### Dataset
The dataset is **not included**. Before running, create the following directory structure locally:

```
data/
train/ (.jpg files named healthy_.jpg or esca_.jpg)
val/ (.jpg files named healthy_.jpg or esca_.jpg)
test/ (.jpg files named healthy_.jpg or esca_*.jpg)
```

---

### Build Docker Image
From the project root:

```bash
docker build -t grape-cnn .
```

---

### Train the Model
Run training inside Docker (mounts current directory into container):
```
docker run --rm -it -v "$(pwd):/workspace" grape-cnn python train.py
```
This will train the model and save `grape_model.keras` to the project directory.

---

### Evaluate the Model
After training completes:
```
docker run --rm -it -v "$(pwd):/workspace" grape-cnn python evaluation.py
```
Outputs include test loss, precision, and recall.

---

## Tools & Skills
- Languages: Python
- Machine Learning: TensorFlow, Convolutional Neural Networks
- DevOps / Containerization: Docker
- Workflow: Training, evaluation, dataset management
