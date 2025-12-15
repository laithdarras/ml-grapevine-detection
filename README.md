# Grapevine Disease Detection (CSE 168: Lab 3)

## How to Run Instructions

### Requirements
- Docker installed and running
- No local Python, TensorFlow, or GPU required

### Dataset
The dataset is not included in the repository.  
You must provide the following directory structure locally before running:

- data/
  - train/   (*.jpg files named healthy_*.jpg or esca_*.jpg)
  - val/     (*.jpg files named healthy_*.jpg or esca_*.jpg)
  - test/    (*.jpg files named healthy_*.jpg or esca_*.jpg)

### Build Docker Image
From the project root directory:

```docker build -t grape-cnn .```

### Train the Model
Run training inside Docker (mounts current directory into container):

```docker run --rm -it -v "$(pwd):/workspace" grape-cnn python train.py```

This will train the model and save `grape_model.keras` to the project directory.

### Evaluate the Model
After training completes:

```docker run --rm -it -v "$(pwd):/workspace" grape-cnn python evaluation.py```

This outputs test loss, precision, and recall.
