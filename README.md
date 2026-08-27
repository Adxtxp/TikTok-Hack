# TikTok-Hack

A reproducible Python repository for TikTok Hackathon project.

## Project Structure

```
.
├── src/                    # Core source code modules
│   ├── __init__.py
│   ├── transforms.py       # Data transforms and augmentation
│   ├── features.py         # Feature extraction
│   ├── model.py            # Model definitions
│   ├── train.py            # Training logic
│   ├── evaluate.py         # Evaluation metrics
│   └── inference.py        # Inference pipeline
├── notebooks/              # Jupyter notebooks
│   └── .gitkeep
├── configs/                # Configuration files
│   └── default.yaml
├── outputs/                # Generated outputs (ignored by git)
│   └── .gitkeep
├── run_on_colab.ipynb      # Original Colab notebook
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

(To be populated as logic is ported from the Colab notebook)

## Configuration

Edit `configs/default.yaml` to customize the pipeline parameters.
