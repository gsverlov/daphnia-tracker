# Fish Tracker

Computer vision pipeline for tracking fish in aquarium videos.

## Setup

This project targets Python 3.12 on Windows.

```powershell
# Create the virtual environment (one-time)
py -3.12 -m venv .venv

# Activate it
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Project structure

```
fish-tracker/
├── data/         # Input video files (gitignored)
├── output/       # Tracking results, CSVs, annotated videos (gitignored)
├── src/          # Source code (Python package)
├── notebooks/    # Jupyter notebooks for exploration and prototyping
├── requirements.txt
├── .gitignore
└── README.md
```
