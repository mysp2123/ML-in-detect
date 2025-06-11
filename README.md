# AI-Powered Casting Defect Detection System

This project uses Deep Learning and Computer Vision to detect and analyze defects in metal casting products.

## Features

1. **Product Classification** (using Deep Learning)
   - Classify products as defective/non-defective using ResNet model
   - Transfer learning for improved performance
   - Data augmentation for increased diversity

2. **Detailed Defect Analysis** (using Computer Vision)
   - Detection of multiple defect types (cracks, holes, surface abnormalities)
   - Measurement of defect size and severity
   - Analysis of detection confidence

3. **Batch Analysis and Reporting**
   - Process multiple images simultaneously
   - Generate detailed statistical reports
   - Visualize results through charts

## Installation

1. Clone repository:
```bash
git clone https://github.com/your-username/casting-defect-detection.git
cd casting-defect-detection
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
```bash
python train_model.py
```

### 2. Predict on a Single Image
```bash
python predict.py path_to_image
```

### 3. Detailed Defect Analysis
```bash
python defect_analyzer.py path_to_image
```

### 4. Batch Analysis
```bash
python batch_analyzer.py
```

## Data Structure

```
casting_512x512/
├── def_front/    # Defective product images
└── ok_front/     # Non-defective product images
```

## Results

Analysis results are saved in the `analysis_results/` directory:
- Images with marked defects
- CSV files with detailed information
- Statistical charts
- Summary reports

## Technologies Used

- PyTorch
- OpenCV
- scikit-image
- NumPy
- Pandas
- Matplotlib
- seaborn

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see the [LICENSE](LICENSE) file for details. 