# Eigenfaces with Eye Normalization

This project implements an Eigenfaces model for facial recognition. It includes preprocessing steps for normalizing face images based on eye positions, Principal Component Analysis (PCA) for dimensionality reduction, and visualization of eigenfaces. The implementation uses OpenCV for image processing and C++ for computational efficiency.

---

## Features
- **Eye Position Normalization**: Aligns and scales images based on eye positions to ensure consistent alignment.
- **Principal Component Analysis (PCA)**: Computes eigenfaces, selecting components based on a specified energy percentage.
- **Visualization**: Displays an overlay of eigenfaces for inspection.
- **Error Handling**: Handles missing or invalid input files gracefully.

---

## Requirements
- **C++ Compiler**: C++17 or higher.
- **Libraries**:
  - OpenCV 4.x or higher
  - C++ Standard Library

---

## Setup
1. **Install OpenCV**:
   Install OpenCV from OpenCV website

2. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

3. **Compile the Program**:
   ```bash
   g++ -o eigenfaces main.cpp -std=c++17 `pkg-config --cflags --libs opencv4`
   ```

---

## Usage

### Command-line Syntax
```bash
./eigenfaces <energyPercent> <datasetFolder>
```

- `<energyPercent>`: Energy threshold for selecting principal components (e.g., `95.0`).
- `<datasetFolder>`: Path to the folder containing PGM images and corresponding `.eye` files.

### Example
```bash
./eigenfaces 95.0 ./dataset
```

---

## Input File Structure

1. **Face Images**:
   - Format: `.pgm`
   - Example: `subject01.pgm`

2. **Eye Position Files**:
   - Format: `.eye`
   - Example: `subject01.eye`
   - Content:
     ```
     left_x left_y right_x right_y
     ```

### Dataset Folder Structure
```
/dataset
├── subject01.pgm
├── subject01.eye
├── subject02.pgm
├── subject02.eye
...
```

---

## Key Functions

1. **`readEyePosition`**:
   Reads eye coordinates from a `.eye` file.

2. **`normalizeImage`**:
   Aligns and scales an image based on eye positions.

3. **`computeEigenfaces`**:
   Performs PCA to compute eigenfaces and selects components based on energy.

4. **`displayEigenfaces`**:
   Visualizes eigenfaces by overlaying the top components.

---

## Outputs

1. **Eigenface Model**:
   - Saved as `eigenface_model.yml`.

2. **Visualization**:
   - Displays overlaid eigenfaces in a new window.

---

## Troubleshooting

### Common Issues
- **"No images found in the dataset folder"**:
  - Ensure the folder contains `.pgm` files and corresponding `.eye` files.

- **"Error reading eye position data"**:
  - Check the `.eye` file format and ensure it contains valid coordinates.

- **PCA Error**:
  - Verify that images have consistent dimensions.

---

## Acknowledgments
- OpenCV: [https://opencv.org/](https://opencv.org/)
- PCA and Eigenfaces: [https://en.wikipedia.org/wiki/Eigenface](https://en.wikipedia.org/wiki/Eigenface)
