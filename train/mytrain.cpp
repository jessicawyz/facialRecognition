#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

void readEyePosition(const string& filename, Point& leftEye, Point& rightEye) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening eye position file: " << filename << endl;
        exit(1);
    }

    string header;
    getline(file, header); // Skip the header line

    int lx, ly, rx, ry;
    file >> lx >> ly >> rx >> ry;

    if (file.fail()) {
        cerr << "Error reading eye position data from file: " << filename << endl;
        exit(1);
    }

    leftEye = Point(lx, ly);
    rightEye = Point(rx, ry);

    file.close();
}

void normalizeImage(const Mat& src, Mat& dst, const Point& leftEye, const Point& rightEye) {
    double eyeDistance = norm(leftEye - rightEye);
    double desiredEyeDistance = 100.0; // Scale factor for normalization
    double scale = desiredEyeDistance / eyeDistance;

    Point2f eyesCenter = (leftEye + rightEye) * 0.5;
    Mat rotMat = getRotationMatrix2D(eyesCenter, 0, scale);

    warpAffine(src, dst, rotMat, Size(src.cols * scale, src.rows * scale));
}

void computeEigenfaces(const vector<Mat>& images, const double energyPercent, Mat& meanFace, Mat& eigenVectors) {
    // Flatten each image into a single row of a matrix
    Mat data(images.size(), images[0].rows * images[0].cols, CV_32F);
    for (size_t i = 0; i < images.size(); i++) {
        Mat row = data.row(i);
        Mat imgRow = images[i].reshape(1, 1); // Flatten image
        imgRow.convertTo(row, CV_32F);
    }

    cout << "Data matrix size: " << data.size() << endl;

    // Perform PCA on the data matrix
    PCA pca(data, Mat(), PCA::DATA_AS_ROW);
    cout << "pca.eigenvectors.rows: " << pca.eigenvectors.rows << endl;
    cout << "pca.eigenvectors.cols: " << pca.eigenvectors.cols << endl;

    if (pca.eigenvalues.empty()) {
        cerr << "Error: PCA failed, eigenvalues are empty!" << endl;
        return;
    }

    cout << "Eigenvalues: " << pca.eigenvalues.t() << endl; // Transpose

    // Calculate total energy from eigenvalues
    double totalEnergy = sum(pca.eigenvalues)[0];
    double cumulativeEnergy = 0.0;
    int numComponents = 0;

    // Select the number of components based on the energy percentage
    for (int i = 0; i < pca.eigenvalues.rows; i++) {
        cumulativeEnergy += pca.eigenvalues.at<float>(i);
        if ((cumulativeEnergy / totalEnergy) >= (energyPercent / 100.0)) {
            numComponents = i + 1;
            break;
        }
    }

    // Ensure numComponents is valid
    if (numComponents <= 0) {
        cerr << "Error: No components selected during PCA." << endl;
        exit(1);
    }

    cout << "Number of components selected: " << numComponents << endl;

    // Limit numComponents to the available eigenvectors
    numComponents = min(numComponents, pca.eigenvectors.rows);

    cout << "Adjusted number of components: " << numComponents << endl;

    // Save the mean face and the selected eigenvectors
    meanFace = pca.mean.clone();
    eigenVectors = pca.eigenvectors.rowRange(0, numComponents).clone();
}

void displayEigenfaces(const cv::Mat& eigenVectors, const cv::Mat& meanFace, int width, int height) {
    // Ensure eigenVectors has valid dimensions
    CV_Assert(!eigenVectors.empty() && eigenVectors.dims == 2);
    CV_Assert(eigenVectors.cols == width * height); // Ensure flattened size matches

    // Determine the number of eigenfaces to overlay (at most 10)
    int numEigenfaces = std::min(10, eigenVectors.rows);

    // Create an empty image for overlay (initialize with zeros)
    cv::Mat overlay = cv::Mat::zeros(height, width, CV_32F);

    for (int i = 0; i < numEigenfaces; i++) {
        // Extract and reshape the i-th eigenvector
        cv::Mat eigenface = eigenVectors.row(i).reshape(1, height);

        // Add the mean face back to the eigenface
        cv::Mat eigenfaceWithMean;
        add(eigenface, meanFace.reshape(1, height), eigenfaceWithMean);

        // Normalize the eigenface for consistent scaling
        cv::normalize(eigenfaceWithMean, eigenfaceWithMean, 0, 1, cv::NORM_MINMAX, CV_32F);

        // Add this normalized eigenface to the overlay
        overlay += eigenfaceWithMean;
    }

    // Normalize the final overlay image
    cv::Mat overlayNormalized;
    cv::normalize(overlay, overlayNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Display the overlaid eigenfaces
    cv::imshow("Overlaid Eigenfaces", overlayNormalized);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <energyPercent> <datasetFolder>" << endl;
        return 1;
    }

    // Parse command-line arguments
    double energyPercent = atof(argv[1]); // Convert the first argument to a double
    string datasetFolder = argv[2];      // Second argument is the dataset folder

    if (energyPercent <= 0.0 || energyPercent > 100.0) {
        cerr << "Error: energyPercent must be in the range (0, 100]." << endl;
        return 1;
    }

    cout << "Using energy percent: " << energyPercent << "%" << endl;
    cout << "Using dataset folder: " << datasetFolder << endl;

    vector<Mat> images;

    for (const auto& entry : fs::directory_iterator(datasetFolder)) {
        string path = entry.path().string();
        if (path.ends_with(".pgm")) {
            string eyeFile = path.substr(0, path.size() - 4) + ".eye";
            if (!fs::exists(eyeFile)) {
                cerr << "Error: Eye file missing for image: " << path << endl;
                continue;
            }

            cout << "Processing image: " << path << endl;

            Point leftEye, rightEye;
            readEyePosition(eyeFile, leftEye, rightEye);
            cout << "Eye positions: Left(" << leftEye << "), Right(" << rightEye << ")" << endl;

            Mat img = imread(path, IMREAD_GRAYSCALE);
            if (img.empty()) {
                cerr << "Error: Failed to load image: " << path << endl;
                continue;
            }

            Mat normalized;
            normalizeImage(img, normalized, leftEye, rightEye);
            if (normalized.empty()) {
                cerr << "Error: Normalized image is empty for image: " << path << endl;
                continue;
            }

            images.push_back(normalized);
        }
    }

    if (images.empty()) {
        cerr << "No images found in the dataset folder." << endl;
        return -1;
    }

    cout << "Total images processed: " << images.size() << endl;

    Mat meanFace, eigenVectors;
    computeEigenfaces(images, energyPercent, meanFace, eigenVectors);

    // Save the model
    string modelFile = "eigenface_model.yml"; // Fixed model file name
    FileStorage fs(modelFile, FileStorage::WRITE);
    fs << "mean" << meanFace;
    fs << "eigenVectors" << eigenVectors;
    fs.release();
    cout << "Model saved successfully to: " << modelFile << endl;

    // Display top 10 Eigenfaces
    int width = images[0].cols;
    int height = images[0].rows;
    displayEigenfaces(eigenVectors, meanFace, width, height);

    return 0;
}

