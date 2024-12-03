#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Function to read the model file
bool loadModel(const string& modelFile, Mat& meanFace, Mat& eigenVectors) {
    FileStorage fs(modelFile, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open model file: " << modelFile << endl;
        return false;
    }

    // Load mean face
    fs["mean"] >> meanFace;
    if (meanFace.empty()) {
        cerr << "Error: meanFace is empty after loading from the model file." << endl;
        return false;
    }
    cout << "Loaded meanFace size: " << meanFace.size() << endl;

    // Load eigenvectors
    fs["eigenVectors"] >> eigenVectors;
    if (eigenVectors.empty()) {
        cerr << "Error: eigenVectors is empty after loading from the model file." << endl;
        return false;
    }
    cout << "Loaded eigenVectors size: " << eigenVectors.size() << endl;

    fs.release();
    return true;
}

// Function to project a face onto the PCA subspace
Mat projectToPCA(const Mat& face, const Mat& meanFace, const Mat& eigenfaces) {
    Mat faceVector = face.reshape(1, 1);
    faceVector.convertTo(faceVector, CV_32F);

    Mat meanVector = meanFace.reshape(1, 1);
    meanVector.convertTo(meanVector, CV_32F);

    Mat projection = (faceVector - meanVector) * eigenfaces.t();
    return projection;
}

// Function to compute the Euclidean distance
double computeEuclideanDistance(const Mat& vec1, const Mat& vec2) {
    return norm(vec1, vec2, NORM_L2);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: HW3_test.exe <face_image> <model_file> <dataset_folder>" << endl;
        return -1;
    }

    string faceImageFile = argv[1];
    string modelFile = argv[2];
    string datasetFolder = argv[3];

    // Load the input face image
    Mat inputFace = imread(faceImageFile, IMREAD_GRAYSCALE);
    if (inputFace.empty()) {
        cerr << "Failed to load face image: " << faceImageFile << endl;
        return -1;
    }

    // Load the trained model
    Mat meanFace, eigenfaces;
    if (!loadModel(modelFile, meanFace, eigenfaces)) {
        return -1;
    }

    // Preprocess the input face (resize to the same size as training images)
    Mat resizedFace;
    try {
        resize(inputFace, resizedFace, Size(meanFace.cols, meanFace.rows));
    }
    catch (const cv::Exception& e) {
        cerr << "Error during resize: " << e.what() << endl;
        return -1;
    }

    // Project the input face onto the PCA subspace
    Mat inputProjection = projectToPCA(resizedFace, meanFace, eigenfaces);

    // Load and preprocess the dataset
    vector<string> datasetLabels;
    vector<Mat> datasetProjections;
    double minDistance = DBL_MAX;
    int bestMatchIndex = -1;

    for (const auto& entry : fs::directory_iterator(datasetFolder)) {
        string path = entry.path().string();
        if (path.ends_with(".pgm")) {
            Mat datasetImage = imread(path, IMREAD_GRAYSCALE);
            if (datasetImage.empty()) {
                cerr << "Error loading dataset image: " << path << endl;
                continue;
            }

            // Resize and project dataset image
            Mat resizedDatasetImage;
            resize(datasetImage, resizedDatasetImage, Size(meanFace.cols, meanFace.rows));
            Mat datasetProjection = projectToPCA(resizedDatasetImage, meanFace, eigenfaces);

            // Compute distance
            double distance = computeEuclideanDistance(inputProjection, datasetProjection);
            if (distance < minDistance) {
                minDistance = distance;
                bestMatchIndex = datasetLabels.size();
            }

            // Store projection and label
            datasetLabels.push_back(path);
            datasetProjections.push_back(datasetProjection);
        }
    }

    // Display the result
    if (bestMatchIndex != -1) {
        cout << "Best match: " << datasetLabels[bestMatchIndex] << " (distance: " << minDistance << ")" << endl;

        Mat bestMatchFace = imread(datasetLabels[bestMatchIndex], IMREAD_GRAYSCALE);

        if (bestMatchFace.empty()) {
            cerr << "Error loading best match face: " << datasetLabels[bestMatchIndex] << endl;
            return -1;
        }

        // Overlay the input face onto the most similar face directly
        Mat overlayImage;
        try {
            addWeighted(inputFace, 0.5, bestMatchFace, 0.5, 0, overlayImage);
        }
        catch (const cv::Exception& e) {
            cerr << "Error during overlay: " << e.what() << endl;
            return -1;
        }

        // Display the two images
        imshow("Most Similar Image", bestMatchFace);
        imshow("Overlayed Image", overlayImage);
        waitKey(0);
    }
    else {
        cerr << "No match found!" << endl;
    }


    return 0;
}
