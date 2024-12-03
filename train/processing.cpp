#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// Function to preprocess images
void preprocessImages(const std::string& inputDir, const std::string& outputDir, int targetWidth, int targetHeight) {
    // Paths for output folders
    std::string outputDirPGM = outputDir + "/pgm";
    std::string outputDirEye = outputDir + "/eye";

    // Create directories if they don't exist
    fs::create_directories(outputDirPGM);
    fs::create_directories(outputDirEye);

    // Iterate over files in input directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            std::string filePath = entry.path().string();
            std::string fileName = entry.path().stem().string(); // File name without extension

            // Read the image in grayscale
            cv::Mat img = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Could not read image: " << filePath << std::endl;
                continue;
            }

            // Resize the image to the target size
            cv::Mat resizedImg;
            cv::resize(img, resizedImg, cv::Size(targetWidth, targetHeight));

            // Save the resized image
            std::string outputPGMPath = outputDirPGM + "/" + fileName + ".pgm";
            cv::imwrite(outputPGMPath, resizedImg);
            std::cout << "Saved resized PGM: " << outputPGMPath << std::endl;

            // Generate a new .eye file
            std::string outputEyePath = outputDirEye + "/" + fileName + ".eye";
            std::ofstream eyeFile(outputEyePath);
            if (eyeFile.is_open()) {
                // Define default eye positions in the resized image
                int lx = targetWidth * 0.35;  // Left eye x position (35% from the left)
                int ly = targetHeight * 0.4;  // Left eye y position (40% from the top)
                int rx = targetWidth * 0.65;  // Right eye x position (65% from the left)
                int ry = targetHeight * 0.4;  // Right eye y position (40% from the top)

                eyeFile << "#LX\tLY\tRX\tRY\n";
                eyeFile << lx << "\t" << ly << "\t" << rx << "\t" << ry << "\n";
                eyeFile.close();

                std::cout << "Saved .eye file: " << outputEyePath << std::endl;
            }
            else {
                std::cerr << "Could not create .eye file: " << outputEyePath << std::endl;
            }
        }
    }
}

int main() {
    // Input and output directories
    std::string inputDir = "C:/CompVision/HW3/test";
    std::string outputDir = "C:/CompVision/HW3/test_processed";

    // Target size for the images
    int targetWidth = 256;
    int targetHeight = 256;

    // Preprocess the images
    preprocessImages(inputDir, outputDir, targetWidth, targetHeight);

    return 0;
}