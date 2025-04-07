#include <opencv2/opencv.hpp>
#include <iostream>

std::vector<cv::KeyPoint> SiftImage(const std::string& imgPath, cv::Ptr<cv::SIFT>& sift, cv::Mat& descriptors) 
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return {};
    }

    std::vector<cv::KeyPoint> keypoints;
    sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    std::cout << "Detected " << keypoints.size() << " keypoints.\n";

    return keypoints;
}

int main() {
    std::string imgPath = "templeRing/templeR0001.png";

    // Create SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Containers for results
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints = SiftImage(imgPath, sift, descriptors);

    // Load image again just to draw
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);

    // Draw keypoints
    cv::Mat output;
    cv::drawKeypoints(img, keypoints, output, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Show and save result
    cv::imshow("SIFT Keypoints", output);
    cv::imwrite("sift_output.png", output);
    cv::waitKey(0);

    return 0;
}
