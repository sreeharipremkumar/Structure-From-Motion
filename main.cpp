#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>


std::vector<cv::KeyPoint> SiftImage(const std::string& imgPath, cv::Ptr<cv::SIFT>& sift, cv::Mat& descriptors) 
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) 
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return {};
    }

    std::vector<cv::KeyPoint> keypoints;
    sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    std::cout << "Detected " << keypoints.size() << " keypoints.\n";

    return keypoints;
}

std::vector<cv::DMatch> MatchFeatures(const cv::Mat& desc1, const cv::Mat& desc2) 
{
    std::vector<std::vector<cv::DMatch>> knnMatches;
    std::vector<cv::DMatch> goodMatches;

    cv::FlannBasedMatcher matcher;
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    const float ratio_thresh = 0.75f;
    for (const auto& match : knnMatches) 
    {
        if (match.size() == 2 && match[0].distance < ratio_thresh * match[1].distance) 
        {
            goodMatches.push_back(match[0]);
        }
    }

    return goodMatches;
}

std::vector<std::tuple<int, int, std::vector<cv::DMatch>>> FindConsecutiveMatches(const std::vector<cv::Mat>& allDescriptors) 
{
    std::vector<std::tuple<int, int, std::vector<cv::DMatch>>> matchPairs;

    for (size_t i = 0; i + 1 < allDescriptors.size(); ++i) 
    {
        auto matches = MatchFeatures(allDescriptors[i], allDescriptors[i + 1]);
        matchPairs.emplace_back(static_cast<int>(i), static_cast<int>(i+1), matches);

    }

    return matchPairs;
}

std::vector<std::tuple<int, int, std::vector<cv::DMatch>>> FindBestImagePairs(const std::vector<cv::Mat>& allDescriptors, int matchThreshold = 100) 
{

    std::vector<std::tuple<int, int, std::vector<cv::DMatch>>> matchPairs;

    for (size_t i = 0; i < allDescriptors.size(); ++i) 
    {
        for (size_t j = i + 1; j < allDescriptors.size(); ++j)
        {
            auto matches = MatchFeatures(allDescriptors[i], allDescriptors[j]);
            if (matches.size() >= matchThreshold) 
            {
                matchPairs.emplace_back(static_cast<int>(i), static_cast<int>(j), matches);
            }
        }
    }

    return matchPairs;
}

void EstimatePoseFromMatches(const std::vector<cv::KeyPoint>& kp1,const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches,const cv::Mat& K,cv::Mat& R, cv::Mat& t)
{
    std::vector<cv::Point2f> pts1, pts2;

    for (const auto& match : matches) 
    {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0);
    cv::recoverPose(E, pts1, pts2, K, R, t);
}

std::vector<cv::Point3d> TriangulatePoints(const std::vector<cv::KeyPoint>& kp1,const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches,const cv::Mat& K,const cv::Mat& R, const cv::Mat& t)
{
    std::vector<cv::Point2f> pts1, pts2;

    for (const auto& match : matches) 
    {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    // Projection matrix for first camera: [I | 0]
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);

    // Projection matrix for second camera: [R | t]
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P2 = K * Rt;

    // Triangulate points
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

    // Convert from homogeneous to 3D
    std::vector<cv::Point3d> points3D;
    for (int i = 0; i < points4D.cols; ++i) 
    {
        cv::Mat col = points4D.col(i);
        col /= col.at<float>(3);
        points3D.emplace_back(col.at<float>(0),col.at<float>(1),col.at<float>(2));
    }

    return points3D;
}

void SavePointCloudXYZ(const std::string& filename, const std::vector<cv::Point3d>& points) 
{
    int invalid_pts = 0;
    std::ofstream file(filename);
    for (const auto& pt : points) 
    {
        if (pt.x >1000 || pt.y >1000 || pt.z>1000 || pt.x <-1000 || pt.y <-1000 || pt.z<-1000)
        {
            invalid_pts = invalid_pts+1;
        }
        else
        {
            file << pt.x << " " << pt.y << " " << pt.z << "\n";
        }
    }
    file.close();
    std::cout << "Saved point cloud with " << points.size()-invalid_pts << " points to " << filename << std::endl;
}

int main() {

    bool useAllPairs = true;
    double fx = 1520.4;
    double fy = 1525.9;
    double cx = 302.32;
    double cy = 246.87;
    
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    std::string imgDir = "templeRing/";

    // Create SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Containers for all results
    std::vector<cv::Mat> allDescriptors;
    std::vector<std::vector<cv::KeyPoint>> allKeypoints;
    std::vector<std::string> allImagePaths;
    std::vector<cv::Mat> allImages;

    // Load image again just to draw

    std::vector<std::string> filePaths;

    for (const auto& entry : std::filesystem::directory_iterator(imgDir)) 
    {
        if (entry.is_regular_file()) 
        {
            filePaths.push_back(entry.path().string());
        }
    }
    
    std::sort(filePaths.begin(), filePaths.end());
    
    for (const auto& imgPath : filePaths) 
    {

        std::cout << "Processing: " << imgPath << std::endl;

        cv::Mat descriptors;
        std::vector<cv::KeyPoint> keypoints = SiftImage(imgPath, sift, descriptors);
    
        if (!keypoints.empty()) 
        {
            allKeypoints.push_back(keypoints);
            allDescriptors.push_back(descriptors);
            allImagePaths.push_back(imgPath);

            allImages.push_back(cv::imread(imgPath, cv::IMREAD_GRAYSCALE));
        }
    }
    std::cout << "Processed " << allImagePaths.size() << " images." << std::endl;

    std::cout << "Matching Pairs in Progress" <<std::endl;

    auto matchPairs = useAllPairs ? FindBestImagePairs(allDescriptors) : FindConsecutiveMatches(allDescriptors);
    if (matchPairs.empty()) 
    {
        std::cerr << "No good image pairs found for matching!" << std::endl;
        return -1;
    }


    for (const auto& [i, j, matches] : matchPairs) {
        std::cout << "Match: Image " << i << " <--> Image " << j 
                  << " | Matches: " << matches.size() << "\n";
    }

    // --- Use best match pair to estimate pose and triangulate
    auto [i, j, matches] = matchPairs[0];
    cv::Mat R, t;
    EstimatePoseFromMatches(allKeypoints[i], allKeypoints[j], matches, K, R, t);

    std::vector<cv::Point3d> points3D = TriangulatePoints(allKeypoints[i], allKeypoints[j], matches, K, R, t);
    std::cout << "Triangulated " << points3D.size() << " 3D points.\n";

    // --- Save point cloud
    SavePointCloudXYZ("output_cloud.xyz", points3D);

    // cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);

    // // Draw keypoints
    // cv::Mat output;
    // cv::drawKeypoints(img, keypoints, output, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // // Show and save result
    // cv::imshow("SIFT Keypoints", output);
    // cv::imwrite("sift_output.png", output);
    // cv::waitKey(0);

    return 0;
}
