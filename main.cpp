#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <map>
#include <queue>
#include <set>

struct PoseEdge {
    int src, dst;
    cv::Mat R, t;
};

//Function to find SIFT features in an image
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

//Function to Match SIFT Features between 2 different images and find good matching features
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

std::set<std::tuple<int, int>> seenMatches;

//Function to compare the all images to find best matching Images
std::vector<std::tuple<int, int, std::vector<cv::DMatch>, cv::Mat, cv::Mat>> FindBestImagePairs(const std::vector<cv::Mat>& allDescriptors,
    const std::vector<std::vector<cv::KeyPoint>>& allKeypoints,
    const cv::Mat& K,
    int matchThreshold = 200,
    double minDisplacementThreshold = 5,
    int minTriangulatedPoints = 50)
{
    std::vector<std::tuple<int, int, std::vector<cv::DMatch>, cv::Mat, cv::Mat>> matchPairs;

    for (size_t i = 0; i < allDescriptors.size(); ++i)
    {
        for (size_t j = i + 1; j < allDescriptors.size(); ++j)
        {
            if (seenMatches.count(std::make_tuple(static_cast<int>(i), static_cast<int>(j))))
                continue;

            auto matches = MatchFeatures(allDescriptors[i], allDescriptors[j]);

            if (matches.size() < matchThreshold)
                continue;

            // Extract matched keypoints
            std::vector<cv::Point2f> pts1, pts2;
            for (const auto& match : matches) 
            {
                pts1.push_back(allKeypoints[i][match.queryIdx].pt);
                pts2.push_back(allKeypoints[j][match.trainIdx].pt);
            }

            double avgDisplacement = 0.0;
            for (size_t k = 0; k < pts1.size(); ++k) 
            {
                avgDisplacement += cv::norm(pts1[k] - pts2[k]);
            }
            avgDisplacement /= pts1.size();

            // Estimate Essential Matrix
            cv::Mat mask;
            cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);
            if (E.empty())
                continue;

            // Recover pose
            cv::Mat R, t;
            int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, mask);
            if (avgDisplacement < minDisplacementThreshold)
                continue;
            
            seenMatches.insert(std::make_tuple(static_cast<int>(i), static_cast<int>(j)));
            matchPairs.emplace_back(static_cast<int>(i), static_cast<int>(j), matches, R.clone(), t.clone());
        }
    }
    return matchPairs;
}

std::vector<PoseEdge> BuildPoseGraph(const std::vector<std::tuple<int, int, std::vector<cv::DMatch>, cv::Mat, cv::Mat>>& matchPairs) {
    std::vector<PoseEdge> graph;
    for (const auto& match : matchPairs) {
        int src = std::get<0>(match);
        int dst = std::get<1>(match);
        const cv::Mat& R = std::get<3>(match);
        const cv::Mat& t = std::get<4>(match);
        graph.push_back({src, dst, R.clone(), t.clone()});
    }
    return graph;
}

void ComputeGlobalPoses(const std::vector<PoseEdge>& graph, std::map<int, cv::Mat>& globalRotations, std::map<int, cv::Mat>& globalTranslations) {
    std::set<int> visited;
    std::queue<int> q;
    globalRotations[0] = cv::Mat::eye(3, 3, CV_64F);
    globalTranslations[0] = cv::Mat::zeros(3, 1, CV_64F);
    visited.insert(0);
    q.push(0);

    while (!q.empty()) {
        int curr = q.front(); q.pop();
        for (const auto& edge : graph) {
            int next = -1;
            bool forward = false;
            if (edge.src == curr && !visited.count(edge.dst)) {
                next = edge.dst; forward = true;
            } else if (edge.dst == curr && !visited.count(edge.src)) {
                next = edge.src; forward = false;
            } else continue;

            const cv::Mat& R_curr = globalRotations[curr];
            const cv::Mat& t_curr = globalTranslations[curr];

            if (forward) {
                globalRotations[next] = R_curr * edge.R;
                globalTranslations[next] = R_curr * edge.t + t_curr;
            } else {
                cv::Mat R_inv = edge.R.t();
                cv::Mat t_inv = -R_inv * edge.t;
                globalRotations[next] = R_curr * R_inv;
                globalTranslations[next] = R_curr * t_inv + t_curr;
            }

            visited.insert(next);
            q.push(next);
        }
    }
}


std::vector<cv::Point3d> TriangulateGlobalPoints(
    const std::vector<std::tuple<int, int, std::vector<cv::DMatch>>>& matches,
    const std::vector<std::vector<cv::KeyPoint>>& allKeypoints,
    const std::map<int, cv::Mat>& globalRot,
    const std::map<int, cv::Mat>& globalTrans,
    const cv::Mat& K)
{
    std::vector<cv::Point3d> points;
    for (const auto& [i, j, pairMatches] : matches) {
        cv::Mat R1 = globalRot.at(i), t1 = globalTrans.at(i);
        cv::Mat R2 = globalRot.at(j), t2 = globalTrans.at(j);

        cv::Mat P1, P2;
        cv::hconcat(R1, t1, P1);
        cv::hconcat(R2, t2, P2);
        P1 = K * P1;
        P2 = K * P2;

        std::vector<cv::Point2f> pts1, pts2;
        for (const auto& m : pairMatches) {
            pts1.push_back(allKeypoints[i][m.queryIdx].pt);
            pts2.push_back(allKeypoints[j][m.trainIdx].pt);
        }

        cv::Mat pts4D;
        cv::triangulatePoints(P1, P2, pts1, pts2, pts4D);

        for (int k = 0; k < pts4D.cols; ++k) {
            cv::Mat x = pts4D.col(k);
            x /= x.at<float>(3);
            points.emplace_back(x.at<float>(0), x.at<float>(1), x.at<float>(2));
        }
    }
    return points;
}


// void SavePointCloudXYZ(const std::string& filename, const std::vector<cv::Point3d>& points) 
// {
//     int invalid_pts = 0;
//     std::ofstream file(filename);
//     for (const auto& pt : points) 
//     {
//         if (pt.x >1000 || pt.y >1000 || pt.z>1000 || pt.x <-1000 || pt.y <-1000 || pt.z<-1000)
//         {
//             invalid_pts = invalid_pts+1;
//         }
//         else
//         {
//             file << pt.x << " " << pt.y << " " << pt.z << "\n";
//         }
//     }
//     file.close();
//     std::cout << "Saved point cloud with " << points.size()-invalid_pts << " points to " << filename << std::endl;
// }

int main() 
{

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
    auto matchPairsFull = FindBestImagePairs(allDescriptors, allKeypoints, K);
    if (matchPairsFull.empty()) 
    {
        std::cerr << "No good image pairs found for matching!" << std::endl;
        return -1;
    }

    auto poseGraph = BuildPoseGraph(matchPairsFull);

    std::map<int, cv::Mat> globalRotations, globalTranslations;
    ComputeGlobalPoses(poseGraph, globalRotations, globalTranslations);

    std::vector<std::tuple<int, int, std::vector<cv::DMatch>>> matchOnly;
    for (const auto& m : matchPairsFull)
        matchOnly.emplace_back(std::get<0>(m), std::get<1>(m), std::get<2>(m));

    auto points = TriangulateGlobalPoints(matchOnly, allKeypoints, globalRotations, globalTranslations, K);

    std::ofstream file("output.xyz");
    int invalid_pts = 0;
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
    std::cout << "Saved point cloud with " << points.size()-invalid_pts<<" removing: "<<invalid_pts<<" invalid pts" << std::endl;
    



    // for (const auto& match : matchPairs)
    // {
    //     int i = std::get<0>(match);
    //     int j = std::get<1>(match);
    //     const auto& matches = std::get<2>(match);
    //     const cv::Mat& R = std::get<3>(match);
    //     const cv::Mat& t = std::get<4>(match);
    
    //     std::cout << "Match: Image " << i << " <--> Image " << j 
    //               << " | Matches: " << matches.size() << "\n";
    //     std::cout << "Rotation matrix R:\n" << R << "\n";
    //     std::cout << "Translation vector t:\n" << t.t() << "\n";
    //     std::cout << "--------------------------------------------------\n";
    // }

    // SavePointCloudXYZ("output.xyz", pointCloud);

    return 0;
}
