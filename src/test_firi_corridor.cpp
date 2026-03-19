#include "perception_tool/grid_map.hpp"
#include "perception_tool/firi_corridor_generator.hpp"
#include "perception_tool/firi_mvie2d.hpp"

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct EllipseShape {
    Eigen::Matrix2d L = Eigen::Matrix2d::Identity();
    Eigen::Vector2d d = Eigen::Vector2d::Zero();
};

double computeInscribedRadius(const std::vector<Eigen::Vector2d>& polygon,
                              const Eigen::Vector2d& center)
{
    double radius = 1e18;
    const int n = static_cast<int>(polygon.size());
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d& a = polygon[i];
        const Eigen::Vector2d& b = polygon[(i + 1) % n];
        const Eigen::Vector2d edge = b - a;
        const double len = edge.norm();
        if (len < 1e-15) {
            continue;
        }
        Eigen::Vector2d normal(-edge.y(), edge.x());
        normal /= len;
        radius = std::min(radius, std::abs(normal.dot(center - a)));
    }
    return radius;
}

double pointToSegmentDistance(const Eigen::Vector2d& p,
                              const Eigen::Vector2d& a,
                              const Eigen::Vector2d& b)
{
    const Eigen::Vector2d ab = b - a;
    const double denom = ab.squaredNorm();
    if (denom < 1e-12) {
        return (p - a).norm();
    }
    const double t = std::max(0.0, std::min(1.0, (p - a).dot(ab) / denom));
    return (p - (a + t * ab)).norm();
}

double minDistancePointToPolygon(const Eigen::Vector2d& p,
                                 const std::vector<Eigen::Vector2d>& polygon)
{
    double min_dist = 1e18;
    for (size_t i = 0; i < polygon.size(); ++i) {
        min_dist = std::min(min_dist,
                            pointToSegmentDistance(p, polygon[i],
                                                   polygon[(i + 1) % polygon.size()]));
    }
    return min_dist;
}

EllipseShape computeInitialEllipse(const std::vector<Eigen::Vector2d>& seed)
{
    EllipseShape ellipse;
    for (const auto& vertex : seed) {
        ellipse.d += vertex;
    }
    ellipse.d /= static_cast<double>(seed.size());
    const double radius = std::max(computeInscribedRadius(seed, ellipse.d) * 0.8, 1e-4);
    ellipse.L = radius * Eigen::Matrix2d::Identity();
    return ellipse;
}

EllipseShape computeFinalEllipse(const std::vector<perception_tools::HalfPlane2D>& planes,
                                 const Eigen::Vector2d& center_hint)
{
    const int m = static_cast<int>(planes.size());
    Eigen::MatrixXd A(m, 2);
    Eigen::VectorXd b(m);
    for (int i = 0; i < m; ++i) {
        A.row(i) = planes[i].normal.transpose();
        b(i) = planes[i].offset;
    }

    perception_tools::MVIE2D mvie;
    const auto ellipse = mvie.solve(A, b, center_hint);
    return {ellipse.L, ellipse.d};
}

std::vector<Eigen::Vector2d> sampleEllipse(const EllipseShape& ellipse, int samples = 180)
{
    std::vector<Eigen::Vector2d> points;
    points.reserve(samples);
    constexpr double kPi = 3.14159265358979323846;
    for (int i = 0; i < samples; ++i) {
        const double theta = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(samples);
        const Eigen::Vector2d unit(std::cos(theta), std::sin(theta));
        points.push_back(ellipse.d + ellipse.L * unit);
    }
    return points;
}

std::vector<std::vector<Eigen::Vector2d>> generateObstaclePolygons(
    const Eigen::Vector2d& env_size,
    int num_obstacles,
    double min_spacing,
    double min_clearance,
    const std::vector<Eigen::Vector2d>& seed,
    const EllipseShape& epsilon0,
    const Eigen::Vector2d& box_min,
    const Eigen::Vector2d& box_max,
    std::mt19937& rng)
{
    constexpr double kPi = 3.14159265358979323846;
    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    std::uniform_real_distribution<double> phase_dist(0.0, 0.25 * kPi);
    std::uniform_real_distribution<double> radius_dist(1.0, 3.0);
    std::uniform_int_distribution<int> vertex_count_dist(3, 8);
    std::uniform_real_distribution<double> x_dist(box_min.x(), box_max.x());
    std::uniform_real_distribution<double> y_dist(box_min.y(), box_max.y());

    const auto ellipse_pts = sampleEllipse(epsilon0, 120);
    std::vector<std::vector<Eigen::Vector2d>> obstacles;
    std::vector<Eigen::Vector2d> placed_centers;

    for (int obs_idx = 0; obs_idx < num_obstacles; ++obs_idx) {
        bool placed = false;
        for (int attempt = 0; attempt < 200 && !placed; ++attempt) {
            const Eigen::Vector2d center(x_dist(rng), y_dist(rng));
            const int num_pts = vertex_count_dist(rng);
            const double phase = phase_dist(rng);

            std::vector<double> angles(num_pts);
            for (int i = 0; i < num_pts; ++i) {
                angles[i] = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(num_pts)
                            + phase + 0.15 * (unit01(rng) - 0.5);
            }
            std::sort(angles.begin(), angles.end());

            std::vector<Eigen::Vector2d> polygon;
            polygon.reserve(num_pts);
            for (double angle : angles) {
                const double radius = radius_dist(rng);
                Eigen::Vector2d vertex = center + radius * Eigen::Vector2d(std::cos(angle), std::sin(angle));
                vertex.x() = std::max(0.0, std::min(env_size.x(), vertex.x()));
                vertex.y() = std::max(0.0, std::min(env_size.y(), vertex.y()));
                polygon.push_back(vertex);
            }

            double dist_to_seed = 1e18;
            for (const auto& vertex : polygon) {
                dist_to_seed = std::min(dist_to_seed, minDistancePointToPolygon(vertex, seed));
            }

            double dist_to_ellipse = 1e18;
            for (const auto& vertex : polygon) {
                for (const auto& e_pt : ellipse_pts) {
                    dist_to_ellipse = std::min(dist_to_ellipse, (vertex - e_pt).norm());
                }
            }

            bool spacing_ok = true;
            for (const auto& other_center : placed_centers) {
                if ((other_center - center).norm() <= min_spacing) {
                    spacing_ok = false;
                    break;
                }
            }

            if (spacing_ok && dist_to_seed > min_clearance && dist_to_ellipse > min_clearance) {
                obstacles.push_back(polygon);
                placed_centers.push_back(center);
                placed = true;
            }
        }
    }

    return obstacles;
}

grid_map::RowMatrixXi rasterizePolygons(const grid_map::GridMap& map,
                                        const std::vector<std::vector<Eigen::Vector2d>>& polygons)
{
    const Eigen::Vector2i voxel_num = map.getVoxelNum();
    cv::Mat mask = cv::Mat::zeros(voxel_num.x(), voxel_num.y(), CV_8UC1);

    for (const auto& polygon : polygons) {
        std::vector<cv::Point> pts;
        pts.reserve(polygon.size());
        for (const auto& vertex : polygon) {
            Eigen::Vector2i idx;
            map.posToIndex(vertex, idx);
            idx.x() = std::max(0, std::min(idx.x(), voxel_num.x() - 1));
            idx.y() = std::max(0, std::min(idx.y(), voxel_num.y() - 1));
            pts.emplace_back(idx.y(), idx.x());
        }
        const std::vector<std::vector<cv::Point>> poly_group{pts};
        cv::fillPoly(mask, poly_group, cv::Scalar(255));
    }

    grid_map::RowMatrixXi occupancy = grid_map::RowMatrixXi::Zero(voxel_num.x(), voxel_num.y());
    for (int x = 0; x < voxel_num.x(); ++x) {
        for (int y = 0; y < voxel_num.y(); ++y) {
            occupancy(x, y) = (mask.at<unsigned char>(x, y) > 0) ? 1 : 0;
        }
    }
    return occupancy;
}

void drawPolyline(cv::Mat& img,
                  const std::vector<Eigen::Vector2d>& polyline,
                  const std::function<cv::Point(const Eigen::Vector2d&)>& worldToPixel,
                  const cv::Scalar& color,
                  int thickness,
                  bool closed)
{
    if (polyline.size() < 2) {
        return;
    }
    for (size_t i = 0; i + 1 < polyline.size(); ++i) {
        cv::line(img, worldToPixel(polyline[i]), worldToPixel(polyline[i + 1]),
                 color, thickness, cv::LINE_AA);
    }
    if (closed) {
        cv::line(img, worldToPixel(polyline.back()), worldToPixel(polyline.front()),
                 color, thickness, cv::LINE_AA);
    }
}

void drawDashedRect(cv::Mat& img,
                    const std::function<cv::Point(const Eigen::Vector2d&)>& worldToPixel,
                    const Eigen::Vector2d& min_pt,
                    const Eigen::Vector2d& max_pt,
                    const cv::Scalar& color)
{
    const std::vector<Eigen::Vector2d> corners = {
        {min_pt.x(), min_pt.y()},
        {max_pt.x(), min_pt.y()},
        {max_pt.x(), max_pt.y()},
        {min_pt.x(), max_pt.y()}
    };

    for (size_t i = 0; i < corners.size(); ++i) {
        const Eigen::Vector2d& a = corners[i];
        const Eigen::Vector2d& b = corners[(i + 1) % corners.size()];
        const Eigen::Vector2d diff = b - a;
        const int segments = 18;
        for (int k = 0; k < segments; k += 2) {
            const double t0 = static_cast<double>(k) / static_cast<double>(segments);
            const double t1 = static_cast<double>(k + 1) / static_cast<double>(segments);
            cv::line(img, worldToPixel(a + t0 * diff), worldToPixel(a + t1 * diff),
                     color, 2, cv::LINE_AA);
        }
    }
}

void drawLegend(cv::Mat& img)
{
    const cv::Rect box(img.cols - 280, 20, 240, 184);
    cv::rectangle(img, box, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(img, box, cv::Scalar(0, 0, 0), 1);

    const int x0 = box.x + 16;
    int y = box.y + 24;
    const int dx = 58;

    cv::line(img, {x0, y}, {x0 + dx, y}, cv::Scalar(0, 220, 0), 3, cv::LINE_AA);
    cv::putText(img, "Seed Polytope", {x0 + 68, y + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    y += 34;
    for (int k = 0; k < 6; ++k) {
        if (k % 2 == 0) {
            cv::line(img, {x0 + k * 10, y}, {x0 + (k + 1) * 10, y},
                     cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
        }
    }
    cv::putText(img, "Bounding Box", {x0 + 68, y + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    y += 34;
    cv::ellipse(img, {x0 + 28, y}, {18, 10}, -25.0, 0.0, 360.0,
                cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    cv::putText(img, "epsilon_0", {x0 + 68, y + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    y += 34;
    cv::ellipse(img, {x0 + 28, y}, {22, 14}, 0.0, 0.0, 360.0,
                cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, "epsilon_final", {x0 + 68, y + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    y += 34;
    cv::line(img, {x0, y}, {x0 + dx, y}, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
    cv::putText(img, "Final Polytope", {x0 + 68, y + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
}

void visualizeCorridor(const grid_map::GridMap& map,
                       const std::vector<std::vector<Eigen::Vector2d>>& obstacle_polygons,
                       const std::vector<Eigen::Vector2d>& seed,
                       const std::vector<Eigen::Vector2d>& final_polytope,
                       const EllipseShape& epsilon0,
                       const EllipseShape& epsilon_final,
                       const Eigen::Vector2d& bbox_min,
                       const Eigen::Vector2d& bbox_max,
                       const Eigen::Vector2d& display_offset)
{
    const double viz_size = map.getMapSize().x();
    const int scale = 16;
    const int viz_pixels = static_cast<int>(viz_size * scale);

    cv::Mat img(viz_pixels, viz_pixels, CV_8UC3, cv::Scalar(245, 245, 245));

    auto worldToPixel = [&](const Eigen::Vector2d& pt) {
        const Eigen::Vector2d display_pt = pt + display_offset;
        return cv::Point(static_cast<int>(display_pt.x() * scale),
                         static_cast<int>(viz_pixels - display_pt.y() * scale));
    };

    for (int k = 0; k <= static_cast<int>(viz_size); ++k) {
        const int px = static_cast<int>(k * scale);
        cv::line(img, {px, 0}, {px, viz_pixels}, cv::Scalar(220, 220, 220), 1);
        cv::line(img, {0, px}, {viz_pixels, px}, cv::Scalar(220, 220, 220), 1);
    }

    for (const auto& polygon : obstacle_polygons) {
        std::vector<cv::Point> pts;
        pts.reserve(polygon.size());
        for (const auto& vertex : polygon) {
            pts.push_back(worldToPixel(vertex));
        }
        const std::vector<std::vector<cv::Point>> poly_group{pts};
        cv::fillPoly(img, poly_group, cv::Scalar(190, 190, 190), cv::LINE_AA);
        cv::polylines(img, poly_group, true, cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    }

    drawDashedRect(img, worldToPixel, bbox_min, bbox_max, cv::Scalar(255, 0, 0));
    drawPolyline(img, seed, worldToPixel, cv::Scalar(0, 220, 0), 3, true);
    drawPolyline(img, sampleEllipse(epsilon0), worldToPixel, cv::Scalar(0, 0, 255), 3, true);
    drawPolyline(img, sampleEllipse(epsilon_final), worldToPixel, cv::Scalar(255, 0, 0), 3, true);
    drawPolyline(img, final_polytope, worldToPixel, cv::Scalar(255, 0, 255), 3, true);

    cv::putText(img, "FIRI Algorithm: Complete Implementation",
                {img.cols / 2 - 220, 36}, cv::FONT_HERSHEY_SIMPLEX,
                0.9, cv::Scalar(0, 0, 0), 2);
    drawLegend(img);

    const char* display_env = std::getenv("DISPLAY");
    if (display_env == nullptr || display_env[0] == '\0') {
        std::cout << "DISPLAY is not available, skip interactive window.\n";
        return;
    }

    try {
        cv::imshow("FIRI Corridor", img);
        cv::waitKey(0);
    } catch (const cv::Exception& e) {
        std::cout << "OpenCV interactive window is unavailable: "
                  << e.what() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    constexpr double kPi = 3.14159265358979323846;
    std::random_device rd;
    unsigned int random_seed = rd();
    if (argc > 1) {
        random_seed = static_cast<unsigned int>(std::stoul(argv[1]));
    }
    std::mt19937 rng(random_seed);

    grid_map::GridMap map;
    const double map_size = 50.0;
    const double resolution = 0.1;
    map.init(map_size, map_size, resolution);

    perception_tools::FootprintSpec footprint;
    footprint.length = 1.5;
    footprint.width = 0.75;
    footprint.offset_x = 0.0;

    perception_tools::BoundingBoxSpec bbox;
    bbox.ahead = 6.0;
    bbox.behind = 6.0;
    bbox.side = 6.0;

    perception_tools::GeneratorOptions options;
    options.max_iter = 15;
    options.convergence_rho = 1e-4;
    options.use_path_seed = false;

    const Eigen::Vector2d display_offset(map_size * 0.5, map_size * 0.5);
    const Eigen::Vector2d robot_pos(25.0 - display_offset.x(), 25.0 - display_offset.y());
    const double robot_yaw = kPi / 6.0;

    perception_tools::CorridorGenerator generator;
    const auto seed = generator.buildFootprintSeed(robot_pos, robot_yaw, footprint);
    const auto epsilon0 = computeInitialEllipse(seed);
    const Eigen::Vector2d bbox_min(robot_pos.x() - bbox.side, robot_pos.y() - bbox.side);
    const Eigen::Vector2d bbox_max(robot_pos.x() + bbox.side, robot_pos.y() + bbox.side);
    const Eigen::Vector2d bbox_min_display = bbox_min + display_offset;
    const Eigen::Vector2d bbox_max_display = bbox_max + display_offset;

    auto obstacle_polygons = generateObstaclePolygons(
        Eigen::Vector2d(map_size, map_size), 15, 0.8, 2.0,
        {seed[0] + display_offset, seed[1] + display_offset, seed[2] + display_offset, seed[3] + display_offset},
        {epsilon0.L, epsilon0.d + display_offset},
        bbox_min_display, bbox_max_display, rng);

    for (auto& polygon : obstacle_polygons) {
        for (auto& vertex : polygon) {
            vertex -= display_offset;
        }
    }

    const auto occupancy = rasterizePolygons(map, obstacle_polygons);
    map.setMap(occupancy);

    const auto result = generator.generate(map, robot_pos, robot_yaw,
                                           footprint, bbox, options);

    assert(!result.obstacles.empty());
    assert(!result.planes.empty());
    assert(result.vertices.size() >= 4);

    for (const auto& seed_point : result.seed) {
        for (const auto& plane : result.planes) {
            assert(plane.normal.dot(seed_point) <= plane.offset + 1e-5);
        }
    }

    const auto epsilon_final = computeFinalEllipse(result.planes, robot_pos);
    assert(epsilon_final.L.determinant() > epsilon0.L.determinant());

    std::cout << "test_firi_corridor passed\n";
    std::cout << "random seed: " << random_seed << '\n';
    std::cout << "random obstacle polygons: " << obstacle_polygons.size() << '\n';
    std::cout << "boundary obstacle samples: " << result.obstacles.size() << '\n';
    std::cout << "iterations: " << result.iterations << '\n';

    visualizeCorridor(map, obstacle_polygons, result.seed, result.vertices,
                      epsilon0, epsilon_final, bbox_min, bbox_max, display_offset);

    return 0;
}
