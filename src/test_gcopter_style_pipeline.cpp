#include "frontend_tool/astar.hpp"
#include "perception_tool/firi_corridor_generator.hpp"
#include "perception_tool/grid_map.hpp"
#include "perception_tool/grid_map_env.hpp"
#include "backend_tool/traj_env.hpp"
#include "backend_tool/traj_opt.hpp"

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Polygon2D = std::vector<Eigen::Vector2d>;

struct PipelineConfig {
    Eigen::Vector2d start = Eigen::Vector2d(-8.7, 1.7);
    Eigen::Vector2d goal = Eigen::Vector2d(8.2, 0.0);

    double map_width = 20.0;
    double map_height = 12.0;
    double map_resolution = 0.08;

    double anchor_spacing = 1.6;
    double nominal_speed = 0.9;

    double astar_inflate_radius = 0.22;
    double astar_max_velocity = 2.0;
    double astar_max_acceleration = 1.0;
    double astar_time_resolution = 0.2;
    int astar_min_trajectory_num = 12;
    int astar_max_iterations = 6000;

    perception_tools::FootprintSpec footprint{0.7, 0.4, 0.0};
    perception_tools::BoundingBoxSpec bbox{2.4 *1.3, 1.4 *1.3, 1.5*1.3 };
    perception_tools::GeneratorOptions generator_options{12, 0.005, true, 5, 4.5};

    double init_velocity_scale = 0.3;

    double rho_v = 50000.0;
    double rho_a = 50000.0;
    double rho_sfc = 200000.0;
    double rho_collision = 100000.0;
    double rho_energy = 200.0;
    double rho_T = 20.0;
    double max_v = 2.0;
    double max_a = 2.0;
    int int_K = 24;

    int render_scale = 34;
};

Polygon2D makeRotatedRectangle(const Eigen::Vector2d& center,
                               double length,
                               double width,
                               double yaw)
{
    const double hl = length * 0.5;
    const double hw = width * 0.5;
    Eigen::Matrix2d rot;
    rot << std::cos(yaw), -std::sin(yaw),
           std::sin(yaw),  std::cos(yaw);

    return {
        center + rot * Eigen::Vector2d(-hl, -hw),
        center + rot * Eigen::Vector2d( hl, -hw),
        center + rot * Eigen::Vector2d( hl,  hw),
        center + rot * Eigen::Vector2d(-hl,  hw)
    };
}

std::vector<Polygon2D> buildGcopterLikeObstacles()
{
    return {
        makeRotatedRectangle(Eigen::Vector2d(-7.8,  3.7), 4.2, 1.2,  0.12),
        makeRotatedRectangle(Eigen::Vector2d(-4.7,  2.1), 3.7, 1.0,  1.28),
        makeRotatedRectangle(Eigen::Vector2d(-2.0,  4.6), 3.0, 1.1,  0.64),
        makeRotatedRectangle(Eigen::Vector2d( 3.7,  2.5), 7.0, 1.1, -0.10),
        makeRotatedRectangle(Eigen::Vector2d(-9.1, -3.8), 4.4, 1.0,  1.45),
        makeRotatedRectangle(Eigen::Vector2d(-3.9, -2.6), 6.5, 1.2,  0.05),
        makeRotatedRectangle(Eigen::Vector2d( 4.9, -1.6), 6.5, 1.2,  0.05),
        makeRotatedRectangle(Eigen::Vector2d( 0.8, -4.1), 3.6, 1.1,  1.22)
    };
}

grid_map::RowMatrixXi rasterizePolygons(const grid_map::GridMap& map,
                                        const std::vector<Polygon2D>& polygons)
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
        cv::fillConvexPoly(mask, pts, cv::Scalar(255), cv::LINE_AA);
    }

    grid_map::RowMatrixXi occupancy = grid_map::RowMatrixXi::Zero(voxel_num.x(), voxel_num.y());
    for (int x = 0; x < voxel_num.x(); ++x) {
        for (int y = 0; y < voxel_num.y(); ++y) {
            occupancy(x, y) = mask.at<unsigned char>(x, y) > 0 ? 1 : 0;
        }
    }
    return occupancy;
}

std::vector<int> sampleAnchorIndices(const std::vector<Eigen::Vector2d>& path,
                                     double spacing)
{
    std::vector<int> indices;
    if (path.empty()) {
        return indices;
    }

    indices.push_back(0);
    double accumulated = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        accumulated += (path[i] - path[i - 1]).norm();
        if (accumulated >= spacing) {
            indices.push_back(static_cast<int>(i));
            accumulated = 0.0;
        }
    }

    const int last_idx = static_cast<int>(path.size()) - 1;
    if (indices.back() != last_idx) {
        indices.push_back(last_idx);
    }
    return indices;
}

double estimateYawAtIndex(const std::vector<Eigen::Vector2d>& path, int idx)
{
    if (path.size() < 2) {
        return 0.0;
    }
    const int prev_idx = std::max(0, idx - 1);
    const int next_idx = std::min(static_cast<int>(path.size()) - 1, idx + 1);
    const Eigen::Vector2d dir = path[next_idx] - path[prev_idx];
    if (dir.norm() < 1.0e-8) {
        return 0.0;
    }
    return std::atan2(dir.y(), dir.x());
}

std::vector<Eigen::Vector2d> localPathWindow(const std::vector<Eigen::Vector2d>& path,
                                             int begin_idx,
                                             int end_idx)
{
    std::vector<Eigen::Vector2d> window;
    const int clamped_begin = std::max(0, begin_idx);
    const int clamped_end = std::min(static_cast<int>(path.size()), end_idx);
    window.reserve(std::max(0, clamped_end - clamped_begin));
    for (int i = clamped_begin; i < clamped_end; ++i) {
        window.push_back(path[i]);
    }
    return window;
}

TrajOpt::SFCPiece2D toSFCPiece(const perception_tools::CorridorResult& sfc)
{
    TrajOpt::SFCPiece2D piece;
    piece.reserve(sfc.planes.size());
    for (const auto& plane : sfc.planes) {
        piece.push_back({plane.normal, plane.offset});
    }
    return piece;
}

TrajOpt::SpatialConstraintMode parseMode(int argc, char** argv)
{
    std::string mode = "sfc";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        }
    }

    if (mode == "esdf") {
        return TrajOpt::SpatialConstraintMode::ESDF;
    }
    if (mode == "hybrid") {
        return TrajOpt::SpatialConstraintMode::ESDFAndSFC;
    }
    return TrajOpt::SpatialConstraintMode::SFC;
}

Eigen::VectorXd buildInitialTimes(const std::vector<Eigen::Vector2d>& anchors,
                                  double nominal_speed)
{
    const int piece_num = static_cast<int>(anchors.size()) - 1;
    Eigen::VectorXd times(piece_num);
    for (int i = 0; i < piece_num; ++i) {
        const double seg_len = (anchors[i + 1] - anchors[i]).norm();
        times(i) = std::max(seg_len / nominal_speed, 0.2);
    }
    return times;
}

void drawHatchedPolygon(cv::Mat& img,
                        const Polygon2D& polygon,
                        const std::function<cv::Point(const Eigen::Vector2d&)>& worldToPixel)
{
    std::vector<cv::Point> pts;
    pts.reserve(polygon.size());
    for (const auto& vertex : polygon) {
        pts.push_back(worldToPixel(vertex));
    }

    cv::Mat obstacle_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::fillConvexPoly(obstacle_mask, pts, cv::Scalar(255), cv::LINE_AA);
    cv::fillConvexPoly(img, pts, cv::Scalar(232, 232, 232), cv::LINE_AA);

    cv::Mat hatch_layer = cv::Mat::zeros(img.size(), img.type());
    for (int x = -img.rows; x < img.cols + img.rows; x += 8) {
        cv::line(hatch_layer, cv::Point(x, 0), cv::Point(x - img.rows, img.rows),
                 cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    }
    hatch_layer.copyTo(img, obstacle_mask);
    cv::polylines(img, pts, true, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
}

cv::Mat renderScene(const grid_map::GridMap& map,
                    const std::vector<Polygon2D>& obstacles,
                    const std::vector<Eigen::Vector2d>& astar_path,
                    const std::vector<Eigen::Vector2d>& optimized_path,
                    const std::vector<Eigen::Vector2d>& anchors,
                    const std::vector<perception_tools::CorridorResult>& sfcs,
                    const Eigen::Vector2d& start,
                    const Eigen::Vector2d& goal,
                    int scale)
{
    const int width_px = static_cast<int>(std::round(map.getMapSize().x() * scale));
    const int height_px = static_cast<int>(std::round(map.getMapSize().y() * scale));
    cv::Mat img(height_px, width_px, CV_8UC3, cv::Scalar(250, 250, 250));

    auto worldToPixel = [&](const Eigen::Vector2d& pt) {
        const double local_x = pt.x() - map.getOrigin().x();
        const double local_y = pt.y() - map.getOrigin().y();
        return cv::Point(static_cast<int>(std::round(local_x * scale)),
                         static_cast<int>(std::round(height_px - local_y * scale)));
    };

    for (int gx = 0; gx <= static_cast<int>(std::round(map.getMapSize().x())); ++gx) {
        const int px = gx * scale;
        cv::line(img, {px, 0}, {px, height_px}, cv::Scalar(228, 228, 228), 1);
    }
    for (int gy = 0; gy <= static_cast<int>(std::round(map.getMapSize().y())); ++gy) {
        const int py = gy * scale;
        cv::line(img, {0, py}, {width_px, py}, cv::Scalar(228, 228, 228), 1);
    }

    for (const auto& obstacle : obstacles) {
        drawHatchedPolygon(img, obstacle, worldToPixel);
    }

    for (const auto& sfc : sfcs) {
        if (sfc.vertices.size() < 3) {
            continue;
        }
        std::vector<cv::Point> pts;
        pts.reserve(sfc.vertices.size());
        for (const auto& v : sfc.vertices) {
            pts.push_back(worldToPixel(v));
        }
        cv::polylines(img, pts, true, cv::Scalar(25, 180, 90), 2, cv::LINE_AA);
    }

    for (size_t i = 1; i < astar_path.size(); ++i) {
        cv::line(img, worldToPixel(astar_path[i - 1]), worldToPixel(astar_path[i]),
                 cv::Scalar(90, 90, 255), 2, cv::LINE_AA);
    }

    for (size_t i = 1; i < optimized_path.size(); ++i) {
        cv::line(img, worldToPixel(optimized_path[i - 1]), worldToPixel(optimized_path[i]),
                 cv::Scalar(255, 60, 0), 3, cv::LINE_AA);
    }

    for (const auto& anchor : anchors) {
        cv::circle(img, worldToPixel(anchor), 4, cv::Scalar(0, 150, 255), -1, cv::LINE_AA);
    }

    cv::circle(img, worldToPixel(start), 6, cv::Scalar(40, 40, 235), -1, cv::LINE_AA);
    cv::circle(img, worldToPixel(goal), 6, cv::Scalar(40, 40, 235), -1, cv::LINE_AA);
    return img;
}

}  // namespace

int main(int argc, char** argv)
{
    const PipelineConfig cfg;
    const TrajOpt::SpatialConstraintMode mode = parseMode(argc, argv);
    const std::string mode_name =
        mode == TrajOpt::SpatialConstraintMode::ESDF ? "esdf" :
        mode == TrajOpt::SpatialConstraintMode::ESDFAndSFC ? "hybrid" : "sfc";

    // 1. Build the synthetic GCOPTER-like scene on a 2D occupancy grid.
    grid_map::GridMap map;
    map.init(cfg.map_width, cfg.map_height, cfg.map_resolution);

    const auto obstacles = buildGcopterLikeObstacles();
    map.setMap(rasterizePolygons(map, obstacles));

    const Eigen::Vector2d& start = cfg.start;
    const Eigen::Vector2d& goal = cfg.goal;

    // 2. Use A* as the front-end to produce a collision-free seed path.
    path_planning::AStar astar(map, cfg.astar_inflate_radius);
    astar.setMaxVelocity(cfg.astar_max_velocity);
    astar.setMaxAcceleration(cfg.astar_max_acceleration);
    astar.setTimeResolution(cfg.astar_time_resolution);
    astar.setMinTrajectoryNumber(cfg.astar_min_trajectory_num);

    const auto astar_traj = astar.planWithPostProcessing(start, goal, cfg.astar_max_iterations);
    if (astar_traj.optimized_path.empty()) {
        std::cerr << "A* failed to find a seed path in the GCOPTER-like scene.\n";
        return -1;
    }

    const auto anchor_indices = sampleAnchorIndices(astar_traj.optimized_path, cfg.anchor_spacing);
    if (anchor_indices.size() < 4) {
        std::cerr << "Anchor sampling produced too few intervals.\n";
        return -1;
    }

    std::vector<Eigen::Vector2d> anchors;
    anchors.reserve(anchor_indices.size());
    for (int idx : anchor_indices) {
        anchors.push_back(astar_traj.optimized_path[idx]);
    }

    // 3. Build one SFC polytope for each path interval using the local heading and path window.
    perception_tools::CorridorGenerator generator;
    std::vector<perception_tools::CorridorResult> sfcs;
    sfcs.reserve(anchor_indices.size() - 1);
    for (size_t i = 0; i + 1 < anchor_indices.size(); ++i) {
        const int center_idx = (anchor_indices[i] + anchor_indices[i + 1]) / 2;
        const double yaw = estimateYawAtIndex(astar_traj.optimized_path, center_idx);
        const auto local_path = localPathWindow(astar_traj.optimized_path,
                                                anchor_indices[i],
                                                std::min(anchor_indices[i + 1] + 6,
                                                         static_cast<int>(astar_traj.optimized_path.size())));
        auto sfc = generator.generate(map,
                                      astar_traj.optimized_path[center_idx],
                                      yaw,
                                      cfg.footprint,
                                      cfg.bbox,
                                      cfg.generator_options,
                                      local_path);
        if (sfc.vertices.size() < 3) {
            std::cerr << "FIRI failed on interval " << i << ".\n";
            return -1;
        }
        sfcs.push_back(std::move(sfc));
    }

    std::vector<TrajOpt::SFCPiece2D> sfc_pieces;
    sfc_pieces.reserve(sfcs.size());
    for (const auto& sfc : sfcs) {
        sfc_pieces.push_back(toSFCPiece(sfc));
    }

    // 4. Select the backend environment according to the requested constraint mode.
    auto map_ptr = std::make_shared<grid_map::GridMap>(map);
    auto distance_env = std::make_shared<TrajOpt::GridMapEnv>(map_ptr);
    auto sfc_env = std::make_shared<TrajOpt::SafeSFCEnv>(sfc_pieces);
    std::shared_ptr<TrajOpt::TrajectoryEnv> env;
    if (mode == TrajOpt::SpatialConstraintMode::ESDF) {
        env = distance_env;
    } else if (mode == TrajOpt::SpatialConstraintMode::ESDFAndSFC) {
        env = std::make_shared<TrajOpt::CompositeEnv>(distance_env, sfc_env);
    } else {
        env = sfc_env;
    }

    Eigen::Matrix2d init_state;
    Eigen::Matrix2d end_state;
    init_state.col(0) = start;
    end_state.col(0) = goal;
    init_state.col(1) = (astar_traj.optimized_path[1] - astar_traj.optimized_path[0]).normalized() * cfg.init_velocity_scale;
    end_state.col(1) =
        (astar_traj.optimized_path.back() - astar_traj.optimized_path[astar_traj.optimized_path.size() - 2]).normalized() * cfg.init_velocity_scale;

    const int piece_num = static_cast<int>(sfc_pieces.size());
    Eigen::MatrixXd inner_pts(2, piece_num - 1);
    for (int i = 0; i < piece_num - 1; ++i) {
        inner_pts.col(i) = anchors[i + 1];
    }

    const Eigen::VectorXd init_T = buildInitialTimes(anchors, cfg.nominal_speed);

    // 5. Configure and run the trajectory optimizer.
    TrajOpt::TrajectoryParams params;
    params.rho_v = cfg.rho_v;
    params.rho_a = cfg.rho_a;
    params.rho_sfc = cfg.rho_sfc;
    params.rho_collision = 0.0;
    params.rho_energy = cfg.rho_energy;
    params.rho_T = cfg.rho_T;
    params.max_v = cfg.max_v;
    params.max_a = cfg.max_a;
    params.int_K = cfg.int_K;
    params.constraint_mode = mode;
    params.use_sfc_parameterization = (mode != TrajOpt::SpatialConstraintMode::ESDF);
    params.use_corridor_parameterization = false;
    if (mode != TrajOpt::SpatialConstraintMode::SFC) {
        params.rho_collision = cfg.rho_collision;
    }

    TrajOpt::TrajectoryOptimizer optimizer(params);
    optimizer.setEnvironment(env);
    if (!optimizer.optimize(init_state, inner_pts, end_state, init_T)) {
        std::cerr << "Trajectory optimization failed, result = "
                  << optimizer.getLastOptResult() << '\n';
        return -1;
    }

    // 6. Render the seed path, generated SFCs and optimized trajectory for inspection.
    const auto optimized_path = optimizer.sampleTrajectory(0.1);
    const cv::Mat viz = renderScene(map, obstacles, astar_traj.optimized_path,
                                    optimized_path, anchors, sfcs, start, goal, cfg.render_scale);
    const std::string output_path = "gcopter_style_pipeline_" + mode_name + ".png";
    cv::imwrite(output_path, viz);

    const auto metrics = optimizer.evaluateTrajectory();
    std::cout << "Saved visualization to " << output_path << '\n';
    std::cout << "Mode: " << mode_name << '\n';
    std::cout << "A* optimized path points: " << astar_traj.optimized_path.size() << '\n';
    std::cout << "Anchor count: " << anchors.size() << '\n';
    std::cout << "SFC count: " << sfcs.size() << '\n';
    std::cout << "Using sfc parameterization: "
              << (optimizer.isUsingSFCParameterization() ? "yes" : "no") << '\n';
    std::cout << "Optimized max velocity: " << metrics.max_velocity << '\n';
    std::cout << "Optimized max acceleration: " << metrics.max_acceleration << '\n';

    const char* display_env = std::getenv("DISPLAY");
    if (display_env != nullptr && display_env[0] != '\0') {
        try {
            cv::imshow("GCOPTER-style pipeline", viz);
            cv::waitKey(0);
        } catch (const cv::Exception& e) {
            std::cout << "OpenCV interactive window is unavailable: " << e.what() << '\n';
        }
    }

    return 0;
}
