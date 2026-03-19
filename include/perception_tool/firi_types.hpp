#pragma once

#include <Eigen/Eigen>

#include <vector>

namespace perception_tools {

struct HalfPlane2D {
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double offset = 0.0;
};

struct FootprintSpec {
    double length = 0.0;
    double width = 0.0;
    double offset_x = 0.0;
};

struct BoundingBoxSpec {
    double ahead = 0.0;
    double behind = 0.0;
    double side = 0.0;
};

struct GeneratorOptions {
    int max_iter = 10;
    double convergence_rho = 0.02;
    bool use_path_seed = true;
    int path_seed_count = 4;
    double path_lookahead = 4.0;
};

struct CorridorResult {
    std::vector<HalfPlane2D> planes;
    std::vector<Eigen::Vector2d> vertices;
    std::vector<Eigen::Vector2d> seed;
    std::vector<Eigen::Vector2d> obstacles;
    int iterations = 0;
    double solve_time_ms = 0.0;
};

}  // namespace perception_tools
