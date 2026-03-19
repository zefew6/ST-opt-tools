#pragma once

#include "perception_tool/grid_map.hpp"
#include "perception_tool/firi_solver.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <vector>

namespace perception_tools {

class CorridorGenerator {
public:
    CorridorResult generate(const grid_map::GridMap& map,
                            const Eigen::Vector2d& robot_pos,
                            double robot_yaw,
                            const FootprintSpec& footprint,
                            const BoundingBoxSpec& bbox,
                            const GeneratorOptions& options = {},
                            const std::vector<Eigen::Vector2d>& path = {}) const
    {
        CorridorResult result;
        result.obstacles = extractBoundaryObstacles(map);
        result.seed = buildFootprintSeed(robot_pos, robot_yaw, footprint);

        if (options.use_path_seed && !path.empty()) {
            auto path_seed = samplePathSeed(path, robot_pos, footprint,
                                            options.path_seed_count,
                                            options.path_lookahead);
            result.seed.insert(result.seed.end(), path_seed.begin(), path_seed.end());
        }

        const auto bbox_planes = buildHeadingAlignedBbox(robot_pos, robot_yaw, bbox);
        auto solve_result = solver_.compute(result.obstacles, result.seed,
                                            bbox_planes, options.max_iter,
                                            options.convergence_rho);

        result.planes = std::move(solve_result.planes);
        result.iterations = solve_result.iterations;
        result.solve_time_ms = solve_result.solve_time_ms;
        result.vertices = computePolytopeVertices(result.planes);
        return result;
    }

    std::vector<Eigen::Vector2d> extractBoundaryObstacles(
        const grid_map::GridMap& map) const
    {
        std::vector<Eigen::Vector2d> obstacles;
        const Eigen::Vector2i voxel_num = map.getVoxelNum();
        const double resolution = map.getResolution();
        const Eigen::Vector2d origin = map.getOrigin();

        const int di[] = {1, -1, 0, 0};
        const int dj[] = {0, 0, 1, -1};
        const double face_dx[] = {0.5, -0.5, 0.0, 0.0};
        const double face_dy[] = {0.0, 0.0, 0.5, -0.5};

        obstacles.reserve(voxel_num.x() * voxel_num.y() / 8);

        for (int i = 0; i < voxel_num.x(); ++i) {
            for (int j = 0; j < voxel_num.y(); ++j) {
                if (!map.isOccupied(Eigen::Vector2i(i, j))) {
                    continue;
                }

                for (int n = 0; n < 4; ++n) {
                    const int ni = i + di[n];
                    const int nj = j + dj[n];

                    bool neighbor_is_free = false;
                    if (ni < 0 || ni >= voxel_num.x() || nj < 0 || nj >= voxel_num.y()) {
                        neighbor_is_free = true;
                    } else {
                        neighbor_is_free = !map.isOccupied(Eigen::Vector2i(ni, nj));
                    }

                    if (!neighbor_is_free) {
                        continue;
                    }

                    const double x = origin.x() + (i + 0.5 + face_dx[n]) * resolution;
                    const double y = origin.y() + (j + 0.5 + face_dy[n]) * resolution;
                    obstacles.emplace_back(x, y);
                }
            }
        }

        return obstacles;
    }

    std::vector<Eigen::Vector2d> buildFootprintSeed(
        const Eigen::Vector2d& robot_pos,
        double robot_yaw,
        const FootprintSpec& footprint) const
    {
        const double hl = footprint.length / 2.0;
        const double hw = footprint.width / 2.0;
        const Eigen::Matrix2d R = rotation(robot_yaw);
        const Eigen::Vector2d center = robot_pos + R.col(0) * footprint.offset_x;

        return {
            center + R * Eigen::Vector2d(hl, hw),
            center + R * Eigen::Vector2d(hl, -hw),
            center + R * Eigen::Vector2d(-hl, -hw),
            center + R * Eigen::Vector2d(-hl, hw)
        };
    }

    std::vector<Eigen::Vector2d> samplePathSeed(
        const std::vector<Eigen::Vector2d>& path,
        const Eigen::Vector2d& robot_pos,
        const FootprintSpec& footprint,
        int path_seed_count,
        double path_lookahead) const
    {
        std::vector<Eigen::Vector2d> pts;
        if (path.empty() || path_seed_count <= 0) {
            return pts;
        }

        double min_dist_sq = 1e18;
        int closest_idx = 0;
        for (int i = 0; i < static_cast<int>(path.size()); ++i) {
            const double dist_sq = (path[i] - robot_pos).squaredNorm();
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                closest_idx = i;
            }
        }

        if (std::sqrt(min_dist_sq) > path_lookahead) {
            return pts;
        }

        const double front_edge_dist = footprint.offset_x + footprint.length / 2.0;
        const double skip_dist_sq = front_edge_dist * front_edge_dist;

        for (int i = closest_idx + 1;
             i < static_cast<int>(path.size()) &&
             static_cast<int>(pts.size()) < path_seed_count;
             ++i) {
            if ((path[i] - robot_pos).squaredNorm() < skip_dist_sq) {
                continue;
            }
            pts.push_back(path[i]);
        }

        return pts;
    }

    std::vector<HalfPlane2D> buildHeadingAlignedBbox(
        const Eigen::Vector2d& robot_pos,
        double robot_yaw,
        const BoundingBoxSpec& bbox) const
    {
        const Eigen::Matrix2d R = rotation(robot_yaw);
        const Eigen::Vector2d fwd = R.col(0);
        const Eigen::Vector2d lft = R.col(1);

        return {
            {fwd, fwd.dot(robot_pos) + bbox.ahead},
            {-fwd, -fwd.dot(robot_pos) + bbox.behind},
            {lft, lft.dot(robot_pos) + bbox.side},
            {-lft, -lft.dot(robot_pos) + bbox.side}
        };
    }

    std::vector<Eigen::Vector2d> computePolytopeVertices(
        const std::vector<HalfPlane2D>& planes) const
    {
        std::vector<Eigen::Vector2d> vertices;
        const int n = static_cast<int>(planes.size());

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                Eigen::Matrix2d M;
                M.row(0) = planes[i].normal.transpose();
                M.row(1) = planes[j].normal.transpose();

                if (std::abs(M.determinant()) < 1e-10) {
                    continue;
                }

                const Eigen::Vector2d rhs(planes[i].offset, planes[j].offset);
                const Eigen::Vector2d vertex = M.inverse() * rhs;

                bool inside = true;
                for (int k = 0; k < n; ++k) {
                    if (planes[k].normal.dot(vertex) > planes[k].offset + 1e-6) {
                        inside = false;
                        break;
                    }
                }

                if (inside && !hasDuplicate(vertices, vertex)) {
                    vertices.push_back(vertex);
                }
            }
        }

        if (vertices.size() < 3) {
            return vertices;
        }

        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const auto& v : vertices) {
            centroid += v;
        }
        centroid /= static_cast<double>(vertices.size());

        std::sort(vertices.begin(), vertices.end(),
                  [&centroid](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                      return std::atan2(a.y() - centroid.y(), a.x() - centroid.x()) <
                             std::atan2(b.y() - centroid.y(), b.x() - centroid.x());
                  });
        return vertices;
    }

private:
    static Eigen::Matrix2d rotation(double yaw)
    {
        const double c = std::cos(yaw);
        const double s = std::sin(yaw);
        Eigen::Matrix2d R;
        R << c, -s,
             s,  c;
        return R;
    }

    static bool hasDuplicate(const std::vector<Eigen::Vector2d>& vertices,
                             const Eigen::Vector2d& candidate,
                             double eps = 1e-8)
    {
        for (const auto& existing : vertices) {
            if ((existing - candidate).norm() <= eps) {
                return true;
            }
        }
        return false;
    }

    mutable FIRISolver solver_;
};

}  // namespace perception_tools
