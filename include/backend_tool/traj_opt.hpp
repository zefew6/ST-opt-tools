/*
    MIT License

    Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)
    Copyright (c) 2025 Deping Zhang (beiyuena@foxmail.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef TRAJ_OPT_HPP
#define TRAJ_OPT_HPP

#include "SplineTrajectory.hpp"
#include "traj_env.hpp"
#include <lbfgs.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

namespace TrajOpt {

enum class SpatialConstraintMode {
    ESDF = 0,
    SFC = 1,
    ESDFAndSFC = 2,
    Corridor = SFC,
    ESDFAndCorridor = ESDFAndSFC
};

struct TrajectoryParams {
    double total_len = 10;
    double total_time = 10;
    double piece_len = 4;
    double rho_v = 10000;           // Velocity penalty weight
    double rho_a = 10000;           // Acceleration penalty weight
    double rho_collision = 100000;  // Collision penalty weight
    double rho_corridor = 100000;   // Backward-compatible alias of rho_sfc
    double rho_sfc = 100000;        // Safe flight corridor penalty weight
    double rho_T = 100;             // Time penalty weight
    double rho_energy = 100;        // Energy (smoothness) penalty weight
    double max_v = 1.0;             // Maximum velocity
    double max_a = 1.0;             // Maximum acceleration
    double safe_threshold = 0.5;    // Safety distance threshold
    double sfc_smooth_factor = 1.0e-2; // GCOPTER-style smoothing factor for corridor penalties
    
    int int_K = 32;                 // Integration sample points
    int mem_size = 256;             // L-BFGS memory size
    int past = 3;                   // L-BFGS parameter
    double g_epsilon = 1e-6;        // Gradient convergence threshold
    double min_step = 1e-32;        // Minimum step size
    double delta = 1e-5;            // Function change convergence threshold
    int max_iter = 10000;           // Maximum iterations
    bool use_corridor_parameterization = false; // Backward-compatible alias
    bool use_sfc_parameterization = false;
    SpatialConstraintMode constraint_mode = SpatialConstraintMode::ESDF;
};

class TrajectoryOptimizer {
public:
    using Vector2d = Eigen::Vector2d;
    using Vector3d = Eigen::Vector3d;
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using CorridorPolytope2D = Eigen::Matrix<double, 2, Eigen::Dynamic>;
    using QuinticSpline2D = SplineTrajectory::QuinticSpline2D;
    using PPoly2D = QuinticSpline2D::TrajectoryType;

    explicit TrajectoryOptimizer(const TrajectoryParams& params = TrajectoryParams())
        : params_(params), in_opt_(false) {}

    TrajectoryOptimizer(const std::vector<Eigen::Vector2d>& guide_path,
                        std::shared_ptr<TrajectoryEnv> env,
                        const TrajectoryParams& params = TrajectoryParams())
        : env_(std::move(env)), guide_path_(guide_path), params_(params), in_opt_(false) {
        preprocessGuidePath();
    }

    void setGuidePath(const std::vector<Eigen::Vector2d>& guide_path) {
        guide_path_ = guide_path;
        current_segment_ = 0;
        preprocessGuidePath();
    }

    void setEnvironment(std::shared_ptr<TrajectoryEnv> env) {
        env_ = std::move(env);
    }

    bool plan() {
        if (guide_path_.size() < 2) return false;
        current_segment_ = 0;
        
        // Prepare initial conditions
        Eigen::Matrix2d init_cond, end_cond;
        init_cond.col(0) = guide_path_.front();
        end_cond.col(0) = guide_path_.back();
        init_cond.col(1) = (guide_path_[1] - guide_path_[0]).normalized() * 0.1;
        end_cond.col(1) = (guide_path_.back() - guide_path_[guide_path_.size()-2]).normalized() * 0.1;

        double total_len = params_.total_len;
        int piece_num = (int)(total_len / params_.piece_len);
        if (piece_num < 2) piece_num = 2;
        params_.piece_len = total_len / piece_num;
        
        // Sample intermediate control points from A* path
        Eigen::MatrixXd inner_pos(2, piece_num-1);
        std::vector<Eigen::Vector2d> inner_pos_node;
        
        double step_len = total_len / piece_num;
        double accumulated_len = 0.0;
        
        for (int i = 1; i < piece_num; ++i) {
            double target_len = i * step_len;
            while (accumulated_len < target_len && current_segment_ < guide_path_.size() - 1) {
                double seg_len = (guide_path_[current_segment_ + 1] - guide_path_[current_segment_]).norm();
                if (accumulated_len + seg_len >= target_len) {
                    double ratio = seg_len > 1.0e-8 ? (target_len - accumulated_len) / seg_len : 0.0;
                    Eigen::Vector2d point = guide_path_[current_segment_] + 
                                          ratio * (guide_path_[current_segment_ + 1] - guide_path_[current_segment_]);
                    inner_pos_node.push_back(point);
                    break;
                }
                accumulated_len += seg_len;
                current_segment_++;
            }
        }
        
        inner_pos.resize(2, inner_pos_node.size());
        for (size_t i = 0; i < inner_pos_node.size(); i++) {
            inner_pos.col(i) = inner_pos_node[i];
        }

        return optimize(init_cond, inner_pos, end_cond, params_.total_time);
    }

    bool optimize(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                  const MatrixXd& endPos, double totalTime) {
        guide_path_.clear();
        current_segment_ = 0;
        return optimizeSE2Traj(initPos, innerPtsPos, endPos, totalTime) >= 0;
    }

    bool optimize(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                  const MatrixXd& endPos, const VectorXd& initTpos) {
        guide_path_.clear();
        current_segment_ = 0;
        return optimizeSE2Traj(initPos, innerPtsPos, endPos, initTpos) >= 0;
    }

    PPoly2D getOptimizedTrajectory() const { return trajectory_; }
    int getLastOptResult() const { return last_opt_result_; }
    double getLastOptCost() const { return last_opt_cost_; }
    bool isUsingCorridorParameterization() const { return use_sfc_parameterization_; }
    bool isUsingSFCParameterization() const { return use_sfc_parameterization_; }

    static bool usesESDFConstraints(SpatialConstraintMode mode) {
        return mode == SpatialConstraintMode::ESDF ||
               mode == SpatialConstraintMode::ESDFAndCorridor;
    }

    static bool usesSFCConstraints(SpatialConstraintMode mode) {
        return mode == SpatialConstraintMode::SFC ||
               mode == SpatialConstraintMode::ESDFAndSFC ||
               mode == SpatialConstraintMode::Corridor ||
               mode == SpatialConstraintMode::ESDFAndCorridor;
    }

    static bool usesCorridorConstraints(SpatialConstraintMode mode) {
        return usesSFCConstraints(mode);
    }

    struct TrajectoryMetrics {
        double max_velocity;
        double max_acceleration;
        double min_clearance;
        double total_time;
        double trajectory_energy;
        double path_deviation;
    };
    
    TrajectoryMetrics evaluateTrajectory() const {
        TrajectoryMetrics metrics;
        metrics.max_velocity = 0.0;
        metrics.max_acceleration = 0.0;
        metrics.min_clearance = std::numeric_limits<double>::max();
        metrics.total_time = trajectory_.getDuration();
        metrics.trajectory_energy = 0.0;
        metrics.path_deviation = 0.0;

        if (!trajectory_.isInitialized()) return metrics;

        const double dt = 0.01;
        int sample_count = 0;
        for (double t = 0.0; t < metrics.total_time; t += dt) {
            Vector2d pos = trajectory_.evaluate(t, 0);
            Vector2d vel = trajectory_.evaluate(t, 1);
            Vector2d acc = trajectory_.evaluate(t, 2);
            
            metrics.trajectory_energy += acc.squaredNorm() * dt;
            metrics.max_velocity = std::max(metrics.max_velocity, vel.norm());
            metrics.max_acceleration = std::max(metrics.max_acceleration, acc.norm());
            if (env_) {
                double clearance;
                Vector2d grad_sdf;
                if (env_->getDistanceAndGradient(pos, clearance, grad_sdf)) {
                    metrics.min_clearance = std::min(metrics.min_clearance, clearance);
                }
            }
            
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& guide_pt : guide_path_) {
                min_dist = std::min(min_dist, (pos - guide_pt).norm());
            }
            if (!guide_path_.empty()) {
                metrics.path_deviation += min_dist;
            }
            sample_count++;
        }
        
        if (sample_count > 0) {
            metrics.path_deviation /= sample_count;
        }
        if (metrics.min_clearance == std::numeric_limits<double>::max()) {
            metrics.min_clearance = -1.0;
        }
        
        return metrics;
    }

    std::vector<Eigen::Vector2d> sampleTrajectory(double dt = 0.1) const {
        std::vector<Eigen::Vector2d> path;
        if (!trajectory_.isInitialized()) return path;
        
        const double total_time = trajectory_.getDuration();
        for (double t = 0.0; t <= total_time; t += dt) {
            path.push_back(trajectory_.evaluate(t, 0));
        }
        
        if (!path.empty() && path.back() != trajectory_.evaluate(total_time, 0)) {
            path.push_back(trajectory_.evaluate(total_time, 0));
        }
        
        return path;
    }

private:
    void preprocessGuidePath() {
        // Prepare guide path for fast nearest neighbor queries
    }

    static bool isApproxDuplicate(const Vector2d& a, const Vector2d& b, double eps = 1.0e-8) {
        return (a - b).norm() <= eps;
    }

    static std::vector<Vector2d> halfspacesToVertices(const CorridorPiece2D& piece) {
        std::vector<Vector2d> vertices;
        const int n = static_cast<int>(piece.size());
        if (n < 3) {
            return vertices;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                Eigen::Matrix2d M;
                M.row(0) = piece[i].normal.transpose();
                M.row(1) = piece[j].normal.transpose();
                if (std::abs(M.determinant()) < 1.0e-10) {
                    continue;
                }

                const Vector2d rhs(piece[i].offset, piece[j].offset);
                const Vector2d vertex = M.inverse() * rhs;

                bool inside = true;
                for (const auto& half_space : piece) {
                    if (half_space.normal.dot(vertex) > half_space.offset + 1.0e-7) {
                        inside = false;
                        break;
                    }
                }
                if (!inside) {
                    continue;
                }

                bool duplicated = false;
                for (const auto& existing : vertices) {
                    if (isApproxDuplicate(existing, vertex)) {
                        duplicated = true;
                        break;
                    }
                }
                if (!duplicated) {
                    vertices.push_back(vertex);
                }
            }
        }

        if (vertices.size() < 3) {
            return {};
        }

        Vector2d centroid = Vector2d::Zero();
        for (const auto& v : vertices) {
            centroid += v;
        }
        centroid /= static_cast<double>(vertices.size());

        std::sort(vertices.begin(), vertices.end(),
                  [&centroid](const Vector2d& a, const Vector2d& b) {
                      return std::atan2(a.y() - centroid.y(), a.x() - centroid.x()) <
                             std::atan2(b.y() - centroid.y(), b.x() - centroid.x());
                  });
        return vertices;
    }

    static CorridorPiece2D intersectPieces(const CorridorPiece2D& lhs,
                                           const CorridorPiece2D& rhs) {
        CorridorPiece2D merged = lhs;
        merged.insert(merged.end(), rhs.begin(), rhs.end());
        return merged;
    }

    static CorridorPiece2D normalizePiece(const CorridorPiece2D& piece) {
        CorridorPiece2D normalized = piece;
        for (auto& half_space : normalized) {
            const double norm = half_space.normal.norm();
            if (norm > 1.0e-10) {
                half_space.normal /= norm;
                half_space.offset /= norm;
            }
        }
        return normalized;
    }

    static CorridorPolytope2D buildVPolytope(const std::vector<Vector2d>& vertices) {
        CorridorPolytope2D poly(2, static_cast<int>(vertices.size()));
        poly.col(0) = vertices.front();
        for (int i = 1; i < static_cast<int>(vertices.size()); ++i) {
            poly.col(i) = vertices[i] - vertices.front();
        }
        return poly;
    }

    static bool isInsidePiece(const CorridorPiece2D& piece,
                              const Vector2d& point,
                              double tol = 1.0e-7) {
        for (const auto& half_space : piece) {
            if (half_space.normal.dot(point) > half_space.offset + tol) {
                return false;
            }
        }
        return true;
    }

    static Vector2d polytopeCentroid(const std::vector<Vector2d>& vertices) {
        Vector2d centroid = Vector2d::Zero();
        if (vertices.empty()) {
            return centroid;
        }
        for (const auto& v : vertices) {
            centroid += v;
        }
        return centroid / static_cast<double>(vertices.size());
    }

    static bool buildShortestPathThroughPolytopes(
        const Vector2d& start,
        const Vector2d& goal,
        const std::vector<CorridorPiece2D>& sfc,
        const std::vector<std::vector<Vector2d>>& overlap_vertices,
        MatrixXd& path_points) {
        const int overlap_num = static_cast<int>(overlap_vertices.size());
        path_points.resize(2, overlap_num + 2);
        path_points.col(0) = start;
        path_points.col(overlap_num + 1) = goal;
        if (overlap_num == 0) {
            path_points.col(0) = start;
            path_points.col(1) = goal;
            return true;
        }

        std::vector<std::vector<Vector2d>> layers(overlap_num + 2);
        layers.front().push_back(start);
        for (int i = 0; i < overlap_num; ++i) {
            layers[i + 1] = overlap_vertices[i];
            if (layers[i + 1].empty()) {
                return false;
            }
        }
        layers.back().push_back(goal);

        const double inf = std::numeric_limits<double>::infinity();
        std::vector<VectorXd> dist(layers.size());
        std::vector<std::vector<int>> prev(layers.size());
        dist[0] = VectorXd::Zero(1);
        prev[0] = std::vector<int>(1, -1);

        for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
            const auto& cur_layer = layers[layer_idx];
            const auto& prev_layer = layers[layer_idx - 1];
            dist[layer_idx] = VectorXd::Constant(static_cast<int>(cur_layer.size()), inf);
            prev[layer_idx] = std::vector<int>(cur_layer.size(), -1);

            int piece_idx = static_cast<int>(layer_idx) - 1;
            piece_idx = std::max(0, std::min(piece_idx, static_cast<int>(sfc.size()) - 1));
            const CorridorPiece2D& piece = sfc[piece_idx];

            for (int j = 0; j < static_cast<int>(cur_layer.size()); ++j) {
                const Vector2d& cur_pt = cur_layer[j];
                for (int k = 0; k < static_cast<int>(prev_layer.size()); ++k) {
                    const Vector2d& prev_pt = prev_layer[k];
                    if (!isInsidePiece(piece, prev_pt) || !isInsidePiece(piece, cur_pt)) {
                        continue;
                    }
                    const double candidate = dist[layer_idx - 1](k) + (cur_pt - prev_pt).norm();
                    if (candidate < dist[layer_idx](j)) {
                        dist[layer_idx](j) = candidate;
                        prev[layer_idx][j] = k;
                    }
                }
            }
        }

        if (!std::isfinite(dist.back()(0))) {
            for (int i = 0; i < overlap_num; ++i) {
                path_points.col(i + 1) = polytopeCentroid(overlap_vertices[i]);
            }
            return true;
        }

        std::vector<int> indices(layers.size(), -1);
        indices.back() = 0;
        for (int layer_idx = static_cast<int>(layers.size()) - 1; layer_idx > 0; --layer_idx) {
            const int prev_idx = prev[layer_idx][indices[layer_idx]];
            if (prev_idx < 0) {
                return false;
            }
            indices[layer_idx - 1] = prev_idx;
        }

        for (int i = 0; i < overlap_num; ++i) {
            path_points.col(i + 1) = layers[i + 1][indices[i + 1]];
        }
        return true;
    }

    static void allocatePiecesByPathLength(const MatrixXd& path,
                                           int total_piece_num,
                                           Eigen::VectorXi& piece_counts) {
        const int interval_num = static_cast<int>(path.cols()) - 1;
        piece_counts = Eigen::VectorXi::Ones(interval_num);
        if (interval_num <= 0) {
            return;
        }

        int remaining = total_piece_num - interval_num;
        if (remaining <= 0) {
            return;
        }

        Eigen::VectorXd seg_lengths(interval_num);
        double total_length = 0.0;
        for (int i = 0; i < interval_num; ++i) {
            seg_lengths(i) = (path.col(i + 1) - path.col(i)).norm();
            total_length += seg_lengths(i);
        }

        if (total_length < 1.0e-10) {
            for (int i = 0; i < remaining; ++i) {
                piece_counts(i % interval_num) += 1;
            }
            return;
        }

        Eigen::VectorXd desired_extra = seg_lengths / total_length * remaining;
        for (int i = 0; i < interval_num; ++i) {
            const int extra = static_cast<int>(std::floor(desired_extra(i)));
            piece_counts(i) += extra;
            remaining -= extra;
            desired_extra(i) -= extra;
        }

        while (remaining > 0) {
            Eigen::Index idx = 0;
            desired_extra.maxCoeff(&idx);
            piece_counts(static_cast<int>(idx)) += 1;
            desired_extra(idx) = 0.0;
            remaining -= 1;
        }
    }

    static void allocatePiecesByLength(const MatrixXd& path,
                                       double piece_len,
                                       Eigen::VectorXi& piece_counts) {
        const int interval_num = static_cast<int>(path.cols()) - 1;
        piece_counts.resize(std::max(0, interval_num));
        if (interval_num <= 0) {
            return;
        }

        const double target_piece_len = std::max(piece_len, 1.0e-3);
        for (int i = 0; i < interval_num; ++i) {
            const double seg_len = (path.col(i + 1) - path.col(i)).norm();
            piece_counts(i) = std::max(1, static_cast<int>(seg_len / target_piece_len) + 1);
        }
    }

    static void buildTimeAllocationFromPath(const MatrixXd& path,
                                            const Eigen::VectorXi& piece_counts,
                                            double speed,
                                            VectorXd& time_alloc) {
        const int interval_num = piece_counts.size();
        const int total_piece_num = piece_counts.sum();
        time_alloc.resize(std::max(0, total_piece_num));
        if (interval_num <= 0 || total_piece_num <= 0) {
            return;
        }

        const double alloc_speed = std::max(speed, 1.0e-3);
        for (int i = 0, offset = 0; i < interval_num; ++i) {
            const int pieces = piece_counts(i);
            const double dt = (path.col(i + 1) - path.col(i)).norm() /
                              (alloc_speed * static_cast<double>(pieces));
            time_alloc.segment(offset, pieces).setConstant(std::max(dt, 1.0e-3));
            offset += pieces;
        }
    }

    static void buildInitialFromPath(const MatrixXd& path,
                                     const Eigen::VectorXi& piece_counts,
                                     MatrixXd& inner_points) {
        const int interval_num = piece_counts.size();
        const int total_piece_num = piece_counts.sum();
        inner_points.resize(2, std::max(0, total_piece_num - 1));
        if (interval_num <= 0 || total_piece_num <= 1) {
            return;
        }

        int point_idx = 0;
        for (int i = 0; i < interval_num; ++i) {
            const int pieces = piece_counts(i);
            const Vector2d a = path.col(i);
            const Vector2d b = path.col(i + 1);
            const Vector2d step = (b - a) / static_cast<double>(pieces);
            for (int j = 0; j < pieces; ++j) {
                if (i > 0 || j > 0) {
                    inner_points.col(point_idx++) = a + step * static_cast<double>(j);
                }
            }
        }
    }

    bool processSFC(const std::vector<CorridorPiece2D>& sfc) {
        sfc_v_polytopes_.clear();
        sfc_v_polytopes_.reserve(std::max(0, 2 * static_cast<int>(sfc.size()) - 1));
        if (sfc.empty()) {
            return false;
        }

        for (int i = 0; i + 1 < static_cast<int>(sfc.size()); ++i) {
            std::vector<Vector2d> vertices = halfspacesToVertices(sfc[i]);
            if (vertices.size() < 3) {
                return false;
            }
            sfc_v_polytopes_.push_back(buildVPolytope(vertices));

            std::vector<Vector2d> overlap_vertices = halfspacesToVertices(intersectPieces(sfc[i], sfc[i + 1]));
            if (overlap_vertices.size() < 3) {
                return false;
            }
            sfc_v_polytopes_.push_back(buildVPolytope(overlap_vertices));
        }

        std::vector<Vector2d> last_vertices = halfspacesToVertices(sfc.back());
        if (last_vertices.size() < 3) {
            return false;
        }
        sfc_v_polytopes_.push_back(buildVPolytope(last_vertices));
        return true;
    }

    static const std::vector<SFCPiece2D>* extractSFC(const std::shared_ptr<TrajectoryEnv>& env) {
        if (!env) {
            return nullptr;
        }
        return env->getSFCPtr();
    }

    static void forwardP(const VectorXd& xi,
                         const Eigen::VectorXi& poly_idx,
                         const std::vector<CorridorPolytope2D>& polytopes,
                         MatrixXd& P) {
        const int point_num = poly_idx.size();
        P.resize(2, point_num);

        VectorXd q;
        for (int i = 0, offset = 0; i < point_num; ++i) {
            const int poly_id = poly_idx(i);
            const int k = polytopes[poly_id].cols();
            q = xi.segment(offset, k).normalized().head(k - 1);
            P.col(i) = polytopes[poly_id].col(0) +
                       polytopes[poly_id].rightCols(k - 1) * q.cwiseProduct(q);
            offset += k;
        }
    }

    static double costTinyNLS(void* ptr,
                              const VectorXd& xi,
                              VectorXd& gradXi) {
        const int n = xi.size();
        const CorridorPolytope2D& ov_poly = *static_cast<CorridorPolytope2D*>(ptr);

        const double sqr_norm_xi = xi.squaredNorm();
        const double inv_norm_xi = 1.0 / std::sqrt(sqr_norm_xi);
        const VectorXd unit_xi = xi * inv_norm_xi;
        const VectorXd r = unit_xi.head(n - 1);
        const Vector2d delta =
            ov_poly.rightCols(n - 1) * r.cwiseProduct(r) + ov_poly.col(1) - ov_poly.col(0);

        double cost = delta.squaredNorm();
        gradXi.resize(n);
        gradXi.head(n - 1) =
            (ov_poly.rightCols(n - 1).transpose() * (2.0 * delta)).array() *
            r.array() * 2.0;
        gradXi(n - 1) = 0.0;
        gradXi = (gradXi - unit_xi.dot(gradXi) * unit_xi).eval() * inv_norm_xi;

        const double sqr_norm_violation = sqr_norm_xi - 1.0;
        if (sqr_norm_violation > 0.0) {
            double c = sqr_norm_violation * sqr_norm_violation;
            const double dc = 3.0 * c;
            c *= sqr_norm_violation;
            cost += c;
            gradXi += dc * 2.0 * xi;
        }

        return cost;
    }

    static void backwardP(const MatrixXd& P,
                          const Eigen::VectorXi& poly_idx,
                          const std::vector<CorridorPolytope2D>& polytopes,
                          VectorXd& xi) {
        int spatial_dim = 0;
        for (int i = 0; i < poly_idx.size(); ++i) {
            spatial_dim += polytopes[poly_idx(i)].cols();
        }
        xi.resize(spatial_dim);

        lbfgs::lbfgs_parameter_t tiny_nls_params;
        tiny_nls_params.past = 0;
        tiny_nls_params.delta = 1.0e-5;
        tiny_nls_params.g_epsilon = std::numeric_limits<double>::epsilon();
        tiny_nls_params.max_iterations = 128;

        for (int i = 0, offset = 0; i < P.cols(); ++i) {
            const int poly_id = poly_idx(i);
            const int k = polytopes[poly_id].cols();
            CorridorPolytope2D ov_poly(2, k + 1);
            ov_poly.col(0) = P.col(i);
            ov_poly.rightCols(k) = polytopes[poly_id];

            VectorXd x(k);
            x.setConstant(std::sqrt(1.0 / static_cast<double>(k)));
            double min_sqr_d = 0.0;
            lbfgs::lbfgs_optimize(x, min_sqr_d,
                                  &TrajectoryOptimizer::costTinyNLS,
                                  nullptr, nullptr,
                                  &ov_poly, tiny_nls_params);
            xi.segment(offset, k) = x;
            offset += k;
        }
    }

    template <typename EIGENVEC>
    static void backwardGradP(const VectorXd& xi,
                              const Eigen::VectorXi& poly_idx,
                              const std::vector<CorridorPolytope2D>& polytopes,
                              const MatrixXd& gradP,
                              EIGENVEC& gradXi) {
        gradXi.resize(xi.size());

        VectorXd q, gradQ, unitQ;
        for (int i = 0, offset = 0; i < gradP.cols(); ++i) {
            const int poly_id = poly_idx(i);
            const int k = polytopes[poly_id].cols();
            q = xi.segment(offset, k);
            const double norm_inv = 1.0 / q.norm();
            unitQ = q * norm_inv;
            gradQ.resize(k);
            gradQ.head(k - 1) =
                (polytopes[poly_id].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                unitQ.head(k - 1).array() * 2.0;
            gradQ(k - 1) = 0.0;
            gradXi.segment(offset, k) =
                (gradQ - unitQ * unitQ.dot(gradQ)) * norm_inv;
            offset += k;
        }
    }

    template <typename EIGENVEC>
    static void normRestrictionLayer(const VectorXd& xi,
                                     const Eigen::VectorXi& poly_idx,
                                     const std::vector<CorridorPolytope2D>& polytopes,
                                     double& extra_cost,
                                     EIGENVEC& gradXi) {
        if (gradXi.size() != xi.size()) {
            gradXi = VectorXd::Zero(xi.size());
        }

        for (int i = 0, offset = 0; i < poly_idx.size(); ++i) {
            const int k = polytopes[poly_idx(i)].cols();
            const VectorXd q = xi.segment(offset, k);
            const double sqr_norm_violation = q.squaredNorm() - 1.0;
            if (sqr_norm_violation > 0.0) {
                double c = sqr_norm_violation * sqr_norm_violation;
                const double dc = 3.0 * c;
                c *= sqr_norm_violation;
                extra_cost += c;
                gradXi.segment(offset, k) += dc * 2.0 * q;
            }
            offset += k;
        }
    }

    static bool smoothedL1(const double& x,
                                 const double& mu,
                                 double& f,
                                 double& df) {
        if (x < 0.0) {
            return false;
        }
        if (x > mu) {
            f = x - 0.5 * mu;
            df = 1.0;
            return true;
        }

        const double xdmu = x / mu;
        const double sqrxdmu = xdmu * xdmu;
        const double mumxd2 = mu - 0.5 * x;
        f = mumxd2 * sqrxdmu * xdmu;
        df = sqrxdmu * (-0.5 * xdmu + 3.0 * mumxd2 / mu);
        return true;
    }

    bool evaluateSFCNodePenalty(const int piece_idx,
                                const Vector2d& pos,
                                double& penalty,
                                Vector2d& gradient) const {
        penalty = 0.0;
        gradient.setZero();
        if (sfc_h_polytopes_.empty()) {
            return false;
        }

        const int seg_idx = std::max(0, std::min(piece_idx, static_cast<int>(sfc_h_polytopes_.size()) - 1));
        const auto& piece = sfc_h_polytopes_[seg_idx];
        const double smooth = std::max(params_.sfc_smooth_factor, 1.0e-6);
        for (const auto& half_space : piece) {
            const double violation = half_space.normal.dot(pos) - half_space.offset;
            double smoothed_cost = 0.0;
            double smoothed_grad = 0.0;
            if (smoothedL1(violation, smooth, smoothed_cost, smoothed_grad)) {
                penalty += smoothed_cost;
                gradient += smoothed_grad * half_space.normal;
            }
        }

        return penalty > 0.0;
    }

    bool buildSFCParameterization() {
        sfc_h_polytopes_.clear();
        sfc_v_polytopes_.clear();
        sfc_v_poly_idx_.resize(0);
        sfc_h_poly_idx_.resize(0);
        sfc_piece_idx_.resize(0);
        sfc_spatial_dim_ = 0;
        sfc_short_path_.resize(2, 0);
        sfc_initial_inner_points_.resize(2, 0);
        sfc_initial_times_.resize(0);

        if (!(params_.use_sfc_parameterization || params_.use_corridor_parameterization) ||
            !usesSFCConstraints(params_.constraint_mode) || !env_) {
            return false;
        }

        const auto* sfc = extractSFC(env_);
        if (sfc == nullptr) {
            return false;
        }

        const int poly_num = static_cast<int>(sfc->size());
        if (poly_num < 1) {
            return false;
        }

        std::vector<std::vector<Vector2d>> overlap_vertices;
        overlap_vertices.reserve(std::max(0, poly_num - 1));
        sfc_h_polytopes_.reserve(poly_num);
        for (int i = 0; i < poly_num; ++i) {
            sfc_h_polytopes_.push_back(normalizePiece((*sfc)[i]));
        }
        for (int i = 0; i < poly_num - 1; ++i) {
            std::vector<Vector2d> vertices =
                halfspacesToVertices(intersectPieces(sfc_h_polytopes_[i], sfc_h_polytopes_[i + 1]));
            if (vertices.size() < 3) {
                return false;
            }
            overlap_vertices.push_back(vertices);
        }

        if (!processSFC(sfc_h_polytopes_)) {
            return false;
        }

        if (!buildShortestPathThroughPolytopes(init_pos_.col(0),
                                               end_pos_.col(0),
                                               sfc_h_polytopes_,
                                               overlap_vertices,
                                               sfc_short_path_)) {
            return false;
        }

        allocatePiecesByLength(sfc_short_path_, params_.piece_len, sfc_piece_idx_);
        if (sfc_piece_idx_.size() != poly_num) {
            return false;
        }

        piece_pos_ = sfc_piece_idx_.sum();
        if (piece_pos_ < 1) {
            return false;
        }

        buildInitialFromPath(sfc_short_path_, sfc_piece_idx_, sfc_initial_inner_points_);
        if (sfc_initial_inner_points_.cols() != piece_pos_ - 1) {
            return false;
        }

        buildTimeAllocationFromPath(sfc_short_path_,
                                    sfc_piece_idx_,
                                    std::max(params_.max_v * 3.0, 1.0e-3),
                                    sfc_initial_times_);
        if (sfc_initial_times_.size() != piece_pos_) {
            return false;
        }
        const double init_total_time = sfc_initial_times_.sum();
        const double target_total_time = std::max(init_total_time, params_.total_time);
        if (init_total_time > 1.0e-8 && target_total_time > init_total_time) {
            sfc_initial_times_ *= target_total_time / init_total_time;
        }

        sfc_v_poly_idx_.resize(std::max(0, piece_pos_ - 1));
        sfc_h_poly_idx_.resize(piece_pos_);
        sfc_spatial_dim_ = 0;
        for (int i = 0, point_idx = 0, piece_id = 0; i < poly_num; ++i) {
            const int pieces = sfc_piece_idx_(i);
            for (int j = 0; j < pieces; ++j, ++piece_id) {
                if (j < pieces - 1) {
                    sfc_v_poly_idx_(point_idx) = 2 * i;
                    sfc_spatial_dim_ += sfc_v_polytopes_[2 * i].cols();
                    point_idx += 1;
                } else if (i < poly_num - 1) {
                    sfc_v_poly_idx_(point_idx) = 2 * i + 1;
                    sfc_spatial_dim_ += sfc_v_polytopes_[2 * i + 1].cols();
                    point_idx += 1;
                }
                sfc_h_poly_idx_(piece_id) = i;
            }
        }

        return sfc_v_poly_idx_.size() == piece_pos_ - 1 &&
               sfc_h_poly_idx_.size() == piece_pos_;
    }

    bool buildCorridorParameterization() {
        return buildSFCParameterization();
    }

    int optimizeSE2Traj(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                        const MatrixXd& endPos, double totalTime) {
        piece_pos_ = innerPtsPos.cols() + 1;
        init_pos_ = initPos;
        end_pos_ = endPos;

        VectorXd initTpos;
        initTpos.resize(piece_pos_);
        initTposByTotalTime(totalTime, innerPtsPos, initTpos);
        return optimizeSE2Traj(initPos, innerPtsPos, endPos, initTpos);
    }

    int failOptimization(int result) {
        in_opt_ = false;
        last_opt_result_ = result;
        last_opt_cost_ = 0.0;
        return result;
    }

    void printInitialTrajectorySummary() const {
        auto metrics = evaluateTrajectory();
        std::cout << "Initial Trajectory:" << std::endl;
        std::cout << "Max velocity: " << metrics.max_velocity << " m/s" << std::endl;
        std::cout << "Max acceleration: " << metrics.max_acceleration << " m/s^2" << std::endl;
        std::cout << "Min clearance: " << metrics.min_clearance << " m" << std::endl;
        std::cout << "Path deviation: " << metrics.path_deviation << " m" << std::endl;
    }

    void configureLBFGS(lbfgs::lbfgs_parameter_t& lbfgs_params) const {
        lbfgs_params.mem_size = params_.mem_size;
        lbfgs_params.past = params_.past;
        lbfgs_params.g_epsilon = params_.g_epsilon;
        lbfgs_params.min_step = params_.min_step;
        lbfgs_params.delta = params_.delta;
        lbfgs_params.max_iterations = params_.max_iter;
    }

    int runLBFGSOptimization(Eigen::VectorXd& x) {
        lbfgs::lbfgs_parameter_t lbfgs_params;
        configureLBFGS(lbfgs_params);

        double final_cost = 0.0;
        const int result = lbfgs::lbfgs_optimize(
            x, final_cost,
            [](void* instance, const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
                return static_cast<TrajectoryOptimizer*>(instance)->costFunction(x, grad);
            },
            nullptr, nullptr, this, lbfgs_params);

        last_opt_result_ = result;
        last_opt_cost_ = final_cost;
        std::cout << "Optimization finished with result: " << result << std::endl;
        std::cout << "Final cost: " << final_cost << std::endl;

        in_opt_ = false;
        return result;
    }

    int resolveSFCPieceIndex(int segment_idx, int segment_num) const {
        if (use_sfc_parameterization_ && sfc_h_poly_idx_.size() == segment_num) {
            return sfc_h_poly_idx_(segment_idx);
        }
        return segment_idx;
    }

    double getSFCWeight() const {
        return params_.rho_sfc > 0.0 ? params_.rho_sfc : params_.rho_corridor;
    }

    int optimizeSE2Traj(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                        const MatrixXd& endPos, const VectorXd& initTpos) {
        in_opt_ = true;
        piece_pos_ = innerPtsPos.cols() + 1;
        init_pos_ = initPos;
        end_pos_ = endPos;
        use_sfc_parameterization_ = false;

        switch (params_.constraint_mode) {
            case SpatialConstraintMode::ESDF:
                return optimizeSE2TrajESDF(initPos, innerPtsPos, endPos, initTpos);
            case SpatialConstraintMode::SFC:
            case SpatialConstraintMode::ESDFAndSFC:
                use_sfc_parameterization_ = buildSFCParameterization();
                if (!use_sfc_parameterization_) {
                    return failOptimization(lbfgs::LBFGSERR_INVALID_N);
                }
                return optimizeSE2TrajSFC(initPos, innerPtsPos, endPos, initTpos);
        }

        return failOptimization(lbfgs::LBFGSERR_INVALID_N);
    }

    int optimizeSE2TrajESDF(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                            const MatrixXd& endPos, const VectorXd& initTpos) {
        dim_T = piece_pos_;
        const int spatial_dim = 2 * (piece_pos_ - 1);
        Eigen::VectorXd x(dim_T + spatial_dim);

        Eigen::Map<Eigen::VectorXd> tau(x.data(), dim_T);
        Eigen::Map<Eigen::MatrixXd> PposMap(x.data() + dim_T, 2, piece_pos_ - 1);
        PposMap = innerPtsPos;

        if (initTpos.size() != piece_pos_) {
            return failOptimization(lbfgs::LBFGSERR_INVALID_N);
        }

        Eigen::VectorXd Tpos = initTpos;
        for (int i = 0; i < Tpos.size(); ++i) {
            Tpos(i) = std::max(Tpos(i), 1.0e-3);
            tau(i) = logC2(Tpos(i));
        }

        generateTrajectory(initPos, endPos, innerPtsPos, Tpos);
        printInitialTrajectorySummary();
        return runLBFGSOptimization(x);
    }

    int optimizeSE2TrajSFC(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                           const MatrixXd& endPos, const VectorXd& initTpos) {
        dim_T = piece_pos_;
        const int spatial_dim = sfc_spatial_dim_;
        Eigen::VectorXd x(dim_T + spatial_dim);

        Eigen::Map<Eigen::VectorXd> tau(x.data(), dim_T);
        MatrixXd Ppos = innerPtsPos;
        if (sfc_initial_inner_points_.cols() == piece_pos_ - 1) {
            Ppos = sfc_initial_inner_points_;
        }

        VectorXd xi;
        backwardP(Ppos, sfc_v_poly_idx_, sfc_v_polytopes_, xi);
        Eigen::Map<Eigen::VectorXd> xiMap(x.data() + dim_T, spatial_dim);
        xiMap = xi;

        Eigen::VectorXd Tpos;
        if (sfc_initial_times_.size() == piece_pos_) {
            Tpos = sfc_initial_times_;
        } else if (initTpos.size() == piece_pos_) {
            Tpos = initTpos;
        } else {
            return failOptimization(lbfgs::LBFGSERR_INVALID_N);
        }

        for (int i = 0; i < Tpos.size(); ++i) {
            Tpos(i) = std::max(Tpos(i), 1.0e-3);
            tau(i) = logC2(Tpos(i));
        }

        generateTrajectory(initPos, endPos, Ppos, Tpos);
        printInitialTrajectorySummary();
        return runLBFGSOptimization(x);
    }

    using ConstraintCostFunction = void (TrajectoryOptimizer::*)(PPoly2D&, double&, Eigen::MatrixXd&, Eigen::VectorXd&);

    double costFunction(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        switch (params_.constraint_mode) {
            case SpatialConstraintMode::ESDF:
                return costFunctionESDF(x, grad);
            case SpatialConstraintMode::SFC:
                return costFunctionSFC(x, grad);
            case SpatialConstraintMode::ESDFAndSFC:
                return costFunctionHybrid(x, grad);
        }

        grad.setZero(x.size());
        return 0.0;
    }

    double costFunctionESDF(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        return evaluateCostFromESDFVariables(x, grad,
                                             &TrajectoryOptimizer::calculateConstraintCostGradESDF);
    }

    double costFunctionSFC(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        return evaluateCostFromSFCVariables(x, grad,
                                            &TrajectoryOptimizer::calculateConstraintCostGradSFC);
    }

    double costFunctionHybrid(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        return evaluateCostFromSFCVariables(x, grad,
                                            &TrajectoryOptimizer::calculateConstraintCostGradHybrid);
    }

    double evaluateCostFromESDFVariables(const Eigen::VectorXd& x,
                                         Eigen::VectorXd& grad,
                                         ConstraintCostFunction constraint_fn) {
        grad.setZero(x.size());

        Eigen::Map<const Eigen::VectorXd> tau(x.data(), dim_T);
        Eigen::Map<Eigen::VectorXd> gradTau(grad.data(), dim_T);
        Eigen::Map<const Eigen::MatrixXd> PposMap(x.data() + dim_T, 2, piece_pos_ - 1);
        const MatrixXd Ppos = PposMap;

        Eigen::VectorXd Tpos(piece_pos_);
        calTfromTau(tau, Tpos);
        generateTrajectory(init_pos_, end_pos_, Ppos, Tpos);

        double constrain_cost = 0.0;
        Eigen::MatrixXd gdCpos_constrain;
        Eigen::VectorXd gdTpos_constrain;
        (this->*constraint_fn)(trajectory_, constrain_cost, gdCpos_constrain, gdTpos_constrain);

        Eigen::MatrixXd gradPpos_constrain;
        Eigen::VectorXd gradTpos_constrain;
        calGradCTtoQT(gdCpos_constrain, gdTpos_constrain, gradPpos_constrain, gradTpos_constrain);

        const double energy = quintic_spline_.getEnergy();
        const double energy_cost = params_.rho_energy * energy;
        const QuinticSpline2D::MatrixType gradP_energy = quintic_spline_.getEnergyGradInnerPoints();
        const Eigen::VectorXd gradT_energy = quintic_spline_.getEnergyGradTimes();

        const MatrixXd gradPposTotal = gradPpos_constrain + params_.rho_energy * gradP_energy.transpose();
        Eigen::VectorXd gradTpos_total = gradTpos_constrain + params_.rho_energy * gradT_energy;

        const double tau_cost = params_.rho_T * Tpos.sum();
        gradTpos_total.array() += params_.rho_T;
        Eigen::VectorXd grad_tau(dim_T);
        calGradtfromT(tau, gradTpos_total, grad_tau);
        gradTau = grad_tau;

        Eigen::Map<Eigen::MatrixXd> gradPpos(grad.data() + dim_T, 2, piece_pos_ - 1);
        gradPpos = gradPposTotal;

        return constrain_cost + energy_cost + tau_cost;
    }

    double evaluateCostFromSFCVariables(const Eigen::VectorXd& x,
                                        Eigen::VectorXd& grad,
                                        ConstraintCostFunction constraint_fn) {
        grad.setZero(x.size());

        Eigen::Map<const Eigen::VectorXd> tau(x.data(), dim_T);
        Eigen::Map<Eigen::VectorXd> gradTau(grad.data(), dim_T);
        Eigen::Map<const Eigen::VectorXd> xi(x.data() + dim_T, sfc_spatial_dim_);
        MatrixXd Ppos(2, piece_pos_ - 1);
        forwardP(xi, sfc_v_poly_idx_, sfc_v_polytopes_, Ppos);

        Eigen::VectorXd Tpos(piece_pos_);
        calTfromTau(tau, Tpos);
        generateTrajectory(init_pos_, end_pos_, Ppos, Tpos);

        double constrain_cost = 0.0;
        Eigen::MatrixXd gdCpos_constrain;
        Eigen::VectorXd gdTpos_constrain;
        (this->*constraint_fn)(trajectory_, constrain_cost, gdCpos_constrain, gdTpos_constrain);

        Eigen::MatrixXd gradPpos_constrain;
        Eigen::VectorXd gradTpos_constrain;
        calGradCTtoQT(gdCpos_constrain, gdTpos_constrain, gradPpos_constrain, gradTpos_constrain);

        const double energy = quintic_spline_.getEnergy();
        const double energy_cost = params_.rho_energy * energy;
        const QuinticSpline2D::MatrixType gradP_energy = quintic_spline_.getEnergyGradInnerPoints();
        const Eigen::VectorXd gradT_energy = quintic_spline_.getEnergyGradTimes();

        const MatrixXd gradPposTotal = gradPpos_constrain + params_.rho_energy * gradP_energy.transpose();
        Eigen::VectorXd gradTpos_total = gradTpos_constrain + params_.rho_energy * gradT_energy;

        const double tau_cost = params_.rho_T * Tpos.sum();
        gradTpos_total.array() += params_.rho_T;
        Eigen::VectorXd grad_tau(dim_T);
        calGradtfromT(tau, gradTpos_total, grad_tau);
        gradTau = grad_tau;

        double total_cost = constrain_cost + energy_cost + tau_cost;
        Eigen::Map<Eigen::VectorXd> gradXi(grad.data() + dim_T, sfc_spatial_dim_);
        backwardGradP(xi, sfc_v_poly_idx_, sfc_v_polytopes_, gradPposTotal, gradXi);
        normRestrictionLayer(xi, sfc_v_poly_idx_, sfc_v_polytopes_, total_cost, gradXi);

        return total_cost;
    }

    void calculateConstraintCostGrad(
        PPoly2D& traj,
        double& cost,
        Eigen::MatrixXd& gdCpos,
        Eigen::VectorXd& gdTpos)
    {
        switch (params_.constraint_mode) {
            case SpatialConstraintMode::ESDF:
                calculateConstraintCostGradESDF(traj, cost, gdCpos, gdTpos);
                return;
            case SpatialConstraintMode::SFC:
                calculateConstraintCostGradSFC(traj, cost, gdCpos, gdTpos);
                return;
            case SpatialConstraintMode::ESDFAndSFC:
                calculateConstraintCostGradHybrid(traj, cost, gdCpos, gdTpos);
                return;
        }
    }

    void calculateConstraintCostGradESDF(
        PPoly2D& traj,
        double& cost,
        Eigen::MatrixXd& gdCpos,
        Eigen::VectorXd& gdTpos)
    {
        cost = 0.0;
        double v_cost = 0.0;
        double a_cost = 0.0;
        double occ_cost = 0.0;

        const int N = traj.getNumSegments();
        gdCpos.resize(6 * N, 2);
        gdCpos.setZero();
        gdTpos.resize(N);
        gdTpos.setZero();

        const auto& breaks = traj.getBreakpoints();
        const MatrixXd& coeffs = traj.getCoefficients();

        Eigen::Vector2d pos, vel, acc, jerk;
        double grad_time = 0.0;
        Eigen::Vector2d grad_p = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_v = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_a = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_sdf = Eigen::Vector2d::Zero();
        Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
        double s1, s2, s3, s4, s5;
        double step, alpha, omg;

        for (int i = 0; i < N; ++i) {
            const Eigen::Matrix<double, 6, 2>& c = coeffs.block<6, 2>(i * 6, 0);
            step = (breaks[i + 1] - breaks[i]) / params_.int_K;
            s1 = 0.0;

            for (int j = 0; j <= params_.int_K; ++j) {
                alpha = 1.0 / params_.int_K * j;
                grad_p.setZero();
                grad_v.setZero();
                grad_a.setZero();
                grad_sdf.setZero();
                grad_time = 0.0;

                s2 = s1 * s1;
                s3 = s2 * s1;
                s4 = s2 * s2;
                s5 = s4 * s1;
                beta0 << 1.0, s1, s2, s3, s4, s5;
                beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
                beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
                beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
                pos = c.transpose() * beta0;
                vel = c.transpose() * beta1;
                acc = c.transpose() * beta2;
                jerk = c.transpose() * beta3;

                omg = (j == 0 || j == params_.int_K) ? 0.5 : 1.0;

                const double vxy_snorm = vel.squaredNorm();
                const double vViola = vxy_snorm - params_.max_v * params_.max_v;
                if (vViola > 0.0) {
                    grad_v += params_.rho_v * 6.0 * vViola * vViola * vel;
                    const double cost_v = params_.rho_v * vViola * vViola * vViola;
                    cost += cost_v * omg * step;
                    v_cost += cost_v * omg * step;
                    grad_time += omg * (cost_v / params_.int_K + step * alpha * grad_v.dot(acc));
                }

                const double axy_snorm = acc.squaredNorm();
                const double aViola = axy_snorm - params_.max_a * params_.max_a;
                if (aViola > 0.0) {
                    grad_a += params_.rho_a * 6.0 * aViola * aViola * acc;
                    const double cost_a = params_.rho_a * aViola * aViola * aViola;
                    cost += cost_a * omg * step;
                    a_cost += cost_a * omg * step;
                    grad_time += omg * (cost_a / params_.int_K + step * alpha * grad_a.dot(jerk));
                }

                if (env_) {
                    double sdf_value = 0.0;
                    if (env_->getDistanceAndGradient(pos, sdf_value, grad_sdf)) {
                        const double cViola = params_.safe_threshold - sdf_value;
                        if (cViola > 0.0 && sdf_value < 5.0) {
                            double penalty = 0.0;
                            Eigen::Vector2d grad_pc = Eigen::Vector2d::Zero();
                            if (cViola < 0.1) {
                                penalty = cViola * cViola;
                                grad_pc = -2.0 * cViola * grad_sdf;
                            } else {
                                penalty = cViola;
                                grad_pc = -grad_sdf;
                            }

                            const double cost_c = params_.rho_collision * penalty;
                            const Eigen::Vector2d grad_pc_scaled = params_.rho_collision * grad_pc;
                            cost += cost_c * omg * step;
                            occ_cost += cost_c * omg * step;
                            grad_time += omg * (cost_c / params_.int_K + step * alpha * grad_pc_scaled.dot(vel));
                            grad_p += grad_pc_scaled;
                        }
                    }
                }

                gdCpos.block<6, 2>(i * 6, 0) +=
                    (beta0 * grad_p.transpose() +
                     beta1 * grad_v.transpose() +
                     beta2 * grad_a.transpose()) * omg * step;
                gdTpos(i) += grad_time;
                s1 += step;
            }
        }

        std::cout << "Cost breakdown - Vel: " << v_cost
                  << ", Acc: " << a_cost
                  << ", Coll: " << occ_cost
                  << ", SFC: 0" << std::endl;
    }

    void calculateConstraintCostGradSFC(
        PPoly2D& traj,
        double& cost,
        Eigen::MatrixXd& gdCpos,
        Eigen::VectorXd& gdTpos)
    {
        cost = 0.0;
        double v_cost = 0.0;
        double a_cost = 0.0;
        double sfc_cost = 0.0;
        const double sfc_weight = getSFCWeight();

        const int N = traj.getNumSegments();
        gdCpos.resize(6 * N, 2);
        gdCpos.setZero();
        gdTpos.resize(N);
        gdTpos.setZero();

        const auto& breaks = traj.getBreakpoints();
        const MatrixXd& coeffs = traj.getCoefficients();

        Eigen::Vector2d pos, vel, acc, jerk;
        double grad_time = 0.0;
        Eigen::Vector2d grad_p = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_v = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_a = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_sfc = Eigen::Vector2d::Zero();
        Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
        double s1, s2, s3, s4, s5;
        double step, alpha, omg;

        for (int i = 0; i < N; ++i) {
            const Eigen::Matrix<double, 6, 2>& c = coeffs.block<6, 2>(i * 6, 0);
            step = (breaks[i + 1] - breaks[i]) / params_.int_K;
            s1 = 0.0;

            for (int j = 0; j <= params_.int_K; ++j) {
                alpha = 1.0 / params_.int_K * j;
                grad_p.setZero();
                grad_v.setZero();
                grad_a.setZero();
                grad_sfc.setZero();
                grad_time = 0.0;

                s2 = s1 * s1;
                s3 = s2 * s1;
                s4 = s2 * s2;
                s5 = s4 * s1;
                beta0 << 1.0, s1, s2, s3, s4, s5;
                beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
                beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
                beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
                pos = c.transpose() * beta0;
                vel = c.transpose() * beta1;
                acc = c.transpose() * beta2;
                jerk = c.transpose() * beta3;

                omg = (j == 0 || j == params_.int_K) ? 0.5 : 1.0;

                const double vxy_snorm = vel.squaredNorm();
                const double vViola = vxy_snorm - params_.max_v * params_.max_v;
                if (vViola > 0.0) {
                    grad_v += params_.rho_v * 6.0 * vViola * vViola * vel;
                    const double cost_v = params_.rho_v * vViola * vViola * vViola;
                    cost += cost_v * omg * step;
                    v_cost += cost_v * omg * step;
                    grad_time += omg * (cost_v / params_.int_K + step * alpha * grad_v.dot(acc));
                }

                const double axy_snorm = acc.squaredNorm();
                const double aViola = axy_snorm - params_.max_a * params_.max_a;
                if (aViola > 0.0) {
                    grad_a += params_.rho_a * 6.0 * aViola * aViola * acc;
                    const double cost_a = params_.rho_a * aViola * aViola * aViola;
                    cost += cost_a * omg * step;
                    a_cost += cost_a * omg * step;
                    grad_time += omg * (cost_a / params_.int_K + step * alpha * grad_a.dot(jerk));
                }

                double sfc_penalty = 0.0;
                const int piece_idx = resolveSFCPieceIndex(i, N);
                if (evaluateSFCNodePenalty(piece_idx, pos, sfc_penalty, grad_sfc)) {
                    const double cost_sc = sfc_weight * sfc_penalty;
                    const Eigen::Vector2d grad_sc = sfc_weight * grad_sfc;
                    cost += cost_sc * omg * step;
                    sfc_cost += cost_sc * omg * step;
                    grad_time += omg * (cost_sc / params_.int_K + step * alpha * grad_sc.dot(vel));
                    grad_p += grad_sc;
                }

                gdCpos.block<6, 2>(i * 6, 0) +=
                    (beta0 * grad_p.transpose() +
                     beta1 * grad_v.transpose() +
                     beta2 * grad_a.transpose()) * omg * step;
                gdTpos(i) += grad_time;
                s1 += step;
            }
        }

        std::cout << "Cost breakdown - Vel: " << v_cost
                  << ", Acc: " << a_cost
                  << ", Coll: 0"
                  << ", SFC: " << sfc_cost << std::endl;
    }

    void calculateConstraintCostGradHybrid(
        PPoly2D& traj,
        double& cost,
        Eigen::MatrixXd& gdCpos,
        Eigen::VectorXd& gdTpos)
    {
        cost = 0.0;
        double v_cost = 0.0;
        double a_cost = 0.0;
        double occ_cost = 0.0;
        double sfc_cost = 0.0;
        const double sfc_weight = getSFCWeight();

        const int N = traj.getNumSegments();
        gdCpos.resize(6 * N, 2);
        gdCpos.setZero();
        gdTpos.resize(N);
        gdTpos.setZero();

        const auto& breaks = traj.getBreakpoints();
        const MatrixXd& coeffs = traj.getCoefficients();

        Eigen::Vector2d pos, vel, acc, jerk;
        double grad_time = 0.0;
        Eigen::Vector2d grad_p = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_v = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_a = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_sdf = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_sfc = Eigen::Vector2d::Zero();
        Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
        double s1, s2, s3, s4, s5;
        double step, alpha, omg;

        for (int i = 0; i < N; ++i) {
            const Eigen::Matrix<double, 6, 2>& c = coeffs.block<6, 2>(i * 6, 0);
            step = (breaks[i + 1] - breaks[i]) / params_.int_K;
            s1 = 0.0;

            for (int j = 0; j <= params_.int_K; ++j) {
                alpha = 1.0 / params_.int_K * j;
                grad_p.setZero();
                grad_v.setZero();
                grad_a.setZero();
                grad_sdf.setZero();
                grad_sfc.setZero();
                grad_time = 0.0;

                s2 = s1 * s1;
                s3 = s2 * s1;
                s4 = s2 * s2;
                s5 = s4 * s1;
                beta0 << 1.0, s1, s2, s3, s4, s5;
                beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
                beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
                beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
                pos = c.transpose() * beta0;
                vel = c.transpose() * beta1;
                acc = c.transpose() * beta2;
                jerk = c.transpose() * beta3;

                omg = (j == 0 || j == params_.int_K) ? 0.5 : 1.0;

                const double vxy_snorm = vel.squaredNorm();
                const double vViola = vxy_snorm - params_.max_v * params_.max_v;
                if (vViola > 0.0) {
                    grad_v += params_.rho_v * 6.0 * vViola * vViola * vel;
                    const double cost_v = params_.rho_v * vViola * vViola * vViola;
                    cost += cost_v * omg * step;
                    v_cost += cost_v * omg * step;
                    grad_time += omg * (cost_v / params_.int_K + step * alpha * grad_v.dot(acc));
                }

                const double axy_snorm = acc.squaredNorm();
                const double aViola = axy_snorm - params_.max_a * params_.max_a;
                if (aViola > 0.0) {
                    grad_a += params_.rho_a * 6.0 * aViola * aViola * acc;
                    const double cost_a = params_.rho_a * aViola * aViola * aViola;
                    cost += cost_a * omg * step;
                    a_cost += cost_a * omg * step;
                    grad_time += omg * (cost_a / params_.int_K + step * alpha * grad_a.dot(jerk));
                }

                if (env_) {
                    double sdf_value = 0.0;
                    if (env_->getDistanceAndGradient(pos, sdf_value, grad_sdf)) {
                        const double cViola = params_.safe_threshold - sdf_value;
                        if (cViola > 0.0 && sdf_value < 5.0) {
                            double penalty = 0.0;
                            Eigen::Vector2d grad_pc = Eigen::Vector2d::Zero();
                            if (cViola < 0.1) {
                                penalty = cViola * cViola;
                                grad_pc = -2.0 * cViola * grad_sdf;
                            } else {
                                penalty = cViola;
                                grad_pc = -grad_sdf;
                            }

                            const double cost_c = params_.rho_collision * penalty;
                            const Eigen::Vector2d grad_pc_scaled = params_.rho_collision * grad_pc;
                            cost += cost_c * omg * step;
                            occ_cost += cost_c * omg * step;
                            grad_time += omg * (cost_c / params_.int_K + step * alpha * grad_pc_scaled.dot(vel));
                            grad_p += grad_pc_scaled;
                        }
                    }

                    double sfc_penalty = 0.0;
                    const int piece_idx = resolveSFCPieceIndex(i, N);
                    if (evaluateSFCNodePenalty(piece_idx, pos, sfc_penalty, grad_sfc)) {
                        const double cost_sc = sfc_weight * sfc_penalty;
                        const Eigen::Vector2d grad_sc = sfc_weight * grad_sfc;
                        cost += cost_sc * omg * step;
                        sfc_cost += cost_sc * omg * step;
                        grad_time += omg * (cost_sc / params_.int_K + step * alpha * grad_sc.dot(vel));
                        grad_p += grad_sc;
                    }
                }

                gdCpos.block<6, 2>(i * 6, 0) +=
                    (beta0 * grad_p.transpose() +
                     beta1 * grad_v.transpose() +
                     beta2 * grad_a.transpose()) * omg * step;
                gdTpos(i) += grad_time;
                s1 += step;
            }
        }

        std::cout << "Cost breakdown - Vel: " << v_cost
                  << ", Acc: " << a_cost
                  << ", Coll: " << occ_cost
                  << ", SFC: " << sfc_cost << std::endl;
    }

    void calGradCTtoQT(
        const Eigen::MatrixXd& gdCpos,
        const Eigen::VectorXd& gdTpos,
        Eigen::MatrixXd& gradPpos,
        Eigen::VectorXd& gradTpos_out)
    {
        QuinticSpline2D::MatrixType gdC_typed = gdCpos;
        auto grad_all = quintic_spline_.propagateGrad(gdC_typed, gdTpos);

        gradPpos = grad_all.inner_points.transpose();
        gradTpos_out = grad_all.times;
        
        std::cout << "Gradient norm - Ppos: " << gradPpos.norm() 
                << ", Tpos: " << gradTpos_out.norm() << std::endl;
    }

    void generateTrajectory(const MatrixXd& initPos, const MatrixXd& endPos,
                          const MatrixXd& innerPts, Eigen::VectorXd Tpos) {
        std::vector<double> times;
        times.reserve(piece_pos_ + 1);
        
        double t = 0;
        for (int i = 0; i < Tpos.size(); ++i) {
            times.push_back(t);
            t += Tpos(i);
        }
        times.push_back(t);

        QuinticSpline2D::MatrixType waypoints(piece_pos_ + 1, 2);
        waypoints.row(0) = initPos.col(0).transpose();
        for (int i = 0; i < innerPts.cols(); ++i) {
            waypoints.row(i + 1) = innerPts.col(i).transpose();
        }
        waypoints.row(piece_pos_) = endPos.col(0).transpose();

        SplineTrajectory::BoundaryConditions<2> bc;
        bc.start_velocity = initPos.col(1);
        bc.end_velocity = endPos.col(1);
        bc.start_acceleration = Eigen::Vector2d::Zero();
        bc.end_acceleration = Eigen::Vector2d::Zero();

        quintic_spline_.update(times, waypoints, bc);
        trajectory_ = quintic_spline_.getTrajectory();
    }

    double calculatePathLength(const std::vector<Eigen::Vector2d>& path) {
        double length = 0.0;
        for (size_t i = 1; i < path.size(); ++i) {
            length += (path[i] - path[i-1]).norm();
        }
        return length;
    }

    void initTposByTotalTime(double totalTime, const Eigen::Ref<const Eigen::MatrixXd>& Ppos, Eigen::VectorXd& T) {
        Eigen::VectorXd seg_lens(piece_pos_);
        Eigen::Vector2d p_last = init_pos_.col(0);
        double total_len = 0.0;

        for (int i = 0; i < piece_pos_ - 1; ++i) {
            Eigen::Vector2d p_next = Ppos.col(i);
            seg_lens(i) = (p_next - p_last).norm();
            total_len += seg_lens(i);
            p_last = p_next;
        }
        seg_lens(piece_pos_ - 1) = (end_pos_.col(0) - p_last).norm();
        total_len += seg_lens(piece_pos_ - 1);

        if (total_len < 1.0e-8) {
            T.setConstant(totalTime / piece_pos_);
            return;
        }

        T = totalTime * seg_lens / total_len;
        for (int i = 0; i < T.size(); ++i) {
            T(i) = std::max(T(i), 1.0e-3);
        }
    }

    inline double logC2(const double& T) {
        return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
    }

    inline void calTfromTau(const Eigen::VectorXd& tau, Eigen::VectorXd& T) {
        T.resize(tau.size());
        for (int i = 0; i < tau.size(); ++i) {
            T(i) = expC2(tau(i));
        }
    }

    inline double expC2(const double& tau) {
        return tau > 0.0 ? ((0.5 * tau + 1.0) * tau + 1.0) : 1.0 / ((0.5 * tau - 1.0) * tau + 1.0);
    }

    inline double getTtoTauGrad(const double& tau) {
        if (tau > 0)
            return tau + 1.0;
        else {
            double denSqrt = (0.5 * tau - 1.0) * tau + 1.0;
            return (1.0 - tau) / (denSqrt * denSqrt);
        } 
    }

    inline void calGradtfromT(const Eigen::VectorXd& tau,
                              const Eigen::VectorXd& gradT,
                              Eigen::VectorXd& gradTau) {
        for (int i = 0; i < tau.size(); ++i) {
            gradTau(i) = gradT(i) * getTtoTauGrad(tau(i));
        }
    }

    std::shared_ptr<TrajectoryEnv> env_;
    std::vector<Eigen::Vector2d> guide_path_;
    TrajectoryParams params_;
    bool in_opt_;
    int piece_pos_;
    int dim_T;
    MatrixXd init_pos_;
    MatrixXd end_pos_;
    PPoly2D trajectory_;
    QuinticSpline2D quintic_spline_;
    int current_segment_ = 0;
    int last_opt_result_ = lbfgs::LBFGSERR_UNKNOWNERROR;
    double last_opt_cost_ = 0.0;
    bool use_sfc_parameterization_ = false;
    std::vector<CorridorPiece2D> sfc_h_polytopes_;
    std::vector<CorridorPolytope2D> sfc_v_polytopes_;
    Eigen::VectorXi sfc_v_poly_idx_;
    Eigen::VectorXi sfc_h_poly_idx_;
    Eigen::VectorXi sfc_piece_idx_;
    int sfc_spatial_dim_ = 0;
    MatrixXd sfc_short_path_;
    MatrixXd sfc_initial_inner_points_;
    VectorXd sfc_initial_times_;
};

} // namespace TrajOpt

#endif // TRAJ_OPT_HPP
