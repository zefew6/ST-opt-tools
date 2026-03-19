#pragma once

#include "perception_tool/firi_mvie2d.hpp"
#include "perception_tool/firi_sdmn2d.hpp"
#include "perception_tool/firi_types.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace perception_tools {

class FIRISolver {
public:
    struct Result {
        std::vector<HalfPlane2D> planes;
        int iterations = 0;
        double solve_time_ms = 0.0;
    };

    Result compute(const std::vector<Eigen::Vector2d>& obstacles,
                   const std::vector<Eigen::Vector2d>& seed_vertices,
                   const std::vector<HalfPlane2D>& bbox_planes,
                   int max_iter = 10,
                   double rho = 0.02)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        if (seed_vertices.empty() || obstacles.empty()) {
            return {bbox_planes, 0, 0.0};
        }

        Eigen::Vector2d d = Eigen::Vector2d::Zero();
        for (const auto& v : seed_vertices) {
            d += v;
        }
        d /= static_cast<double>(seed_vertices.size());

        double r_init = inscribed_radius(seed_vertices, d);
        r_init = std::max(r_init * 0.8, 1e-4);
        Eigen::Matrix2d L = r_init * Eigen::Matrix2d::Identity();

        constexpr double kPi = 3.14159265358979323846;
        double prev_vol = r_init * r_init * kPi;
        std::vector<HalfPlane2D> best_planes = bbox_planes;
        int iters = 0;

        for (int k = 0; k < max_iter; ++k) {
            iters = k + 1;
            auto planes = run_rsi(obstacles, seed_vertices, L, d, bbox_planes);
            best_planes = planes;

            const int m = static_cast<int>(planes.size());
            Eigen::MatrixXd A(m, 2);
            Eigen::VectorXd b(m);
            for (int i = 0; i < m; ++i) {
                A.row(i) = planes[i].normal.transpose();
                b(i) = planes[i].offset;
            }

            auto mvie = mvie_solver_.solve(A, b, d);
            const double new_vol = mvie.volume();
            if (k > 0 && (new_vol - prev_vol) / (prev_vol + 1e-15) < rho) {
                break;
            }

            prev_vol = new_vol;
            L = mvie.L;
            d = mvie.d;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();
        return {best_planes, iters, ms};
    }

private:
    struct ObsHalfPlane {
        Eigen::Vector2d b_sol = Eigen::Vector2d::Zero();
        Eigen::Vector2d a = Eigen::Vector2d::Zero();
        double a_norm = 0.0;
        int obs_idx = -1;
    };

    double inscribed_radius(const std::vector<Eigen::Vector2d>& verts,
                            const Eigen::Vector2d& c) const
    {
        double r = 1e18;
        const int n = static_cast<int>(verts.size());

        for (int i = 0; i < n; ++i) {
            const Eigen::Vector2d& a = verts[i];
            const Eigen::Vector2d& b = verts[(i + 1) % n];
            const Eigen::Vector2d edge = b - a;
            const double len = edge.norm();
            if (len < 1e-15) {
                continue;
            }

            Eigen::Vector2d normal(-edge.y(), edge.x());
            normal /= len;
            r = std::min(r, std::abs(normal.dot(c - a)));
        }

        return r;
    }

    std::vector<HalfPlane2D> run_rsi(
        const std::vector<Eigen::Vector2d>& obstacles,
        const std::vector<Eigen::Vector2d>& seed_vertices,
        const Eigen::Matrix2d& L,
        const Eigen::Vector2d& d,
        const std::vector<HalfPlane2D>& bbox_planes)
    {
        const Eigen::Matrix2d L_inv = L.inverse();
        const Eigen::Matrix2d L_inv_t = L_inv.transpose();

        std::vector<Eigen::Vector2d> seed_bar;
        seed_bar.reserve(seed_vertices.size());
        for (const auto& v : seed_vertices) {
            seed_bar.push_back(L_inv * (v - d));
        }

        std::vector<Eigen::Vector2d> obs_bar;
        obs_bar.reserve(obstacles.size());
        for (const auto& u : obstacles) {
            obs_bar.push_back(L_inv * (u - d));
        }

        const int n_seed = static_cast<int>(seed_bar.size());
        std::vector<Eigen::Vector2d> base_normals(n_seed + 1);
        std::vector<double> base_bounds(n_seed + 1);
        for (int i = 0; i < n_seed; ++i) {
            base_normals[i] = seed_bar[i];
            base_bounds[i] = 1.0;
        }

        std::vector<ObsHalfPlane> candidates;
        candidates.reserve(obs_bar.size());

        for (size_t i = 0; i < obs_bar.size(); ++i) {
            base_normals[n_seed] = -obs_bar[i];
            base_bounds[n_seed] = -1.0;

            auto result = sdmn_.solve(base_normals, base_bounds);
            if (!result.feasible) {
                continue;
            }

            const double b_sq = result.y.squaredNorm();
            if (b_sq <= 1e-10) {
                continue;
            }

            const Eigen::Vector2d a = result.y / b_sq;
            candidates.push_back({result.y, a, a.norm(), static_cast<int>(i)});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const ObsHalfPlane& lhs, const ObsHalfPlane& rhs) {
                      return lhs.a_norm < rhs.a_norm;
                  });

        std::vector<bool> separated(obs_bar.size(), false);
        std::vector<HalfPlane2D> result_planes = bbox_planes;

        for (const auto& hp : candidates) {
            if (separated[hp.obs_idx]) {
                continue;
            }

            Eigen::Vector2d n_orig = L_inv_t * hp.a;
            const double d_orig = hp.a.squaredNorm() + n_orig.dot(d);
            const double n_norm = n_orig.norm();
            if (n_norm < 1e-15) {
                continue;
            }

            result_planes.push_back({n_orig / n_norm, d_orig / n_norm});

            for (size_t i = 0; i < obs_bar.size(); ++i) {
                if (!separated[i] && hp.b_sol.dot(obs_bar[i]) >= 1.0 - 1e-8) {
                    separated[i] = true;
                }
            }

            if (result_planes.size() > 50) {
                break;
            }
        }

        return result_planes;
    }

    SDMN2D sdmn_;
    MVIE2D mvie_solver_;
};

}  // namespace perception_tools
