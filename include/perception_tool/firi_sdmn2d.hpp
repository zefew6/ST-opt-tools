#pragma once

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace perception_tools {

class SDMN2D {
public:
    struct Result {
        Eigen::Vector2d y = Eigen::Vector2d::Zero();
        bool feasible = false;
    };

    explicit SDMN2D(unsigned int seed = 42) : rng_(seed) {}

    Result solve(const std::vector<Eigen::Vector2d>& e,
                 const std::vector<double>& f)
    {
        const size_t d = e.size();
        if (d == 0) {
            return {Eigen::Vector2d::Zero(), true};
        }

        std::vector<size_t> perm(d);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);

        Eigen::Vector2d y = Eigen::Vector2d::Zero();

        for (size_t ii = 0; ii < d; ++ii) {
            const size_t idx = perm[ii];
            if (e[idx].dot(y) <= f[idx] + 1e-12) {
                continue;
            }

            const Eigen::Vector2d& eh = e[idx];
            const double fh = f[idx];
            const double e_te = eh.squaredNorm();
            if (e_te < 1e-15) {
                return {Eigen::Vector2d::Zero(), false};
            }

            const Eigen::Vector2d v = (fh / e_te) * eh;
            const int j = (std::abs(v.x()) >= std::abs(v.y())) ? 0 : 1;
            const int k = 1 - j;

            Eigen::Vector2d m_col;
            const double v_norm = v.norm();

            if (v_norm < 1e-15) {
                m_col = Eigen::Vector2d(-eh.y(), eh.x());
                const double mn = m_col.norm();
                if (mn <= 1e-15) {
                    return {Eigen::Vector2d::Zero(), false};
                }
                m_col /= mn;
            } else {
                const double sign_vj = (v(j) >= 0.0) ? 1.0 : -1.0;
                const Eigen::Vector2d u_ref =
                    v + sign_vj * v_norm * Eigen::Vector2d::Unit(j);
                const double u_t_u = u_ref.squaredNorm();
                if (u_t_u < 1e-15) {
                    m_col = Eigen::Vector2d(-eh.y(), eh.x()).normalized();
                } else {
                    m_col = Eigen::Vector2d::Unit(k) - (2.0 * u_ref(k) / u_t_u) * u_ref;
                }
            }

            double lo = -1e18;
            double hi = 1e18;
            bool feasible = true;

            for (size_t pp = 0; pp < ii; ++pp) {
                const size_t pidx = perm[pp];
                const double a_1d = e[pidx].dot(m_col);
                const double b_1d = f[pidx] - e[pidx].dot(v);

                if (std::abs(a_1d) < 1e-15) {
                    if (b_1d < -1e-10) {
                        feasible = false;
                        break;
                    }
                    continue;
                }

                const double bound = b_1d / a_1d;
                if (a_1d > 0.0) {
                    hi = std::min(hi, bound);
                } else {
                    lo = std::max(lo, bound);
                }
            }

            if (!feasible || lo > hi + 1e-10) {
                return {Eigen::Vector2d::Zero(), false};
            }

            double t = 0.0;
            if (lo <= 0.0 && 0.0 <= hi) {
                t = 0.0;
            } else if (lo > 0.0) {
                t = lo;
            } else {
                t = hi;
            }

            y = m_col * t + v;
        }

        return {y, true};
    }

private:
    std::mt19937 rng_;
};

}  // namespace perception_tools
