#pragma once

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>

namespace perception_tools {

class MVIE2D {
public:
    struct Ellipsoid {
        Eigen::Matrix2d L = Eigen::Matrix2d::Identity();
        Eigen::Vector2d d = Eigen::Vector2d::Zero();

        double volume() const
        {
            constexpr double kPi = 3.14159265358979323846;
            return kPi * std::abs(L(0, 0) * L(1, 1));
        }
    };

    Ellipsoid solve(const Eigen::MatrixXd& A,
                    const Eigen::VectorXd& b,
                    const Eigen::Vector2d& center_hint)
    {
        const int m = static_cast<int>(A.rows());

        Eigen::Vector2d c = center_hint;
        double r = 1e18;
        for (int i = 0; i < m; ++i) {
            const double norm_ai = A.row(i).norm();
            if (norm_ai > 1e-10) {
                const double gap = b(i) - A.row(i).dot(c);
                r = std::min(r, gap / norm_ai);
            }
        }
        if (r <= 0.0) {
            r = 1e-4;
        }
        r *= 0.9;
        r = std::max(r, 1e-6);

        Eigen::VectorXd x(5);
        x << r, 0.0, r, c.x(), c.y();

        double t = 1.0;
        const double mu = 4.0;

        for (int outer = 0; outer < 20; ++outer) {
            for (int inner = 0; inner < 40; ++inner) {
                Eigen::VectorXd grad = gradient(A, b, x, t);
                Eigen::MatrixXd H = hessian(A, b, x, t);
                H += 1e-8 * Eigen::MatrixXd::Identity(5, 5);

                Eigen::VectorXd dx = H.ldlt().solve(-grad);
                const double lambda_sq = -grad.dot(dx);
                if (lambda_sq < 1e-6) {
                    break;
                }

                double alpha = 1.0;
                const double f0 = objective(A, b, x, t);

                for (int ls = 0; ls < 32; ++ls) {
                    Eigen::VectorXd xn = x + alpha * dx;
                    if (xn(0) > 1e-10 && xn(2) > 1e-10) {
                        const double fn = objective(A, b, xn, t);
                        if (std::isfinite(fn) && fn < f0 + 0.3 * alpha * grad.dot(dx)) {
                            x = xn;
                            break;
                        }
                    }
                    alpha *= 0.5;
                    if (alpha < 1e-12) {
                        break;
                    }
                }
            }

            if (static_cast<double>(m) / t < 1e-3) {
                break;
            }
            t *= mu;
        }

        Ellipsoid ellipsoid;
        ellipsoid.L << x(0), 0.0,
                       x(1), x(2);
        ellipsoid.d << x(3), x(4);
        return ellipsoid;
    }

private:
    double objective(const Eigen::MatrixXd& A,
                     const Eigen::VectorXd& b,
                     const Eigen::VectorXd& x,
                     double t)
    {
        const double L11 = x(0);
        const double L21 = x(1);
        const double L22 = x(2);
        const double d1 = x(3);
        const double d2 = x(4);

        if (L11 <= 0.0 || L22 <= 0.0) {
            return 1e18;
        }

        double val = -t * (std::log(L11) + std::log(L22));

        for (int i = 0; i < A.rows(); ++i) {
            const double a1 = A(i, 0);
            const double a2 = A(i, 1);
            const double r1 = L11 * a1 + L21 * a2;
            const double r2 = L22 * a2;
            const double gap =
                b(i) - a1 * d1 - a2 * d2 - std::sqrt(r1 * r1 + r2 * r2);
            if (gap <= 0.0) {
                return 1e18;
            }
            val -= std::log(gap);
        }

        return val;
    }

    Eigen::VectorXd gradient(const Eigen::MatrixXd& A,
                             const Eigen::VectorXd& b,
                             const Eigen::VectorXd& x,
                             double t)
    {
        const double L11 = x(0);
        const double L21 = x(1);
        const double L22 = x(2);
        const double d1 = x(3);
        const double d2 = x(4);

        Eigen::VectorXd g = Eigen::VectorXd::Zero(5);
        g(0) = -t / L11;
        g(2) = -t / L22;

        for (int i = 0; i < A.rows(); ++i) {
            const double a1 = A(i, 0);
            const double a2 = A(i, 1);
            const double r1 = L11 * a1 + L21 * a2;
            const double r2 = L22 * a2;
            const double nr = std::sqrt(r1 * r1 + r2 * r2);

            double gap = b(i) - a1 * d1 - a2 * d2 - nr;
            if (gap < 1e-15) {
                gap = 1e-15;
            }
            const double inv_gap = 1.0 / gap;

            if (nr > 1e-15) {
                const double inv_nr = 1.0 / nr;
                g(0) += inv_gap * r1 * a1 * inv_nr;
                g(1) += inv_gap * r1 * a2 * inv_nr;
                g(2) += inv_gap * r2 * a2 * inv_nr;
            }
            g(3) += inv_gap * a1;
            g(4) += inv_gap * a2;
        }

        return g;
    }

    Eigen::MatrixXd hessian(const Eigen::MatrixXd& A,
                            const Eigen::VectorXd& b,
                            const Eigen::VectorXd& x,
                            double t)
    {
        constexpr double kEps = 1e-6;
        Eigen::MatrixXd H(5, 5);

        for (int j = 0; j < 5; ++j) {
            Eigen::VectorXd xp = x;
            Eigen::VectorXd xm = x;
            xp(j) += kEps;
            xm(j) -= kEps;
            H.col(j) = (gradient(A, b, xp, t) - gradient(A, b, xm, t)) / (2.0 * kEps);
        }

        return 0.5 * (H + H.transpose());
    }
};

}  // namespace perception_tools
