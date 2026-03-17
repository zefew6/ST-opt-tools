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

struct TrajectoryParams {
    double total_len = 10;
    double total_time = 10;
    double piece_len = 4;
    double rho_v = 10000;           // Velocity penalty weight
    double rho_a = 10000;           // Acceleration penalty weight
    double rho_collision = 100000;  // Collision penalty weight
    double rho_corridor = 100000;   // Safe corridor penalty weight
    double rho_T = 100;             // Time penalty weight
    double rho_energy = 100;        // Energy (smoothness) penalty weight
    double max_v = 1.0;             // Maximum velocity
    double max_a = 1.0;             // Maximum acceleration
    double safe_threshold = 0.5;    // Safety distance threshold
    
    int int_K = 32;                 // Integration sample points
    int mem_size = 256;             // L-BFGS memory size
    int past = 3;                   // L-BFGS parameter
    double g_epsilon = 1e-6;        // Gradient convergence threshold
    double min_step = 1e-32;        // Minimum step size
    double delta = 1e-5;            // Function change convergence threshold
    int max_iter = 10000;           // Maximum iterations
};

class TrajectoryOptimizer {
public:
    using Vector2d = Eigen::Vector2d;
    using Vector3d = Eigen::Vector3d;
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using PPoly2D = SplineTrajectory::PPolyND<2>;
    using QuinticSpline2D = SplineTrajectory::QuinticSpline2D;

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

    int optimizeSE2Traj(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                        const MatrixXd& endPos, const VectorXd& initTpos) {
        in_opt_ = true;
        piece_pos_ = innerPtsPos.cols() + 1;

        init_pos_ = initPos;
        end_pos_ = endPos;

        dim_T = piece_pos_;
        int variable_num = 2 * (piece_pos_ - 1) + dim_T;

        Eigen::VectorXd x;
        x.resize(variable_num);

        Eigen::Map<Eigen::VectorXd> tau(x.data(), dim_T);
        Eigen::Map<Eigen::MatrixXd> Ppos(x.data() + dim_T, 2, piece_pos_ - 1);
        Ppos = innerPtsPos;

        Eigen::VectorXd Tpos;
        if (initTpos.size() != piece_pos_) {
            in_opt_ = false;
            last_opt_result_ = lbfgs::LBFGSERR_INVALID_N;
            last_opt_cost_ = 0.0;
            return last_opt_result_;
        }
        Tpos = initTpos;
        for (int i = 0; i < Tpos.size(); ++i) {
            Tpos(i) = std::max(Tpos(i), 1.0e-3);
        }
        for (int i = 0; i < dim_T; ++i) {
            tau(i) = logC2(Tpos(i));
        }

        generateTrajectory(initPos, endPos, Ppos, Tpos);
        auto metrics = evaluateTrajectory();
        std::cout << "Initial Trajectory:" << std::endl;
        std::cout << "Max velocity: " << metrics.max_velocity << " m/s" << std::endl;
        std::cout << "Max acceleration: " << metrics.max_acceleration << " m/s^2" << std::endl;
        std::cout << "Min clearance: " << metrics.min_clearance << " m" << std::endl;
        std::cout << "Path deviation: " << metrics.path_deviation << " m" << std::endl;

        lbfgs::lbfgs_parameter_t lbfgs_params;
        lbfgs_params.mem_size = params_.mem_size;
        lbfgs_params.past = params_.past;
        lbfgs_params.g_epsilon = params_.g_epsilon;
        lbfgs_params.min_step = params_.min_step;
        lbfgs_params.delta = params_.delta;
        lbfgs_params.max_iterations = params_.max_iter;

        double final_cost;
        int result = lbfgs::lbfgs_optimize(
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

    double costFunction(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        double cost = 0.0;

        Eigen::Map<const Eigen::VectorXd> tau(x.data(), dim_T);
        Eigen::Map<Eigen::VectorXd> gradTau(grad.data(), dim_T);
        Eigen::Map<const Eigen::MatrixXd> Ppos(x.data() + dim_T, 2, piece_pos_ - 1);
        Eigen::Map<Eigen::MatrixXd> gradPpos(grad.data() + dim_T, 2, piece_pos_ - 1);
     
        Eigen::VectorXd Tpos;
        Tpos.resize(piece_pos_);
        calTfromTau(tau, Tpos);
        generateTrajectory(init_pos_, end_pos_, Ppos, Tpos);

        double constrain_cost = 0.0;
        Eigen::MatrixXd gdCpos_constrain;
        Eigen::VectorXd gdTpos_constrain;
        calculateConstraintCostGrad(trajectory_, constrain_cost, gdCpos_constrain, gdTpos_constrain);
        
        Eigen::MatrixXd gradPpos_constrain;
        Eigen::VectorXd gradTpos_constrain;
        calGradCTtoQT(gdCpos_constrain, gdTpos_constrain, gradPpos_constrain, gradTpos_constrain);
        
        double energy = quintic_spline_.getEnergy();
        double energy_cost = params_.rho_energy * energy;
        
        QuinticSpline2D::MatrixType gradP_energy = quintic_spline_.getEnergyGradInnerP();
        Eigen::VectorXd gradT_energy = quintic_spline_.getEnergyGradTimes();
        
        gradPpos = gradPpos_constrain + params_.rho_energy * gradP_energy.transpose();
        Eigen::VectorXd gradTpos_total = gradTpos_constrain + params_.rho_energy * gradT_energy;

        double tau_cost = params_.rho_T * Tpos.sum();
        gradTpos_total.array() += params_.rho_T;
        Eigen::VectorXd grad_tau(dim_T);
        calGradtfromT(tau, gradTpos_total, grad_tau);
        gradTau = grad_tau;

        cost = constrain_cost + energy_cost + tau_cost;
        return cost;
    }

    void calculateConstraintCostGrad(
        PPoly2D& traj,
        double& cost,
        Eigen::MatrixXd& gdCpos,
        Eigen::VectorXd& gdTpos)
    {
        cost = 0.0;
        double v_cost = 0.0;
        double a_cost = 0.0;
        double occ_cost = 0.0;
        double corridor_cost = 0.0;
        
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
        Eigen::Vector2d grad_corridor = Eigen::Vector2d::Zero();
        Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
        double s1, s2, s3, s4, s5;
        double step, alpha, omg;

        for (int i = 0; i < N; ++i) {
            const Eigen::Matrix<double, 6, 2>& c = coeffs.block<6, 2>(i * 6, 0);
            step = (breaks[i+1] - breaks[i]) / params_.int_K;
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

                // 1. Velocity constraint
                double vxy_snorm = vel.squaredNorm();
                double vViola = vxy_snorm - params_.max_v * params_.max_v;
                if (vViola > 0) {
                    grad_v += params_.rho_v * 6 * vViola * vViola * vel;
                    double cost_v = params_.rho_v * vViola * vViola * vViola;
                    cost += cost_v * omg * step;
                    v_cost += cost_v * omg * step;
                    grad_time += omg * (cost_v / params_.int_K + step * alpha * grad_v.dot(acc));
                }

                // 2. Acceleration constraint
                double axy_snorm = acc.squaredNorm();
                double aViola = axy_snorm - params_.max_a * params_.max_a;
                if (aViola > 0) {
                    grad_a += params_.rho_a * 6 * aViola * aViola * acc;
                    double cost_a = params_.rho_a * aViola * aViola * aViola;
                    cost += cost_a * omg * step;
                    a_cost += cost_a * omg * step;
                    grad_time += omg * (cost_a / params_.int_K + step * alpha * grad_a.dot(jerk));
                }

                // 3. Collision constraint
                if (env_) {
                    double sdf_value;
                    if (env_->getDistanceAndGradient(pos, sdf_value, grad_sdf)) {
                        double cViola = params_.safe_threshold - sdf_value;
                        if (cViola > 0 && sdf_value < 5) {
                            double penalty;
                            Eigen::Vector2d grad_pc;
                            
                            if (cViola < 0.1) {
                                penalty = cViola * cViola;
                                grad_pc = -2.0 * cViola * grad_sdf;
                            } else {
                                penalty = cViola;
                                grad_pc = -grad_sdf;
                            }
                            
                            double cost_c = params_.rho_collision * penalty;
                            Eigen::Vector2d grad_pc_scaled = params_.rho_collision * grad_pc;
                            
                            cost += cost_c * omg * step;
                            occ_cost += cost_c * omg * step;
                            grad_time += omg * (cost_c / params_.int_K + step * alpha * grad_pc_scaled.dot(vel));
                            grad_p += grad_pc_scaled;
                        }
                    }

                    double corridor_viola;
                    if (env_->getSafeCorridorViolation(i, pos, corridor_viola, grad_corridor) &&
                        corridor_viola > 0.0) {
                        const double cost_sc = params_.rho_corridor * corridor_viola * corridor_viola;
                        const Eigen::Vector2d grad_sc = params_.rho_corridor * 2.0 * corridor_viola * grad_corridor;
                        cost += cost_sc * omg * step;
                        corridor_cost += cost_sc * omg * step;
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
                  << ", Corridor: " << corridor_cost << std::endl;
    }

    void calGradCTtoQT(
        const Eigen::MatrixXd& gdCpos,
        const Eigen::VectorXd& gdTpos,
        Eigen::MatrixXd& gradPpos,
        Eigen::VectorXd& gradTpos_out)
    {
        QuinticSpline2D::MatrixType gdC_typed = gdCpos;
        QuinticSpline2D::MatrixType gradByPoints;
        Eigen::VectorXd gradByTimes;
        
        quintic_spline_.propagateGrad(gdC_typed, gdTpos, gradByPoints, gradByTimes);
        
        gradPpos = gradByPoints.transpose();
        gradTpos_out = gradByTimes;
        
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

        SplineTrajectory::SplineVector2D waypoints;
        waypoints.push_back(initPos.col(0));
        for (int i = 0; i < innerPts.cols(); ++i) {
            waypoints.push_back(innerPts.col(i));
        }
        waypoints.push_back(endPos.col(0));

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
};

} // namespace TrajOpt

#endif // TRAJ_OPT_HPP
