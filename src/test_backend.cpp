/*
    MIT License

    Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)
*/

#include "traj_env.hpp"
#include "traj_opt.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

int main() {
    using TrajOpt::CorridorHalfSpace2D;
    using TrajOpt::CorridorPiece2D;
    using TrajOpt::SafeCorridorEnv;
    using TrajOpt::TrajectoryOptimizer;
    using TrajOpt::TrajectoryParams;

    Eigen::Matrix2d init_state;
    Eigen::Matrix2d end_state;
    init_state.col(0) << 0.0, 0.0;
    init_state.col(1) << 0.6, 0.0;
    end_state.col(0) << 6.0, 2.0;
    end_state.col(1) << 0.4, 0.0;

    Eigen::MatrixXd inner_pts(2, 2);
    inner_pts.col(0) << 2.0, 0.4;
    inner_pts.col(1) << 4.0, 1.6;

    Eigen::VectorXd init_T(3);
    init_T << 2.0, 2.2, 2.0;

    std::vector<CorridorPiece2D> corridor(3);
    corridor[0] = {
        CorridorHalfSpace2D{Eigen::Vector2d(1.0, 0.0), 2.4},
        CorridorHalfSpace2D{Eigen::Vector2d(-1.0, 0.0), 0.2},
        CorridorHalfSpace2D{Eigen::Vector2d(0.0, 1.0), 0.8},
        CorridorHalfSpace2D{Eigen::Vector2d(0.0, -1.0), 0.6}
    };
    corridor[1] = {
        CorridorHalfSpace2D{Eigen::Vector2d(1.0, 0.0), 4.4},
        CorridorHalfSpace2D{Eigen::Vector2d(-1.0, 0.0), -1.6},
        CorridorHalfSpace2D{Eigen::Vector2d(0.0, 1.0), 2.0},
        CorridorHalfSpace2D{Eigen::Vector2d(0.0, -1.0), -0.1}
    };
    corridor[2] = {
        CorridorHalfSpace2D{Eigen::Vector2d(1.0, 0.0), 6.2},
        CorridorHalfSpace2D{Eigen::Vector2d(-1.0, 0.0), -3.6},
        CorridorHalfSpace2D{Eigen::Vector2d(0.0, 1.0), 2.4},
        CorridorHalfSpace2D{Eigen::Vector2d(0.0, -1.0), -1.0}
    };

    auto env = std::make_shared<SafeCorridorEnv>(corridor);

    TrajectoryParams params;
    params.rho_corridor = 50000.0;
    params.rho_energy = 200.0;
    params.rho_T = 20.0;
    params.max_v = 2.0;
    params.max_a = 2.0;
    params.int_K = 24;

    TrajectoryOptimizer optimizer(params);
    optimizer.setEnvironment(env);

    if (!optimizer.optimize(init_state, inner_pts, end_state, init_T)) {
        std::cerr << "Trajectory optimization failed, result = "
                  << optimizer.getLastOptResult() << std::endl;
        return -1;
    }

    const auto metrics = optimizer.evaluateTrajectory();
    std::cout << "=== Pure Backend Demo ===" << std::endl;
    std::cout << "Result code: " << optimizer.getLastOptResult() << std::endl;
    std::cout << "Final cost: " << optimizer.getLastOptCost() << std::endl;
    std::cout << "Total time: " << metrics.total_time << " s" << std::endl;
    std::cout << "Max velocity: " << metrics.max_velocity << " m/s" << std::endl;
    std::cout << "Max acceleration: " << metrics.max_acceleration << " m/s^2" << std::endl;

    const auto sample_path = optimizer.sampleTrajectory(0.5);
    std::cout << "Sampled trajectory points:" << std::endl;
    for (size_t i = 0; i < sample_path.size(); ++i) {
        std::cout << i << ": " << sample_path[i].transpose() << std::endl;
    }

    return 0;
}
