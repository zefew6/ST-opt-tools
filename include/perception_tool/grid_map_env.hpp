#pragma once

#include "grid_map.hpp"
#include "traj_env.hpp"

namespace TrajOpt {

class GridMapEnv : public TrajectoryEnv {
public:
    explicit GridMapEnv(std::shared_ptr<grid_map::GridMap> map)
        : map_(std::move(map)) {}

    bool getDistanceAndGradient(const Eigen::Vector2d& pos,
                                double& distance,
                                Eigen::Vector2d& gradient) const override {
        if (!map_) {
            distance = std::numeric_limits<double>::max();
            gradient.setZero();
            return false;
        }
        return map_->getDistanceAndGradient(pos, distance, gradient);
    }

    std::shared_ptr<grid_map::GridMap> getMap() const { return map_; }

private:
    std::shared_ptr<grid_map::GridMap> map_;
};

}  // namespace TrajOpt
