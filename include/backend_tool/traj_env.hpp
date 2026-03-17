#pragma once

#include <Eigen/Dense>
#include <limits>
#include <memory>
#include <vector>

namespace TrajOpt {

class TrajectoryEnv {
public:
    virtual ~TrajectoryEnv() = default;

    virtual bool getDistanceAndGradient(const Eigen::Vector2d& pos,
                                        double& distance,
                                        Eigen::Vector2d& gradient) const {
        distance = std::numeric_limits<double>::max();
        gradient.setZero();
        return false;
    }

    virtual bool getSafeCorridorViolation(const int piece_idx,
                                          const Eigen::Vector2d& pos,
                                          double& violation,
                                          Eigen::Vector2d& gradient) const {
        (void)piece_idx;
        (void)pos;
        violation = 0.0;
        gradient.setZero();
        return false;
    }
};

struct CorridorHalfSpace2D {
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double offset = 0.0;  // normal.dot(pos) <= offset
};

using CorridorPiece2D = std::vector<CorridorHalfSpace2D>;

class SafeCorridorEnv : public TrajectoryEnv {
public:
    SafeCorridorEnv() = default;

    explicit SafeCorridorEnv(const std::vector<CorridorPiece2D>& corridor)
        : corridor_(corridor) {}

    void setCorridor(const std::vector<CorridorPiece2D>& corridor) {
        corridor_ = corridor;
    }

    bool empty() const { return corridor_.empty(); }

    bool getSafeCorridorViolation(const int piece_idx,
                                  const Eigen::Vector2d& pos,
                                  double& violation,
                                  Eigen::Vector2d& gradient) const override {
        if (corridor_.empty()) {
            violation = 0.0;
            gradient.setZero();
            return false;
        }

        const int seg_idx = std::max(0, std::min(piece_idx, static_cast<int>(corridor_.size()) - 1));
        const CorridorPiece2D& piece = corridor_[seg_idx];
        if (piece.empty()) {
            violation = 0.0;
            gradient.setZero();
            return false;
        }

        violation = 0.0;
        gradient.setZero();
        for (const auto& half_space : piece) {
            const double cur_violation = half_space.normal.dot(pos) - half_space.offset;
            if (cur_violation > violation) {
                violation = cur_violation;
                gradient = half_space.normal;
            }
        }
        return violation > 0.0;
    }

private:
    std::vector<CorridorPiece2D> corridor_;
};

}  // namespace TrajOpt
