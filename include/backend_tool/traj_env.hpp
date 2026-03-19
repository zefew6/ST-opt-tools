#pragma once

#include <Eigen/Dense>
#include <limits>
#include <memory>
#include <vector>

namespace TrajOpt {

struct CorridorHalfSpace2D {
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double offset = 0.0;  // normal.dot(pos) <= offset
};

using CorridorPiece2D = std::vector<CorridorHalfSpace2D>;
using SFCHalfSpace2D = CorridorHalfSpace2D;
using SFCPiece2D = CorridorPiece2D;

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

    virtual bool getSFCViolation(const int piece_idx,
                                 const Eigen::Vector2d& pos,
                                 double& violation,
                                 Eigen::Vector2d& gradient) const {
        (void)piece_idx;
        (void)pos;
        violation = 0.0;
        gradient.setZero();
        return false;
    }

    virtual bool getSafeCorridorViolation(const int piece_idx,
                                          const Eigen::Vector2d& pos,
                                          double& violation,
                                          Eigen::Vector2d& gradient) const {
        return getSFCViolation(piece_idx, pos, violation, gradient);
    }

    virtual const std::vector<SFCPiece2D>* getSFCPtr() const {
        return nullptr;
    }
};

class SafeCorridorEnv : public TrajectoryEnv {
public:
    SafeCorridorEnv() = default;

    explicit SafeCorridorEnv(const std::vector<CorridorPiece2D>& corridor)
        : corridor_(corridor) {}

    void setCorridor(const std::vector<CorridorPiece2D>& corridor) {
        corridor_ = corridor;
    }

    void setSFC(const std::vector<SFCPiece2D>& sfc) {
        corridor_ = sfc;
    }

    const std::vector<CorridorPiece2D>& getCorridor() const {
        return corridor_;
    }

    const std::vector<SFCPiece2D>& getSFC() const {
        return corridor_;
    }

    bool empty() const { return corridor_.empty(); }

    bool getSFCViolation(const int piece_idx,
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

    const std::vector<SFCPiece2D>* getSFCPtr() const override {
        return &corridor_;
    }

private:
    std::vector<CorridorPiece2D> corridor_;
};

using SafeSFCEnv = SafeCorridorEnv;

class CompositeEnv : public TrajectoryEnv {
public:
    CompositeEnv() = default;

    CompositeEnv(std::shared_ptr<TrajectoryEnv> distance_env,
                 std::shared_ptr<TrajectoryEnv> corridor_env)
        : distance_env_(std::move(distance_env)),
          corridor_env_(std::move(corridor_env)) {}

    void setDistanceEnv(std::shared_ptr<TrajectoryEnv> env) {
        distance_env_ = std::move(env);
    }

    void setCorridorEnv(std::shared_ptr<TrajectoryEnv> env) {
        corridor_env_ = std::move(env);
    }

    void setSFCEnv(std::shared_ptr<TrajectoryEnv> env) {
        corridor_env_ = std::move(env);
    }

    const std::shared_ptr<TrajectoryEnv>& getSFCEnv() const {
        return corridor_env_;
    }

    bool getDistanceAndGradient(const Eigen::Vector2d& pos,
                                double& distance,
                                Eigen::Vector2d& gradient) const override {
        if (!distance_env_) {
            distance = std::numeric_limits<double>::max();
            gradient.setZero();
            return false;
        }
        return distance_env_->getDistanceAndGradient(pos, distance, gradient);
    }

    bool getSFCViolation(const int piece_idx,
                         const Eigen::Vector2d& pos,
                         double& violation,
                         Eigen::Vector2d& gradient) const override {
        if (!corridor_env_) {
            violation = 0.0;
            gradient.setZero();
            return false;
        }
        return corridor_env_->getSFCViolation(piece_idx, pos, violation, gradient);
    }

    const std::vector<SFCPiece2D>* getSFCPtr() const override {
        if (!corridor_env_) {
            return nullptr;
        }
        return corridor_env_->getSFCPtr();
    }

private:
    std::shared_ptr<TrajectoryEnv> distance_env_;
    std::shared_ptr<TrajectoryEnv> corridor_env_;
};

}  // namespace TrajOpt
