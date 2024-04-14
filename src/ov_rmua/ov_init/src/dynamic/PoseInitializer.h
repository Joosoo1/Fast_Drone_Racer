#ifndef OV_INIT_POSEINITIALIZER_H
#define OV_INIT_POSEINITIALIZER_H

#include "init/InertialInitializerOptions.h"

namespace ov_core {
class FeatureDatabase;
struct ImuData;
struct PoseData;
} // namespace ov_core
namespace ov_type {
class Type;
class IMU;
class Landmark;
} // namespace ov_type

namespace ov_init {

class PoseInitializer {
public:
  explicit PoseInitializer(InertialInitializerOptions &params_, std::shared_ptr<ov_core::FeatureDatabase> db,
                           std::shared_ptr<std::vector<ov_core::ImuData>> imu_data_,
                           std::shared_ptr<std::map<double, Eigen::Matrix<double, 7, 1>>> pose_data_)
      : params(params_), _db(db), imu_data(imu_data_), pose_data(pose_data_) {}

  bool initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                  std::shared_ptr<ov_type::IMU> t_imu);

private:
  /// Initialization parameters
  InertialInitializerOptions params;

  /// Feature tracker database with all features in it
  std::shared_ptr<ov_core::FeatureDatabase> _db;

  /// Our history of IMU messages (time, angular, linear)
  std::shared_ptr<std::vector<ov_core::ImuData>> imu_data;

  /// Our history of pose estimated by FCU (time, pose)
  std::shared_ptr<std::map<double, Eigen::Matrix<double, 7, 1>>> pose_data;
};

} // namespace ov_init

#endif // OV_INIT_POSEINITIALIZER_H