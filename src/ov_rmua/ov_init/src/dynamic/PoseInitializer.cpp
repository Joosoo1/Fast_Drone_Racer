#include "PoseInitializer.h"

#include "feat/FeatureHelper.h"
#include "types/IMU.h"
#include "utils/helper.h"
#include "utils/sensor_data.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_init;

bool PoseInitializer::initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                                 std::shared_ptr<ov_type::IMU> t_imu) {

  /// Find the newest image frame as k time instant
  double newest_cam_time = -1;
  double oldest_cam_time = INFINITY;
  for (auto const &feat : _db->get_internal_data()) {
    for (auto const &camtimepair : feat.second->timestamps) {
      for (auto const &time : camtimepair.second) {
        newest_cam_time = std::max(newest_cam_time, time);
        oldest_cam_time = std::min(oldest_cam_time, time);
      }
    }
  }

  // Or we can just set a fixed length
  double oldest_pose_time = newest_cam_time - 0.5;
  bool found_oldest_pose = false;
  auto lower_bound_oldest = pose_data->lower_bound(oldest_pose_time);
  if (lower_bound_oldest != pose_data->end()) {
    if (lower_bound_oldest->first == oldest_pose_time) {
      found_oldest_pose = true;
    } else if (lower_bound_oldest != pose_data->begin()) {
      --lower_bound_oldest;
      found_oldest_pose = true;
    }
  }

  if (found_oldest_pose) {
    oldest_pose_time = lower_bound_oldest->first;
    auto it0 = pose_data->begin();
    while (it0 != pose_data->end()) {
      if (it0->first < oldest_pose_time) {
        it0 = pose_data->erase(it0);
      } else {
        it0++;
      }
    }
  }

  // std::cout << "newest_cam_time: " << std::fixed << std::setprecision(3) << newest_cam_time << std::endl;
  // std::cout << "oldest_pose_time: " << std::fixed << std::setprecision(3) << oldest_pose_time << std::endl;

  /// Interpolate the pose at the newest time using two bounding poses estimated by FCU

  auto lower_bound = pose_data->lower_bound(newest_cam_time);
  auto upper_bound = pose_data->upper_bound(newest_cam_time);

  bool found_older = false;
  bool found_newer = false;

  if (lower_bound != pose_data->end()) {
    if (lower_bound->first == newest_cam_time) {
      found_older = true;
    } else if (lower_bound != pose_data->begin()) {
      --lower_bound;
      found_older = true;
    }
  }

  double t0 = -1;
  double t1 = -1;

  Eigen::Matrix<double, 7, 1> pose0, pose1;
  pose0(3) = 1;
  pose1(3) = 1;

  if (upper_bound != pose_data->end()) {
    found_newer = true;
  }

  if (found_older) {
    t0 = lower_bound->first;
    pose0 = lower_bound->second;
  }

  if (found_newer) {
    t1 = upper_bound->first;
    pose1 = upper_bound->second;
  }

  if (found_older && found_newer) {
    assert(t0 < t1);
  } else {
    return false;
    std::cout << "didn't find the bounding poses" << std::endl;
  }

  Eigen::Matrix<double, 3, 3> R_GtoI_0 = quat_2_Rot(pose0.block(0, 0, 4, 1));
  Eigen::Matrix<double, 3, 3> R_GtoI_1 = quat_2_Rot(pose1.block(0, 0, 4, 1));
  Eigen::Matrix<double, 3, 1> p_IinG_0 = pose0.block(4, 0, 3, 1);
  Eigen::Matrix<double, 3, 1> p_IinG_1 = pose1.block(4, 0, 3, 1);
  double lambda = (newest_cam_time - t0) / (t1 - t0);
  Eigen::Matrix<double, 3, 1> p_IinG = (1 - lambda) * p_IinG_0 + lambda * p_IinG_1;
  Eigen::Matrix<double, 3, 1> theta_0to1 = log_so3(R_GtoI_1 * R_GtoI_0.transpose());
  Eigen::Matrix<double, 3, 3> R_GtoI = exp_so3(lambda * theta_0to1) * R_GtoI_0;
  Eigen::MatrixXd newest_cam_pose = Eigen::MatrixXd::Zero(7, 1);
  newest_cam_pose.block(0, 0, 4, 1) = rot_2_quat(R_GtoI);
  newest_cam_pose.block(4, 0, 3, 1) = p_IinG;
  pose_data->insert({newest_cam_time, newest_cam_pose});
  // std::cout << "size of pose_data: " << pose_data->size() << std::endl;
  // std::cout << "p_IinG: " << p_IinG << std::endl;

  /// Construct a linear system to solve v_I0 in {G} frame
  /// First the collect imu readings in the initialization window
  std::vector<ov_core::ImuData> readings = InitializerHelper::select_imu_readings(*imu_data, oldest_pose_time, newest_cam_time);
  assert(readings.size() > 2);
  // std::cout << "size of imu readings: " << readings.size() << std::endl;
  // First integrate from the first to the last
  std::map<double, bool> map_pose_times;
  std::map<double, std::shared_ptr<ov_core::CpiV1>> map_pose_cpi_I0toIi;
  std::map<double, bool> valid_pose_time;
  valid_pose_time[newest_cam_time] = true;
  for (auto const pose : *pose_data) {
    // No preintegration at the first timestamp
    double current_time = pose.first;
    if (current_time == oldest_pose_time) {
      map_pose_cpi_I0toIi.insert({current_time, nullptr});
      continue;
    }
    if (current_time > newest_cam_time)
      continue;
    // Perform integration from I0 to Ii
    double cpiI0toIi1_time0_in_imu = oldest_pose_time;
    double cpiI0toIi1_time1_in_imu = current_time;
    auto cpiI0toIi1 = std::make_shared<ov_core::CpiV1>(params.sigma_w, params.sigma_wb, params.sigma_a, params.sigma_ab, true);
    cpiI0toIi1->setLinearizationPoints(params.init_dyn_bias_g, params.init_dyn_bias_a);
    std::vector<ov_core::ImuData> cpiI0toIi1_readings =
        InitializerHelper::select_imu_readings(*imu_data, cpiI0toIi1_time0_in_imu, cpiI0toIi1_time1_in_imu);
    if (cpiI0toIi1_readings.size() < 2) {
      PRINT_DEBUG(YELLOW "[init-d]: camera %.2f in has %zu IMU readings!\n" RESET, (cpiI0toIi1_time1_in_imu - cpiI0toIi1_time0_in_imu),
                  cpiI0toIi1_readings.size());
      std::cout << "cpiI0toIi1_readings.size() < 2" << std::endl;
      return false;
    }
    // std::cout << "cpiI0toIi1_readings size: " << cpiI0toIi1_readings.size() << std::endl;
    double cpiI0toIi1_dt_imu = cpiI0toIi1_readings.at(cpiI0toIi1_readings.size() - 1).timestamp - cpiI0toIi1_readings.at(0).timestamp;
    if (std::abs(cpiI0toIi1_dt_imu - (cpiI0toIi1_time1_in_imu - cpiI0toIi1_time0_in_imu)) > 0.01) {
      PRINT_DEBUG(YELLOW "[init-d]: camera IMU was only propagated %.3f of %.3f\n" RESET, cpiI0toIi1_dt_imu,
                  (cpiI0toIi1_time1_in_imu - cpiI0toIi1_time0_in_imu));
      std::cout << "std::abs(cpiI0toIi1_dt_imu - (cpiI0toIi1_time1_in_imu - cpiI0toIi1_time0_in_imu)) > 0.01" << std::endl;
      return false;
    }
    for (size_t k = 0; k < cpiI0toIi1_readings.size() - 1; k++) {
      auto imu0 = cpiI0toIi1_readings.at(k);
      auto imu1 = cpiI0toIi1_readings.at(k + 1);
      cpiI0toIi1->feed_IMU(imu0.timestamp, imu1.timestamp, imu0.wm, imu0.am, imu1.wm, imu1.am);
    }
    map_pose_cpi_I0toIi.insert({current_time, cpiI0toIi1});
    valid_pose_time.insert({current_time, true});
  }

  // Construct a linear system to solve the velocity
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * valid_pose_time.size(), 3);
  Eigen::MatrixXd b = Eigen::VectorXd::Zero(3 * valid_pose_time.size());
  Eigen::Vector3d gravity;
  gravity << 0.0, 0.0, params.gravity_mag;
  Eigen::Matrix<double, 7, 1> pose_0 = pose_data->at(oldest_pose_time);
  Eigen::Vector3d p_I0inG = pose_0.block(4, 0, 3, 1);
  Eigen::Matrix3d R_GtoI0 = quat_2_Rot(pose_0.block(0, 0, 4, 1));

  int index_meas = 0;
  for (auto const timepair : valid_pose_time) {
    double time_pose = timepair.first;
    double DT = time_pose - oldest_pose_time;
    Eigen::MatrixXd alpha_I0toIk = map_pose_cpi_I0toIi.at(time_pose)->alpha_tau;
    auto pose_k = pose_data->at(time_pose);
    Eigen::Vector3d p_IkinG = pose_k.block(4, 0, 3, 1);
    A.block(index_meas, 0, 3, 3) = DT * Eigen::MatrixXd::Identity(3, 3);
    b.block(index_meas, 0, 3, 1) = p_IkinG - p_I0inG + 0.5 * gravity * DT * DT - R_GtoI0.transpose() * alpha_I0toIk;
    index_meas += 3;
  }

  Eigen::MatrixXd AtA = A.transpose() * A;
  Eigen::MatrixXd Atb = A.transpose() * b;
  Eigen::MatrixXd v_I0inG = AtA.colPivHouseholderQr().solve(Atb);
  std::cout << "v_I0inG: " << v_I0inG.transpose() << std::endl;

  /// Integrate the velocity to k time
  double DTk = newest_cam_time - oldest_pose_time;
  auto cpi_k = map_pose_cpi_I0toIi.at(newest_cam_time);
  Eigen::MatrixXd beta_I0toIk = cpi_k->beta_tau;
  Eigen::Vector3d v_IkinG = v_I0inG - gravity * DTk + R_GtoI0.transpose() * beta_I0toIk;
  std::cout << "v_IkinG: " << v_IkinG.transpose() << std::endl;
  std::cout << "p_IinG.z(): " << p_IinG.z() << std::endl;

  // if (v_IkinG.norm() > 3.0 && p_IinG.z() > 35) {
  if (v_IkinG.norm() > 0.1) {
    /// Set the imu state and covariance, finish the initialization
    timestamp = newest_cam_time;
    Eigen::VectorXd imu_state = Eigen::VectorXd::Zero(16);
    imu_state.block(0, 0, 4, 1) = rot_2_quat(R_GtoI);
    imu_state.block(4, 0, 3, 1) = p_IinG;
    imu_state.block(7, 0, 3, 1) = v_IkinG;
    imu_state.block(10, 0, 3, 1) = params.init_dyn_bias_g;
    imu_state.block(13, 0, 3, 1) = params.init_dyn_bias_a;
    assert(t_imu != nullptr);
    t_imu->set_value(imu_state);
    t_imu->set_fej(imu_state);
    // Create base covariance and its covariance ordering
    order.clear();
    order.push_back(t_imu);
    covariance = std::pow(0.02, 2) * Eigen::MatrixXd::Identity(t_imu->size(), t_imu->size());
    covariance.block(0, 0, 3, 3) = std::pow(0.02, 2) * Eigen::Matrix3d::Identity(); // q
    covariance.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity(); // p
    covariance.block(6, 6, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity(); // v
    return true;
  } else {
    PRINT_INFO("Not fast enough to initialize the state\n");
    return false;
  }
}