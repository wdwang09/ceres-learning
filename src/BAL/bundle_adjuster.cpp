#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "./bal_problem.h"

struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x_(observed_x), observed_y_(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera, const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x_;
    residuals[1] = predicted_y - observed_y_;

    return true;
  }

  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y));
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

struct SnavelyReprojectionErrorWithQuaternions {
  SnavelyReprojectionErrorWithQuaternions(double observed_x, double observed_y)
      : observed_x_(observed_x), observed_y_(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera, const T* const point,
                  T* residuals) const {
    // camera[0,1,2,3] is are the rotation of the camera as a quaternion.
    //
    // We use QuaternionRotatePoint as it does not assume that the
    // quaternion is normalized, since one of the ways to run the
    // bundle adjuster is to let Ceres optimize all 4 quaternion
    // parameters without using a Quaternion manifold.
    T p[3];
    ceres::QuaternionRotatePoint(camera, point, p);

    p[0] += camera[4];
    p[1] += camera[5];
    p[2] += camera[6];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[8];
    const T& l2 = camera[9];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[7];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x_;
    residuals[1] = predicted_y - observed_y_;

    return true;
  }

  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return new ceres::AutoDiffCostFunction<
        SnavelyReprojectionErrorWithQuaternions, 2, 10, 3>(
        new SnavelyReprojectionErrorWithQuaternions(observed_x, observed_y));
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

void solve_problem(const char* filename) {
  bool is_using_quaternions = false;
  ceres_examples::BALProblem bal_problem(filename, is_using_quaternions);
  // bal_problem.Normalize();
  // bal_problem.Perturb(0.1, 0.5, 0.5);

  // std::cout << std::filesystem::current_path() << std::endl;
  bal_problem.WriteToPLYFile("before.ply");

  const int point_block_size = bal_problem.point_block_size();    // 9 or 10
  const int camera_block_size = bal_problem.camera_block_size();  // 3
  double* points = bal_problem.mutable_points();    // 3 * num_points_
  double* cameras = bal_problem.mutable_cameras();  // 9 * num_cameras_

  // Observations is 2*num_observations long array observations =
  // [u_1, u_2, ... , u_n], where each u_i is two dimensional, the x
  // and y positions of the observation.
  const double* observations = bal_problem.observations();

  // add residual blocks
  ceres::Problem ceres_problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    ceres::CostFunction* cf;
    if (is_using_quaternions) {
      cf = SnavelyReprojectionErrorWithQuaternions::Create(
          observations[2 * i + 0], observations[2 * i + 1]);
    } else {
      cf = SnavelyReprojectionError::Create(observations[2 * i + 0],
                                            observations[2 * i + 1]);
    }
    double* camera =
        cameras + camera_block_size * bal_problem.camera_index()[i];
    double* point = points + point_block_size * bal_problem.point_index()[i];
    ceres_problem.AddResidualBlock(cf, new ceres::HuberLoss(1.0), camera,
                                   point);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.max_num_iterations = 20;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &ceres_problem, &summary);

  std::cout << summary.BriefReport() << std::endl;
  bal_problem.WriteToPLYFile("after.ply");
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(FATAL) << "Need BAL file in argv[1]!";
    return 1;
  }
  solve_problem(argv[1]);
  return 0;
}
