#include <ceres/ceres.h>

#include <iostream>

struct NumericDiffCostFunctor {
  bool operator()(const double* const x, double* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  double initial_x = 5.0;
  double x = initial_x;

  ceres::Problem problem;
  ceres::CostFunction* cost_unction =
      new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL,
                                         1, 1>(new NumericDiffCostFunctor);
  problem.AddResidualBlock(cost_unction, nullptr, &x);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "=====" << std::endl;
  std::cout << summary.BriefReport() << std::endl;
  std::cout << "x: " << initial_x << " -> " << x << std::endl;
  return 0;
}
