#include <ceres/ceres.h>

#include <iostream>

struct QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
  ~QuadraticCostFunction() override = default;
  bool Evaluate(double const* const* parameters, double* residual,
                double** jacobians) const override {
    const double x = parameters[0][0];
    residual[0] = 10 - x;

    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  double initial_x = 5.0;
  double x = initial_x;

  ceres::Problem problem;
  ceres::CostFunction* cost_function = new QuadraticCostFunction;
  problem.AddResidualBlock(cost_function, nullptr, &x);

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
