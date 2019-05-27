//
// Created by timian on 12/19/18.
//

#include "linearReg.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;


struct CostFunctor {
    CostFunctor(double x, double y)
            : x_(x), y_(y) {}

    template<typename T>
    bool operator()(const T *const a, const T *const b, T *residual) const {
        residual[0] = T(y_) - (a[0] * T(x_) + b[0]);
        return true;
    }

private:
    const double x_;
    const double y_;
};

linearReg:: linearReg(std::vector<std::pair<double, int>> inputData) {
    type = LR;
    data = std::move(inputData);
}

auto dataNeedsBtree = 0;
auto k = 0;

void linearReg::train() {
    double kNumObservations = data.size();
    double error = 0;
    Problem problem;

    for (int i = 0; i < kNumObservations; ++i) {
        problem.AddResidualBlock(
                new AutoDiffCostFunction<CostFunctor, 1, 1, 1>(
                        new CostFunctor(data[i].first, data[i].second)),
                NULL,
                &a, &b);
    }

    Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    for (auto key : data) {

        double prediction = predict(key.first);
        double calcError = abs(prediction - key.second);
        if (calcError > error) {
            error = calcError;
        }
    }
//    if (error > 128) {
//        k++;
//        dataNeedsBtree += data.size();
//        std::cout << k << " with error above 128 \n";
//        std::cout << dataNeedsBtree << " Datapoints need to be indexed by b-tree \n";
//
//    }

}

float linearReg::predict(float key) {
    return a * key + b;
}

