#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


//______________________________________________________________________________________________________________________
// Пример из http://ceres-solver.org/nnls_modeling.html#introduction
// 0.5*(10-x)^2
//______________________________________________________________________________________________________________________

class CostFunctor1 {
public:
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};

TEST (CeresSolver, HelloWorld1) {
    double initial_x = 5.0;
    double cur_x = initial_x;

    CostFunctor1 *f = new CostFunctor1();
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor1, 1, 1>(f);
    ceres::LossFunction* loss_function = new ceres::TrivialLoss();

    ceres::Problem problem;
    problem.AddResidualBlock(cost_function, loss_function, &cur_x);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    const int N_RESIDUAL_BLOCKS = 1;
    double* params[N_RESIDUAL_BLOCKS];
    double* jacobians[N_RESIDUAL_BLOCKS];

    double initial_residual = 0.0;
    double initial_jacobian = 0.0;
    params[0] = &initial_x;
    jacobians[0] = &initial_jacobian;
    cost_function->Evaluate(params, &initial_residual, jacobians);

    double final_residual = 0.0;
    double final_jacobian = 0.0;
    params[0] = &cur_x;
    jacobians[0] = &final_jacobian;
    cost_function->Evaluate(params, &final_residual, jacobians);

    std::cout << "x:     " << initial_x        << " -> " << cur_x << std::endl;
    std::cout << "f(x):  " << initial_residual << " -> " << final_residual << std::endl;
    std::cout << "f'(x): " << initial_jacobian << " -> " << final_jacobian << std::endl;

    ASSERT_NEAR(cur_x, 10.0, 1e-6);
}

//______________________________________________________________________________________________________________________
// Пусть есть два фиксированных 3D объекта - параболоид и прямая.
// Хотим найти их точку пересечения. Да, это из пушки по воробьям, но как иллюстрация для тренировки - полезно :)
// Значит у нас две невязки (Residual) - расстояние до параболоида и до прямой.
// И всего один блок параметров состоящий из трех чисел - (x,y,z) - координаты точки (искомого пересечения).
//______________________________________________________________________________________________________________________

class DistanceToFixedLine {
public:
    DistanceToFixedLine(const double linePoint[3], const double lineDirection[3]) {
        double normal_len2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            normal_len2 += lineDirection[d] * lineDirection[d];
        }
        double normal_len = std::sqrt(normal_len2);

        for (int d = 0; d < 3; ++d) {
            this->linePoint[d] = linePoint[d];
            this->lineDirection[d] = lineDirection[d] / normal_len;
        }
    }

    template <typename T>
    bool operator()(const T* const queryPoint, T* residual) const {
        T linePointToQuery[3];
        T n[3];
        for (int d = 0; d < 3; ++d) {
            linePointToQuery[d] = queryPoint[d] - T(linePoint[d]);
            n[d] = T(lineDirection[d]);
        }
        T crossProduct[3];
        ceres::CrossProduct<T>(linePointToQuery, n, crossProduct);

        T distance = ceres::sqrt(ceres::DotProduct(crossProduct, crossProduct));
        residual[0] = distance;
        return true;
    }

protected:
    double linePoint[3];
    double lineDirection[3];
};

class ResidualToParaboloid {
public:
    ResidualToParaboloid(const double center[3], const double a, const double b) : a(a), b(b) {
        for (int d = 0; d < 3; ++d) {
            this->center[d] = center[d];
        }
    }

    template <typename T>
    bool operator()(const T* const queryPoint, T* residual) const {
        T dx = queryPoint[0] - T(center[0]);
        T dy = queryPoint[1] - T(center[1]);
        residual[0] = T(a) * dx * dx + T(b) * dy * dy + T(center[2]) - queryPoint[2];
        return true;
    }

protected:
    double center[3];
    double a;
    double b;
};

TEST (CeresSolver, HelloWorld2) {
    const double line_point[3]  = {10.0, 5.0, 0.0};
    const double line_direction[3] = {0.0, 0.0, 1.0};
    ceres::CostFunction* line_cost_function = new ceres::AutoDiffCostFunction<DistanceToFixedLine, 1, 3>(
            new DistanceToFixedLine(line_point, line_direction));

    const double paraboloid_center[3] = {5.0, 10.0, 100.0};
    const double paraboloid_a = 2.0;
    const double paraboloid_b = 2.0;
    ceres::CostFunction* paraboloid_cost_function = new ceres::AutoDiffCostFunction<ResidualToParaboloid, 1, 3>(
            new ResidualToParaboloid(paraboloid_center, paraboloid_a, paraboloid_b));

    const double expected_point_solution[3] = {10.0, 5.0, 200.0};
    {
        const double* params[1];
        double residual;
        params[0] = expected_point_solution;

        residual = -1.0;
        line_cost_function->Evaluate(params, &residual, nullptr);
        ASSERT_NEAR(residual, 0.0, 1e-6);

        residual = -1.0;
        paraboloid_cost_function->Evaluate(params, &residual, nullptr);
        ASSERT_NEAR(residual, 0.0, 1e-6);
    }

    double point[3] = {0.0, 0.0, 0.0};

    {
        const double* params[1];
        double residual;
        params[0] = point;

        residual = -1.0;
        line_cost_function->Evaluate(params, &residual, nullptr);
        ASSERT_GT(std::abs(residual), 1.0);

        residual = -1.0;
        paraboloid_cost_function->Evaluate(params, &residual, nullptr);
        ASSERT_GT(std::abs(residual), 1.0);
    }

    ceres::Problem problem;
    problem.AddResidualBlock(line_cost_function, new ceres::TrivialLoss(), point);
    problem.AddResidualBlock(paraboloid_cost_function, new ceres::TrivialLoss(), point);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Found intersection point: (" << point[0] << ", " << point[1] << ", " << point[2] << ")" << std::endl;

    {
        const double* params[1];
        double residual;
        params[0] = point;

        residual = -1.0;
        line_cost_function->Evaluate(params, &residual, nullptr);
        ASSERT_NEAR(residual, 0.0, 1e-6);

        residual = -1.0;
        paraboloid_cost_function->Evaluate(params, &residual, nullptr);
        ASSERT_NEAR(residual, 0.0, 1e-6);
    }

    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(point[d], expected_point_solution[d], 1e-4);
    }
}

//______________________________________________________________________________________________________________________
// Пусть есть сколько-то шумных замеров (потенциально включающих еще и выбросы), хочется их зафиттить прямой
//______________________________________________________________________________________________________________________

typedef std::array<double, 2> double_2;

class PointObservationError {
public:
    explicit PointObservationError(const double_2 point) {
        for (int d = 0; d < 2; ++d) {
            samplePoint[d] = point[d];
        }
    }

    template <typename T>
    bool operator()(const T* const line, T* residual) const {
        residual[0] = (line[0] * T(samplePoint[0]) + line[1] * T(samplePoint[1]) + line[2]) /
                      ceres::sqrt(line[0] * line[0] + line[1] * line[1]);
        return true;
    }

protected:
    double samplePoint[2];
};

double calcLineY(double x, const double* abc) {
    return -(abc[0] * x + abc[2]) / abc[1];
}

double calcDistanceToLine2D(double x, double y, const double* abc) {
    double dist = abc[0] * x + abc[1] * y + abc[2];
    dist /= std::sqrt(abc[0] * abc[0] + abc[1] * abc[1]);
    return dist;
}

void evaluateLine(const std::vector<double_2> &points, const double* line, double sigma,
                  double &fitted_inliers_fraction, double &mean_inliers_distance);

void evaluateLineFitting(double sigma, double &fitted_inliers_fraction, double &mean_inliers_distance,
                         double outliers_fraction = 0.0, bool use_huber = false) {
    const double ideal_line[3] = {0.5, -1.0, 100.0};

    const size_t n_points = 1000;
    const size_t n_points_outliers = static_cast<size_t>(n_points * outliers_fraction);

    std::vector<double_2> points(n_points);

    std::default_random_engine r(212512512391);

    double min_x = -sigma * n_points;
    double max_x =  sigma * n_points;
    double min_y = calcLineY(min_x, ideal_line);
    double max_y = calcLineY(max_x, ideal_line);
    if (min_y > max_y) std::swap(min_y, max_y);
    min_y -= sigma * n_points;
    max_y += sigma * n_points;

    std::uniform_real_distribution<double> uniform_x(min_x, max_x);
    std::uniform_real_distribution<double> uniform_y(min_y, max_y);
    std::normal_distribution<double> sigma_shift(0.0, sigma);

    for (size_t i = 0; i < n_points; ++i) {
        double x = uniform_x(r);
        double y;

        if (i < n_points - n_points_outliers) {
            y = calcLineY(x, ideal_line);

            double shift_distance = sigma_shift(r);

            double line_normal_x = ideal_line[0];
            double line_normal_y = ideal_line[1];
            double line_normal_norm = std::sqrt(line_normal_x * line_normal_x + line_normal_y * line_normal_y);
            line_normal_x /= line_normal_norm;
            line_normal_y /= line_normal_norm;

            x = x + line_normal_x * shift_distance;
            y = y + line_normal_y * shift_distance;
        } else {
            y = uniform_y(r);
        }

        points[i][0] = x;
        points[i][1] = y;
    }

    ceres::Problem problem;

    double line_params[3] = {0.0, 1.0, 0.0};

    for (size_t i = 0; i < n_points; ++i) {
        ceres::CostFunction* point_residual = new ceres::AutoDiffCostFunction<PointObservationError, 1, 3>(
                new PointObservationError(points[i]));

        ceres::LossFunction* loss = use_huber
                                    ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(3.0 * sigma))
                                    : static_cast<ceres::LossFunction*>(new ceres::TrivialLoss());

        problem.AddResidualBlock(point_residual, loss, line_params);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    if (std::abs(line_params[1]) > 1e-12) {
        double scale = ideal_line[1] / line_params[1];
        for (double &v : line_params) {
            v *= scale;
        }
    } else if (std::abs(line_params[0]) > 1e-12) {
        double scale = ideal_line[0] / line_params[0];
        for (double &v : line_params) {
            v *= scale;
        }
    }

    std::cout << "Found line: (a=" << line_params[0] << ", b=" << line_params[1] << ", c=" << line_params[2] << ")" << std::endl;

    const double a_threshold = (outliers_fraction > 0.0 && !use_huber) ? 0.35 : 0.05;
    const double b_threshold = 1e-8;
    const double c_threshold = (outliers_fraction > 0.0 && !use_huber) ? 100.0 : 1.0;

    ASSERT_NEAR(line_params[0], ideal_line[0], a_threshold);
    ASSERT_NEAR(line_params[1], ideal_line[1], b_threshold);
    ASSERT_NEAR(line_params[2], ideal_line[2], c_threshold);

    double inliers_fraction, mse;
    evaluateLine(points, ideal_line, sigma, inliers_fraction, mse);
    if (outliers_fraction == 0.0) {
        ASSERT_GT(inliers_fraction, 0.99);
        ASSERT_LT(mse, 1.1 * sigma * sigma);
    } else {
        ASSERT_GT(inliers_fraction, 0.75);
        ASSERT_LT(mse, 1.1 * sigma * sigma);
    }

    evaluateLine(points, line_params, sigma, inliers_fraction, mse);
    if (outliers_fraction == 0.0) {
        ASSERT_GT(inliers_fraction, 0.99);
        ASSERT_LT(mse, 1.1 * sigma * sigma);
    } else if (use_huber) {
        ASSERT_GT(inliers_fraction, 0.75);
        ASSERT_LT(mse, 1.1 * sigma * sigma);
    }
}

void evaluateLine(const std::vector<double_2> &points, const double* line,
                  double sigma, double &fitted_inliers_fraction, double &mse_inliers_distance) {
    const size_t n = points.size();
    size_t inliers = 0;
    mse_inliers_distance = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double dist = calcDistanceToLine2D(points[i][0], points[i][1], line);
        if (std::abs(dist) <= 3.0 * sigma) {
            ++inliers;
            mse_inliers_distance += dist * dist;
        }
    }
    fitted_inliers_fraction = 1.0 * inliers / n;
    mse_inliers_distance /= inliers;
}

TEST (CeresSolver, FitLineNoise) {
    const double sigma = 1.0;

    double no_outliers_trivial_loss_inliers;
    double no_outliers_trivial_loss_mean_inliers_distance;
    evaluateLineFitting(sigma, no_outliers_trivial_loss_inliers, no_outliers_trivial_loss_mean_inliers_distance);
}

TEST (CeresSolver, FitLineNoiseAndOutliers) {
    const double sigma = 1.0;
    const double outliers_fraction = 0.20;

    double trivial_loss_inliers;
    double trivial_loss_mean_inliers_distance;
    evaluateLineFitting(sigma, trivial_loss_inliers, trivial_loss_mean_inliers_distance, outliers_fraction);
}

TEST (CeresSolver, FitLineNoiseAndOutliersWithHuberLoss) {
    const double sigma = 1.0;
    const double outliers_fraction = 0.20;
    const bool use_huber = true;

    double huber_loss_inliers;
    double huber_loss_mean_inliers_distance;
    evaluateLineFitting(sigma, huber_loss_inliers, huber_loss_mean_inliers_distance, outliers_fraction, use_huber);
}