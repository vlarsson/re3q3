#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <time.h>
#include "re3q3/re3q3.h"

#define TEST(FUNC) if(!FUNC()) { std::cout << #FUNC"\033[1m\033[31m FAILED!\033[0m\n"; } else { std::cout << #FUNC"\033[1m\033[32m PASSED!\033[0m\n"; passed++;} num_tests++; 
#define REQUIRE(COND) if(!(COND)) { std::cout << "Failure: "#COND" was not satisfied.\n"; return false; }


void compute_equation_residuals(const Eigen::Matrix<double, 3, 10> & coeffs, const Eigen::Matrix<double, 3, 8> & solution, int n_sols, double res[8]) {
	Eigen::Matrix<double, 10, 1> mons;

	for (int i = 0; i < n_sols; i++) {
		double x = solution(0, i);
		double y = solution(1, i);
		double z = solution(2, i);
		mons << x * x, x* y, x* z, y* y, y* z, z* z, x, y, z, 1.0;
		Eigen::Matrix<double, 3, 1> residuals = coeffs * mons;
		res[i] = residuals.cwiseAbs().maxCoeff();
	}
}

bool verify_solutions(const Eigen::Matrix<double, 3, 10> &coeffs,
					const Eigen::Matrix<double, 3, 8> &solutions,
					int n_sols, double tol) {
	bool ok = true;

	double res[8];
	compute_equation_residuals(coeffs, solutions, n_sols, res);

	for (int i = 0; i < n_sols; i++) {		
		ok &= res[i] < tol;
	}
	return ok;
}


bool test_random_coefficients() {
	Eigen::Matrix<double, 3, 10> coeffs;
	Eigen::Matrix<double, 3, 8> solutions;

	coeffs.setRandom();

	//std::cout << "coeffs: " << coeffs << "\n";

	int n_sols = re3q3::re3q3(coeffs, &solutions);
	
	return verify_solutions(coeffs, solutions, n_sols, 1e-8);
}


bool test_degenerate_for_x() {
	Eigen::Matrix<double, 3, 10> coeffs;
	Eigen::Matrix<double, 3, 8> solutions;

	coeffs.setRandom();
	coeffs.col(3) = 0.5 * (coeffs.col(5) + coeffs.col(4));

	int n_sols = re3q3::re3q3(coeffs, &solutions);

	return verify_solutions(coeffs, solutions, n_sols, 1e-8);
}


bool test_degenerate_for_y() {
	Eigen::Matrix<double, 3, 10> coeffs;
	Eigen::Matrix<double, 3, 8> solutions;

	coeffs.setRandom();
	coeffs.col(0) = 0.5 * (coeffs.col(5) + coeffs.col(2));

	int n_sols = re3q3::re3q3(coeffs, &solutions);

	return verify_solutions(coeffs, solutions, n_sols, 1e-8);
}

bool test_degenerate_for_z() {
	Eigen::Matrix<double, 3, 10> coeffs;
	Eigen::Matrix<double, 3, 8> solutions;

	coeffs.setRandom();
	coeffs.col(0) = 0.5 * (coeffs.col(1) + coeffs.col(3));

	int n_sols = re3q3::re3q3(coeffs, &solutions);

	return verify_solutions(coeffs, solutions, n_sols, 1e-8);
}

bool test_degenerate_for_xy() {
	Eigen::Matrix<double, 3, 10> coeffs;
	Eigen::Matrix<double, 3, 8> solutions;

	coeffs.setRandom();
	coeffs.col(0) = 0.5 * (coeffs.col(5) + coeffs.col(2));
	coeffs.col(3) = 0.5 * (coeffs.col(5) + coeffs.col(4));

	int n_sols = re3q3::re3q3(coeffs, &solutions);

	return verify_solutions(coeffs, solutions, n_sols, 1e-8);
}


bool test_pure_squares() {
	Eigen::Matrix<double, 3, 10> coeffs;
	Eigen::Matrix<double, 3, 8> solutions;

	coeffs.setZero();
	coeffs(0, 0) = 1.0;
	coeffs(0, 9) = -1.0;
	coeffs(1, 3) = 1.0;
	coeffs(1, 9) = -1.0;
	coeffs(2, 5) = 1.0;
	coeffs(2, 9) = -1.0;
		
	int ok = 0;

	// We run multiple tests here since the change of variables is random.
	for(int test = 0; test < 1000; ++test) {
		int n_sols = re3q3::re3q3(coeffs, &solutions);
		

		REQUIRE(n_sols == 8);

		if(verify_solutions(coeffs, solutions, n_sols, 1e-8))
			++ok;
	}	
	return ok == 1000;
}

bool benchmark_random_coeffs() {

	std::vector<double> residuals;
	residuals.reserve(10000 * 8);

	for (int iter = 0; iter < 10000; ++iter) {
		Eigen::Matrix<double, 3, 10> coeffs;
		Eigen::Matrix<double, 3, 8> solutions;
		coeffs.setRandom();
		
		int n_sols = re3q3::re3q3(coeffs, &solutions);

		double res[8];
		compute_equation_residuals(coeffs, solutions, n_sols, res);
		for (int i = 0; i < n_sols; ++i)
			residuals.push_back(std::log10(res[i]));
	}


	std::sort(residuals.begin(), residuals.end());

	double q90 = residuals[static_cast<int>(residuals.size() * 0.90)];
	double q95 = residuals[static_cast<int>(residuals.size() * 0.95)];
	double q99 = residuals[static_cast<int>(residuals.size() * 0.99)];


	std::cout << "q90: " << q90 << ", q95: " << q95 << ", q99: " << q99 << "\n";
	
	return q99 < -6;
}


bool benchmark_degen_rotation_homogeneous() {

	std::vector<double> residuals;
	residuals.reserve(10000 * 8);

	Eigen::Matrix<double, 3, 10> coeffs;
	Eigen::Matrix<double, 3, 9> Rcoeffs;
	Eigen::Matrix<double, 4, 8> solutions;

	for (int iter = 0; iter < 10000; ++iter) {

		// Generate random 180 degree rotation
		Eigen::Quaterniond q_gt;
		q_gt.coeffs().setRandom();
		q_gt.coeffs()(0) = 0.0;
		q_gt.coeffs().normalize();

		// Problem is x1'*R*x2 = 0
		// => kron(x2',x1') * vec(R) = 0

		for (int i = 0; i < 3; ++i) {
			Eigen::Vector3d x1, x2, v;
			x2.setRandom().normalize();
			v.setRandom();
			x1 = v.cross(q_gt.toRotationMatrix() * x2).normalized();

			Rcoeffs.row(i) << x2(0) * x1.transpose(), x2(1)* x1.transpose(), x2(2)* x1.transpose();
		}

		
		int n_sols = re3q3::re3q3_rotation(Rcoeffs, &solutions);
		
		double res = 1.0;		
		for (int i = 0; i < n_sols; ++i) {			
			Eigen::Vector4d q = solutions.col(i);
			Eigen::Matrix3d R = Eigen::Quaterniond(q).toRotationMatrix();			
			res = std::min(res, (R - q_gt.toRotationMatrix()).norm());
		}
		residuals.push_back(std::log10(res));
	}


	std::sort(residuals.begin(), residuals.end());

	double q90 = residuals[static_cast<int>(residuals.size() * 0.90)];
	double q95 = residuals[static_cast<int>(residuals.size() * 0.95)];
	double q99 = residuals[static_cast<int>(residuals.size() * 0.99)];


	std::cout << "q90: " << q90 << ", q95: " << q95 << ", q99: " << q99 << "\n";

	return q99 < -6;
}


bool benchmark_degen_rotation_inhomogeneous() {

	std::vector<double> residuals;
	residuals.reserve(10000 * 8);

	Eigen::Matrix<double, 3, 10> coeffs;
	Eigen::Matrix<double, 3, 10> Rcoeffs;
	Eigen::Matrix<double, 4, 8> solutions;

	for (int iter = 0; iter < 10000; ++iter) {

		// Generate random 180 degree rotation
		Eigen::Quaterniond q_gt;
		q_gt.coeffs().setRandom();
		q_gt.coeffs()(0) = 0.0;
		q_gt.coeffs().normalize();

		// Problem is x1'*R*x2 = d
		// => kron(x2',x1') * vec(R) = d

		for (int i = 0; i < 3; ++i) {
			Eigen::Vector3d x1, x2, v;
			x2.setRandom().normalize();
			x1.setRandom().normalize();
			double d = -x1.dot(q_gt.toRotationMatrix() * x2);
			Rcoeffs.row(i) << x2(0) * x1.transpose(), x2(1)* x1.transpose(), x2(2)* x1.transpose(), d;
		}


		int n_sols = re3q3::re3q3_rotation(Rcoeffs, &solutions);

		double res = 1.0;
		for (int i = 0; i < n_sols; ++i) {
			Eigen::Vector4d q = solutions.col(i);
			Eigen::Matrix3d R = Eigen::Quaterniond(q).toRotationMatrix();
			res = std::min(res, (R - q_gt.toRotationMatrix()).norm());
		}
		residuals.push_back(std::log10(res));
	}


	std::sort(residuals.begin(), residuals.end());

	double q90 = residuals[static_cast<int>(residuals.size() * 0.90)];
	double q95 = residuals[static_cast<int>(residuals.size() * 0.95)];
	double q99 = residuals[static_cast<int>(residuals.size() * 0.99)];


	std::cout << "q90: " << q90 << ", q95: " << q95 << ", q99: " << q99 << "\n";

	return q99 < -6;
}

int main() {
	
	unsigned int seed = (unsigned int)time(0);		
	srand(seed);

	std::cout << "Running tests... (seed = " << seed << ")\n\n";

	int passed = 0;
	int num_tests = 0;
	
	TEST(test_random_coefficients);
	TEST(test_degenerate_for_x);
	TEST(test_degenerate_for_y);
	TEST(test_degenerate_for_z);
	TEST(test_degenerate_for_xy);
	TEST(test_pure_squares);
	TEST(benchmark_random_coeffs);
	TEST(benchmark_degen_rotation_homogeneous);
	TEST(benchmark_degen_rotation_inhomogeneous);

	std::cout << "\nDone! Passed " << passed << "/" << num_tests << " tests.\n";
}
