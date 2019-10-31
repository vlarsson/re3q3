#include <Eigen/Dense>
#include "examples/p4pf.h"
#include <iostream>
#include <time.h>

#define TEST(FUNC) if(!FUNC()) { std::cout << #FUNC"\033[1m\033[31m FAILED!\033[0m\n"; } else { std::cout << #FUNC"\033[1m\033[32m PASSED!\033[0m\n"; passed++;} num_tests++; 
#define REQUIRE(COND) if(!(COND)) { std::cout << "Failure: "#COND" was not satisfied.\n"; return false; }

static const double TOL = 1e-6;


void setup_random_scene(double depth_low, double depth_high, double focal, int n_points,
		Eigen::Matrix<double,2,Eigen::Dynamic> *points2d,
		Eigen::Matrix<double,3,Eigen::Dynamic> *points3d,
		Camera *pose_gt)
{

	// Setup random pose
	pose_gt->R = Eigen::Quaternion<double>::UnitRandom().toRotationMatrix();
	pose_gt->t.setRandom();
	pose_gt->focal = focal;

	points2d->resize(2, n_points);
	points3d->resize(3, n_points);

	// Generate random data in camera coordinate system
	points3d->setRandom();
	points3d->row(2) = ((depth_high - depth_low) * (points3d->row(2).array() + 1.0) / 2.0 + depth_low).eval();

	// Project
	points2d->row(0) = focal * points3d->row(0).array() / points3d->row(2).array();
	points2d->row(1) = focal * points3d->row(1).array() / points3d->row(2).array();
	
	// Transform into world coordinate system
	points3d->colwise() -= pose_gt->t;
	*points3d = pose_gt->R.transpose() * (*points3d);
}



bool test_p4pf_multiple_trials() {
	int success = 0;

	Eigen::Matrix<double, 2, Eigen::Dynamic> points2d;
	Eigen::Matrix<double, 3, Eigen::Dynamic> points3d;
	Camera pose_gt;

	for(int trials = 0; trials < 1000; ++trials) {
		setup_random_scene(2.0, 10.0, 2000.0, 4, &points2d, &points3d, &pose_gt);

		std::vector<Camera> poses;
		int n_sols = p4pf(points2d, points3d, &poses);

		if(n_sols == 0)
			continue;

		for(int i = 0; i < poses.size(); ++i) {
			double err_r = (poses[i].R - pose_gt.R).norm();
			double err_t = (poses[i].t - pose_gt.t).norm();
			double err_f = std::abs(poses[i].focal - pose_gt.focal) / pose_gt.focal;
			
			if(err_r < TOL && err_t < TOL && err_f < TOL) {
				success++;
				break;
			}
		}
	}

	std::cout << "Success: " << success << " / 1000\n";
	return success >= 995;
}

bool test_p4pf_simple() {
	int success = 0;

	Eigen::Matrix<double, 2, Eigen::Dynamic> points2d;
	Eigen::Matrix<double, 3, Eigen::Dynamic> points3d;
	Camera pose_gt;

	setup_random_scene(2.0, 10.0, 2.0, 4, &points2d, &points3d, &pose_gt);

	std::vector<Camera> poses;
	int n_sols = p4pf(points2d, points3d, &poses);

	REQUIRE(n_sols > 0);

	for (int i = 0; i < poses.size(); ++i) {
		double err_r = (poses[i].R - pose_gt.R).norm();
		double err_t = (poses[i].t - pose_gt.t).norm();
		double err_f = std::abs(poses[i].focal - pose_gt.focal) / pose_gt.focal;

		if (err_r < TOL && err_t < TOL && err_f < TOL) {
			success++;
			break;
		}
	}
	
	return success > 0;
}



int main() {
	
	unsigned int seed = (unsigned int)time(0);	
	srand(seed);

	std::cout << "Running tests... (seed = " << seed << ")\n\n";

	int passed = 0;
	int num_tests = 0;
	
	TEST(test_p4pf_simple);
	TEST(test_p4pf_multiple_trials);

	std::cout << "\nDone! Passed " << passed << "/" << num_tests << " tests.\n";
}
