/*
 * p4pf.h
 * Author: Viktor Larsson
 */
#pragma once
#include <Eigen/Dense>
#include <vector>

struct Camera {
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
	double focal;
};

/* Absolute pose with unknown focal length from four 2D-3D point correspondences */
int p4pf(const Eigen::Matrix<double, 2, 4> &points2d,
		 const Eigen::Matrix<double, 3, 4> &points3d, 
		 std::vector<Camera> *output,
		 bool normalize_input = false);