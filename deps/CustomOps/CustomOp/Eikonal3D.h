#include <vector>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/SparseLU>
#include <set>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

namespace Eikonal3D{




double calculate_unique_solution(double a1, double a2, 
        double a3, double f, double h);

class Eikonal3D{
public:
    Eikonal3D(const double *u0, const double *f, double h, 
            int m, int n, int j);

    void solve(double *u, double tol = 1e-6, bool verbose = false);    

    const double *u0;
    const double *f;
    double h;
    int m, n, l;

private:

    void sweeping(double *u);
    void sweeping_over_I_J_K(double *u, int dirI, int dirJ, int dirK);
    
};

void forward(double *u, const double *u0, const double *f, double h, 
            int m, int n, int l, double tol, bool verbose);

void backward(
            double * grad_u0, double * grad_f, 
            const double * grad_u, 
            const double *u, const double *u0,  const double*f, double h, 
            int m, int n, int l);

}