#include "Eikonal3D.h"

namespace Eikonal3D{


#define u(i,j,k) u[(i) * n * l + (j) * l + (k)]
#define f(i,j,k) f[(i) * n * l + (j) * l + (k)]
#define get_id(i,j,k) ((i) * n * l + (j) * l + (k))


double calculate_unique_solution(double a1_, double a2_, 
        double a3_, double f, double h){
    double a1, a2, a3;
    a3 = std::max(std::max(a1_, a2_), a3_);
    a1 = std::min(std::min(a1_, a2_), a3_);
    a2 = a1_ + a2_ + a3_ - a1 - a3;

    double x = a1 + f * h;
    if (x <= a2) return x;
    double B = - (a1 + a2);
    double C = (a1 * a1 + a2 * a2 - f * f * h * h)/2.0;
    double x1 = (-B + sqrt(B * B - 4 * C))/2.0;
    double x2 = (-B - sqrt(B * B - 4 * C))/2.0;
    x = x1 > a2 ? x1 : x2;
    if (x <= a3) return x;
    B = -2.0*(a1 + a2 + a3)/3.0;
    C = (a1 * a1 + a2 * a2 + a3 * a3 - f * f * h * h)/3.0;
    x1 = (-B + sqrt(B * B - 4 * C))/2.0;
    x2 = (-B - sqrt(B * B - 4 * C))/2.0;
    x = x1 > a3 ? x1 : x2;
    return x;
}


Eikonal3D::Eikonal3D(const double *u0, const double *f_, double h, 
            int m, int n, int l)
    : h(h), m(m), n(n), l(l), u0(u0){ f = f_;}

void Eikonal3D::sweeping_over_I_J_K(double *u, 
        int dirI,
        int dirJ,
        int dirK){
        
    auto I = std::make_tuple(dirI==1?0:m-1, dirI==1?m:-1, dirI);
    auto J = std::make_tuple(dirJ==1?0:n-1, dirJ==1?n:-1, dirJ);
    auto K = std::make_tuple(dirK==1?0:l-1, dirK==1?l:-1, dirK);

    for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
        for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
            for (int k = std::get<0>(K); k != std::get<1>(K); k += std::get<2>(K)){
                double uxmin = i==0 ? u(i+1, j, k) : \
                                (i==m-1 ? u(i-1, j, k) : std::min(u(i+1, j, k), u(i-1, j, k)));
                double uymin = j==0 ? u(i, j+1, k) : \
                                (j==n-1 ? u(i, j-1, k) : std::min(u(i, j+1, k), u(i, j-1, k)));
                double uzmin = k==0 ? u(i, j, k+1) : \
                                (k==l-1 ? u(i, j, k-1) : std::min(u(i, j, k+1), u(i, j, k-1)));
                double u_new = calculate_unique_solution(uxmin, uymin, uzmin, f(i, j, k), h);
                u(i, j, k) = std::min(u_new, u(i, j, k));
            }

}

void Eikonal3D::sweeping(double *u){
    sweeping_over_I_J_K(u, 1, 1, 1);
    sweeping_over_I_J_K(u, -1, 1, 1);
    sweeping_over_I_J_K(u, -1, -1, 1);
    sweeping_over_I_J_K(u, 1, -1, 1);
    sweeping_over_I_J_K(u, 1, -1, -1);
    sweeping_over_I_J_K(u, 1, 1, -1);
    sweeping_over_I_J_K(u, -1, 1, -1);
    sweeping_over_I_J_K(u, -1, -1, -1);
}


void Eikonal3D::solve(double *u, double tol, bool verbose){
    memcpy(u, u0, sizeof(double)*m*n*l);
    auto u_old = new double[m*n*l];
    for (int i = 0; i < 20; i++){
        memcpy(u_old, u, sizeof(double)*m*n*l);
        sweeping(u);
        double err = 0.0;

        for (int j = 0; j < m*n*l; j++){
            err = std::max(fabs(u[j]-u_old[j]), err);
        }
        if (verbose){
            printf("Iteration %d, Error = %0.6e\n", i, err);
        }
        if (err < tol) break;
    }
    delete [] u_old;
}

void forward(double *u, const double *u0, const double *f, double h, 
            int m, int n, int l, double tol = 1e-6, bool verbose = false){
    Eikonal3D ek(u0, f, h, m, n, l);
    ek.solve(u, tol, verbose);
};

void backward(
            double * grad_u0, double * grad_f, 
            const double * grad_u, 
            const double *u, const double *u0,  const double*f, double h, 
            int m, int n, int l){
    
    Eigen::VectorXd g(m*n*l);
    memcpy(g.data(), grad_u, sizeof(double)*m*n*l);

    // calculate gradients for \partial L/\partial u0
    for (int i = 0; i < m*n*l; i++){
        if (fabs(u[i] - u0[i])<1e-6) grad_u0[i] = g[i];
        else grad_u0[i] = 0.0;
    }

    // calculate gradients for \partial L/\partial f

    Eigen::VectorXd rhs(m*n*l);
    for (int i=0;i<m*n*l;i++){
      rhs[i] = -2*f[i]*h*h;
    }

    std::vector<T> triplets;
    std::set<int> zero_id;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            for (int k = 0; k < l; k++){
                double uxmin = i==0 ? u(i+1, j, k) : \
                                (i==m-1 ? u(i-1, j, k) : std::min(u(i+1, j, k), u(i-1, j, k)));
                double uymin = j==0 ? u(i, j+1, k) : \
                                (j==n-1 ? u(i, j-1, k) : std::min(u(i, j+1, k), u(i, j-1, k)));
                double uzmin = k==0 ? u(i, j, k+1) : \
                                (k==l-1 ? u(i, j, k-1) : std::min(u(i, j, k+1), u(i, j, k-1)));

                int this_id = get_id(i, j, k);
                int idx = i==0 ? get_id(i+1, j, k) : \
                                (i==m-1 ? get_id(i-1, j, k) : \
                                ( u(i+1, j, k) > u(i-1, j, k) ? get_id(i-1, j, k) : get_id(i+1, j, k)));
                int idy = j==0 ? get_id(i, j+1, k) : \
                                (j==n-1 ? get_id(i, j-1, k) : \
                                ( u(i, j+1, k) > u(i, j-1, k) ? get_id(i, j-1, k) : get_id(i, j+1, k)));
                int idz = k==0 ? get_id(i, j, k+1) : \
                                (k==l-1 ? get_id(i, j, k-1) : \
                                ( u(i, j, k+1) > u(i, j, k-1) ? get_id(i, j, k-1) : get_id(i, j, k+1)));


                bool this_id_is_not_zero = false;
                if (u(i, j, k) > uxmin){
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (u(i, j, k) - uxmin)));
                    triplets.push_back(T(this_id, idx, -2.0 * (u(i, j, k) - uxmin)));
                }

                if (u(i, j, k) > uymin){
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (u(i, j, k) - uymin)));
                    triplets.push_back(T(this_id, idy, -2.0 * (u(i, j, k) - uymin)));
                }

                if (u(i, j, k) > uzmin){
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (u(i, j, k) - uzmin)));
                    triplets.push_back(T(this_id, idz, -2.0 * (u(i, j, k) - uzmin)));
                }

                if (!this_id_is_not_zero){
                    zero_id.insert(this_id);
                    g[this_id] = 0.0;
                }

            }
        }
    }

    // H = sum c_{ijk} * u_{ijk} = c^T u 
    // partial H/partial f_{lmn} = c^T partial u/partial f_{lmn}
    // c^T [pu/pf_1 pu/pf_2 ... pu/pf_N] 
    
    // i -- source
    // Gi(u) = g(fi)
    // Gi(u*) = 0, p Gi(u^*) / p u = 0
    // 
    // Gi(u) = g(fi) i = 1, 2, ... N
    // ==> Gi'(u) * pu/pf1 = p g(fi) / pf1 = 0 (i != 1)
    // Gi'(u) * [pu/pf1 pu/pf2 ... pu/pf_N]  = [0 0 0 ... 0 g'(fi) 0 ... 0]
    // [G1'(u); G2'(u);...] * [pu/pf1 pu/pf2 ... pu/pf_N] = diag(g'(f1), g'(f2), ..)
    //  [pu/pf1 pu/pf2 ... pu/pf_N] = [G1'(u); G2'(u);...] ^{-1} * diag(g'(f1), g'(f2), ..)
    // output = c^T [G1'(u); G2'(u);...] ^{-1} * diag(g'(f1), g'(f2), ..)
    // x_row = c^T [G1'(u); G2'(u);...] ^{-1} ==> [G1'(u); G2'(u);...]^T x_col = c
    // x_row_i x 
    // output = x_col *  diag(g'(f1), g'(f2), ..) = [x_col1 * g'(f1), x_col2 * g'(f2), ...]

    SpMat A(m*n*l, m*n*l);

    if (zero_id.size()>0){
        for (auto& t : triplets){
            if (zero_id.count(t.col()) || zero_id.count(t.row())) t = T(t.col(), t.row(), 0.0);
        }
        for (auto idx: zero_id){
            triplets.push_back(T(idx, idx, 1.0));
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    A = A.transpose();
    Eigen::SparseLU<SpMat> solver;
     
    solver.analyzePattern(A);
    solver.factorize(A);
    Eigen::VectorXd res = solver.solve(g);
    for(int i=0;i<m*n*l;i++){
      grad_f[i] = -res[i] * rhs[i];
    }

}

}
// int main(){
//     printf("Hello world\n");


//     int m = 21, n = 21, l = 21;
//     auto u0 = new double[m*n*l];
//     auto f = new double[m*n*l];
//     auto u = new double[m*n*l];
//     auto grad_u = new double[m*n*l];
//     auto grad_f = new double[m*n*l];
//     auto grad_u0 = new double[m*n*l];
//     for(int i = 0; i < m*n*l; i++){
//         u0[i] = 1000.0;
//         f[i] = 1.0;
//         grad_u[i] = 1.0;
//     }
//     u0[10 * n * l + 10 * l + 10] = 0.0;
//     double h = 0.01;
//     Eikonal3D::Eikonal3D ek(u0, f, h, m, n, l);

//     ek.solve(u, 1e-6, true);

    

    
//     Eikonal3D::backward(grad_u0, grad_f, grad_u, u, u0, f, h, m, n, l);

//     std::ofstream ofile("solution.txt");
//     for(int i = 0; i < m*n*l; i++){
//         ofile << u[i] << std::endl;
//     }
//     ofile.close();
//     ofile.clear();

//     ofile.open("gradu.txt");
//     for(int i = 0; i < m*n*l; i++){
//         ofile << grad_u0[i] << std::endl;
//     }
//     ofile.close();
//     ofile.clear();

//     ofile.open("grad_f.txt");
//     for(int i = 0; i < m*n*l; i++){
//         ofile << grad_f[i] << std::endl;
//     }
//     ofile.close();
//     ofile.clear();


//     printf("Exit normally\n");

//     return 1;
// }