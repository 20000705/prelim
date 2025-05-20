#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace Eigen;

// Construct the sparse matrix A using a 5-point finite difference stencil
SparseMatrix<double> build_matrix(int N) {
    int size = N * N;
    SparseMatrix<double> A(size, size);
    std::vector<Triplet<double>> triplets;
    double h2 = 1.0 / std::pow(N + 1, 2);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            triplets.emplace_back(idx, idx, 4.0 / h2);
            if (i > 0) triplets.emplace_back(idx, idx - N, -1.0 / h2);
            if (i < N - 1) triplets.emplace_back(idx, idx + N, -1.0 / h2);
            if (j > 0) triplets.emplace_back(idx, idx - 1, -1.0 / h2);
            if (j < N - 1) triplets.emplace_back(idx, idx + 1, -1.0 / h2);
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

// Construct the RHS vector b using f(x, y) = sin(pi x) * sin(pi y)
VectorXd build_rhs(int N) {
    VectorXd b(N * N);
    double h = 1.0 / (N + 1);
    for (int i = 0; i < N; ++i) {
        double x = (i + 1) * h;
        for (int j = 0; j < N; ++j) {
            double y = (j + 1) * h;
            b[i * N + j] = std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }
    return b;
}

int main() {
    std::vector<int> Ns = {40, 80, 160}; // Different grid sizes to test

    for (int N : Ns) {
        std::cout << "=== N = " << N << " ===\n";

        // Assemble matrix A and RHS vector b
        SparseMatrix<double> A = build_matrix(N);
        VectorXd b = build_rhs(N);

        // Compute Incomplete Cholesky preconditioner
        IncompleteCholesky<double> ic;
        ic.compute(A);
        if (ic.info() != Success) {
            std::cerr << "IC failed\n";
            continue;
        }

        // Initial guess x = 0
        VectorXd x = VectorXd::Zero(b.size());
        VectorXd r = b - A * x;
        VectorXd z = ic.solve(r);
        VectorXd p = z;

        double b_norm = b.norm();
        std::vector<double> rel_residuals = {r.norm() / b_norm};

        int max_iter = 500;
        double tol = 1e-7;

        // Start timing the solver
        auto start = std::chrono::high_resolution_clock::now();

        double rz_old = r.dot(z);
        int iter = 0;
        for (; iter < max_iter; ++iter) {
            VectorXd Ap = A * p;
            double alpha = rz_old / p.dot(Ap);

            x += alpha * p;
            r -= alpha * Ap;

            double res_norm = r.norm();
            double rel_res = res_norm / b_norm;
            rel_residuals.push_back(rel_res);
            if (rel_res < tol) break;

            z = ic.solve(r);
            double rz_new = r.dot(z);
            double beta = rz_new / rz_old;
            p = z + beta * p;
            rz_old = rz_new;
        }

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Output residual history to text file for plotting
        std::ofstream fout("residuals_N" + std::to_string(N) + ".txt");
        for (double val : rel_residuals)
            fout << val << "\n";
        fout.close();

        // Print performance summary
        std::cout << "Final residual: " << rel_residuals.back()
                  << ", Iterations: " << iter
                  << ", Runtime: " << elapsed.count() << " seconds\n\n";
    }

    return 0;
}