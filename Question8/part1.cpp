#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>

// Function for thermal diffusivity alpha(x)
double alpha(double x) {
    return 1.0 + sin(2.0 * M_PI * x);
}

// Function for domain decomposition (1D)
void decompose1d(int n, int m, int i, int* s, int* e) {
    const int base = n / m;
    const int rem = n % m;
    *s = i * base + (i < rem ? i : rem);
    *e = *s + base - 1 + (i < rem ? 1 : 0);
    if (*e >= n || i == m - 1) *e = n - 1;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double T = 20.0;  // Final simulation time

    std::vector<double> global_x;
    int N;

    // Read grid.txt on rank 0
    if (rank == 0) {
        std::ifstream in("grid.txt");
        if (!in) {
            std::cerr << "Error: Could not open grid.txt" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        double val;
        while (in >> val) {
            global_x.push_back(val);
        }
        N = global_x.size();
        std::cout << "Read " << N << " grid points" << std::endl;
        
        // Verify grid is ordered
        for (int i = 0; i < N-1; i++) {
            if (global_x[i] >= global_x[i+1]) {
                std::cerr << "Error: Grid points not in ascending order at index " << i << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N == 0) {
        if (rank == 0) std::cerr << "Error: Empty grid" << std::endl;
        MPI_Finalize();
        return 1;
    }

    // Decomposition info
    int start, end;
    decompose1d(N, size, rank, &start, &end);
    int local_N = end - start + 1;

    // Allocate local grid with 2 ghost cells
    std::vector<double> x(local_N + 2);
    std::vector<double> u(local_N + 2);
    std::vector<double> u_new(local_N + 2);

    // Scatter data
    std::vector<int> counts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        int s, e;
        decompose1d(N, size, i, &s, &e);
        counts[i] = e - s + 1;
        displs[i] = s;
    }

    MPI_Scatterv(global_x.data(), counts.data(), displs.data(), MPI_DOUBLE,
                &x[1], local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Set ghost x values (extrapolation if no neighbor)
    if (rank > 0) {
        MPI_Sendrecv(&x[1], 1, MPI_DOUBLE, rank - 1, 0, 
                    &x[0], 1, MPI_DOUBLE, rank - 1, 1, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        x[0] = x[1] - (x[2] - x[1]);
    }

    if (rank < size - 1) {
        MPI_Sendrecv(&x[local_N], 1, MPI_DOUBLE, rank + 1, 1, 
                    &x[local_N + 1], 1, MPI_DOUBLE, rank + 1, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        x[local_N + 1] = x[local_N] + (x[local_N] - x[local_N - 1]);
    }

    // Calculate local minimum grid spacing
    double min_dx_local = 1.0;
    for (int i = 1; i <= local_N + 1; ++i) {
        min_dx_local = std::min(min_dx_local, x[i] - x[i-1]);
    }
    
    // Find global minimum grid spacing
    double min_dx;
    MPI_Allreduce(&min_dx_local, &min_dx, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    std::cout << "Minimum grid spacing (Δx_min): " << std::setprecision(10) << min_dx << std::endl;
    
    // Calculate maximum stable time step
    double max_alpha = 2.0;  // since α(x) = 1 + sin(2πx) ∈ [0,2]
    double dt = 0.25 * min_dx * min_dx / max_alpha;  // Conservative CFL condition
    
    const int steps = static_cast<int>(T / dt) + 1;

    // Initial condition: u(x, 0) = x*(1 - x)
    for (int i = 1; i <= local_N; ++i) {
        u[i] = x[i] * (1.0 - x[i]);
    }

    // Boundary conditions
    if (rank == 0) u[1] = 0.0;
    if (rank == size - 1) u[local_N] = 0.0;

    // Time stepping
    for (int step = 0; step < steps; ++step) {
        // Exchange ghost values with non-blocking communication
        MPI_Request reqs[4];
        if (rank > 0) {
            MPI_Isend(&u[1], 1, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(&u[0], 1, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &reqs[2]);
        }
        if (rank < size - 1) {
            MPI_Isend(&u[local_N], 1, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &reqs[1]);
            MPI_Irecv(&u[local_N + 1], 1, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &reqs[3]);
        }
        
        // Wait for all communications to complete
        if (rank > 0) {
            MPI_Waitall(2, &reqs[0], MPI_STATUSES_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Waitall(2, &reqs[1], MPI_STATUSES_IGNORE);
        }
        
        // Enforce boundary conditions on ghost cells
        if (rank == 0) u[0] = 0.0;
        if (rank == size - 1) u[local_N + 1] = 0.0;

        // Update u_new
        for (int i = 1; i <= local_N; ++i) {
            double dxp = x[i + 1] - x[i];
            double dxm = x[i] - x[i - 1];
            double dxi = 0.5 * (dxp + dxm);
            
            double alpha_p = 0.5 * (alpha(x[i]) + alpha(x[i + 1]));
            double alpha_m = 0.5 * (alpha(x[i]) + alpha(x[i - 1]));
            
            double flux_p = alpha_p * (u[i + 1] - u[i]) / dxp;
            double flux_m = alpha_m * (u[i] - u[i - 1]) / dxm;
            
            u_new[i] = u[i] + dt / dxi * (flux_p - flux_m);
            
            // Check for numerical instability
            if (std::isnan(u_new[i])) {
                std::cerr << "Rank " << rank << ": NaN detected at step " << step 
                          << ", x = " << x[i] << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Enforce boundary conditions
        if (rank == 0) u_new[1] = 0.0;
        if (rank == size - 1) u_new[local_N] = 0.0;

        std::swap(u, u_new);
    }

    // Output results
    std::ofstream out("result_rank" + std::to_string(rank) + ".txt");
    out << std::scientific << std::setprecision(10);
    for (int i = 1; i <= local_N; ++i) {
        out << x[i] << " " << u[i] << "\n";
    }
    out.close();

    MPI_Finalize();
    return 0;
}