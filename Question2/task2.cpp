#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <mpi.h>

// Check if a point (x, y, z) is inside a sphere
bool inside_sphere(double x, double y, double z, double cx, double cy, double cz, double r) {
    double dx = x - cx;
    double dy = y - cy;
    double dz = z - cz;
    return dx * dx + dy * dy + dz * dz <= r * r;
}

// Approximate intersection volume formula (used as ground truth)
double reference_intersection_volume(double r, double d) {
    if (d >= 2 * r) return 0.0;
    return (M_PI * (4 * r + d) * std::pow(2 * r - d, 2)) / 12;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double r = 5.0;
    double lower = -6.0;
    double upper = 6.0;
    double volume_box = std::pow(upper - lower, 3);
    
    // Use known reference volume based on distance between centers
    double d = std::sqrt(2 * 2 + 2 * 2 + 2 * 2);  // Distance between (0,0,0) and (2,2,2)
    double true_volume = reference_intersection_volume(r, d);

    long long N_total = 1e8;
    long long N_local = N_total / size;

    std::mt19937 gen(rank + 12345);
    std::uniform_real_distribution<double> dist(lower, upper);

    double t_start = MPI_Wtime();

    long long count_local = 0;
    for (long long i = 0; i < N_local; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        double z = dist(gen);
        if (inside_sphere(x, y, z, 0, 0, 0, r) &&
            inside_sphere(x, y, z, 2, 2, 2, r)) {
            count_local++;
        }
    }

    long long count_total = 0;
    MPI_Reduce(&count_local, &count_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (rank == 0) {
        double p = static_cast<double>(count_total) / N_total;
        double volume_estimate = volume_box * p;
        double abs_error = std::abs(volume_estimate - true_volume);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Estimated intersection volume: " << volume_estimate << "\n";
        std::cout << "Absolute error: " << abs_error << "\n";
        std::cout << "Execution time: " << elapsed << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}