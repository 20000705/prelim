#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <chrono>

// Check if point is inside the projection of a sphere (2D circle)
bool inside_projection(double x, double y, double cx, double cy, double r) {
    double dx = x - cx;
    double dy = y - cy;
    return dx * dx + dy * dy <= r * r;
}

// Calculate overlap area of two circles with equal radius and distance d
double compute_overlap_area(double r, double d) {
    if (d >= 2 * r) return 0.0; // No overlap
    double part1 = 2 * r * r * std::acos(d / (2 * r));
    double part2 = 0.5 * d * std::sqrt(4 * r * r - d * d);
    return part1 - part2;
}

int main() {
    const double r = 5.0;
    const double d = std::sqrt(8.0); // Distance between (0,0) and (2,2)

    // Theoretical reference area
    double circle_area = M_PI * r * r;
    double overlap_area = compute_overlap_area(r, d);
    double expected_area = 2 * circle_area - overlap_area;

    // Bounding box and area
    double lower = -5.0, upper = 7.0;
    double area_box = std::pow(upper - lower, 2);

    // Monte Carlo samples
    long long N = 1e8;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(lower, upper);

    // Start timing
    auto t_start = std::chrono::high_resolution_clock::now();

    // Count how many points fall in either projection
    long long count = 0;
    for (long long i = 0; i < N; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        if (inside_projection(x, y, 0, 0, r) || inside_projection(x, y, 2, 2, r)) {
            count++;
        }
    }

    // Estimate area
    double p = static_cast<double>(count) / N;
    double area_estimate = area_box * p;

    // Compute absolute error
    double abs_error = std::abs(area_estimate - expected_area);

    // End timing
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    // Output results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Estimated shadow area: " << area_estimate << " units²\n";
    std::cout << "Expected shadow area:  " << expected_area << " units²\n";
    std::cout << "Absolute error:        " << abs_error << "\n";
    std::cout << "Execution time:        " << elapsed.count() << " seconds\n";

    return 0;
}