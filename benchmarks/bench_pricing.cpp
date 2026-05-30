// Loop-vs-vectorised pricing benchmark using a hand-rolled steady_clock timer
// (no external benchmark dependency). Reports best-of-N wall time, throughput
// in options/second, and the speedup of the Eigen grid over the scalar loop.
#include <algorithm>
#include <chrono>
#include <cstdio>

#include <Eigen/Core>

#include "opt/market_data.hpp"
#include "opt/option.hpp"
#include "opt/pricing_grid.hpp"

namespace {

// Runs `fn` `repeats` times and returns the fastest elapsed time in seconds.
// `sink` accumulates a value from each run so the optimiser cannot elide it.
template <typename Fn>
double best_time(int repeats, double& sink, Fn&& fn) {
    double best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        const Eigen::MatrixXd result = fn();
        const auto stop = std::chrono::steady_clock::now();
        sink += result(0, 0);
        best = std::min(best, std::chrono::duration<double>(stop - start).count());
    }
    return best;
}

}  // namespace

int main() {
    const opt::MarketData base{/*spot ignored=*/0.0, 0.05, 0.02, 0.20};
    const double expiry = 1.0;
    constexpr int kRepeats = 5;
    double sink = 0.0;

    std::printf("%-12s %14s %14s %10s %16s %16s\n", "grid", "scalar (ms)", "eigen (ms)",
                "speedup", "scalar opt/s", "eigen opt/s");
    std::printf("--------------------------------------------------------------------------------\n");

    for (const int dim : {100, 250, 500, 1000}) {
        const Eigen::VectorXd spots = Eigen::VectorXd::LinSpaced(dim, 50.0, 150.0);
        const Eigen::VectorXd strikes = Eigen::VectorXd::LinSpaced(dim, 60.0, 140.0);
        const double cells = static_cast<double>(dim) * dim;

        // Warm up (and validate the two agree on the first cell).
        sink += opt::price_grid(base, spots, strikes, expiry, opt::OptionType::Call)(0, 0);

        const double scalar_s = best_time(kRepeats, sink, [&] {
            return opt::price_grid_scalar(base, spots, strikes, expiry, opt::OptionType::Call);
        });
        const double eigen_s = best_time(kRepeats, sink, [&] {
            return opt::price_grid(base, spots, strikes, expiry, opt::OptionType::Call);
        });

        std::printf("%5dx%-6d %14.3f %14.3f %9.2fx %16.3e %16.3e\n", dim, dim, scalar_s * 1e3,
                    eigen_s * 1e3, scalar_s / eigen_s, cells / scalar_s, cells / eigen_s);
    }

    // Keep the sink observable so nothing above is optimised away.
    std::printf("\n(checksum %.6f)\n", sink);
    return 0;
}
