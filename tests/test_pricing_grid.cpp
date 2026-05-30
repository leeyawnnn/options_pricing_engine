// Unit tests for the vectorised Black-Scholes pricing grid.
#include "opt/pricing_grid.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "opt/black_scholes.hpp"
#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace {

const opt::MarketData kBase{/*spot ignored=*/0.0, 0.05, 0.02, 0.20};

TEST(PriceGrid, VectorisedMatchesScalarLoop) {
    const Eigen::VectorXd spots = Eigen::VectorXd::LinSpaced(40, 50.0, 150.0);
    const Eigen::VectorXd strikes = Eigen::VectorXd::LinSpaced(35, 60.0, 140.0);

    // The vectorised grid uses the SIMD-friendly A&S CDF (|err| < 7.5e-8 in
    // Phi), so it differs from the erfc-based scalar engine by at most
    // ~forward * Phi-error ~ 1e-5; a 1e-4 bound is comfortable.
    for (const opt::OptionType type : {opt::OptionType::Call, opt::OptionType::Put}) {
        const Eigen::MatrixXd vec = opt::price_grid(kBase, spots, strikes, 1.0, type);
        const Eigen::MatrixXd loop = opt::price_grid_scalar(kBase, spots, strikes, 1.0, type);
        ASSERT_EQ(vec.rows(), spots.size());
        ASSERT_EQ(vec.cols(), strikes.size());
        EXPECT_LT((vec - loop).cwiseAbs().maxCoeff(), 1e-4);
    }
}

TEST(PriceGrid, CellsMatchScalarBlackScholes) {
    const Eigen::VectorXd spots = Eigen::VectorXd::LinSpaced(5, 80.0, 120.0);
    const Eigen::VectorXd strikes = Eigen::VectorXd::LinSpaced(5, 80.0, 120.0);
    const double expiry = 0.75;
    const Eigen::MatrixXd grid = opt::price_grid(kBase, spots, strikes, expiry, opt::OptionType::Call);

    for (Eigen::Index i = 0; i < spots.size(); ++i) {
        opt::MarketData m = kBase;
        m.spot = spots(i);
        for (Eigen::Index j = 0; j < strikes.size(); ++j) {
            EXPECT_NEAR(grid(i, j), opt::black_scholes_call(m, strikes(j), expiry), 1e-4);
        }
    }
}

}  // namespace
