// Unit tests for the standard normal helpers and Black-Scholes pricing.
#include "opt/black_scholes.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "opt/market_data.hpp"
#include "opt/normal_distribution.hpp"

namespace {

// ---------------------------------------------------------------------------
// Standard normal distribution
// ---------------------------------------------------------------------------

TEST(NormalDistribution, CdfKnownValues) {
    // The erfc-based CDF is accurate to ~machine epsilon; Phi(0) is exact.
    EXPECT_NEAR(opt::cumulative_normal(0.0), 0.5, 1e-15);
    EXPECT_NEAR(opt::cumulative_normal(1.0), 0.8413447, 1e-7);
    EXPECT_NEAR(opt::cumulative_normal(-1.0), 0.1586553, 1e-7);
    EXPECT_NEAR(opt::cumulative_normal(1.96), 0.9750021, 1e-7);
    EXPECT_NEAR(opt::cumulative_normal(-2.5758), 0.0049998, 1e-6);
}

TEST(NormalDistribution, CdfTailsSaturate) {
    EXPECT_GT(opt::cumulative_normal(10.0), 1.0 - 1e-7);
    EXPECT_LT(opt::cumulative_normal(-10.0), 1e-7);
}

TEST(NormalDistribution, CdfReflectionSymmetry) {
    for (const double x : {0.1, 0.5, 1.0, 2.5, 4.0}) {
        EXPECT_NEAR(opt::cumulative_normal(-x), 1.0 - opt::cumulative_normal(x), 1e-12);
    }
}

TEST(NormalDistribution, PdfKnownValues) {
    EXPECT_NEAR(opt::standard_normal_pdf(0.0), 0.3989422804, 1e-10);
    EXPECT_NEAR(opt::standard_normal_pdf(1.0), 0.2419707245, 1e-10);
    EXPECT_NEAR(opt::standard_normal_pdf(-1.0), 0.2419707245, 1e-10);
}

// ---------------------------------------------------------------------------
// Black-Scholes pricing
// ---------------------------------------------------------------------------

// Three worked examples from Hull, "Options, Futures, and Other Derivatives",
// 8th ed., Ch. 15 (Black-Scholes-Merton), all on non-dividend-paying stocks.
TEST(BlackScholes, MatchesHullTextbookValues) {
    {
        const opt::MarketData mkt{/*spot=*/42.0, /*rate=*/0.10, /*dividend=*/0.0,
                                  /*volatility=*/0.20};
        EXPECT_NEAR(opt::black_scholes_call(mkt, 40.0, 0.5), 4.7594, 1e-3);
        EXPECT_NEAR(opt::black_scholes_put(mkt, 40.0, 0.5), 0.8086, 1e-3);
    }
    {
        const opt::MarketData mkt{60.0, 0.08, 0.0, 0.30};
        EXPECT_NEAR(opt::black_scholes_call(mkt, 65.0, 0.25), 2.1334, 1e-3);
        EXPECT_NEAR(opt::black_scholes_put(mkt, 65.0, 0.25), 5.8463, 1e-3);
    }
    {
        const opt::MarketData mkt{100.0, 0.05, 0.0, 0.20};
        EXPECT_NEAR(opt::black_scholes_call(mkt, 100.0, 1.0), 10.4506, 1e-3);
        EXPECT_NEAR(opt::black_scholes_put(mkt, 100.0, 1.0), 5.5735, 1e-3);
    }
}

TEST(BlackScholes, PriceDispatchMatchesNamedFunctions) {
    const opt::MarketData mkt{100.0, 0.05, 0.02, 0.25};
    EXPECT_DOUBLE_EQ(opt::black_scholes_price(mkt, 95.0, 1.5, opt::OptionType::Call),
                     opt::black_scholes_call(mkt, 95.0, 1.5));
    EXPECT_DOUBLE_EQ(opt::black_scholes_price(mkt, 95.0, 1.5, opt::OptionType::Put),
                     opt::black_scholes_put(mkt, 95.0, 1.5));
}

TEST(BlackScholes, PutCallParityHoldsAcrossRandomParameters) {
    std::mt19937_64 rng{0xC0FFEEULL};
    std::uniform_real_distribution<double> spot(1.0, 200.0);
    std::uniform_real_distribution<double> strike(1.0, 200.0);
    std::uniform_real_distribution<double> rate(-0.05, 0.15);
    std::uniform_real_distribution<double> dividend(0.0, 0.05);
    std::uniform_real_distribution<double> vol(0.01, 1.0);
    std::uniform_real_distribution<double> expiry(0.01, 5.0);

    for (int i = 0; i < 100; ++i) {
        const opt::MarketData mkt{spot(rng), rate(rng), dividend(rng), vol(rng)};
        const double residual = opt::put_call_parity_residual(mkt, strike(rng), expiry(rng));
        EXPECT_LT(std::abs(residual), 1e-10);
    }
}

TEST(BlackScholes, ZeroVolatilityEqualsPresentValueOfCertainPayoff) {
    const opt::MarketData mkt{100.0, 0.05, 0.02, /*volatility=*/0.0};
    const double strike = 95.0;
    const double expiry = 2.0;
    const double forward = 100.0 * std::exp(-0.02 * expiry);
    const double pv_strike = strike * std::exp(-0.05 * expiry);

    EXPECT_NEAR(opt::black_scholes_call(mkt, strike, expiry),
                std::max(forward - pv_strike, 0.0), 1e-12);
    EXPECT_NEAR(opt::black_scholes_put(mkt, strike, expiry),
                std::max(pv_strike - forward, 0.0), 1e-12);

    // A deep out-of-the-money call must clamp to zero, not go negative.
    EXPECT_DOUBLE_EQ(opt::black_scholes_call(mkt, 1000.0, expiry), 0.0);
}

TEST(BlackScholes, ZeroTimeToExpiryEqualsIntrinsicValue) {
    {
        const opt::MarketData itm{110.0, 0.05, 0.02, 0.20};
        EXPECT_NEAR(opt::black_scholes_call(itm, 100.0, 0.0), 10.0, 1e-12);
        EXPECT_NEAR(opt::black_scholes_put(itm, 100.0, 0.0), 0.0, 1e-12);
    }
    {
        const opt::MarketData otm{90.0, 0.05, 0.02, 0.20};
        EXPECT_NEAR(opt::black_scholes_call(otm, 100.0, 0.0), 0.0, 1e-12);
        EXPECT_NEAR(opt::black_scholes_put(otm, 100.0, 0.0), 10.0, 1e-12);
    }
}

}  // namespace
