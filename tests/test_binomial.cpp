// Unit tests for the Cox-Ross-Rubinstein binomial tree pricer.
#include "opt/binomial_tree.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "opt/black_scholes.hpp"
#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace {

// European prices converge to Black-Scholes, with error shrinking like ~1/N.
TEST(Binomial, EuropeanConvergesToBlackScholes) {
    const opt::MarketData mkt{100.0, 0.05, 0.0, 0.20};
    const opt::EuropeanCall call{100.0, 1.0};
    const double bs = opt::black_scholes_call(mkt, call.strike(), call.expiry());

    const double err_100 = std::abs(opt::binomial_price(call, mkt, 100) - bs);
    const double err_1000 = std::abs(opt::binomial_price(call, mkt, 1000) - bs);
    const double err_10000 = std::abs(opt::binomial_price(call, mkt, 10000) - bs);

    // Monotone improvement across the decade jumps.
    EXPECT_LT(err_1000, err_100);
    EXPECT_LT(err_10000, err_1000);
    EXPECT_LT(err_10000, 1e-3);

    // First-order convergence: error * N stays roughly constant (it would grow
    // by ~10x per decade for the 1/sqrt(N) of a poor scheme).
    EXPECT_LT(err_10000 * 10000.0, 3.0 * err_100 * 100.0);
}

TEST(Binomial, EuropeanPutConvergesToBlackScholes) {
    const opt::MarketData mkt{100.0, 0.05, 0.0, 0.20};
    const opt::EuropeanPut put{100.0, 1.0};
    const double bs = opt::black_scholes_put(mkt, put.strike(), put.expiry());
    EXPECT_NEAR(opt::binomial_price(put, mkt, 5000), bs, 1e-3);
}

// Merton: an American call on a non-dividend-paying stock is never exercised
// early, so it equals the European call -- and on the lattice it is in fact
// computed by the identical roll-back, hence bit-for-bit equal.
TEST(Binomial, AmericanCallEqualsEuropeanCallWithoutDividends) {
    const opt::MarketData mkt{100.0, 0.08, 0.0, 0.25};
    const opt::EuropeanCall euro{95.0, 1.0};
    const opt::AmericanCall amer{95.0, 1.0};
    for (const int n : {50, 200, 1000}) {
        EXPECT_DOUBLE_EQ(opt::binomial_price(amer, mkt, n), opt::binomial_price(euro, mkt, n));
    }
}

// The early-exercise premium on an American put is strictly positive.
TEST(Binomial, AmericanPutExceedsEuropeanPut) {
    const opt::MarketData mkt{100.0, 0.08, 0.0, 0.20};
    const opt::EuropeanPut euro{110.0, 1.0};  // in the money
    const opt::AmericanPut amer{110.0, 1.0};
    const int n = 1000;
    const double euro_price = opt::binomial_price(euro, mkt, n);
    const double amer_price = opt::binomial_price(amer, mkt, n);
    EXPECT_GT(amer_price, euro_price);
    EXPECT_GT(amer_price - euro_price, 1e-3);      // a meaningful premium
    EXPECT_GE(amer_price, amer.payoff(mkt.spot));  // never worth less than intrinsic
}

TEST(Binomial, ZeroTimeReturnsIntrinsicValue) {
    const opt::MarketData mkt{110.0, 0.05, 0.0, 0.20};
    const opt::EuropeanCall call{100.0, 0.0};
    EXPECT_DOUBLE_EQ(opt::binomial_price(call, mkt, 100), 10.0);
}

TEST(Binomial, RejectsNonPositiveSteps) {
    const opt::MarketData mkt{100.0, 0.05, 0.0, 0.20};
    const opt::EuropeanCall call{100.0, 1.0};
    EXPECT_THROW((void)opt::binomial_price(call, mkt, 0), std::invalid_argument);
    EXPECT_THROW((void)opt::binomial_price(call, mkt, -5), std::invalid_argument);
}

}  // namespace
