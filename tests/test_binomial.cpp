// Unit tests for the Cox-Ross-Rubinstein binomial tree pricer.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "opt/binomial_tree.hpp"
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

// -----------------------------------------------------------------------------
// The zero-volatility limit
// -----------------------------------------------------------------------------

// With sigma = 0 the CRR parameterisation degenerates: u = d = 1, so the
// risk-neutral probability (growth - d)/(u - d) is 0/0 and the recursion cannot
// be evaluated. The pricer used to walk straight into that division and return
// NaN. These tests pin the deterministic branch that replaces it.

TEST(BinomialTree, ZeroVolatilityEuropeanMatchesClosedForm) {
    for (const double rate : {-0.01, 0.0, 0.05}) {
        for (const double div : {0.0, 0.03}) {
            const opt::MarketData market{100.0, rate, div, 0.0};
            for (const double strike : {70.0, 100.0, 130.0}) {
                const opt::EuropeanCall call{strike, 1.5};
                const opt::EuropeanPut put{strike, 1.5};
                EXPECT_NEAR(opt::binomial_price(call, market, 200),
                            opt::black_scholes_call(market, strike, 1.5), 1e-12)
                    << "K=" << strike << " r=" << rate << " q=" << div;
                EXPECT_NEAR(opt::binomial_price(put, market, 200),
                            opt::black_scholes_put(market, strike, 1.5), 1e-12)
                    << "K=" << strike << " r=" << rate << " q=" << div;
            }
        }
    }
}

// An American option with no volatility is not simply the European one: the
// holder still chooses when to exercise along the deterministic forward path.
// A deep in-the-money put takes the money now rather than watch the forward
// drift away from the strike.
TEST(BinomialTree, ZeroVolatilityAmericanPutExercisesImmediately) {
    const opt::MarketData market{100.0, 0.05, 0.01, 0.0};
    const opt::AmericanPut american{125.0, 0.8};
    const opt::EuropeanPut european{125.0, 0.8};

    EXPECT_NEAR(opt::binomial_price(american, market, 400), 25.0, 1e-9);
    EXPECT_GT(opt::binomial_price(american, market, 400),
              opt::binomial_price(european, market, 400));
}

// The deterministic branch has to join up with the stochastic one, not sit
// beside it: the lattice price must approach the zero-vol value as vol shrinks.
//
// This cannot be tested by driving vol to zero at a fixed step count. A CRR
// lattice needs sigma >= |r-q| sqrt(T/N) to keep its risk-neutral probability
// in [0, 1], and binomial_price refuses the regime below that -- see
// RejectsVolatilityBelowTheCrrStabilityFloor. So the approach is tested over
// the range where the lattice is a lattice.
//
// The option is struck at the forward, which makes the zero-vol price exactly
// zero: every cent of the lattice price is then time value, the quantity that
// has to vanish with sigma. It is an American call on an underlying yielding
// less than the rate, so early exercise is never optimal and the value is a
// smooth function of sigma rather than one pinned by an exercise boundary --
// which is what the old deep-in-the-money put was, making this test vacuous.
TEST(BinomialTree, PriceApproachesTheZeroVolatilityLimit) {
    constexpr int kSteps = 400;
    constexpr double kExpiry = 0.8;
    opt::MarketData market{100.0, 0.05, 0.01, 0.0};
    const double forward = market.spot * std::exp((market.rate - market.dividend) * kExpiry);
    const opt::AmericanCall call{forward, kExpiry};

    const double limit_value = opt::binomial_price(call, market, kSteps);
    ASSERT_NEAR(limit_value, 0.0, 1e-12);

    double previous_gap = std::numeric_limits<double>::infinity();
    double previous_vol = 0.0;
    for (const double vol : {0.2, 0.1, 0.05, 0.02, 0.01}) {
        market.volatility = vol;
        const double gap = std::abs(opt::binomial_price(call, market, kSteps) - limit_value);
        EXPECT_LT(gap, previous_gap) << "vol=" << vol;
        if (previous_vol > 0.0) {
            // Time value is linear in sigma to leading order, so halving sigma
            // has to halve the gap. This is the statement that the gap really
            // is heading for zero, rather than merely decreasing.
            EXPECT_NEAR(gap / previous_gap, vol / previous_vol, 0.02) << "vol=" << vol;
        }
        previous_gap = gap;
        previous_vol = vol;
    }
}

// Below the CRR stability floor the risk-neutral probability leaves [0, 1] and
// the rollback stops being an expectation -- one branch carries a negative
// weight. It does not fail loudly: it silently returned 0.000 for an
// at-the-forward call worth 0.035. A refused answer beats a wrong one.
TEST(BinomialTree, RejectsVolatilityBelowTheCrrStabilityFloor) {
    constexpr int kSteps = 400;
    constexpr double kExpiry = 0.8;
    opt::MarketData market{100.0, 0.05, 0.01, 0.0};
    const double forward = market.spot * std::exp((market.rate - market.dividend) * kExpiry);
    const opt::AmericanCall call{forward, kExpiry};

    const double floor_vol = std::abs(market.rate - market.dividend) * std::sqrt(kExpiry / kSteps);

    market.volatility = floor_vol * 1.01;
    EXPECT_NO_THROW((void)opt::binomial_price(call, market, kSteps));

    market.volatility = floor_vol * 0.99;
    EXPECT_THROW((void)opt::binomial_price(call, market, kSteps), std::invalid_argument);

    // The floor falls as the tree is refined, so the same quote becomes
    // priceable with more steps -- which is what the message tells the caller
    // to do. N must grow as 1/sigma^2 for that, hence the jump.
    EXPECT_NO_THROW((void)opt::binomial_price(call, market, 4 * kSteps));
}

// The lattice's spot ladder spans S * e^{+/- sigma sqrt(T) sqrt(N)}. Once that
// exponent passes exp's range the top of the ladder is +inf and the bottom is
// 0, and the rollback turns either into a confident number: an infinite
// terminal payoff propagates all the way down, and a ladder built by
// multiplying up from an underflowed base is zero everywhere, pricing a
// valuable call at nothing. Both were silent, so the regime is refused instead.
TEST(BinomialTree, RejectsLatticesThatLeaveDoublesRange) {
    const opt::MarketData market{100.0, 0.05, 0.01, 25.0};
    const opt::EuropeanCall call{100.0, 1.0};

    // 25 * sqrt(256) = 400 is inside the range; 25 * sqrt(1024) = 800 is not.
    double priced = 0.0;
    EXPECT_NO_THROW(priced = opt::binomial_price(call, market, 256));
    EXPECT_TRUE(std::isfinite(priced));
    EXPECT_NEAR(priced, opt::black_scholes_call(market, call.strike(), call.expiry()), 1e-6);

    EXPECT_THROW((void)opt::binomial_price(call, market, 1024), std::invalid_argument);
}

TEST(BinomialTree, ZeroVolatilityPricesAreFinite) {
    const opt::MarketData market{100.0, 0.05, 0.0, 0.0};
    for (const double strike : {50.0, 100.0, 200.0}) {
        const opt::AmericanCall call{strike, 1.0};
        const opt::AmericanPut put{strike, 1.0};
        EXPECT_TRUE(std::isfinite(opt::binomial_price(call, market, 100)));
        EXPECT_TRUE(std::isfinite(opt::binomial_price(put, market, 100)));
    }
}
