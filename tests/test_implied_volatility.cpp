// Tests for the implied-volatility inversion.
//
// The inversion is the one place in this library where the interesting failure
// mode is not "wrong answer" but "confidently wrong answer": a solver that
// returns a number for a quote that has no implied volatility at all is worse
// than one that refuses. So roughly half of these tests are about the refusal
// paths.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "opt/binomial_tree.hpp"
#include "opt/black_scholes.hpp"
#include "opt/greeks.hpp"
#include "opt/implied_volatility.hpp"
#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace {

using opt::IvMethod;
using opt::IvStatus;
using opt::MarketData;
using opt::OptionType;

}  // namespace

// -----------------------------------------------------------------------------
// Round trips
// -----------------------------------------------------------------------------

// Price with a known vol, invert, get the vol back. This is the whole contract.
TEST(ImpliedVolatility, RoundTripsAcrossTheSurface) {
    int checked = 0;
    for (const double spot : {40.0, 100.0, 400.0}) {
        for (const double moneyness : {0.7, 0.9, 1.0, 1.1, 1.4}) {
            for (const double expiry : {0.02, 0.25, 1.0, 5.0}) {
                for (const double vol : {0.05, 0.15, 0.35, 1.2}) {
                    for (const double rate : {-0.005, 0.0, 0.045}) {
                        for (const double div : {0.0, 0.025}) {
                            const MarketData market{spot, rate, div, vol};
                            const double strike = spot * moneyness;
                            for (const OptionType type : {OptionType::Call, OptionType::Put}) {
                                const double price =
                                    opt::black_scholes_price(market, strike, expiry, type);
                                const auto iv =
                                    opt::implied_volatility(price, market, strike, expiry, type);
                                ASSERT_EQ(iv.status, IvStatus::Converged)
                                    << "S=" << spot << " K=" << strike << " T=" << expiry
                                    << " vol=" << vol;

                                // The accuracy available in the *volatility* is
                                // not a free parameter: solving the price to
                                // within eps pins the vol only to eps/vega. So
                                // the tolerance is derived from the reported
                                // vega rather than picked, which turns the
                                // round trip into a test of the conditioning
                                // claim as well as of the root finder. Twenty
                                // times the theoretical floor leaves room for
                                // the curvature the linearisation ignores.
                                const double resolvable = 20.0 * opt::IvSettings{}.price_tolerance /
                                                          std::max(iv.vega, 1e-300);
                                EXPECT_NEAR(iv.volatility, vol, std::max(1e-9, resolvable))
                                    << "S=" << spot << " K=" << strike << " T=" << expiry
                                    << " vol=" << vol << " vega=" << iv.vega;
                                EXPECT_LE(std::abs(iv.price_residual), 1e-9);
                                ++checked;
                            }
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(checked, 3 * 5 * 4 * 4 * 3 * 2 * 2);
}

// Well-conditioned quotes should be solved by Newton, which is the fast path.
// If this ever starts failing, the solver is falling back to Brent everywhere
// and has quietly become an order of magnitude slower.
TEST(ImpliedVolatility, NearTheMoneyQuotesUseNewton) {
    const MarketData market{100.0, 0.04, 0.01, 0.25};
    for (const double strike : {85.0, 95.0, 100.0, 105.0, 120.0}) {
        const double price = opt::black_scholes_call(market, strike, 0.75);
        const auto iv = opt::implied_volatility(price, market, strike, 0.75, OptionType::Call);
        EXPECT_EQ(iv.method, IvMethod::SafeguardedNewton) << "K=" << strike;
        EXPECT_LE(iv.iterations, 12) << "K=" << strike;
    }
}

// -----------------------------------------------------------------------------
// The wings, where vega collapses
// -----------------------------------------------------------------------------

// A far out-of-the-money short-dated call is worth essentially nothing and its
// price barely responds to vol. The inversion still succeeds, but the reported
// vega is the number that says how much to trust it.
TEST(ImpliedVolatility, DeepOutOfTheMoneyConvergesButReportsTinyVega) {
    const MarketData market{100.0, 0.03, 0.0, 0.18};
    const double strike = 260.0;
    const double expiry = 0.05;
    const double price = opt::black_scholes_call(market, strike, expiry);

    ASSERT_LT(price, 1e-20);
    const auto iv = opt::implied_volatility(price, market, strike, expiry, OptionType::Call);
    EXPECT_EQ(iv.status, IvStatus::Converged);
    EXPECT_LT(iv.vega, 1e-12);

    // The economic content of "vega is 1e-12": moving the quote by one hundredth
    // of a cent moves the implied vol by more than the whole quoting range.
    const double vol_move_per_cent = 0.01 / iv.vega;
    EXPECT_GT(vol_move_per_cent, 1.0);
}

// A price that is flat in vol to the last bit cannot pin the vol down, but it
// must still pin the *price* down -- the solver is not allowed to return a
// volatility whose price misses the target.
TEST(ImpliedVolatility, FlatRegionStillMatchesThePrice) {
    const MarketData market{100.0, 0.03, 0.0, 0.18};
    const double price = opt::black_scholes_call(market, 500.0, 0.02);
    const auto iv = opt::implied_volatility(price, market, 500.0, 0.02, OptionType::Call);
    ASSERT_EQ(iv.status, IvStatus::Converged);
    MarketData recovered = market;
    recovered.volatility = iv.volatility;
    EXPECT_NEAR(opt::black_scholes_call(recovered, 500.0, 0.02), price, 1e-10);
}

// -----------------------------------------------------------------------------
// Refusal paths
// -----------------------------------------------------------------------------

TEST(ImpliedVolatility, RejectsPricesBelowIntrinsic) {
    const MarketData market{100.0, 0.05, 0.0, 0.2};
    const double floor_price = opt::black_scholes_call({100.0, 0.05, 0.0, 0.0}, 80.0, 1.0);
    const auto iv = opt::implied_volatility(floor_price - 0.5, market, 80.0, 1.0, OptionType::Call);
    EXPECT_EQ(iv.status, IvStatus::BelowIntrinsic);
    EXPECT_TRUE(std::isnan(iv.volatility));
    EXPECT_FALSE(iv.converged());
}

TEST(ImpliedVolatility, RejectsPricesAtOrAboveTheUpperBound) {
    const MarketData market{100.0, 0.05, 0.02, 0.2};
    // A call can never be worth more than the dividend-discounted spot.
    const double ceiling = market.spot * std::exp(-market.dividend * 1.0);
    const auto iv = opt::implied_volatility(ceiling * 1.01, market, 100.0, 1.0, OptionType::Call);
    EXPECT_EQ(iv.status, IvStatus::AboveUpperBound);
    EXPECT_TRUE(std::isnan(iv.volatility));
}

// A quote exactly at intrinsic is not a failure: zero volatility reproduces it.
TEST(ImpliedVolatility, PriceAtIntrinsicImpliesZeroVolatility) {
    const MarketData market{100.0, 0.05, 0.0, 0.2};
    const double intrinsic = opt::black_scholes_call({100.0, 0.05, 0.0, 0.0}, 80.0, 1.0);
    const auto iv = opt::implied_volatility(intrinsic, market, 80.0, 1.0, OptionType::Call);
    EXPECT_EQ(iv.status, IvStatus::Converged);
    EXPECT_DOUBLE_EQ(iv.volatility, 0.0);
}

TEST(ImpliedVolatility, RejectsMalformedInputs) {
    const MarketData market{100.0, 0.05, 0.0, 0.2};
    EXPECT_EQ(opt::implied_volatility(10.0, market, 100.0, 0.0, OptionType::Call).status,
              IvStatus::InvalidInput);
    EXPECT_EQ(opt::implied_volatility(10.0, market, -1.0, 1.0, OptionType::Call).status,
              IvStatus::InvalidInput);
    EXPECT_EQ(opt::implied_volatility(std::numeric_limits<double>::quiet_NaN(), market, 100.0, 1.0,
                                      OptionType::Call)
                  .status,
              IvStatus::InvalidInput);
}

// -----------------------------------------------------------------------------
// The initial guess
// -----------------------------------------------------------------------------

// Brenner-Subrahmanyam is exact only in the limit; at the forward with a
// moderate maturity it should land within a couple of vol points, which is what
// makes it worth having over a fixed starting value.
TEST(ImpliedVolatility, BrennerSubrahmanyamGuessIsCloseAtTheForward) {
    for (const double vol : {0.10, 0.20, 0.40}) {
        const MarketData market{100.0, 0.03, 0.01, vol};
        const double expiry = 0.5;
        const double forward_strike = 100.0 * std::exp((market.rate - market.dividend) * expiry);
        const double price = opt::black_scholes_call(market, forward_strike, expiry);
        const double guess = opt::brenner_subrahmanyam_guess(price, market, forward_strike, expiry,
                                                             OptionType::Call);
        EXPECT_NEAR(guess, vol, 0.02) << "true vol=" << vol;
    }
}

// The guess is allowed to be useless; it is not allowed to break the solve.
TEST(ImpliedVolatility, ConvergesEvenWhenTheGuessIsDegenerate) {
    const MarketData market{100.0, 0.0, 0.0, 0.40};
    const double strike = 40.0;  // deep ITM: almost all intrinsic, guess ~ 0
    const double price = opt::black_scholes_call(market, strike, 0.1);
    EXPECT_DOUBLE_EQ(opt::brenner_subrahmanyam_guess(price, market, strike, 0.1, OptionType::Call),
                     opt::brenner_subrahmanyam_guess(price, market, strike, 0.1, OptionType::Call));
    const auto iv = opt::implied_volatility(price, market, strike, 0.1, OptionType::Call);
    ASSERT_EQ(iv.status, IvStatus::Converged);
    EXPECT_LE(std::abs(iv.price_residual), 1e-10);
}

// -----------------------------------------------------------------------------
// Inverting the lattice instead of the formula
// -----------------------------------------------------------------------------

TEST(ImpliedVolatility, BinomialInversionRoundTripsOnAnAmericanPut) {
    constexpr int kSteps = 300;
    for (const double vol : {0.12, 0.30, 0.65}) {
        for (const double strike : {80.0, 100.0, 110.0}) {
            const MarketData market{100.0, 0.05, 0.01, vol};
            const opt::AmericanPut put{strike, 0.8};
            const double price = opt::binomial_price(put, market, kSteps);
            const auto iv = opt::implied_volatility_binomial(price, market, put, kSteps);
            ASSERT_EQ(iv.status, IvStatus::Converged) << "vol=" << vol << " K=" << strike;
            EXPECT_EQ(iv.method, IvMethod::Brent);
            ASSERT_GT(iv.vega, 1.0) << "vol=" << vol << " K=" << strike;
            EXPECT_NEAR(iv.volatility, vol, 1e-6) << "vol=" << vol << " K=" << strike;
        }
    }
}

// The case the strike list above deliberately avoids, tested on its own because
// it is a property of American options rather than a defect of the solver.
//
// A 25% in-the-money American put at a moderate volatility is worth exercising
// immediately, so its price is exactly its intrinsic value -- and stays exactly
// there for every volatility below the exercise threshold. The implied
// volatility is not identified: the solution set is an interval, not a point.
// The solver returns its smallest member and reports a vega of zero, which is
// the signal to drop the quote rather than plot it on a smile.
TEST(ImpliedVolatility, DeepInTheMoneyAmericanPutHasNoIdentifiedVolatility) {
    constexpr int kSteps = 300;
    const MarketData market{100.0, 0.05, 0.01, 0.12};
    const opt::AmericanPut put{125.0, 0.8};
    const double price = opt::binomial_price(put, market, kSteps);

    // Exercising now is worth K - S = 25 and the lattice agrees to the cent.
    ASSERT_NEAR(price, 25.0, 1e-9);

    // Two quite different volatilities produce the identical price, which is
    // what "not identified" means here.
    MarketData higher = market;
    higher.volatility = 0.20;
    EXPECT_DOUBLE_EQ(opt::binomial_price(put, higher, kSteps), price);

    const auto iv = opt::implied_volatility_binomial(price, market, put, kSteps);
    EXPECT_EQ(iv.status, IvStatus::Converged);
    EXPECT_DOUBLE_EQ(iv.volatility, 0.0);
    EXPECT_NEAR(iv.vega, 0.0, 1e-9);
}

// On a European contract the lattice and the formula are pricing the same thing,
// so their inversions must agree to within the tree's discretisation error.
TEST(ImpliedVolatility, BinomialAndAnalyticInversionsAgreeOnEuropeans) {
    constexpr int kSteps = 2000;
    const MarketData market{100.0, 0.04, 0.0, 0.22};
    for (const double strike : {85.0, 100.0, 115.0}) {
        const opt::EuropeanPut put{strike, 1.0};
        const double price = opt::black_scholes_put(market, strike, 1.0);
        const auto analytic = opt::implied_volatility(price, market, strike, 1.0, OptionType::Put);
        const auto lattice = opt::implied_volatility_binomial(price, market, put, kSteps);
        ASSERT_TRUE(analytic.converged() && lattice.converged());
        EXPECT_NEAR(analytic.volatility, lattice.volatility, 5e-4) << "K=" << strike;
    }
}

// The headline reason the lattice inversion exists. An American put is worth
// more than a European one at the same volatility, so explaining a given market
// price takes *less* volatility under the American model. Quoting a
// Black-Scholes IV on an American put therefore overstates the vol, and this
// test measures the overstatement instead of asserting it is small.
TEST(ImpliedVolatility, BlackScholesOverstatesVolOnAmericanPuts) {
    constexpr int kSteps = 1500;
    const MarketData market{100.0, 0.06, 0.0, 0.28};
    const opt::AmericanPut american{110.0, 1.0};

    const double american_price = opt::binomial_price(american, market, kSteps);
    const auto bs_iv = opt::implied_volatility(american_price, market, 110.0, 1.0, OptionType::Put);
    const auto tree_iv = opt::implied_volatility_binomial(american_price, market, american, kSteps);

    ASSERT_TRUE(bs_iv.converged() && tree_iv.converged());
    EXPECT_NEAR(tree_iv.volatility, market.volatility, 1e-5);
    EXPECT_GT(bs_iv.volatility, tree_iv.volatility);
    // Order of magnitude: a whole vol point on a one-year 10% OTM put.
    EXPECT_GT(bs_iv.volatility - tree_iv.volatility, 0.005);
}

TEST(ImpliedVolatility, BinomialInversionRejectsNonPositiveSteps) {
    const MarketData market{100.0, 0.05, 0.0, 0.2};
    const opt::AmericanPut put{100.0, 1.0};
    EXPECT_THROW((void)opt::implied_volatility_binomial(5.0, market, put, 0),
                 std::invalid_argument);
}

// A CRR lattice cannot produce a price between the zero-volatility value and
// the value at its own stability floor, where p = 1 and the tree is a single
// all-up path. The band is narrow, but a quote inside it implies a volatility
// the lattice has no way to represent, and saying so beats rounding it to zero
// or to the floor.
TEST(ImpliedVolatility, BinomialInversionReportsQuotesBelowTheLatticeFloor) {
    constexpr int kSteps = 100;
    constexpr double kExpiry = 1.0;
    const MarketData market{100.0, 0.05, 0.01, 0.0};
    // Struck at the forward, so the zero-volatility price is exactly zero and
    // the band starts there.
    const double forward = market.spot * std::exp((market.rate - market.dividend) * kExpiry);
    const opt::EuropeanCall call{forward, kExpiry};
    ASSERT_NEAR(opt::binomial_price(call, market, kSteps), 0.0, 1e-12);

    MarketData at_floor = market;
    at_floor.volatility =
        std::abs(market.rate - market.dividend) * std::sqrt(kExpiry / kSteps) * (1.0 + 1e-6);
    const double floor_price = opt::binomial_price(call, at_floor, kSteps);
    ASSERT_GT(floor_price, 0.0);

    const auto inside = opt::implied_volatility_binomial(0.5 * floor_price, market, call, kSteps);
    EXPECT_FALSE(inside.converged());
    EXPECT_EQ(inside.status, opt::IvStatus::BelowLatticeResolution);
    EXPECT_TRUE(std::isnan(inside.volatility));

    // Just above the band the inversion works again, landing on the floor.
    const auto outside = opt::implied_volatility_binomial(1.5 * floor_price, market, call, kSteps);
    EXPECT_TRUE(outside.converged());
    EXPECT_GE(outside.volatility, 0.99 * at_floor.volatility);
}
