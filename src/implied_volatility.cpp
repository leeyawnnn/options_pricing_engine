#include "opt/implied_volatility.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "opt/binomial_tree.hpp"
#include "opt/black_scholes.hpp"
#include "opt/greeks.hpp"

namespace opt {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// How far the upper bracket is allowed to grow, and how many doublings it gets
/// to grow in. 5.0 * 2^10 is 5120 vol points; a quote that needs more than that
/// is not a quote.
constexpr int kMaxBracketExpansions = 10;

[[nodiscard]] IvResult failure(IvStatus status, IvMethod method, int iterations) noexcept {
    return IvResult{kNaN, kNaN, kNaN, iterations, status, method};
}

/// Brent's method on a bracketed root of a continuous function.
///
/// Brent combines bisection (which always converges but only linearly) with
/// inverse quadratic interpolation and the secant method (which converge
/// superlinearly but can leave the bracket). It keeps the bracket invariant at
/// every step, so it inherits bisection's guarantee while usually running at
/// interpolation speed. `contracted` is Brent's bookkeeping for "the last step
/// made real progress": an interpolated step is only accepted if it is less than
/// half the size of the step before last, which is what stops the method from
/// stalling on a flat region.
///
/// Brent, R. P. (1973), Algorithms for Minimization Without Derivatives,
/// Prentice-Hall, chapter 4.
///
/// @param f      Objective; f(lo) and f(hi) must straddle zero.
/// @param x_tol  Absolute tolerance on the bracket width.
/// @param y_tol  Absolute tolerance on |f(x)|.
/// @return The root, and the iteration count written through @p iterations.
[[nodiscard]] double brent_root(const std::function<double(double)>& f, double lo, double hi,
                                double f_lo, double f_hi, double x_tol, double y_tol,
                                int max_iterations, int& iterations, bool& converged) {
    double a = lo;
    double b = hi;
    double fa = f_lo;
    double fb = f_hi;

    // b is kept as the better of the two endpoints.
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    double last_step = hi - lo;
    bool contracted = true;

    for (iterations = 0; iterations < max_iterations; ++iterations) {
        if (std::abs(fb) <= y_tol || std::abs(b - a) <= x_tol) {
            converged = true;
            return b;
        }

        double step = 0.0;
        const bool can_interpolate = fa != fc && fb != fc;
        if (can_interpolate) {
            // Inverse quadratic interpolation through (a, fa), (b, fb), (c, fc).
            step = a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc)) +
                   c * fa * fb / ((fc - fa) * (fc - fb));
            step -= b;
        } else if (fb != fa) {
            step = -fb * (b - a) / (fb - fa);  // secant
        }

        const double candidate = b + step;
        const double bracket_lo = std::min(a, b);
        const double bracket_hi = std::max(a, b);
        const bool inside = candidate > bracket_lo && candidate < bracket_hi;
        const bool shrinking = std::abs(step) < 0.5 * std::abs(last_step);
        if (!(inside && shrinking && contracted)) {
            step = 0.5 * (a - b);  // bisect
            contracted = false;
        } else {
            contracted = true;
        }

        last_step = step;
        c = b;
        fc = fb;
        b += step;
        fb = f(b);

        // Re-establish the bracket around the new b.
        if ((fa > 0.0) == (fb > 0.0)) {
            a = c;
            fa = fc;
        }
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    converged = false;
    return b;
}

/// Shared front end for both inversions: validate, bracket, then hand off.
///
/// @param price_at   Model price as a function of volatility. Must be
///                   non-decreasing, which every European and American
///                   Black-Scholes-family price is.
/// @param vega_at    Analytic dPrice/dSigma, or nullptr when none is available
///                   (the lattice), in which case Brent is always used.
/// @param guess      Starting volatility for the Newton phase; ignored when
///                   @p vega_at is nullptr. Correctness never depends on it.
/// @param min_vol    Smallest volatility @p price_at may be evaluated at, other
///                   than zero itself. Zero for a closed-form price, which is
///                   defined everywhere; positive for the lattice, which is not
///                   a lattice below |r-q| sqrt(T/N).
/// @param max_vol    Largest volatility @p price_at may be evaluated at.
///                   Infinite for a closed-form price; finite for the lattice,
///                   whose spot ladder leaves double's range at high vol. The
///                   bracket never widens past it.
[[nodiscard]] IvResult invert_with_guess(double target,
                                         const std::function<double(double)>& price_at,
                                         const std::function<double(double)>* vega_at, double guess,
                                         double min_vol, double max_vol,
                                         const IvSettings& settings) {
    if (!std::isfinite(target) || !(settings.max_volatility > 0.0) || settings.max_iterations < 1) {
        return failure(IvStatus::InvalidInput, IvMethod::Brent, 0);
    }

    // Lower end of the bracket: sigma = 0 is the discounted forward intrinsic,
    // the cheapest an arbitrage-free option can be.
    const double price_at_zero = price_at(0.0);
    if (target < price_at_zero - settings.price_tolerance) {
        return failure(IvStatus::BelowIntrinsic, IvMethod::Brent, 0);
    }
    if (target <= price_at_zero + settings.price_tolerance) {
        // The quote is the intrinsic value; zero vol reproduces it exactly and
        // there is nothing to solve.
        return IvResult{0.0, 0.0, price_at_zero - target, 0, IvStatus::Converged, IvMethod::Brent};
    }

    // Upper end: widen geometrically until the target is enclosed. The price is
    // bounded above as sigma -> infinity, so a target at or beyond that ceiling
    // never gets bracketed and is reported rather than iterated on forever.
    double hi = std::min(settings.max_volatility, max_vol);
    if (!(hi > min_vol)) {
        return failure(IvStatus::InvalidInput, IvMethod::Brent, 0);
    }
    double price_at_hi = price_at(hi);
    for (int i = 0; i < kMaxBracketExpansions && price_at_hi < target && hi < max_vol; ++i) {
        hi = std::min(hi * 2.0, max_vol);
        price_at_hi = price_at(hi);
    }
    if (price_at_hi < target) {
        return failure(IvStatus::AboveUpperBound, IvMethod::Brent, kMaxBracketExpansions);
    }

    const auto residual = [&](double vol) { return price_at(vol) - target; };

    // The search runs from the floor rather than from zero, because below it
    // price_at is not defined (the lattice) or not needed (the closed form,
    // where the floor is zero). A target between the zero-volatility price and
    // the price at the floor has a root in a gap the model cannot represent;
    // that is a real answer about the model, not a solver failure, so it is
    // reported rather than rounded to whichever end is nearer.
    double lo = min_vol;
    double f_lo = min_vol > 0.0 ? residual(min_vol) : price_at_zero - target;
    if (f_lo > settings.price_tolerance) {
        return failure(IvStatus::BelowLatticeResolution, IvMethod::Brent, 0);
    }
    double f_hi = price_at_hi - target;

    // Safeguarded Newton, when a usable derivative exists.
    if (vega_at != nullptr) {
        // A guess of zero (the formula's answer when the quote has no time
        // value to speak of) would start Newton exactly where vega is zero, so
        // fall back to the middle of the bracket.
        double vol = guess > lo && guess < hi ? guess : 0.5 * (lo + hi);

        for (int iteration = 1; iteration <= settings.max_iterations; ++iteration) {
            const double f = residual(vol);
            if (f > 0.0) {
                hi = vol;
                f_hi = f;
            } else {
                lo = vol;
                f_lo = f;
            }
            if (std::abs(f) <= settings.price_tolerance || (hi - lo) <= settings.vol_tolerance) {
                return IvResult{vol,       (*vega_at)(vol),     f,
                                iteration, IvStatus::Converged, IvMethod::SafeguardedNewton};
            }

            const double vega = (*vega_at)(vol);
            if (!(std::abs(vega) > settings.min_vega_for_newton)) {
                break;  // vega has collapsed; hand the live bracket to Brent
            }

            const double next = vol - f / vega;
            // Reject a Newton step that leaves the bracket and bisect instead.
            vol = (next > lo && next < hi) ? next : 0.5 * (lo + hi);
        }
    }

    int iterations = 0;
    bool converged = false;
    const double root =
        brent_root(residual, lo, hi, f_lo, f_hi, settings.vol_tolerance, settings.price_tolerance,
                   settings.max_iterations, iterations, converged);
    if (!converged) {
        return failure(IvStatus::MaxIterations, IvMethod::Brent, iterations);
    }
    return IvResult{root,
                    vega_at != nullptr ? (*vega_at)(root) : kNaN,
                    residual(root),
                    iterations,
                    IvStatus::Converged,
                    IvMethod::Brent};
}

}  // namespace

const char* to_string(IvStatus status) noexcept {
    switch (status) {
        case IvStatus::Converged:
            return "converged";
        case IvStatus::BelowIntrinsic:
            return "below_intrinsic";
        case IvStatus::AboveUpperBound:
            return "above_upper_bound";
        case IvStatus::MaxIterations:
            return "max_iterations";
        case IvStatus::InvalidInput:
            return "invalid_input";
        case IvStatus::BelowLatticeResolution:
            return "below_lattice_resolution";
    }
    return "unknown";
}

const char* to_string(IvMethod method) noexcept {
    switch (method) {
        case IvMethod::SafeguardedNewton:
            return "newton";
        case IvMethod::Brent:
            return "brent";
    }
    return "unknown";
}

double brenner_subrahmanyam_guess(double price, const MarketData& market, double strike,
                                  double expiry, OptionType type) noexcept {
    if (!(expiry > 0.0) || !(market.spot > 0.0)) {
        return 0.0;
    }
    const double forward = market.spot * std::exp(-market.dividend * expiry);
    const double pv_strike = strike * std::exp(-market.rate * expiry);
    const double intrinsic = type == OptionType::Call ? std::max(forward - pv_strike, 0.0)
                                                      : std::max(pv_strike - forward, 0.0);
    const double time_value = price - intrinsic;
    if (!(time_value > 0.0)) {
        return 0.0;
    }
    constexpr double two_pi = 6.283185307179586;
    return std::sqrt(two_pi / expiry) * time_value / forward;
}

IvResult implied_volatility(double price, const MarketData& market, double strike, double expiry,
                            OptionType type, const IvSettings& settings) {
    if (!(expiry > 0.0) || !(strike > 0.0) || !(market.spot > 0.0) || !std::isfinite(price)) {
        return IvResult{kNaN, kNaN, kNaN, 0, IvStatus::InvalidInput, IvMethod::Brent};
    }

    MarketData scratch = market;
    const std::function<double(double)> price_at = [&](double vol) {
        scratch.volatility = vol;
        return black_scholes_price(scratch, strike, expiry, type);
    };
    const std::function<double(double)> vega_at = [&](double vol) {
        scratch.volatility = vol;
        return bs_vega(scratch, strike, expiry);
    };
    const double guess = brenner_subrahmanyam_guess(price, market, strike, expiry, type);
    return invert_with_guess(price, price_at, &vega_at, guess, 0.0,
                             std::numeric_limits<double>::infinity(), settings);
}

IvResult implied_volatility_binomial(double price, const MarketData& market, const Option& option,
                                     int steps, const IvSettings& settings) {
    if (steps < 1) {
        throw std::invalid_argument("implied_volatility_binomial: steps must be >= 1");
    }
    if (!(option.expiry() > 0.0) || !(market.spot > 0.0) || !std::isfinite(price)) {
        return IvResult{kNaN, kNaN, kNaN, 0, IvStatus::InvalidInput, IvMethod::Brent};
    }

    MarketData scratch = market;
    const std::function<double(double)> price_at = [&](double vol) {
        scratch.volatility = vol;
        return binomial_price(option, scratch, steps);
    };

    // A CRR lattice keeps its risk-neutral probability in [0, 1] only for
    // sigma >= |r-q| sqrt(dt); binomial_price throws below that. The tiny pad
    // keeps the solver off the boundary itself, where the comparison is decided
    // by rounding in sqrt(dt) * sqrt(dt) rather than by the inputs.
    const double dt = option.expiry() / static_cast<double>(steps);
    const double min_vol = std::abs(market.rate - market.dividend) * std::sqrt(dt) * (1.0 + 1e-6);

    // Mirror of the floor: above this the lattice's spot ladder overflows a
    // double and binomial_price refuses, so the bracket must not widen past it.
    // The pad plays the same role as the one on the floor -- the two sides
    // compute the same boundary by different roundings, and landing exactly on
    // it must not decide the throw.
    const double max_vol = 700.0 / (std::sqrt(dt) * static_cast<double>(steps)) * (1.0 - 1e-6);

    IvResult result = invert_with_guess(price, price_at, nullptr, 0.0, min_vol, max_vol, settings);
    if (result.converged()) {
        // The lattice has no closed-form vega, but callers need the same
        // conditioning diagnostic the analytic path gives them -- above all for
        // a deep in-the-money American option, whose price is exactly flat in
        // volatility throughout the immediate-exercise region, so that every
        // volatility below the exercise threshold reproduces the quote and the
        // implied volatility is not identified at all. A reported vega of zero
        // is how that shows up.
        if (!(result.volatility > 0.0)) {
            // The quote is the sigma = 0 price. Differencing around it would
            // mean pricing a lattice that does not exist down there, and the
            // answer is known anyway: flat price, no vega, vol not identified.
            result.vega = 0.0;
        } else {
            const double bump = std::max(1e-4, 1e-3 * result.volatility);
            const double lo = std::max(result.volatility - bump, min_vol);
            const double hi = result.volatility + bump;
            result.vega = (price_at(hi) - price_at(lo)) / (hi - lo);
        }
    }
    return result;
}

}  // namespace opt
