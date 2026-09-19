#pragma once

#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace opt {

/// Why an implied-volatility inversion stopped.
///
/// The failure modes are enumerated rather than collapsed into a single "did
/// not converge" because on a real option chain they are not equally
/// interesting. A quote below intrinsic is a bad quote; a quote that needs more
/// than 500% vol to explain it is a bad quote; hitting the iteration limit on a
/// well-posed problem would be a bug in this solver.
enum class IvStatus {
    Converged,        ///< A volatility reproducing the target price was found.
    BelowIntrinsic,   ///< Target is under the sigma -> 0 price; no root exists.
    AboveUpperBound,  ///< Target is at or above the sigma -> inf limit.
    MaxIterations,    ///< Ran out of iterations without meeting the tolerance.
    InvalidInput,     ///< Non-finite input, or a non-positive time to expiry.
    /// The quote implies a volatility the pricing model cannot resolve. Only
    /// the lattice inversion reports this; see @ref implied_volatility_binomial
    /// for the (narrow) band of prices it covers.
    BelowLatticeResolution,
};

/// Which root-finder produced the answer.
enum class IvMethod {
    SafeguardedNewton,  ///< Newton steps, with bisection whenever one escapes.
    Brent,              ///< Derivative-free fallback used where vega collapses.
};

/// Tuning for an inversion. The defaults are deliberately tight; a market maker
/// would loosen @ref price_tolerance to something like a tick.
struct IvSettings {
    /// Absolute tolerance on |model price - target price|, in price units.
    double price_tolerance = 1e-10;
    /// Absolute tolerance on the width of the volatility bracket.
    double vol_tolerance = 1e-12;
    /// Hard iteration cap. The safeguarded methods below are guaranteed to
    /// converge, so hitting this means the tolerances are below what double
    /// precision can deliver for this quote.
    int max_iterations = 100;
    /// Upper end of the initial search bracket, in annualised vol units.
    double max_volatility = 5.0;
    /// Below this vega (per unit vol) Newton is not trustworthy and the solver
    /// switches to Brent. Deep out-of-the-money quotes live here.
    double min_vega_for_newton = 1e-8;
};

/// Outcome of an inversion.
struct IvResult {
    double volatility;      ///< The implied vol; NaN unless status is Converged.
    double vega;            ///< Vega at the solution, per unit vol. See below.
    double price_residual;  ///< model(volatility) - target, at the solution.
    int iterations;         ///< Root-finder iterations actually used.
    IvStatus status;        ///< Why the solve stopped.
    IvMethod method;        ///< Which root-finder was used.

    [[nodiscard]] bool converged() const noexcept { return status == IvStatus::Converged; }
};

/// Human-readable names, for CSV output and error messages.
[[nodiscard]] const char* to_string(IvStatus status) noexcept;
[[nodiscard]] const char* to_string(IvMethod method) noexcept;

/// Brenner-Subrahmanyam initial guess for the implied volatility.
///
/// For an option struck at the forward, \f$C \approx S e^{-qT}\sigma\sqrt{T} /
/// \sqrt{2\pi}\f$, which inverts to \f$\sigma_0 \approx \sqrt{2\pi/T}\,\cdot\,
/// C/(S e^{-qT})\f$. Away from the forward the formula is applied to the time
/// value (price minus the discounted forward intrinsic), which understates the
/// vol but stays on the right side of the root often enough to be a useful
/// starting point. It is only ever a starting point: the solver brackets the
/// root independently, so a bad guess costs iterations and never correctness.
///
/// Brenner, M. and Subrahmanyam, M. G. (1988), "A Simple Formula to Compute the
/// Implied Standard Deviation", Financial Analysts Journal 44(5), 80-83.
[[nodiscard]] double brenner_subrahmanyam_guess(double price, const MarketData& market,
                                                double strike, double expiry,
                                                OptionType type) noexcept;

/// Invert the Black-Scholes-Merton price for volatility.
///
/// The Black-Scholes price is strictly increasing in \f$\sigma\f$ on
/// \f$(0,\infty)\f$, bounded below by the discounted forward intrinsic and above
/// by \f$Se^{-qT}\f$ (calls) or \f$Ke^{-rT}\f$ (puts), so the root is unique
/// whenever the target lies strictly inside those bounds. The solver brackets it
/// on \f$[0, \text{max\_volatility}]\f$, widening the upper end geometrically if
/// the target is not yet enclosed, and then runs safeguarded Newton: a Newton
/// step is accepted only if it lands inside the current bracket, otherwise the
/// step is replaced by bisection. Where vega is below
/// @ref IvSettings::min_vega_for_newton -- the deep wings, where the price is
/// numerically flat in vol -- it uses Brent's method instead, which needs no
/// derivative.
///
/// @p market's `volatility` field is ignored; it is the unknown being solved for.
///
/// The returned @ref IvResult::vega is the conditioning diagnostic and should be
/// checked before the number is used: a converged inversion with vega of 1e-6
/// means a one-cent quote error moves the implied vol by ten thousand vol
/// points, so the answer is arithmetically correct and economically meaningless.
[[nodiscard]] IvResult implied_volatility(double price, const MarketData& market, double strike,
                                          double expiry, OptionType type,
                                          const IvSettings& settings = {});

/// Invert a Cox-Ross-Rubinstein lattice price for volatility.
///
/// This exists because listed single-name and ETF equity options are American,
/// so a Black-Scholes inversion of their quotes is inverting the wrong model.
/// For a put -- where early exercise has real value -- the two answers differ by
/// materially more than the bid-ask spread, and this function is what lets that
/// claim be measured rather than asserted.
///
/// The lattice price has no closed-form vega, so the inversion always uses
/// Brent, and @ref IvResult::vega is filled in afterwards by a central
/// difference on the lattice price. It costs @p steps times more work per
/// iteration than the closed-form inversion, which is why the analytic path
/// remains the default.
///
/// The search is confined to volatilities the lattice can actually represent.
/// A CRR tree needs \f$\sigma \ge |r-q|\sqrt{T/N}\f$ to keep its
/// risk-neutral probability in \f$[0,1]\f$, so the solver brackets just above
/// that floor rather than at zero.
///
/// That floor is less of a limitation than it looks. At \f$\sigma\f$ exactly
/// on it \f$p = 1\f$: the tree collapses to its single all-up path, which is
/// worth exactly the zero-volatility price. So the prices the lattice cannot
/// produce are a sliver immediately above that price, not a gap that more steps
/// would close -- raising @p steps lowers the floor *volatility* while leaving
/// the unreachable price band where it is. A quote landing inside it is one this
/// discretisation cannot tell apart from the deterministic price, and is
/// reported as @ref IvStatus::BelowLatticeResolution rather than rounded to
/// whichever side is nearer.
///
/// One case deserves naming because it is not an error and is easy to mistake
/// for one. A deep in-the-money American put is worth exercising immediately, so
/// its price is *exactly* flat in volatility over the whole immediate-exercise
/// region: every volatility below the exercise threshold reproduces the quote,
/// and the implied volatility is genuinely not identified. This function then
/// returns the smallest member of the solution set (zero) with a reported vega
/// of zero, which is the signal to discard the quote rather than plot it.
///
/// @param steps Lattice steps, \f$\ge 1\f$.
/// @throws std::invalid_argument if @p steps < 1.
[[nodiscard]] IvResult implied_volatility_binomial(double price, const MarketData& market,
                                                   const Option& option, int steps,
                                                   const IvSettings& settings = {});

}  // namespace opt
