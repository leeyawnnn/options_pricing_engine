#pragma once

namespace opt {

/// Market observables needed to price an option under a Black-Scholes-style
/// model. This is a plain data carrier (no global state): callers construct one
/// and pass it by `const&` into the pricing routines.
///
/// All rates and the volatility are expressed on a continuously-compounded,
/// annualised basis. Time to expiry is *not* stored here — it is a property of
/// the option contract (see opt::Option), so the same MarketData can price a
/// whole book of options with different maturities.
struct MarketData {
    double spot{};        ///< Current price of the underlying, \f$S_0 \ge 0\f$.
    double rate{};        ///< Continuously-compounded risk-free rate \f$r\f$.
    double dividend{};    ///< Continuous dividend yield \f$q\f$.
    double volatility{};  ///< Annualised volatility \f$\sigma \ge 0\f$.

    /// True when the observables are economically sensible: a non-negative spot
    /// and a non-negative volatility. Rates and the dividend yield may legally
    /// be negative, so they are not constrained here.
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return spot >= 0.0 && volatility >= 0.0;
    }
};

}  // namespace opt
