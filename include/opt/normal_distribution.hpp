#pragma once

#include <cmath>

namespace opt {

/// Probability density function of the standard normal distribution,
/// \f$\phi(x) = \tfrac{1}{\sqrt{2\pi}}\,e^{-x^2/2}\f$.
[[nodiscard]] inline double standard_normal_pdf(double x) noexcept {
    constexpr double inv_sqrt_2pi = 0.39894228040143267794;  // 1 / sqrt(2*pi)
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

/// Cumulative distribution function of the standard normal distribution,
/// \f$\Phi(x) = \int_{-\infty}^{x}\phi(u)\,du\f$.
///
/// Evaluated via the complementary error function,
/// \f$\Phi(x) = \tfrac{1}{2}\,\operatorname{erfc}\!\bigl(-x/\sqrt{2}\bigr)\f$.
/// `std::erfc` is a high-quality library primitive accurate to ~1e-16, far
/// inside the 7.5e-8 target — and, unlike a low-order rational approximation,
/// its derivative is just as accurate, which keeps the finite-difference
/// Greeks consistent with the closed-form ones.
[[nodiscard]] inline double cumulative_normal(double x) noexcept {
    constexpr double inv_sqrt_2 = 0.70710678118654752440;  // 1 / sqrt(2)
    return 0.5 * std::erfc(-x * inv_sqrt_2);
}

}  // namespace opt
