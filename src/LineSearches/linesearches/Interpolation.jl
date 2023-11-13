function interpolation_VS_V(
  x₋::Real, v₋::Real, s₋::Real,
  x₊::Real, v₊::Real
)
  # x̄, Δx = middif(x₋, x₊)
  # Δv = (v₊ - v₋) / 2

  # A = Δv - s₋ * Δx
  # B = Δv

  # x̄ - B * Δx / A

  # Simplification from x₋
  Δx = (x₊ - x₋) / 2
  Δv = (v₊ - v₋) / 2
  p = -s₋ * Δx
  x₋ + p * Δx / (Δv + p)
end

function interpolation_VS0_V(
  v₀::Real, s₀::Real,
  x₁::Real, v₁::Real
)
  # Simplification when x₋ = 0
  p = -s₀ * x₁
  num = p * x₁
  den = 2 * (v₁ - v₀ + p)
  num / den
end

function interpolation_VS_V_V(
  x₀::Real, v₀::Real, s₀::Real,
  x₁::Real, v₁::Real,
  x₂::Real, v₂::Real
)
  # TODO ?
end

function interpolation_VS0_V_V(
  v₀::Real, s₀::Real,
  x₁::Real, v₁::Real,
  x₂::Real, v₂::Real
)
  x₁² = abs2(x₁)
  x₂² = abs2(x₂)
  invdet = inv(x₁² * x₂² * (x₂ - x₁))

  u₁ = v₁ - v₀ - s₀ * x₁
  u₂ = v₂ - v₀ - s₀ * x₂

  u₁x₂² = u₁ * x₂²
  u₁x₂³ = u₁x₂² * x₂
  u₂x₁² = u₂ * x₁²
  u₂x₁³ = u₂x₁² * x₁

  A = (u₂x₁² - u₁x₂²) * invdet
  B = (u₁x₂³ - u₂x₁³) * invdet

  if isapprox(A, 0)
    -s₀ / (2 * B)
  else
    Δ = abs2(B) - 3 * A * s₀
    if Δ < 0
      # Very unlikely, we must have
      # 0 < x₁ < x₂ and v₀ > v₁ > v₂ (up to permutation of x₁ and x₂)
      max(x₁, x₂)
    else
      # The first formula is more accurate
      # (robust to canceling sqrt(Δ) - B)
      # but it performs worse

      # -s₀ / (B + sqrt(Δ))
      (sqrt(Δ) - B) / (3 * A)
    end
  end
end
