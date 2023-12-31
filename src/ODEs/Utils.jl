function _make_us(t::Real, u::AbstractVector, u̇::AbstractVector)
  (SVector(t), u, u̇)
end

function _make_u̇!(v, u, u₋, dt⁻¹)
  @inbounds @simd for i in eachindex(v, u, u₋)
    v[i] = (u[i] - u₋[i]) * dt⁻¹
  end
  v
end

function _make_u!(u₊, u̇, u₋, dt)
  @inbounds @simd for i in eachindex(u₊, u̇, u₋)
    u₊[i] = u₋[i] + dt * u̇[i]
  end
  u₊
end
