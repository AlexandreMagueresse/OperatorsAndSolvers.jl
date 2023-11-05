macro abstractmethod(message="This function belongs to an interface definition and cannot be used.")
  quote
    error($(esc(message)))
  end
end

function _u̇_disc!(v, u, u₋, dt⁻¹)
  copy!(v, u)
  rmul!(v, dt⁻¹)
  axpy!(-dt⁻¹, u₋, v)
  v
end

function combine!(dst, src, weight)
  axpy!(weight, src, dst)
  dst
end
