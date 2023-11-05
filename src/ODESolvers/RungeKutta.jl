#######################
# Runge-Kutta Wrapper #
#######################
function RungeKutta(args...; butcher::AbstractButcherTableau, kwargs...)
  if butcher isa AbstractExplicitButcherTableau
    ERK(args...; butcher, kwargs...)
  elseif butcher isa AbstractDiagonallyImplicitButcherTableau
    DIRK(args...; butcher, kwargs...)
  end
end
