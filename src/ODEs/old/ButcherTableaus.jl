abstract type AbstractButcherTableau end
abstract type AbstractExplicitButcherTableau <: AbstractButcherTableau end
abstract type AbstractImplicitButcherTableau <: AbstractButcherTableau end
abstract type AbstractDiagonallyImplicitButcherTableau <: AbstractImplicitButcherTableau end

function _check_tableau(as, bs, cs)
  msg = "This Butcher tableau is ill-defined."
  la, lb, lc = length(as), length(bs), length(cs)
  @assert la == lb == lc msg

  msg = "This Butcher tableau is not consistent: the weights must sum to one."
  @assert sum(bs) ≈ 1 msg
end

function tableaudisp(::AbstractButcherTableau)
  ""
end

function tableauname(tableau::AbstractButcherTableau)
  println(tableau.name)
end

function tableaudesc(tableau::AbstractButcherTableau)
  println(tableau.desc)
end

##########################
# ExplicitButcherTableau #
##########################
struct ExplicitButcherTableau{A,B,C} <: AbstractExplicitButcherTableau
  as::A
  bs::B
  cs::C
  name::String
  desc::String

  function ExplicitButcherTableau(as, bs, cs, name="", desc="")
    _check_tableau(as, bs, cs)

    msg = "This Butcher tableau is not explicit."
    for (i, ais) in enumerate(as)
      @assert length(ais) <= i - 1 msg
    end

    A = typeof(as)
    B = typeof(bs)
    C = typeof(cs)
    new{A,B,C}(as, bs, cs, name, desc)
  end
end

Base.length(butcher::ExplicitButcherTableau) = length(butcher.as)

########
# EB_1 #
########
function EB_1_1(::Type{T}) where {T}
  as = [T[]]
  bs = T[1]
  cs = T[0]

  name = "Explicit Euler scheme"
  desc = """
  Euler's first-order explicit scheme. Same as ExplicitEuler solver.
  """
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

EB_1_1_Euler = EB_1_1

########
# EB_2 #
########
function EB_2_2(::Type{T}, α=1 / 2, name="", desc="") where {T}
  b2 = inv(2 * α)
  b1 = 1 - b2

  as = [T[], T[α]]
  bs = T[b1, b2]
  cs = T[0, α]
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

function EB_2_2_Midpoint(::Type{T}) where {T}
  name = "Midpoint scheme"
  desc = """
  Midpoint/Trapezoidal second-order explicit scheme.
  """
  EB_2_2(T, 1 / 2, name, desc)
end

function EB_2_2_Ralston(::Type{T}) where {T}
  name = "Ralston's scheme"
  desc = """
  Ralston's second-order explicit scheme.
  """
  EB_2_2(T, 2 / 3, name, desc)
end

function EB_2_2_Heun(::Type{T}) where {T}
  name = "Heun's second-order scheme"
  desc = """
  Heun's second-order explicit scheme. Same as ExplicitHeun solver.
  """
  EB_2_2(T, 1, name, desc)
end

########
# EB_3 #
########
function EB_3_3(::Type{T}, α=1 / 2, name="", desc="") where {T}
  msg = "α cannot belong to (0, 2/3, 1) for a general third-order explicit Butcher tableau."
  @assert !(α in (0, 2 / 3, 1)) msg

  β = (1 - α) / (α * (3 * α - 2))
  b2 = inv(6 * α * (1 - α))
  b3 = (2 - 3 * α) / (6 * (1 - α))
  b1 = 1 - b2 - b3

  as = [T[], T[α], T[1+β, -β]]
  bs = T[b1, b2, b3]
  cs = T[0, α, 1]
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

function EB_3_3_Kutta(::Type{T}) where {T}
  name = "Kutta's third-order explicit scheme"
  desc = """
  Kutta's third-order explicit scheme.
  """
  EB_3_3(T, 1 / 2, name, desc)
end

function EB_3_3_Ralston(::Type{T}) where {T}
  as = [T[], T[1/2], T[0, 3/4]]
  bs = T[2/9, 3/9, 4/9]
  cs = T[0, 1/2, 3/4]

  name = "Ralston's third-order explicit scheme"
  desc = """
  Ralston's third-order explicit scheme.
  """
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

function EB_3_3_Heun(::Type{T}) where {T}
  as = [T[], T[1/3], T[0, 2/3]]
  bs = T[1/4, 0, 3/4]
  cs = T[0, 1/3, 2/3]

  name = "Heun's third-order explicit scheme"
  desc = """
  Heun's third-order explicit scheme.
  """
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

function EB_3_3_Wray(::Type{T}) where {T}
  as = [T[], T[8/15], T[1/4, 5/12]]
  bs = T[1/4, 0, 3/4]
  cs = T[0, 8/15, 2/3]

  name = "Wray's / Van der Houwen's third-order explicit scheme"
  desc = """
  Wray's / Van der Houwen's third-order explicit scheme.
  """
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

EB_3_3_Houwen = EB_3_3_Wray

function EB_3_3_SSPRK(::Type{T}) where {T}
  as = [T[], T[1], T[1/4, 1/4]]
  bs = T[1/6, 1/6, 4/6]
  cs = T[0, 1, 1/2]

  name = "Strong Stability Preserving Runge-Kutta (SSPRK) third-order explicit scheme"
  desc = """
  Strong Stability Preserving Runge-Kutta (SSPRK) third-order explicit scheme.
  """
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

########
# EB_4 #
########
function EB_4_4_Kutta(::Type{T}) where {T}
  as = [T[], T[1/2], T[0, 1/2], T[0, 0, 1]]
  bs = T[1/6, 2/6, 2/6, 1/6]
  cs = T[0, 1/2, 1/2, 1]

  name = "Kutta's fourth-order explicit scheme"
  desc = """
  Kutta's fourth-order explicit scheme.
  """
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

function EB_4_4_Simpson(::Type{T}) where {T}
  as = [T[], T[1/3], T[-1/3, 1], T[1, -1, 1]]
  bs = T[1/8, 3/8, 3/8, 1/8]
  cs = T[0, 1/3, 2/3, 1]

  name = "Kutta's/Simpson's fourth-order explicit scheme."
  desc = """
  Kutta's/Simpson's fourth-order explicit scheme.
  """
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

function EB_4_4_Ralston(::Type{T}) where {T}
  c1 = 0
  c2 = 2 / 5
  c3 = (14 - 3 * sqrt(5)) / 16
  c4 = 1

  b1 = 1 / 2 + (1 - 2 * (c2 + c3)) / (12 * c2 * c3)
  b2 = (2 * c3 - 1) / (12 * c2 * (c3 - c2) * (1 - c2))
  b3 = (1 - 2 * c2) / (12 * c3 * (c3 - c2) * (1 - c3))
  b4 = 1 / 2 + (2 * (c2 + c3) - 3) / (12 * (1 - c2) * (1 - c3))

  a21 = c2
  a32 = c3 * (c3 - c2) / (2 * c2 * (1 - 2 * c2))
  a31 = c3 - a32
  a42 = (1 - c2) * (c2 + c3 - 1 - (2 * c3 - 1)^2) / (2 * c2 * (c3 - c2) * (6 * c2 * c3 - 4 * (c2 + c3) + 3))
  a43 = (1 - 2 * c2) * (1 - c2) * (1 - c3) / (c3 * (c3 - c2) * (6 * c2 * c3 - 4 * (c2 + c3) + 3))
  a41 = c4 - a42 - a43

  as = [T[], T[a21], T[a31, a32], T[a41, a42, a43]]
  bs = T[b1, b2, b3, b4]
  cs = T[c1, c2, c3, c4]

  name = "Ralston's fourth-order explicit scheme"
  desc = """
  Ralston's fourth-order explicit scheme
  """
  ExplicitButcherTableau(as, bs, cs, name, desc)
end

####################################
# DiagonallyImplicitButcherTableau #
####################################
struct DiagonallyImplicitButcherTableau{A,B,C} <: AbstractDiagonallyImplicitButcherTableau
  as::A
  bs::B
  cs::C
  name::String
  desc::String

  function DiagonallyImplicitButcherTableau(as, bs, cs, name="", desc="")
    _check_tableau(as, bs, cs)

    msg = "This Butcher tableau is not diagonally implicit."
    for (i, ais) in enumerate(as)
      @assert length(ais) <= i msg
    end

    A = typeof(as)
    B = typeof(bs)
    C = typeof(cs)
    new{A,B,C}(as, bs, cs, name, desc)
  end
end

Base.length(butcher::DiagonallyImplicitButcherTableau) = length(butcher.as)

#########
# DIB_1 #
#########
function DIB_1_1(::Type{T}) where {T}
  as = [T[1]]
  bs = T[1]
  cs = T[1]

  name = "Implicit Euler scheme"
  desc = """
  Euler's first-order implicit scheme. Same as ImplicitEuler solver.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

DIB_1_1_Euler = DIB_1_1

#########
# DIB_2 #
#########
function DIB_2_1_MidPoint(::Type{T}) where {T}
  as = [T[1/2]]
  bs = T[1]
  cs = T[1/2]

  name = "Implicit midpoint scheme"
  desc = """
  One-stage second-order implicit scheme.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

function DIB_2_2_CrankNicolson(::Type{T}) where {T}
  as = [T[], T[1/2, 1/2]]
  bs = T[1/2, 1/2]
  cs = T[0, 1]

  name = "Implicit Crank-Nicolson scheme"
  desc = """
  Crank-Nicolson's second-order implicit scheme. Same as CrankNicolson solver.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

function DIB_2_2_PareschiRusso(::Type{T}, α=1 + sqrt(2) / 2) where {T}
  α = T(α)
  as = [T[α], T[1-2*α, α]]
  bs = T[1/2, 1/2]
  cs = T[α, 1-α]

  name = "Implicit Pareschi-Russo scheme"
  desc = """
  Two-stage, second-order scheme.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

DIB_2_2_QinZhang(T) = DIB_2_2_PareschiRusso(T, 1 / 4)

function DIB_2_2_Unknown(::Type{T}, α=1 + sqrt(2) / 2) where {T}
  α = T(α)
  as = [T[α], T[1-α, α]]
  bs = T[1-α, α]
  cs = T[α, 1]

  name = "Implicit scheme"
  desc = """
  Two-stage, second-order scheme.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

#########
# DIB_3 #
#########
function DIB_3_2_Crouzeix(::Type{T}) where {T}
  as = [T[1/2+sqrt(3)/6], T[-sqrt(3)/3, 1/2+sqrt(3)/6]]
  bs = T[1/2, 1/2]
  cs = T[1/2+sqrt(3)/6, 1/2-sqrt(3)/6]

  name = "Implicit Crouzeix scheme"
  desc = """
  Two-stage, third-order DIRK.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

function DIB_3_2_HammerHollingsworth(::Type{T}) where {T}
  as = [T[], T[1/3, 1/3]]
  bs = T[1/4, 3/4]
  cs = T[0, 2/3]

  name = "Implicit Hammer-Hollingsworth scheme"
  desc = """
  Two-stage, third-order ESDIRK (Radau I).
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

function DIB_3_3_CeschinoKunzmann(::Type{T}) where {T}
  as = [T[], T[1/4, 1/4], T[1/6, 4/6, 1/6]]
  bs = T[1/6, 4/6, 1/6]
  cs = T[0, 1/2, 1]

  name = "Implicit Ceschino-Kunzmann scheme"
  desc = """
  Three-stage, third-order EDIRK.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

function DIB_3_3_Alt(::Type{T}) where {T}
  as = [T[], T[3/4, 3/4], T[7/18, -4/18, 15/18]]
  bs = T[7/18, -4/18, 15/18]
  cs = T[0, 3/2, 1]

  name = "Implicit Alt scheme"
  desc = """
  Three-stage, third-order EDIRK.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

#########
# DIB_4 #
#########
function DIB_4_3_Crouzeix(::Type{T}) where {T}
  α = T(2 * cospi(1 / 18) / sqrt(3))
  as = [T[(1+α)/2], T[-α/2, (1+α)/2], T[1+α, -(1 + 2 * α), (1+α)/2]]
  bs = T[inv(6 * α^2), 1-inv(3 * α^2), inv(6 * α^2)]
  cs = T[(1+α)/2, 1/2, (1-α)/2]

  name = "Implicit Crouzeix scheme"
  desc = """
  Trhee-stage, fourth-order DIRK.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

function DIB_4_3_Butcher(::Type{T}) where {T}
  as = [T[], T[1/4, 1/4], T[0, 1]]
  bs = T[1/6, 4/6, 1/6]
  cs = T[0, 1/2, 1]

  name = "Implicit Butcher scheme"
  desc = """
  Three-stage, fourth-order EDIRK (Lobatto III).
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

function DIB_4_4_CeschinoKunzmann(::Type{T}) where {T}
  as = [T[], T[1/6, 1/6], T[1/12, 6/12, 1/12], T[1/8, 3/8, 3/8, 1/8]]
  bs = T[1/8, 3/8, 3/8, 1/8]
  cs = T[0, 1/3, 2/3, 1]

  name = "Implicit Ceschino-Kunzmann scheme"
  desc = """
  Four-stage, fourth-order EDIRK.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end

function DIB_4_4_Alt(::Type{T}) where {T}
  as = [T[], T[3/4, 3/4], T[447/675, -357/675, 855/675], T[13/42, 84/42, -125/42, 70/42]]
  bs = T[13/42, 84/42, -125/42, 70/42]
  cs = T[0, 3/2, 7/5, 1]

  name = "Implicit Alt scheme"
  desc = """
  Four-stage, fourth-order EDIRK.
  """
  DiagonallyImplicitButcherTableau(as, bs, cs, name, desc)
end
