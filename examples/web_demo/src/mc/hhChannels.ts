/**
 * Hodgkin-Huxley (squid axon, 1952) channel kinetics.
 *
 * Voltage in mV, time in ms throughout. The α/β rate functions are the
 * canonical squid-axon parameterisation from Hodgkin & Huxley's J. Physiol.
 * paper — the same numbers you'll find in every computational-neuroscience
 * textbook. Real fly central-complex channel densities aren't published at
 * single-cell-type resolution; we use the squid values knowing they're a
 * qualitative match, not a quantitative one.
 *
 * The functions here are pure and side-effect free; they're the inner loop
 * of the membrane integrator in `hhSolver.ts`.
 */

// Resting potential we initialise membrane voltage to, plus the implicit
// "v-shift" baked into the H&H α/β rates (they're written assuming v=0 is
// rest, but we use absolute mV). Standard textbook offset: -65 mV.
export const V_REST_MV = -65;
export const E_NA_MV = 50;
export const E_K_MV = -77;
export const E_L_MV = -54.387;

/** Channel-density defaults (mS/cm²) — soma/dendrite. AIS gets ~10× Na/K. */
export const G_NA_DEFAULT_MS_PER_CM2 = 120;
export const G_K_DEFAULT_MS_PER_CM2 = 36;
export const G_L_DEFAULT_MS_PER_CM2 = 0.3;
export const AIS_BOOST = 10;

/** Membrane capacitance (μF/cm²). */
export const CM_UF_PER_CM2 = 1.0;

/** Axial resistivity (Ω·cm). */
export const RA_OHM_CM = 150;

/**
 * `safeExpm1Ratio(x, y) = (x - y) / (exp((x - y)/k) - 1) · k`
 * with a Taylor-series fallback near the removable singularity at x == y.
 *
 * H&H's α/β rate functions contain (v - v0) / (1 - exp((v - v0)/k)) terms
 * that look like 0/0 at v == v0. The true limiting value is finite (`-k`
 * for the sign convention used below); we hand-roll that limit so the
 * solver doesn't NaN-out when voltage happens to land on the singularity
 * during integration.
 */
function rateAlpha(v: number, v0: number, k: number, scale: number): number {
  const x = (v - v0) / k;
  if (Math.abs(x) < 1e-6) {
    // Limit of scale·k·x / (1 - exp(-x)) as x → 0: scale·k.
    return scale * k * (1 - 0.5 * x);
  }
  return (scale * (v - v0)) / (1 - Math.exp(-x));
}

/* ---- Sodium activation (m) ---- */
export function alphaM(v: number): number {
  // 0.1·(v + 40) / (1 − exp(−(v + 40)/10))
  return rateAlpha(v, -40, 10, 0.1);
}
export function betaM(v: number): number {
  return 4 * Math.exp(-(v + 65) / 18);
}

/* ---- Sodium inactivation (h) ---- */
export function alphaH(v: number): number {
  return 0.07 * Math.exp(-(v + 65) / 20);
}
export function betaH(v: number): number {
  return 1 / (1 + Math.exp(-(v + 35) / 10));
}

/* ---- Potassium activation (n) ---- */
export function alphaN(v: number): number {
  // 0.01·(v + 55) / (1 − exp(−(v + 55)/10))
  return rateAlpha(v, -55, 10, 0.01);
}
export function betaN(v: number): number {
  return 0.125 * Math.exp(-(v + 65) / 80);
}

/** Steady-state gating variable for the rate constants (α / (α + β)). */
export function gateSteadyState(alpha: number, beta: number): number {
  return alpha / (alpha + beta);
}

/** Resting-state gating values at v == V_REST_MV. Cached. */
export const M_REST = gateSteadyState(alphaM(V_REST_MV), betaM(V_REST_MV));
export const H_REST = gateSteadyState(alphaH(V_REST_MV), betaH(V_REST_MV));
export const N_REST = gateSteadyState(alphaN(V_REST_MV), betaN(V_REST_MV));
