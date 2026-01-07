/**
 * SICM-01R-FULL: STOCHASTIC INTER-ARRIVAL COMMUNICATION MODEL
 * CORRECTED MATHEMATICAL KERNEL
 * * @author Mathematical Systems Architect
 * @version 3.0.0-RIGOR
 */

// ============================================================================
// 1. MATHEMATICAL UTILITIES (STOCHASTIC CALCULUS SUPPORT)
// ============================================================================

const MathLib = {
    /**
     * Log-Sum-Exp for numerical stability in HMM Forward Algorithm.
     * log(a + b) = log(exp(ln_a) + exp(ln_b))
     */
    logSumExp: (arr) => {
        if (arr.length === 0) return -Infinity;
        const max = Math.max(...arr);
        if (max === -Infinity) return -Infinity;
        const sum = arr.reduce((acc, val) => acc + Math.exp(val - max), 0);
        return max + Math.log(sum);
    },

    /**
     * Modified Bessel Function of the first kind, order 0 (I0).
     * Required for normalizing Von Mises distribution.
     * Approximation: Polynomial approximation for small |z| and asymptotic for large |z|.
     */
    besselI0: (x) => {
        const ax = Math.abs(x);
        if (ax < 3.75) {
            let y = x / 3.75;
            y = y * y;
            return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 +
                   y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
        } else {
            let y = 3.75 / ax;
            return (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 +
                   y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 +
                   y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 +
                   y * 0.392377e-2))))))));
        }
    },

    /**
     * Adaptive Simpson's Rule for integration.
     * Calculates Integral_a^b f(x) dx with error control.
     */
    integrateAdaptive: (func, a, b, tol = 1e-6) => {
        const m = (a + b) / 2;
        const h = b - a;
        const fa = func(a), fb = func(b), fm = func(m);
        const S = (h / 6) * (fa + 4 * fm + fb);

        const recursive = (a, b, fa, fb, fm, S, tol) => {
            const m = (a + b) / 2;
            const h = b - a;
            const lm = (a + m) / 2, rm = (m + b) / 2;
            const flm = func(lm), frm = func(rm);
            const leftS = (h / 12) * (fa + 4 * flm + fm);
            const rightS = (h / 12) * (fm + 4 * frm + fb);

            if (Math.abs(leftS + rightS - S) <= 15 * tol) {
                return leftS + rightS + (leftS + rightS - S) / 15;
            }
            return recursive(a, m, fa, fm, flm, leftS, tol / 2) +
                   recursive(m, b, fm, fb, frm, rightS, tol / 2);
        };
        return recursive(a, b, fa, fb, fm, S, tol);
    }
};

// ============================================================================
// 2. DISTRIBUTION KERNELS (NORMALIZED)
// ============================================================================

class WeibullKernel {
    constructor(lambda, k) {
        this.lambda = lambda;
        this.k = k;
    }

    logPdf(t) {
        if (t <= 0) return -Infinity;
        // ln f(t) = ln k - ln lambda + (k-1)(ln t - ln lambda) - (t/lambda)^k
        return Math.log(this.k) - Math.log(this.lambda) + 
               (this.k - 1) * (Math.log(t) - Math.log(this.lambda)) - 
               Math.pow(t / this.lambda, this.k);
    }

    survival(t) {
        if (t <= 0) return 1.0;
        return Math.exp(-Math.pow(t / this.lambda, this.k));
    }
    
    hazard(t) {
        if (t <= 0) return 0;
        return (this.k / this.lambda) * Math.pow(t / this.lambda, this.k - 1);
    }
}

class CircadianModulator {
    constructor(peakHour, concentration) {
        this.mu = peakHour;
        this.kappa = concentration;
        // ISSUE 5 RESOLVED: Explicit Normalization
        // Integral of exp(k*cos(x)) over [0,24] is 24 * I0(k) (if mapped 1:1)
        // Average value is I0(k). We divide by I0(k) so mean intensity is 1.0.
        this.normConst = MathLib.besselI0(this.kappa);
    }

    // Normalized intensity: m_tilde(t)
    intensity(hour) {
        const theta = (hour % 24) * (2 * Math.PI / 24);
        const muRad = this.mu * (2 * Math.PI / 24);
        const raw = Math.exp(this.kappa * Math.cos(theta - muRad));
        return raw / this.normConst;
    }
    
    getMultiplier(timestamp) {
        const date = new Date(timestamp);
        const hour = date.getHours() + date.getMinutes() / 60;
        return this.intensity(hour); 
    }
}

/**
 * Handles time-varying hazard integration.
 * S(u+t | u) = exp( - Integral_u^{u+t} h_0(tau) * m(tau) dtau )
 */
class ModulatedProcess {
    constructor(kernel, modulator, startTime) {
        this.kernel = kernel;
        this.modulator = modulator;
        this.startTime = startTime;
    }

    // Hazard at relative time t (hours since last event)
    hazard(t) {
        const absTime = this.startTime + t * 3600000;
        return this.kernel.hazard(t) * this.modulator.getMultiplier(absTime);
    }

    // S(u+t | u)
    conditionalSurvival(u, t) {
        const integral = MathLib.integrateAdaptive(
            (tau) => this.hazard(tau),
            u,
            u + t,
            1e-5
        );
        return Math.exp(-integral);
    }
}

// ============================================================================
// 3. CORE ENGINE (FULL BAYESIAN HMM)
// ============================================================================

class SICMEngine {
    constructor(config = {}) {
        this.config = {
            // High resolution required for 6D space? 
            // We use coarse-to-fine or strictly bounded grid to manage complexity.
            // Constraint: DO NOT optimize. We implement the loops.
            gridRes: 6, // 6^6 = 46,656 combinations. Manageable in JS.
            minLambda: 1, maxLambda: 720,
            minK: 0.5, maxK: 5.0,
            ...config
        };
        this.history = [];
        this.currentSilence = 0;
    }

    ingestData(rawEvents) {
        const sorted = rawEvents.sort((a, b) => a.timestamp - b.timestamp);
        this.history = [];
        for (let i = 1; i < sorted.length; i++) {
            const diff = (sorted[i].timestamp - sorted[i-1].timestamp) / 36e5;
            if (diff > 0) this.history.push(diff);
        }
        const lastTime = sorted[sorted.length - 1].timestamp;
        this.currentSilence = (Date.now() - lastTime) / 36e5;
        this.startTime = lastTime; // Reference for Circadian

        // Safety: Minimum data
        if (this.history.length < 5) return { valid: false, reason: "INSUFFICIENT_DATA" };
        return { valid: true };
    }

    // ISSUE 1 & 2 RESOLVED: Joint Inference of Regimes & Transition Matrix
    computePosterior() {
        const { gridRes, minLambda, maxLambda, minK, maxK } = this.config;
        const posterior = [];
        let logEvidence = -Infinity;

        // Grid Generators
        const range = (min, max, steps) => 
            Array.from({length: steps}, (_, i) => min + (max-min)*(i/(steps-1)));
        
        const lambdas = range(minLambda, maxLambda, gridRes);
        const ks = range(minK, maxK, gridRes);
        const probs = range(0.1, 0.9, 5); // Transition probs

        // THE MONSTER LOOP (6D)
        // L_A, K_A, L_D, K_D, P_AA, P_DD
        for (const lA of lambdas) {
        for (const kA of ks) {
            const kernelA = new WeibullKernel(lA, kA);
            
            for (const lD of lambdas) {
            // Constraint: Dormant should be slower than Active to avoid identifiability issues (label switching)
            // Strict inequality enforcement:
            if (lD <= lA) continue; 
            
            for (const kD of ks) {
                const kernelD = new WeibullKernel(lD, kD);
                
                for (const pAA of probs) {
                for (const pDD of probs) {
                    
                    // Hidden Markov Model Parameters
                    const transMat = [[pAA, 1-pAA], [1-pDD, pDD]];
                    // Stationary distribution for initial state
                    // pi_A = (1-pDD) / (2 - pAA - pDD)
                    const pi_A = (1 - pDD) / (2 - pAA - pDD);
                    const initProbs = [pi_A, 1 - pi_A];

                    // FORWARD ALGORITHM (Exact Likelihood)
                    // alpha_t(j) = P(Obs_1...t, Z_t = j)
                    let logAlpha = [Math.log(initProbs[0]), Math.log(initProbs[1])];

                    for (const w of this.history) {
                        const logPdfA = kernelA.logPdf(w);
                        const logPdfD = kernelD.logPdf(w);
                        
                        // alpha_t(j) = [sum_i alpha_{t-1}(i) * P_ij] * pdf_j(w)
                        const nextLogAlpha = [];
                        
                        // State 0 (Active)
                        const logTransToA = MathLib.logSumExp([
                            logAlpha[0] + Math.log(transMat[0][0]),
                            logAlpha[1] + Math.log(transMat[1][0])
                        ]);
                        nextLogAlpha[0] = logTransToA + logPdfA;

                        // State 1 (Dormant)
                        const logTransToD = MathLib.logSumExp([
                            logAlpha[0] + Math.log(transMat[0][1]),
                            logAlpha[1] + Math.log(transMat[1][1])
                        ]);
                        nextLogAlpha[1] = logTransToD + logPdfD;

                        logAlpha = nextLogAlpha;
                    }

                    // Total Log Likelihood of History
                    const logLikelihoodHist = MathLib.logSumExp(logAlpha);

                    // Add Censored Observation (Current Silence u)
                    // We must propagate state one step further: P(Z_{last} | Data)
                    // then apply survival S_j(u)
                    
                    // Normalized probability of being in state j at time N
                    const P_Zn_A = Math.exp(logAlpha[0] - logLikelihoodHist);
                    const P_Zn_D = Math.exp(logAlpha[1] - logLikelihoodHist);

                    // Probability of state at N+1 (start of silence)
                    // P(Z_{n+1}=A) = P(Z_n=A)*P_AA + P(Z_n=D)*P_DA
                    const P_ZnPlus1_A = P_Zn_A * transMat[0][0] + P_Zn_D * transMat[1][0];
                    const P_ZnPlus1_D = P_Zn_A * transMat[0][1] + P_Zn_D * transMat[1][1];

                    // Likelihood of silence u
                    // L_cens = P(Z=A)*S_A(u) + P(Z=D)*S_D(u)
                    const S_A_u = kernelA.survival(this.currentSilence);
                    const S_D_u = kernelD.survival(this.currentSilence);
                    const likelihoodCens = P_ZnPlus1_A * S_A_u + P_ZnPlus1_D * S_D_u;

                    if (likelihoodCens <= 0) continue;

                    const totalLogProb = logLikelihoodHist + Math.log(likelihoodCens);
                    
                    // Posterior Object
                    const entry = {
                        kA: kernelA, kD: kernelD,
                        transMat,
                        // Regime probabilities at start of silence (u=0)
                        regimeProbs: [P_ZnPlus1_A, P_ZnPlus1_D], 
                        logProb: totalLogProb
                    };
                    
                    posterior.push(entry);
                    logEvidence = MathLib.logSumExp([logEvidence, totalLogProb]);
                }}}}
            }
            }
        }}

        // Normalize
        this.posterior = posterior.map(e => ({
            ...e,
            prob: Math.exp(e.logProb - logEvidence)
        }));

        if (this.posterior.length === 0) return { valid: false, reason: "MODEL_COLLAPSE" };
        
        // ISSUE 7 PRE-CHECK: Effective Sample Size
        const ess = 1 / this.posterior.reduce((acc, x) => acc + x.prob**2, 0);
        if (ess < 2) return { valid: false, reason: "UNCERTAINTY_DOMINATED" };

        return { valid: true };
    }

    // ISSUE 3 & 4 RESOLVED: Dynamic Regime Update & Exact Residuals
    predict(deltaT) {
        if (!this.posterior) return { status: "UNDEFINED" };

        const circadian = new CircadianModulator(15, 1.0); // Assume fixed for now (or infer 7th param)
        
        // Predictive CDF components for Quantiles
        // We will sample the CDF at fine resolution
        const cdfPoints = [];
        const resolution = 50; 
        
        let expectedResidual = 0;
        let probContact = 0;

        for (const model of this.posterior) {
            const { kA, kD, regimeProbs, prob: modelProb } = model;

            // Construct processes
            const procA = new ModulatedProcess(kA, circadian, this.startTime);
            const procD = new ModulatedProcess(kD, circadian, this.startTime);

            // ISSUE 3: Update Regime Probabilities based on Survival to u
            // P(Z=j | T > u) = P(Z=j @ start) * S_j(u) / Sum(...)
            // Note: computePosterior already gave us P(Z @ start of silence). 
            // We need P(Z @ u) which is the start of the prediction window.
            
            // Survival from 0 to u (relative to start of silence)
            const S_A_u = procA.conditionalSurvival(0, this.currentSilence);
            const S_D_u = procD.conditionalSurvival(0, this.currentSilence);
            
            const numA = regimeProbs[0] * S_A_u;
            const numD = regimeProbs[1] * S_D_u;
            const denom = numA + numD;

            if (denom <= 1e-12) continue; // Collapsed hypothesis

            const pi_A_u = numA / denom;
            const pi_D_u = numD / denom;

            // 1. Probability of Contact in [u, u+deltaT]
            // P(C) = pi_A * (1 - S_A(u+dT|u)) + pi_D * (1 - S_D(u+dT|u))
            const S_A_cond = procA.conditionalSurvival(this.currentSilence, deltaT);
            const S_D_cond = procD.conditionalSurvival(this.currentSilence, deltaT);
            
            const pContact_Hypothesis = pi_A_u * (1 - S_A_cond) + pi_D_u * (1 - S_D_cond);
            probContact += modelProb * pContact_Hypothesis;

            // 2. Exact Residual Life (ISSUE 4)
            // E = Integral_0^inf S_mix(u+t | u) dt
            // S_mix(u+t|u) = pi_A_u * S_A(u+t|u) + pi_D_u * S_D(u+t|u)
            const integrand = (t) => {
                const sA = procA.conditionalSurvival(this.currentSilence, t);
                const sD = procD.conditionalSurvival(this.currentSilence, t);
                return pi_A_u * sA + pi_D_u * sD;
            };

            // Integrate until convergence (S < 1e-4)
            // Use chunks to avoid infinite loops
            let t_chunk = 0;
            let integralChunk = 0;
            let currentS = 1;
            while(currentS > 1e-4 && t_chunk < 10000) { // Safety cap
                const chunkVal = MathLib.integrateAdaptive(integrand, t_chunk, t_chunk + 24, 1e-4);
                integralChunk += chunkVal;
                t_chunk += 24;
                currentS = integrand(t_chunk);
            }
            expectedResidual += modelProb * integralChunk;
        }

        // ISSUE 7: Quantiles (Re-loop for CDF)
        // For CI95, we need t such that P(T < t) = 0.025 and 0.975
        // This requires constructing the aggregate CDF P(T < t | Data)
        // CDF(t) = 1 - Sum_models [ P(model) * S_mix(u+t | u) ]
        
        const getCDF = (t) => {
            let survival = 0;
            for (const model of this.posterior) {
                const { kA, kD, regimeProbs, prob: modelProb } = model;
                const procA = new ModulatedProcess(kA, circadian, this.startTime);
                const procD = new ModulatedProcess(kD, circadian, this.startTime);
                
                // Recalculate mixing weights at u (optimization: cache these)
                const S_A_u = procA.conditionalSurvival(0, this.currentSilence);
                const S_D_u = procD.conditionalSurvival(0, this.currentSilence);
                const denom = regimeProbs[0] * S_A_u + regimeProbs[1] * S_D_u;
                if (denom < 1e-12) continue;
                
                const pi_A = (regimeProbs[0] * S_A_u) / denom;
                const pi_D = (regimeProbs[1] * S_D_u) / denom;
                
                const s_mix = pi_A * procA.conditionalSurvival(this.currentSilence, t) +
                              pi_D * procD.conditionalSurvival(this.currentSilence, t);
                survival += modelProb * s_mix;
            }
            return 1 - survival;
        };

        // Binary search for quantiles
        const solveQuantile = (q) => {
            let low = 0, high = 5000; // Search range hours
            for(let i=0; i<20; i++) {
                const mid = (low + high) / 2;
                if (getCDF(mid) < q) low = mid;
                else high = mid;
            }
            return (low + high) / 2;
        };

        const ciLower = solveQuantile(0.025);
        const ciUpper = solveQuantile(0.975);

        return {
            status: "OK",
            probability: probContact,
            expectedResidual: expectedResidual,
            ci95: [ciLower, ciUpper]
        };
    }

    generateReport(deltaT) {
        const train = this.computePosterior();
        if (!train.valid) return { status: "UNDEFINED", reason: train.reason };
        
        const pred = this.predict(deltaT);
        if (pred.status === "UNDEFINED") return { status: "UNDEFINED" };

        return {
            status: "OK",
            result: {
                P_contact: pred.probability,
                E_residual: pred.expectedResidual,
                CI_95: pred.ci95
            },
            meta: {
                method: "SICM-01R-FULL (Exact Bayesian HMM)",
                posteriorSize: this.posterior.length
            }
        };
    }
}

// ============================================================================
// 4. INTERFACE
// ============================================================================

function runAnalysis(events, deltaT) {
    const engine = new SICMEngine();
    const ingest = engine.ingestData(events);
    if (!ingest.valid) return { status: "UNDEFINED", reason: ingest.reason };
    return engine.generateReport(deltaT);
}

if (typeof module !== 'undefined') module.exports = { SICMEngine, runAnalysis };
