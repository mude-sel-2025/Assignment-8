# ==========================================
# Assignment 8 (MUDE, IITM): Sampling and reliability
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import time

"""
1) Implements solution functions for:
   - Task 1: Circle area (Monte Carlo)
   - Task 2: Beam deflection Pf using Monte Carlo
   - Task 3: Beam deflection Pf using Latin Hypercube Sampling
   - Task 4: 3-bar truss FEA Pf using MC and LHS (uses analyze_truss)

Also contains:
 - lhs_sample() : simple Latin Hypercube sampler for independent marginals
 - analyze_truss() : small planar 3-bar truss linear-elastic solver

"""

# ---------------------------
# Function: to ensure positive samples from normal by resampling
# ---------------------------
def positive_normal(mu, sigma, size, rng=None):
    """Draw from Normal(mu,sigma) but resample any non-positive draws."""
    if rng is None:
        rng = np.random.default_rng()
    out = rng.normal(mu, sigma, size=size)
    # resample nonpositive values
    while np.any(out <= 0):
        mask = out <= 0
        out[mask] = rng.normal(mu, sigma, size=mask.sum())
    return out

# ---------------------------
# LHS sampler for independent marginals
# ---------------------------
def lhs_sample(distributions, N, rng=None):
    """
    Latin Hypercube Sampling for independent marginals.
    distributions: list of dicts, each dict has keys 'dist' (scipy.stats frozen) or
                   a tuple ('norm', mu, sigma) or ('uniform', low, high) etc.
    N: number of samples
    Returns: array of shape (N, d)
    NOTE: This is a simple LHS without correlation between marginals.
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(distributions)
    # Create N strata [0..N-1] and random offsets in each stratum
    u = np.empty((N, d))
    for j, spec in enumerate(distributions):
        # draw uniform samples inside stratified intervals
        cut = (np.arange(N) + rng.random(N)) / N  # values in (0,1)
        rng.shuffle(cut)  # randomly permute strata positions
        # Transform via specified distribution
        # spec can be a scipy.stats frozen distribution or a tuple describing distribution
        if hasattr(spec, 'rvs'):
            # assume frozen distribution, can accept size argument
            # use ppf to map uniform sample to distrib
            u[:, j] = spec.ppf(cut)
        else:
            # spec as tuple: ('norm', mu, sigma), ('uniform', low, high), ('lognorm', s, scale)
            typ = spec[0].lower()
            if typ == 'norm':
                mu, sigma = spec[1], spec[2]
                u[:, j] = stats.norm.ppf(cut, loc=mu, scale=sigma)
            elif typ == 'uniform':
                low, high = spec[1], spec[2]
                u[:, j] = stats.uniform.ppf(cut, loc=low, scale=(high-low))
            elif typ == 'lognorm':
                s, scale = spec[1], spec[2]
                u[:, j] = stats.lognorm.ppf(cut, s=s, scale=scale)
            elif typ == 'triangular':
                # ('triangular', left, mode, right)
                left, mode, right = spec[1], spec[2], spec[3]
                u[:, j] = stats.triang.ppf(cut, (mode-left)/(right-left), loc=left, scale=(right-left))
            else:
                raise ValueError("Unsupported spec: " + str(spec))
    return u

# ---------------------------
# analyze_truss: simple 3-bar planar truss FEA
# ---------------------------
def analyze_truss(node_coords, elements, E_vec, A_vec, loads, supports):
    """
    Simple linear-elastic 2D truss finite element solver.
    node_coords: array (n_nodes,2)
    elements: list of (n1, n2) zero-based node indices
    E_vec: array of length n_elems (Pa)
    A_vec: array of length n_elems (m^2)
    loads: global force vector shape (2*n_nodes,) (N)
    supports: dict {node_index : (ux_fixed_bool, uy_fixed_bool)}  True means fixed
    Returns: displacements vector (2*n_nodes,) in meters
    Notes:
      - builds global stiffness K and solves K_reduced * u_free = f_free
      - small-strain linear truss with axial stiffness Ae*E/L in local coords
    """
    n_nodes = node_coords.shape[0]
    n_dof = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))
    # assemble
    for e, (n1, n2) in enumerate(elements):
        x1, y1 = node_coords[n1]
        x2, y2 = node_coords[n2]
        dx = x2 - x1
        dy = y2 - y1
        L = math.hypot(dx, dy)
        cx = dx / L
        cy = dy / L
        k_local = (A_vec[e] * E_vec[e] / L) * np.array([[ cx*cx, cx*cy, -cx*cx, -cx*cy],
                                                       [ cx*cy, cy*cy, -cx*cy, -cy*cy],
                                                       [-cx*cx, -cx*cy,  cx*cx,  cx*cy],
                                                       [-cx*cy, -cy*cy,  cx*cy,  cy*cy]])
        # global dof indices
        dof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            for j in range(4):
                K[dof[i], dof[j]] += k_local[i, j]

    # apply supports: reduce system
    fixed = []
    for node, (fix_x, fix_y) in supports.items():
        if fix_x:
            fixed.append(2*node)
        if fix_y:
            fixed.append(2*node + 1)
    free = [i for i in range(n_dof) if i not in fixed]

    # Ensure loads has correct shape
    f = np.zeros(n_dof)
    f[:len(loads)] = loads

    # Partition K
    Kff = K[np.ix_(free, free)]
    Kfc = K[np.ix_(free, fixed)]
    ff = f[free]

    # Solve for free displacements
    if Kff.size == 0:
        u = np.zeros(n_dof)
    else:
        u_free = np.linalg.solve(Kff, ff)
        u = np.zeros(n_dof)
        u[free] = u_free
        u[fixed] = 0.0
    return u

# ---------------------------
# Task 1: Circle area with Monte Carlo
# ---------------------------
def problem1_circle_area_mc(N=10000, seed=0, show_plot=True):
    """
    Monte Carlo estimation of unit-circle area (circle radius 0.5 inside [0,1]^2).
    Returns estimate and standard error.
    """
    rng = np.random.default_rng(seed)
    # sample N uniform points
    XY = rng.random((N,2))  # uniform in [0,1]^2
    dx = XY[:,0] - 0.5
    dy = XY[:,1] - 0.5

    # WRITE_YOUR_CODE HERE TO COUNT POINTS INSIDE CIRCLE
    inside = 

    count_inside = inside.sum()
    p_hat = count_inside / N
    area_hat = p_hat * 1.0  # area of square is 1
    se = math.sqrt(p_hat*(1-p_hat)/N)
    print(f"[problem1] N={N}, points inside={count_inside}, area_hat={area_hat:.6f}, SE={se:.6e}")
    if show_plot:
        plt.figure(figsize=(5,5))
        plt.scatter(XY[~inside,0], XY[~inside,1], s=5, color='lightgray', label='outside')
        plt.scatter(XY[inside,0], XY[inside,1], s=5, color='steelblue', label='inside')
        circle = plt.Circle((0.5,0.5),0.5, color='r', fill=False, linewidth=1.5)
        ax = plt.gca()
        ax.add_patch(circle)
        ax.set_aspect('equal')
        plt.title(f"MC circle area estimate: {area_hat:.5f} (SE {se:.2e})")
        plt.legend()
    return area_hat, se

# ---------------------------
# Task 2: Beam deflection Pf using Monte Carlo
# ---------------------------
def problem2_beam_deflection_mc(N=10000, seed=1, delta_allow=0.02, show_plots=True):
    """
    Monte Carlo estimation of Pf = P(delta > delta_allow) for beam under uncertain w,L,E,I.
    All units in SI (w N/m, L m, E Pa, I m^4).
    """
    rng = np.random.default_rng(seed)
    # Suggested distribution parameters (users can modify)
    # Note: We'll draw positive values; use positive_normal helper for S/E/I where needed.
    w_mean, w_sd = 5.0e3, 0.5e3     # 5 kN/m in N/m
    L_mean, L_sd = 5.0, 0.05
    E_mean, E_sd = 200e9, 10e9
    I_mean, I_sd = 8.0e-6, 0.5e-6

    # Draw samples (ensure positive)
    w = positive_normal(w_mean, w_sd, N, rng)
    L = positive_normal(L_mean, L_sd, N, rng)
    E = positive_normal(E_mean, E_sd, N, rng)
    I = positive_normal(I_mean, I_sd, N, rng)

    # Compute delta for each sample
    # WRITE_YOUR_CODE HERE TO COMPUTE deflection delta in meters
    delta = 

    # compute Pf
    failures = delta > delta_allow
    Nf = failures.sum()
    Pf = Nf / N
    se = math.sqrt(Pf*(1-Pf)/N)
    print(f"[problem2] N={N}, Pf={Pf:.6e}, Nf={Nf}, SE={se:.3e}")

    if show_plots:
        plt.figure(figsize=(8,4))
        plt.hist(delta, bins=40, density=False, alpha=0.7)
        plt.axvline(delta_allow, color='r', linestyle='--', linewidth=2, label=f'allow={delta_allow} m')
        plt.title(f"Histogram of midspan delta (MC), Pf={Pf:.3e}")
        plt.xlabel("delta (m)")
        plt.ylabel("count")
        plt.legend()

        # quick convergence plot with increasing N
        Ns = [500, 1000, 2000, 5000, N]
        estimates = []
        for n in Ns:
            w_n = positive_normal(w_mean, w_sd, n, rng)
            L_n = positive_normal(L_mean, L_sd, n, rng)
            E_n = positive_normal(E_mean, E_sd, n, rng)
            I_n = positive_normal(I_mean, I_sd, n, rng)
            delta_n = (5.0*w_n*(L_n**4)) / (384.0 * E_n * I_n)
            Pf_n = (delta_n > delta_allow).sum() / n
            estimates.append(Pf_n)
        plt.figure(figsize=(6,4))
        plt.plot(Ns, estimates, 'o-')
        plt.xscale('log')
        plt.xlabel('N (log scale)')
        plt.ylabel('Pf estimate')
        plt.title('Convergence of Pf estimate (MC)')
        plt.grid(True)

    # Simple sensitivity: Pearson corr between inputs and delta
    corr_w = np.corrcoef(w, delta)[0,1]
    corr_L = np.corrcoef(L, delta)[0,1]
    corr_E = np.corrcoef(E, delta)[0,1]
    corr_I = np.corrcoef(I, delta)[0,1]
    print("[problem2] Pearson corr with delta: w:{:.3f}, L:{:.3f}, E:{:.3f}, I:{:.3f}".format(corr_w, corr_L, corr_E, corr_I))
    return {'Pf': Pf, 'SE': se, 'Nf': Nf, 'delta': delta, 'inputs': {'w':w,'L':L,'E':E,'I':I}}

# ---------------------------
# Task 3: Beam deflection Pf using LHS
# ---------------------------
def problem3_beam_deflection_lhs(N=500, replicates=20, seed=2, delta_allow=0.02, show_plots=True):
    """
    Use Latin Hypercube Sampling to estimate Pf = P(delta > delta_allow).
    Compare estimator variance vs MC using same sample sizes.
    """
    rng = np.random.default_rng(seed)
    # Distribution specs for LHS (we will use tuples for rhs)
    # Note: using kN->N conversion for w
    distributions = [
        ('norm', 5.0e3, 0.5e3),    # w (N/m)
        ('norm', 5.0, 0.05),       # L (m)
        ('norm', 200e9, 10e9),     # E (Pa)
        ('norm', 8.0e-6, 0.5e-6)   # I (m^4)
    ]

    # Helper: single run of LHS (returns Pf)
    def run_lhs_once(N_local, rng_local):
        sample = lhs_sample(distributions, N_local, rng=rng_local)  # shape (N_local,4)
        w = sample[:,0]
        L = sample[:,1]
        E = sample[:,2]
        I = sample[:,3]
        delta = (5.0 * w * (L**4)) / (384.0 * E * I)
        Pf = np.mean(delta > delta_allow)
        return Pf, delta

    # Run replicates to compare spread of estimator for LHS vs MC
    pf_lhs = []
    pf_mc = []
    for rep in range(replicates):
        # use different RNG streams for each replicate
        rng_rep = np.random.default_rng(seed + rep + 100)
        pf_l, _ = run_lhs_once(N, rng_rep)
        pf_lhs.append(pf_l)

        # Monte Carlo baseline
        w = positive_normal(5.0e3, 0.5e3, N, rng_rep)
        L = positive_normal(5.0, 0.05, N, rng_rep)
        E = positive_normal(200e9, 10e9, N, rng_rep)
        I = positive_normal(8.0e-6, 0.5e-6, N, rng_rep)
        delta_mc = (5.0*w*(L**4)) / (384.0 * E * I)
        pf_mc.append(np.mean(delta_mc > delta_allow))

    pf_lhs = np.array(pf_lhs)
    pf_mc = np.array(pf_mc)
    print(f"[problem3] LHS: mean Pf={pf_lhs.mean():.3e}, std of estimator={pf_lhs.std(ddof=1):.3e}")
    print(f"[problem3] MC : mean Pf={pf_mc.mean():.3e}, std of estimator={pf_mc.std(ddof=1):.3e}")

    if show_plots:
        plt.figure(figsize=(8,4))
        plt.boxplot([pf_mc, pf_lhs], tick_labels=['MC', 'LHS'])
        plt.ylabel('Estimated Pf')
        plt.title(f'Comparison of Pf estimator spread (N={N}, replicates={replicates})')
        plt.grid(True)

    return {'pf_lhs': pf_lhs, 'pf_mc': pf_mc}

# ---------------------------
# Task 4: 3-bar truss FEA using MC and LHS
# ---------------------------
def problem4_truss_mcs_lhs(N=500, seed=10, show_plots=True):
    """
    Use analyze_truss to compute displacement at node C for 3-bar truss and compute Pf via MC and LHS.
    Returns dict with results.
    """
    rng = np.random.default_rng(seed)
    # Node coords A(0,0), B(4,0), C(2,2)
    node_coords = np.array([[0.0, 0.0],
                            [4.0, 0.0],
                            [2.0, 2.0]])
    # elements (0-based): AC, BC, AB
    elements = [(0,2), (1,2), (0,1)]
    # baseline areas in m^2 (converted from mm^2, e.g., 2000 mm2 = 0.002 m2)
    A1 = 2000e-6   # m^2 for AC
    A2 = 2000e-6   # m^2 for BC
    A3 = 3000e-6   # m^2 for AB
    A_nom = np.array([A1, A2, A3])
    E_nom = 200e9  # Pa
    # loads: global DOF = 2*n_nodes = 6
    # Apply downward point load P at node C (index 2) in y-direction
    # We'll set loads vector later per sample
    # supports: A pinned (ux,uy fixed), B roller (ux free, uy fixed)
    supports = {0: (True, True), 1: (False, True)}

    # random variables: P (kN -> N), E variation, A_i variation
    P_mean, P_sd = 50e3, 5e3  # N
    E_mean, E_sd = 200e9, 10e9
    # area variation: 10% SD
    A_sd_ratio = 0.10

    delta_allow = 0.02  # 20 mm

    # Helper: evaluate one sample and return delta_C
    def evaluate_sample(P, E_val, A_vec):
        # global loads vector (N)
        f = np.zeros(6)
        f[2*2 + 1] = -P  # node 2 (index 2), y-direction negative
        E_vec = np.array([E_val, E_val, E_val])
        u = analyze_truss(node_coords, elements, E_vec, A_vec, f, supports)
        # displacement of node C vertical is DOF index 2*2+1 = 5
        delta_C = u[5]
        return delta_C

    # Monte Carlo sampling
    t0 = time.time()
    deltas_mc = np.zeros(N)
    for i in range(N):
        P = positive_normal(P_mean, P_sd, 1, rng)[0]
        E_val = positive_normal(E_mean, E_sd, 1, rng)[0]
        A_vec = positive_normal(A_nom, A_sd_ratio*A_nom, 3, rng)
        deltas_mc[i] = evaluate_sample(P, E_val, A_vec)
    t_mc = time.time() - t0
    Pf_mc = np.mean(deltas_mc > delta_allow)
    print(f"[problem4 MC] N={N}, Pf={Pf_mc:.6e}, time={t_mc:.2f}s")

    # LHS sampling: for simplicity sample P, E, and one area scale factor (apply to all areas) or sample each A
    # We'll sample P, E, A1, A2, A3 as independent marginals using LHS
    distributions = [
        ('norm', P_mean, P_sd),      # P
        ('norm', E_mean, E_sd),      # E
        ('norm', A_nom[0], A_sd_ratio*A_nom[0]),  # A1
        ('norm', A_nom[1], A_sd_ratio*A_nom[1]),  # A2
        ('norm', A_nom[2], A_sd_ratio*A_nom[2])   # A3
    ]
    sample_lhs = lhs_sample(distributions, N, rng=rng)  # shape (N,5)
    deltas_lhs = np.zeros(N)
    t0 = time.time()
    for i in range(N):
        P = sample_lhs[i,0]
        E_val = sample_lhs[i,1]
        A_vec = sample_lhs[i,2:5]
        deltas_lhs[i] = evaluate_sample(P, E_val, A_vec)
    t_lhs = time.time() - t0
    Pf_lhs = np.mean(deltas_lhs > delta_allow)
    print(f"[problem4 LHS] N={N}, Pf={Pf_lhs:.6e}, time={t_lhs:.2f}s")

    if show_plots:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.hist(deltas_mc, bins=30, alpha=0.7)
        plt.axvline(delta_allow, color='r', linestyle='--', label='allow')
        plt.title(f"MC deltas (Pf={Pf_mc:.3e})")
        plt.xlabel('delta_C (m)')
        plt.legend()

        plt.subplot(1,2,2)
        plt.hist(deltas_lhs, bins=30, alpha=0.7)
        plt.axvline(delta_allow, color='r', linestyle='--', label='allow')
        plt.title(f"LHS deltas (Pf={Pf_lhs:.3e})")
        plt.xlabel('delta_C (m)')
        plt.legend()
        plt.tight_layout()

    return {'Pf_mc': Pf_mc, 'time_mc': t_mc, 'deltas_mc': deltas_mc,
            'Pf_lhs': Pf_lhs, 'time_lhs': t_lhs, 'deltas_lhs': deltas_lhs}

# ==========================================
# Run if executed
# ==========================================
if __name__ == "__main__":
    # Uncomment as needed

    # problem1_circle_area_mc(N=2000, seed=0, show_plot=True)
    # problem2_beam_deflection_mc(N=5000, seed=1, delta_allow=0.02, show_plots=True)
    # problem3_beam_deflection_lhs(N=500, replicates=20, seed=2, delta_allow=0.02, show_plots=True)
    # problem4_truss_mcs_lhs(N=400, seed=10, show_plots=True)

    plt.show()
