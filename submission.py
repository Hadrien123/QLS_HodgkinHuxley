# %% SECTION 1: Packages import
import os
import numpy as np

# Import other packages as necessary
from scipy.optimize import least_squares

# %% SECTION 2: Helper functions
# Define functions if needed
# Bounds: [e_r, g_bar, m_inf steepness, m_inf center, h_inf steepness, h_inf center, tau_m scale, center, width, tau_h scale, center, width, g_l, e_l]
lower = np.array([-150, 0,   0, -150, -1, -150, 0.01, -200, 1,   0.01, -200, 1,   0, -150])
upper = np.array([ 150, 100, 1,  150,  0,  150, 1e3,   200, 1e5, 1e4,   200, 1e5, 1,  150])
rng = np.random.default_rng(0)   # fixed seed -> reproducible
n_starts = 20

def gate_model(tau, x_0,x_inf,dt):
    y = x_0 + (x_inf - x_0) * (1 - np.exp(-dt / tau))
    return y

# Per-segment model: one voltage segment, its own m_inf and h_inf, carries m/h between segments
def get_current_segment(e_r, g_bar, tau_m, tau_h, m_inf, h_inf, v, t, m_0, h_0):
    m = np.zeros(len(v))
    h = np.zeros(len(v))

    m[0] = m_0
    h[0] = h_0

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]

        m[i] = gate_model(tau_m, m[i-1], m_inf, dt)
        h[i] = gate_model(tau_h, h[i-1], h_inf, dt)

    return g_bar * (v - e_r) * m * h, m[-1], h[-1]

# [tau_m, tau_h, m_inf, h_inf] at step voltage number k. tau_h is stored as a ratio >= 1 to tau_m
# so that m is always the fast gate (the current only depends on m * h, so the two could swap).
def segment_params(params, k):
    tau_m, ratio, m_inf, h_inf = params[2+4*k:6+4*k]
    return tau_m, tau_m * ratio, m_inf, h_inf

# All 10 segments are fitted together, with one shared e_r and g_bar.
# Segments at the same voltage share their gate parameters: seg_v[j-1] is the voltage index of segment j.
def compute_error_segment(params, data, past, seg_v, k_hold):
    e_r, g_bar = params[:2]
    _, _, m_0, h_0 = segment_params(params, k_hold)   # start from the holding steady state
    errors = []
    for j in range(1, 11):
        start = j*100-past
        stop = (j+1)*100
        tau_m, tau_h, m_inf, h_inf = segment_params(params, seg_v[j-1])
        current, m_0, h_0 = get_current_segment(e_r, g_bar, tau_m, tau_h, m_inf, h_inf, data[start:stop, 1], data[start:stop, 0], m_0, h_0)
        errors.append(current - data[start:stop, 2])
    return np.concatenate(errors)

# Whole-trace model: m_inf/h_inf are sigmoids of voltage, tau_m/tau_h are Gaussians of voltage
def get_current(e_r, g_bar, m_params, h_params, tau_m_params, tau_h_params, g_l, e_l, v, t):
    m = np.zeros(len(v))
    h = np.zeros(len(v))

    m_inf = sigmoid(*m_params, v)
    h_inf = sigmoid(*h_params, v)
    tau_m = np.maximum(gaussian(*tau_m_params, v), 1e-3)   # floor: a Gaussian goes to 0 far from its center
    tau_h = np.maximum(gaussian(*tau_h_params, v), 1e-3)

    m[0] = m_inf[0]   # the trace starts at the holding voltage, so start at steady state
    h[0] = h_inf[0]

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]

        m[i] = gate_model(tau_m[i], m[i-1], m_inf[i], dt)
        h[i] = gate_model(tau_h[i], h[i-1], h_inf[i], dt)

    return g_bar * (v - e_r) * m * h + g_l * (v - e_l)   # channel current + leak

def compute_error(params, t, v, test_i):
    e_r, g_bar = params[:2]
    return get_current(e_r, g_bar, params[2:4], params[4:6], params[6:9], params[9:12], params[12], params[13], v, t) - test_i

def random_start():
    e_r   = rng.uniform(-100, 50)
    g_bar = 10 ** rng.uniform(-3, 0)    # log-uniform: spans orders of magnitude
    m_inf = [ 10 ** rng.uniform(-2, -0.5), rng.uniform(-80, 20)]    # steepness > 0: rises with voltage
    h_inf = [-10 ** rng.uniform(-2, -0.5), rng.uniform(-80, 20)]    # steepness < 0: falls with voltage
    tau_m = [10 ** rng.uniform(-1, 1), rng.uniform(-80, 20), 10 ** rng.uniform(2, 4)]
    tau_h = [10 ** rng.uniform(0, 2), rng.uniform(-80, 20), 10 ** rng.uniform(2, 4)]
    g_l   = 10 ** rng.uniform(-4, -2)
    e_l   = rng.uniform(-80, 0)
    return [e_r, g_bar, *m_inf, *h_inf, *tau_m, *tau_h, g_l, e_l]

def gaussian(scale,center,width,v):
    return scale*np.exp(-((v-center)**2)/width)

def compute_error_tau(params,v,tau):
    scale,center,width = params
    return gaussian(scale,center,width,v) - tau # Gaussian formula

def sigmoid(steepness,center,v):
    return 1/(1+np.exp(-steepness*(v-center)))

def compute_error_x_inf(params,v,x_inf):
    steepness,center = params
    return sigmoid(steepness,center,v) - x_inf

# Sigmoid through the segment fits' x_inf values
def fit_x_inf(v, x_inf):
    return least_squares(compute_error_x_inf, args=(np.asarray(v), x_inf), x0=[1, 0]).x

# Gaussians through the segment fits' tau_m and tau_h values
def fit_taus(v, tau_m, tau_h):
    v = np.asarray(v)
    fit_m = least_squares(compute_error_tau, args=(v, tau_m), x0=[3, 20, 2000])
    fit_h = least_squares(compute_error_tau, args=(v, tau_h), x0=[30, -40, 2000])
    return fit_m.x, fit_h.x

# Model comparison (lower is better). Assumes Gaussian noise, so the likelihood
# reduces to the residual sum of squares; constant terms are dropped.
def compute_aic(residuals, n_params):
    n = len(residuals)
    ss_res = np.sum(np.asarray(residuals) ** 2)
    return n * np.log(ss_res / n) + 2 * n_params

def compute_bic(residuals, n_params):
    n = len(residuals)
    ss_res = np.sum(np.asarray(residuals) ** 2)
    return n * np.log(ss_res / n) + n_params * np.log(n)

# The training data: looked up in `folder` first, then in data/ next to this file
TRAIN_FILE = "CookLab1UnknownChannelExtra.txt"

def find_train_file(folder):
    candidates = [os.path.join(folder, TRAIN_FILE),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", TRAIN_FILE)]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{TRAIN_FILE} not found in: {candidates}")

def r_squared(residuals, observed):
    return 1 - np.sum(residuals ** 2) / np.sum((observed - np.mean(observed)) ** 2)

# %% SECTION 3: Main function
def run_vclamp_model(folder: str, data: str = "CookLab1UnknownChannelExtra.txt", train_frac: float = 0.75):
    """
    Compute current from voltage-clamp input.

    Parameters
    ----------
    folder :
        Directory with input file for voltage-clamp model.
    data :
        Name of input file. The model is run on this file; it is never used for fitting.
    # Please list any parameters added to the function.
    train_frac :
        Fraction of the Extra file's trace (from the start) used to fit the model; the
        rest is held out and only used to report hold-out R2 and RMSE.

    Returns
    -------
    np.ndarray
        Output current computed by the voltage-clamp model.
    """
    # %% SECTION 3-a: Load provided input data.
    # NO MODIFICATIONS: This section should be kept as is.
    data = np.loadtxt(os.path.join(folder, data))
    v = data[:, 1]

    # i has the same shape as array v, add code in the next section to fill it
    i = np.empty_like(v)

    # %% SECTION 3-b: Compute i [to be filled by students]
    # The model is always trained on the Extra file (first train_frac of it), then run on `data`.
    rng.bit_generator.state = np.random.default_rng(0).bit_generator.state   # reseed: every call uses the same random starts
    extra = np.loadtxt(find_train_file(folder))
    tau_m = []
    tau_h = []
    m_inf = []
    h_inf = []
    past=1

    # One fit for all 10 segments (see compute_error_segment)
    seg_volts = [extra[j*100+50, 1] for j in range(1, 11)]
    tau_v = sorted(set(seg_volts))
    seg_v = [tau_v.index(x) for x in seg_volts]
    k_hold = tau_v.index(extra[0, 1])
    # [e_r, g_bar], then [tau_m, tau_h / tau_m, m_inf, h_inf] for each voltage
    lower_seg = [-150, 0]   + [0.01, 1,   0, 0] * len(tau_v)
    upper_seg = [ 150, 100] + [100,  1e4, 1, 1] * len(tau_v)
    fit = least_squares(compute_error_segment, args = (extra, past, seg_v, k_hold), x0 = [-20, 0.01] + [1, 5, 0.5, 0.5] * len(tau_v), bounds = (lower_seg, upper_seg), max_nfev = 400)
    print("Segment fit succeeded:", fit.success)

    for k in range(len(tau_v)):
        seg = segment_params(fit.x, k)
        print(f"Fitted parameters for V={tau_v[k]}:", seg)
        tau_m.append(seg[0])
        tau_h.append(seg[1])
        m_inf.append(seg[2])
        h_inf.append(seg[3])

    # Rescale the gates to peak at 1 and move the factor into g_bar (the current only depends on g_bar * m * h)
    e_r = fit.x[0]
    g_bar = fit.x[1] * max(m_inf) * max(h_inf)
    m_inf = np.array(m_inf) / max(m_inf)
    h_inf = np.array(h_inf) / max(h_inf)

    tau_m_fit, tau_h_fit = fit_taus(tau_v,tau_m,tau_h)
    m_fit = fit_x_inf(tau_v,m_inf)
    h_fit = fit_x_inf(tau_v,h_inf)

    t_x = extra[:, 0]
    v_x = extra[:, 1]
    i_x = extra[:, 2]

    # Whole-trace fit on the first train_frac of the Extra trace only. The gates carry state, so the
    # full trace is always simulated and the mask picks which points count.
    train = np.arange(len(t_x)) < len(t_x) * train_frac
    err = lambda p: compute_error(p, t_x, v_x, i_x)[train]

    fits = []
    for k in range(n_starts):
        if k == 0:
            x0 = np.clip([e_r, g_bar, *m_fit, *h_fit, *tau_m_fit, *tau_h_fit, 1e-3, -60], lower, upper)   # the segment fits' curves, small leak
        else:
            x0 = random_start()
        fits.append(least_squares(err, x0=x0, bounds=(lower, upper), max_nfev=40, x_scale='jac'))
    best = min(fits, key=lambda f: f.cost)   # best start, judged on the training points only
    fit = least_squares(err, x0=best.x, bounds=(lower, upper), max_nfev=400, x_scale='jac')   # let the best start converge

    res = compute_error(fit.x, t_x, v_x, i_x)
    print("Fitted parameters:", fit.x)
    print("Fit succeeded:", fit.success)
    print(f"Extra file, train ({train_frac:.0%}): R2 = {r_squared(res[train], i_x[train]):.3f}")
    print(f"Extra file, hold-out ({1 - train_frac:.0%}): R2 = {r_squared(res[~train], i_x[~train]):.3f}, RMSE = {np.sqrt(np.mean(res[~train] ** 2)):.4f}")
    print("AIC (train):", compute_aic(res[train], len(fit.x)))
    print("BIC (train):", compute_bic(res[train], len(fit.x)))

    # Run the trained model on the input file
    i = get_current(fit.x[0], fit.x[1], fit.x[2:4], fit.x[4:6], fit.x[6:9], fit.x[9:12], fit.x[12], fit.x[13], v, data[:, 0])
    if data.shape[1] > 2:   # the input file has a measured current: report how well the model predicts it
        print(f"Input file: R2 = {r_squared(i - data[:, 2], data[:, 2]):.3f}, RMSE = {np.sqrt(np.mean((i - data[:, 2]) ** 2)):.4f}")

    # %% SECTION 3-c: Assert validity of output and return it.
    # NO MODIFICATIONS: This section should be kept as is.
    assert len(i.shape) == 1, "Currrent array should be 1-dimensional"
    return i

# %% SECTION 4: Function calls
if __name__ == "__main__":
    # Replace this with the path to the folder where provided input .txt file is!
    folder = "data"

    # Run function on the voltage-clamp data
    i = run_vclamp_model(folder)