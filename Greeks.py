import numpy as np
import matplotlib
matplotlib.use('Agg')  # pas d'affichage, sauvegarde uniquement
import matplotlib.pyplot as plt
import os
os.makedirs('figures', exist_ok=True)

# ============================================================
# Q2) Prix du Call vanille par Monte Carlo
# P = e^{-rT} E[(S(T)-K)+]
# ============================================================

r=0.01
sigma=0.3
S0=1
K=1
T=2
N_simulations = 100000

# Simulation de Z ~ N(0,1) pour estimer S(T) = S0 * exp((r - sigma²/2)*T + sigma*sqrt(T)*Z)
Z = np.random.normal(0, 1, N_simulations)

ST=S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*Z)
payoff=np.maximum(ST-K,0)  # payoff du call : (S(T)-K)+
P=np.exp(-r*T)*np.mean(payoff)
print("Le prix du call (avec loi normale et np.mean) est : ", P)
#calcul avec loi normal : Le prix du call est :  0.2081737621858164

# Même simulation via Box-Muller : U1,U2 ~ U[0,1] → Z gaussien
U1 = np.random.uniform(0, 1, N_simulations)
U2 = np.random.uniform(0, 1, N_simulations)
Z2 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
ST=S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*Z2)
payoff2=np.maximum(ST-K,0)
P2=np.exp(-r*T)*np.mean(payoff2)
print("Le prix du call (avec loi uniform et np.mean) est : ", P2)
#calcul avec loi uniforme U1,U2: Le prix du call est :  0.2081737621858164

# Formule analytique Black-Scholes via la CDF de la loi normale (Abramowitz & Stegun)
def A(x):
    """CDF gaussienne via Abramowitz & Stegun (valide pour tout x reel)."""
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    if x < 0:
        return 1.0 - A(-x)
    t = 1 / (1 + b0 * x)
    a = 1 - (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2) * (
        b1*t + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5)
    return a
    

#d1=0.259272 
#d2=-0.164992 

# d1 et d2 : arguments de la formule de Black-Scholes
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
P3 = S0 * A(d1) - K * np.exp(-r * T) * A(d2)


print("Le prix du call (avec loi uniform et formule analytique) est : ", P3)

print("---" *50)

# ============================================================
# Q3) Grecques théoriques : Delta, Gamma, Vega
# ============================================================

# Delta = dP/dS0 : sensibilité du prix au cours initial
delta_th = A(d1)
print("Delta_theo : ", delta_th)

# Gamma = d²P/dS0² : courbure du prix par rapport à S0
gamma_th = 1/(np.sqrt(2*np.pi)*sigma*S0*np.sqrt(T))*np.exp(-0.5*d1**2)
print("Gamma_theo : ", gamma_th)

# Vega = dP/dsigma : sensibilité du prix à la volatilité
vega_th = S0 * np.sqrt(T) * 1/(np.sqrt(2*np.pi)) * np.exp(-0.5*d1**2)
print("Vega_theo : ", vega_th)

print("---" *50)

# ============================================================
# Q4-5) Estimateurs par différences finies et méthode trajectorielle
# ============================================================

# Z commun aux deux évaluations pour éviter l'effet tirage (variables communes)
Z = np.random.normal(0, 1, N_simulations)

def price_call(s, sig=sigma, Z=Z):
    ST = s * np.exp((r - 0.5 * sig**2) * T + sig * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

epsi = 0.00001  # epsilon petit pour approcher la dérivée (différences finies centrées)

# Delta par différences finies centrées : (P(S0+ε) - P(S0-ε)) / 2ε
delta_df = (price_call(S0 + epsi) - price_call(S0 - epsi)) / (2 * epsi)
print("Delta par méthode de différences finies, mc_standard : ", delta_df)

# Gamma par différences finies du second ordre : (P(S0+ε) - 2P(S0) + P(S0-ε)) / ε²
gamma_df = (price_call(S0 + epsi) - 2 * price_call(S0) + price_call(S0 - epsi)) / (epsi**2)
print("Gamma par méthode de différences finies, mc_standard : ", gamma_df)

# Vega : différences finies sur sigma
epsi_sig = 0.001
vega_df = (price_call(S0, sig=sigma + epsi_sig) - price_call(S0, sig=sigma - epsi_sig)) / (2 * epsi_sig)
print("Vega par méthode de différences finies, mc_standard : ", vega_df)

print("---" *50)

# Méthode trajectorielle : on dérive sous l'espérance (Q4)
Z = np.random.normal(0, 1, N_simulations)
ST=(S0)*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*Z)

# Delta trajectioriel : E[e^{-rT} * 1_{S(T)>K} * S(T)/S0]
indicatrise = ST > K 
delta_mf=np.exp(-r*T)*np.mean(indicatrise*ST/S0)
print("Delta par méthode trajectorielle, mc_standard : ", delta_mf)

# Gamma trajectioriel : Dirac approché par 1_{|S(T)-K|<=xi} / 2xi
xi = 0.05
Dirac_approx = (np.abs(ST - K) <= xi) / (2 * xi)
gamma_mf=np.exp(-r*T)*np.mean(Dirac_approx*(ST/S0)**2)
print("Gamma par méthode trajectorielle, mc_standard : ", gamma_mf)

# Vega trajectioriel : dérivation par rapport à sigma, fait apparaître (W_T - sigma*T)
WT = np.sqrt(T) * Z
vega_mf=np.exp(-r*T)*np.mean(indicatrise*ST*(WT-sigma*T))
print("Vega par méthode trajectorielle, mc_standard : ", vega_mf)


# ============================================================
# Q6) Réduction de variance par variables antithétiques
# ============================================================

# Principe : pour chaque Z, on utilise aussi -Z → réduction de variance si f(Z)+f(-Z) corrélés négativement
N_demi = N_simulations // 2
U1 = np.random.uniform(0, 1, N_demi) 
U2 = np.random.uniform(0, 1, N_demi) 

Z = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
Z_anti = -Z  # trajectoire antithétique

WT = np.sqrt(T) * Z
WT_anti = np.sqrt(T) * Z_anti
ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * WT)
ST_anti = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * WT_anti)

ind = ST > K
ind_anti = ST_anti > K

xi = 0.05
dirac = (np.abs(ST - K) <= xi) / (2 * xi) 
dirac_anti = (np.abs(ST_anti - K) <= xi) / (2 * xi) 

# Estimateur antithétique : moyenne des contributions de Z et -Z
delta_Z = np.exp(-r * T) * ind * (ST / S0)
delta_anti = np.exp(-r * T) * ind_anti * (ST_anti / S0)
delta_paire = (delta_Z + delta_anti) / 2  
delta_p=np.mean(delta_paire)

gamma_Z = np.exp(-r * T) * dirac * (ST / S0)**2
gamma_anti = np.exp(-r * T) * dirac_anti * (ST_anti / S0)**2
gamma_paire = (gamma_Z + gamma_anti) / 2  
gamma_p=np.mean(gamma_paire)

vega_Z = np.exp(-r * T) * ind * ST * (WT - sigma * T)
vega_anti = np.exp(-r * T) * ind_anti * ST_anti * (WT_anti - sigma * T)
vega_paire = (vega_Z + vega_anti) / 2     
vega_p=np.mean(vega_paire)  

print("---" *50)

print("Delta par méthode trajectorielle, antithétique : ", delta_p)
print("Gamma par méthode trajectorielle, antithétique : ", gamma_p)
print("Vega par méthode trajectorielle, antithétique : ", vega_p)


# Comparaison des 4 estimateurs en fonction de N, avec IC à 90%
N_values = [1000, 5000, 10000, 20000, 50000, 100000]
z90 = 1.645  # quantile à 90% de la loi normale
eps = 0.001 
xi = 0.05   

# Stockage : 4 méthodes → [DF Std, Traj Std, Traj Anti, DF Anti]
res = {'delta': {'mean': [[], [], [], []], 'ci': [[], [], [], []]},
       'gamma': {'mean': [[], [], [], []], 'ci': [[], [], [], []]},
       'vega':  {'mean': [[], [], [], []], 'ci': [[], [], [], []]}}

for N in N_values:
    U1 = np.random.uniform(0, 1, N)
    U2 = np.random.uniform(0, 1, N)
    Z_full = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    N_demi = N // 2
    Z_half = Z_full[:N_demi]
    Z_anti = -Z_half  # trajectoires antithétiques
    
    ST_std = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_full)
    WT_std = np.sqrt(T) * Z_full
    
    ST_H = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_half)
    ST_A = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_anti)
    WT_H = np.sqrt(T) * Z_half
    WT_A = np.sqrt(T) * Z_anti

    def p_eval_Z(s, sig, Z_arr):
        """Payoff avec un Z donné (pour DF antithétique)."""
        return np.exp(-r*T) * np.maximum(s * np.exp((r - 0.5*sig**2)*T + sig*np.sqrt(T)*Z_arr) - K, 0)

    def p_eval(s, sig):
        return p_eval_Z(s, sig, Z_full)

    # DF antithétique : moyenne des DF calculés avec Z_half et -Z_half
    df_delta_H = (p_eval_Z(S0+eps, sigma, Z_half) - p_eval_Z(S0-eps, sigma, Z_half)) / (2*eps)
    df_delta_A = (p_eval_Z(S0+eps, sigma, Z_anti) - p_eval_Z(S0-eps, sigma, Z_anti)) / (2*eps)
    df_gamma_H = (p_eval_Z(S0+eps, sigma, Z_half) - 2*p_eval_Z(S0, sigma, Z_half) + p_eval_Z(S0-eps, sigma, Z_half)) / eps**2
    df_gamma_A = (p_eval_Z(S0+eps, sigma, Z_anti) - 2*p_eval_Z(S0, sigma, Z_anti) + p_eval_Z(S0-eps, sigma, Z_anti)) / eps**2
    df_vega_H  = (p_eval_Z(S0, sigma+eps, Z_half) - p_eval_Z(S0, sigma-eps, Z_half)) / (2*eps)
    df_vega_A  = (p_eval_Z(S0, sigma+eps, Z_anti) - p_eval_Z(S0, sigma-eps, Z_anti)) / (2*eps)

    delta_evals = [
        (p_eval(S0+eps, sigma) - p_eval(S0-eps, sigma)) / (2*eps),                          # 0: DF Std
        np.exp(-r*T) * (ST_std > K) * (ST_std / S0),                                        # 1: Traj Std
        (np.exp(-r*T)*(ST_H>K)*(ST_H/S0) + np.exp(-r*T)*(ST_A>K)*(ST_A/S0)) / 2,           # 2: Traj Anti
        (df_delta_H + df_delta_A) / 2                                                        # 3: DF Anti
    ]

    dirac_std = (np.abs(ST_std - K) <= xi) / (2*xi)
    dirac_H = (np.abs(ST_H - K) <= xi) / (2*xi)
    dirac_A = (np.abs(ST_A - K) <= xi) / (2*xi)

    gamma_evals = [
        (p_eval(S0+eps, sigma) - 2*p_eval(S0, sigma) + p_eval(S0-eps, sigma)) / (eps**2),   # 0: DF Std
        np.exp(-r*T) * dirac_std * (ST_std / S0)**2,                                        # 1: Traj Std
        (np.exp(-r*T)*dirac_H*(ST_H/S0)**2 + np.exp(-r*T)*dirac_A*(ST_A/S0)**2) / 2,        # 2: Traj Anti
        (df_gamma_H + df_gamma_A) / 2                                                        # 3: DF Anti
    ]

    vega_evals = [
        (p_eval(S0, sigma+eps) - p_eval(S0, sigma-eps)) / (2*eps),                          # 0: DF Std
        np.exp(-r*T) * (ST_std > K) * ST_std * (WT_std - sigma*T),                          # 1: Traj Std
        (np.exp(-r*T)*(ST_H>K)*ST_H*(WT_H-sigma*T) + np.exp(-r*T)*(ST_A>K)*ST_A*(WT_A-sigma*T)) / 2, # 2: Traj Anti
        (df_vega_H + df_vega_A) / 2                                                          # 3: DF Anti
    ]

    evaluations = [('delta', delta_evals), ('gamma', gamma_evals), ('vega', vega_evals)]
    for greek_name, evals in evaluations:
        for i in range(4):
            m = np.mean(evals[i])
            n_size = N_demi if i in (2, 3) else N  # antithétique utilise N/2 paires
            ci = z90 * np.std(evals[i], ddof=1) / np.sqrt(n_size)
            res[greek_name]['mean'][i].append(m)
            res[greek_name]['ci'][i].append(ci)

labels = ['DF Standard', 'Traj Standard', 'Traj Antithétique', 'DF Antithétique']
colors = ['blue', 'red', 'orange', 'green']

# Tracé de la convergence des 4 estimateurs + valeur théorique pour chaque grecque
for greek_name, th_val, title in [('delta', delta_th, 'Delta'), 
                                  ('gamma', gamma_th, 'Gamma'), 
                                  ('vega', vega_th, 'Vega')]:
    plt.figure(figsize=(10, 6))
    
    for i in range(4):
        means = np.array(res[greek_name]['mean'][i])
        cis = np.array(res[greek_name]['ci'][i])
        
        plt.plot(N_values, means, label=labels[i], color=colors[i], marker='o')
        plt.fill_between(N_values, means - cis, means + cis, color=colors[i], alpha=0.15)
        
    plt.axhline(th_val, color='black', linestyle='--', linewidth=2, label=f'Théorique = {th_val:.4f}')
    
    plt.title(f'Convergence des estimateurs pour {title}')
    plt.xlabel('Nombre de trajectoires (N)')
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/fig_q6_{greek_name}.pdf')
    plt.close()

# ============================================================
# Q7) Grecques vanille en fonction de S0 ∈ [0, 2]
# Critère : N=50000 → IC 90% < 1% de la valeur
# ============================================================
print("=" * 80)
print("Q7) Grecques vanille (antithetique, N=50000) en fonction de S0 in [0,2]")
N_q7 = 50000
S0_vals_q7 = np.linspace(0.01, 2.0, 60)

d_q7_mc, g_q7_mc, v_q7_mc = [], [], []
d_q7_th, g_q7_th, v_q7_th = [], [], []

for s0v in S0_vals_q7:
    N2 = N_q7 // 2
    U1v = np.random.uniform(0, 1, N2)
    U2v = np.random.uniform(0, 1, N2)
    Zv  = np.sqrt(-2 * np.log(U1v)) * np.cos(2 * np.pi * U2v)
    Za  = -Zv  # antithétique
    WTv = np.sqrt(T) * Zv;  WTa = np.sqrt(T) * Za
    STv = s0v * np.exp((r - 0.5*sigma**2)*T + sigma*WTv)
    STa = s0v * np.exp((r - 0.5*sigma**2)*T + sigma*WTa)
    xiv = 0.05
    ind_v = STv > K;  ind_a = STa > K
    dirac_v = (np.abs(STv - K) <= xiv) / (2*xiv)
    dirac_a = (np.abs(STa - K) <= xiv) / (2*xiv)
    d_mc = np.mean((np.exp(-r*T)*ind_v*STv/s0v + np.exp(-r*T)*ind_a*STa/s0v) / 2)
    g_mc = np.mean((np.exp(-r*T)*dirac_v*(STv/s0v)**2 + np.exp(-r*T)*dirac_a*(STa/s0v)**2) / 2)
    v_mc = np.mean((np.exp(-r*T)*ind_v*STv*(WTv-sigma*T) + np.exp(-r*T)*ind_a*STa*(WTa-sigma*T)) / 2)
    d_q7_mc.append(d_mc); g_q7_mc.append(g_mc); v_q7_mc.append(v_mc)
    # Valeurs theoriques B&S
    d1v = (np.log(s0v/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d_q7_th.append(A(d1v))
    g_q7_th.append((1/np.sqrt(2*np.pi))*np.exp(-0.5*d1v**2) / (s0v*sigma*np.sqrt(T)))
    v_q7_th.append(s0v*np.sqrt(T)*(1/np.sqrt(2*np.pi))*np.exp(-0.5*d1v**2))

# Delta ∈ [0,1] croissant (OTM→ITM), Gamma et Vega maximaux en ATM (S0=K)
for mc, th, title_q7 in [(d_q7_mc, d_q7_th, 'Delta'), (g_q7_mc, g_q7_th, 'Gamma'), (v_q7_mc, v_q7_th, 'Vega')]:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(S0_vals_q7, mc, label='MC antithetique', color='darkorange')
    ax.plot(S0_vals_q7, th, label='Theorique B&S', color='black', linestyle='--')
    ax.axvline(K, color='gray', linestyle=':', linewidth=1, label='ATM (S0=K=1)')
    ax.set_title(f'Q7 — {title_q7} en fonction de S0')
    ax.set_xlabel('S0'); ax.set_ylabel(title_q7); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(f'figures/fig_q7_{title_q7.lower()}.pdf'); plt.close()
print("  Critere : N=50000 trajectoires => IC 90% < 1% de la valeur absolue")
print("  Delta croît de 0 (OTM) à 1 (ITM), Gamma et Vega maximaux à l'ATM")
print("=" * 80)

# ============================================================
# Q8) Call panier : grecques trajectorielle (dimension 2)
# Panier B(T) = alpha*S1(T) + beta*S2(T)
# ============================================================

alpha = 0.5
beta  = 0.5
rho   = 0.5
sigma1 = 0.25
sigma2 = 0.30
S1_0  = 1.0
S2_0  = 1.0
N_sim = 100000
xi_basket = 0.05


np.random.seed(42)
# Box-Muller : loi uniforme uniquement (consigne du sujet)
U1_q8 = np.random.uniform(0, 1, N_sim); U2_q8 = np.random.uniform(0, 1, N_sim)
U3_q8 = np.random.uniform(0, 1, N_sim); U4_q8 = np.random.uniform(0, 1, N_sim)
Z1 = np.sqrt(-2 * np.log(U1_q8)) * np.cos(2 * np.pi * U2_q8)
Z2 = np.sqrt(-2 * np.log(U3_q8)) * np.cos(2 * np.pi * U4_q8)

# Cholesky : W2 = rho*W1 + sqrt(1-rho²)*Z2 pour obtenir Corr(W1,W2)=rho
W1T = np.sqrt(T) * Z1
W2T = np.sqrt(T) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

S1T = S1_0 * np.exp((r - 0.5 * sigma1**2) * T + sigma1 * W1T)
S2T = S2_0 * np.exp((r - 0.5 * sigma2**2) * T + sigma2 * W2T)

BT = alpha * S1T + beta * S2T   # valeur du panier à maturité

payoff_basket = np.maximum(BT - K, 0)
P_basket = np.exp(-r * T) * np.mean(payoff_basket)
print("=" * 80)
print("Q8) Grecques du Call Panier — méthode trajectiorielle")
print(f"Prix du call panier  : {P_basket:.6f}")

# Delta1 = dP/dS1,0 : même forme que le cas vanille avec alpha*S1(T)/S1,0
ind_basket = BT > K
delta1_traj = np.exp(-r * T) * np.mean(ind_basket * alpha * S1T / S1_0)
print(f"Delta1 (trajectioriel) : {delta1_traj:.6f}")

# Gamma1 : Dirac approché en BT (et non ST)
dirac_basket = (np.abs(BT - K) <= xi_basket) / (2 * xi_basket)
gamma1_traj = np.exp(-r * T) * np.mean(dirac_basket * (alpha * S1T / S1_0)**2)
print(f"Gamma1 (trajectioriel, Dirac approx xi={xi_basket}) : {gamma1_traj:.6f}")

# Vega1 = dP/dsigma1 : fait apparaître (W1T - sigma1*T)
vega1_traj = np.exp(-r * T) * np.mean(ind_basket * alpha * S1T * (W1T - sigma1 * T))
print(f"Vega1  (trajectioriel) : {vega1_traj:.6f}")

z90 = 1.645

# IC à 90% : z90 * std / sqrt(N)
samples_delta = np.exp(-r * T) * ind_basket * alpha * S1T / S1_0
ci_delta = z90 * np.std(samples_delta, ddof=1) / np.sqrt(N_sim)
print(f"IC 90% Delta1 : [{delta1_traj - ci_delta:.6f}, {delta1_traj + ci_delta:.6f}]")

samples_gamma = np.exp(-r * T) * dirac_basket * (alpha * S1T / S1_0)**2
ci_gamma = z90 * np.std(samples_gamma, ddof=1) / np.sqrt(N_sim)
print(f"IC 90% Gamma1 : [{gamma1_traj - ci_gamma:.6f}, {gamma1_traj + ci_gamma:.6f}]")

samples_vega = np.exp(-r * T) * ind_basket * alpha * S1T * (W1T - sigma1 * T)
ci_vega = z90 * np.std(samples_vega, ddof=1) / np.sqrt(N_sim)
print(f"IC 90% Vega1  : [{vega1_traj - ci_vega:.6f}, {vega1_traj + ci_vega:.6f}]")
print("=" * 80)

# ============================================================
# Q9) Comparaison DF vs trajectioriel (antithétique) sur panier
# ============================================================

# PARAMETRES COMMUNS q9-q12
r_   = 0.01
sig1 = 0.25
sig2 = 0.30
al   = 0.5
be   = 0.5
rho_ = 0.5
S10  = 1.0
S20  = 1.0
K_   = 1.0
T_   = 2.0
z90  = 1.645
xi_  = 0.05
eps_ = 0.001
M_   = 50      # pas pour simuler W(t)

def simulate_BM(N, rho=rho_, T=T_, M=M_):
    """Simule W1(T) et W2(T) correles (Cholesky) via Box-Muller (loi uniforme)."""
    dt = T / M
    # Box-Muller : N*M paires de gaussiennes depuis des uniformes
    U1bm = np.random.uniform(0, 1, (N, M))
    U2bm = np.random.uniform(0, 1, (N, M))
    U3bm = np.random.uniform(0, 1, (N, M))
    U4bm = np.random.uniform(0, 1, (N, M))
    dZ1 = np.sqrt(-2 * np.log(U1bm)) * np.cos(2 * np.pi * U2bm)
    dZ2 = np.sqrt(-2 * np.log(U3bm)) * np.cos(2 * np.pi * U4bm)
    W1T = np.sum(np.sqrt(dt) * dZ1, axis=1)
    W2T = np.sum(np.sqrt(dt) * (rho * dZ1 + np.sqrt(1 - rho**2) * dZ2), axis=1)
    return W1T, W2T

def basket_assets(W1T, W2T, s10=S10, s20=S20, sg1=sig1, sg2=sig2, T=T_):
    S1 = s10 * np.exp((r_ - 0.5*sg1**2)*T + sg1*W1T)
    S2 = s20 * np.exp((r_ - 0.5*sg2**2)*T + sg2*W2T)
    return S1, S2

def greeks_antithetic(N, s10=S10, s20=S20, sg1=sig1, sg2=sig2,
                      rho=rho_, K=K_, T=T_):
    W1T, W2T = simulate_BM(N, rho=rho, T=T)
    N2 = N // 2
    w1h, w1a = W1T[:N2], -W1T[:N2]  # paires antithétiques
    w2h, w2a = W2T[:N2], -W2T[:N2]

    def contrib(w1, w2):
        s1, s2 = basket_assets(w1, w2, s10, s20, sg1, sg2, T)
        B  = al*s1 + be*s2
        ind = (B > K).astype(float)
        dirac = (np.abs(B - K) <= xi_) / (2*xi_)
        d = np.exp(-r_*T) * ind * al * s1 / s10
        g = np.exp(-r_*T) * dirac * (al * s1 / s10)**2
        v = np.exp(-r_*T) * ind * al * s1 * (w1 - sg1*T)
        return d, g, v

    dh, gh, vh = contrib(w1h, w2h)
    da, ga, va = contrib(w1a, w2a)
    delta_s = (dh + da) / 2  # moyenne des deux contributions
    gamma_s = (gh + ga) / 2
    vega_s  = (vh + va) / 2
    return delta_s, gamma_s, vega_s

def greeks_fd(N, s10=S10, s20=S20, sg1=sig1, sg2=sig2,
              rho=rho_, K=K_, T=T_):
    W1T, W2T = simulate_BM(N, rho=rho, T=T)  # Z commun pour DF (variables communes)

    def payoff(s10_v, sg1_v):
        s1, s2 = basket_assets(W1T, W2T, s10_v, s20, sg1_v, sg2, T)
        return np.exp(-r_*T) * np.maximum(al*s1 + be*s2 - K, 0)

    P0  = payoff(s10,        sg1)
    Pp  = payoff(s10+eps_,   sg1)
    Pm  = payoff(s10-eps_,   sg1)
    Psp = payoff(s10,        sg1+eps_)
    Psm = payoff(s10,        sg1-eps_)

    d = (Pp - Pm) / (2*eps_)
    g = (Pp - 2*P0 + Pm) / (eps_**2)
    v = (Psp - Psm) / (2*eps_)
    return d, g, v

def mean_ci(arr, n):
    m  = np.mean(arr)
    ci = z90 * np.std(arr, ddof=1) / np.sqrt(n)
    return m, ci

# Convergence en fonction de N pour les deux estimateurs (DF et trajectioriel)
N_vals = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
res9 = {g: {'fd': {'m':[], 'ci':[]}, 'tr': {'m':[], 'ci':[]}}
        for g in ['delta','gamma','vega']}

for N in N_vals:
    fd  = greeks_fd(N)
    tr  = greeks_antithetic(N)
    N2  = N // 2
    for idx, gname in enumerate(['delta','gamma','vega']):
        mf, cif = mean_ci(fd[idx],  N)
        mt, cit = mean_ci(tr[idx], N2)
        res9[gname]['fd']['m'].append(mf);  res9[gname]['fd']['ci'].append(cif)
        res9[gname]['tr']['m'].append(mt);  res9[gname]['tr']['ci'].append(cit)

print("=" * 80)
print("Q9) Grecques call panier — valeurs convergées (N=100 000)")
print(f"  sigma=(0.25,0.30)  alpha=beta=0.5  rho=0.5  r=0.01  K=1  T=2")
for gname, gtitle in [('delta','Delta1'), ('gamma','Gamma1'), ('vega','Vega1')]:
    mfd = res9[gname]['fd']['m'][-1];  cfd = res9[gname]['fd']['ci'][-1]
    mtr = res9[gname]['tr']['m'][-1];  ctr = res9[gname]['tr']['ci'][-1]
    print(f"  {gtitle:7s}  DF={mfd:.5f} ± {cfd:.5f}   Traj={mtr:.5f} ± {ctr:.5f}")
print("=" * 80)

for gname, gtitle in [('delta','Delta1'), ('gamma','Gamma1'), ('vega','Vega1')]:
    fig, ax = plt.subplots(figsize=(9,5))
    for key, label, col in [('fd','DF (var. communes)','steelblue'),
                              ('tr','Trajectioriel (antith.)','darkorange')]:
        ms  = np.array(res9[gname][key]['m'])
        cis = np.array(res9[gname][key]['ci'])
        ax.plot(N_vals, ms, marker='o', color=col, label=label)
        ax.fill_between(N_vals, ms-cis, ms+cis, color=col, alpha=0.2)
    ax.set_xscale('log')
    ax.set_title(f'Q9 — {gtitle} en fonction de N')
    ax.set_xlabel('N'); ax.set_ylabel(gtitle); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(f'figures/fig_q9_{gname}.pdf'); plt.close()

# ============================================================
# Q10) Grecques en fonction de S1,0 — comparaison panier vs alpha*call BS
# ============================================================
N_ref  = 50000
S10_vals = np.linspace(0.01, 1.0, 40)

d10_tr, g10_tr, v10_tr = [], [], []
d10_fd, g10_fd, v10_fd = [], [], []
d10_bs, g10_bs, v10_bs = [], [], []   # comparaison alpha*call(K_eff)

from scipy.stats import norm

def bs_greeks_call(S, K_eff, sg, r=r_, T=T_):
    if S <= 0 or K_eff <= 0:
        return 0.0, 0.0, 0.0
    d1 = (np.log(S/K_eff) + (r + 0.5*sg**2)*T) / (sg*np.sqrt(T))
    d2 = d1 - sg*np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sg * np.sqrt(T))
    vega  = S * np.sqrt(T) * norm.pdf(d1)
    return delta, gamma, vega

for s10v in S10_vals:
    tr = greeks_antithetic(N_ref, s10=s10v)
    fd = greeks_fd(N_ref, s10=s10v)
    d10_tr.append(np.mean(tr[0])); g10_tr.append(np.mean(tr[1])); v10_tr.append(np.mean(tr[2]))
    d10_fd.append(np.mean(fd[0])); g10_fd.append(np.mean(fd[1])); v10_fd.append(np.mean(fd[2]))
    # Référence analytique : alpha * call BS de strike K_eff = K - beta*S2,0
    # (approximation valide si S2 était déterministe)
    K_eff = K_ - be * S20
    if K_eff > 0:
        dbs, gbs, vbs = bs_greeks_call(s10v, K_eff, sig1)
        d10_bs.append(al * dbs); g10_bs.append(al * gbs); v10_bs.append(al * vbs)
    else:
        d10_bs.append(np.nan); g10_bs.append(np.nan); v10_bs.append(np.nan)

# Index S1_0=1.0 (dernier point)
print("=" * 80)
print("Q10) Grecques à S1,0=1.0 — panier MC vs alpha*call BS (K_eff=K-beta*S2_0=0.5)")
for arr_tr, arr_bs, gname in [(d10_tr, d10_bs, 'Delta1'), (g10_tr, g10_bs, 'Gamma1'), (v10_tr, v10_bs, 'Vega1')]:
    print(f"  {gname:7s}  panier MC={arr_tr[-1]:.5f}   alpha*BS={arr_bs[-1]:.5f}   ecart={abs(arr_tr[-1]-arr_bs[-1]):.5f}")
print("  -> Le call panier != alpha*call(K_eff) car S2 est stochastique (randomisation du strike)")
print("=" * 80)

for arr_tr, arr_fd, arr_bs, gname in [
        (d10_tr, d10_fd, d10_bs, 'Delta1'),
        (g10_tr, g10_fd, g10_bs, 'Gamma1'),
        (v10_tr, v10_fd, v10_bs, 'Vega1')]:
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(S10_vals, arr_tr, label='Trajectioriel (antith.)', color='darkorange')
    ax.plot(S10_vals, arr_fd, label='DF (var. communes)', color='steelblue', linestyle='--')
    ax.plot(S10_vals, arr_bs, label=r'$\alpha$·call(K-$\beta S_{2,0}$) BS', color='green', linestyle=':')
    ax.set_title(f'Q10 — {gname} en fonction de S1,0')
    ax.set_xlabel('S1,0'); ax.set_ylabel(gname); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(f'figures/fig_q10_{gname.lower().replace("1","")}.pdf'); plt.close()

# ============================================================
# Q11) Grecques en fonction de K ∈ [0, 2]
# ============================================================
K_vals = np.linspace(0.01, 2.0, 40)
d11_tr, g11_tr, v11_tr = [], [], []
d11_fd, g11_fd, v11_fd = [], [], []

# W fixé pour réduire la variance entre les évaluations à différents K
W1T_fixed, W2T_fixed = simulate_BM(N_ref)

def greeks_antithetic_fixed_W(W1T, W2T, K, s10=S10):
    N = len(W1T); N2 = N // 2
    w1h, w1a = W1T[:N2], -W1T[:N2]
    w2h, w2a = W2T[:N2], -W2T[:N2]
    def contrib(w1, w2):
        s1, s2 = basket_assets(w1, w2, s10)
        B  = al*s1 + be*s2
        ind = (B > K).astype(float)
        dirac = (np.abs(B - K) <= xi_) / (2*xi_)
        d = np.exp(-r_*T_) * ind * al * s1 / s10
        g = np.exp(-r_*T_) * dirac * (al * s1 / s10)**2
        v = np.exp(-r_*T_) * ind * al * s1 * (w1 - sig1*T_)
        return d, g, v
    dh,gh,vh = contrib(w1h,w2h); da,ga,va = contrib(w1a,w2a)
    return np.mean((dh+da)/2), np.mean((gh+ga)/2), np.mean((vh+va)/2)

def greeks_fd_fixed_W(W1T, W2T, K, s10=S10):
    def payoff(s10v, sg1v):
        s1, s2 = basket_assets(W1T, W2T, s10v, S20, sg1v)
        return np.exp(-r_*T_) * np.maximum(al*s1 + be*s2 - K, 0)
    P0=payoff(s10,sig1); Pp=payoff(s10+eps_,sig1); Pm=payoff(s10-eps_,sig1)
    Psp=payoff(s10,sig1+eps_); Psm=payoff(s10,sig1-eps_)
    return np.mean((Pp-Pm)/(2*eps_)), np.mean((Pp-2*P0+Pm)/eps_**2), np.mean((Psp-Psm)/(2*eps_))

for Kv in K_vals:
    dt, gt, vt = greeks_antithetic_fixed_W(W1T_fixed, W2T_fixed, Kv)
    df, gf, vf = greeks_fd_fixed_W(W1T_fixed, W2T_fixed, Kv)
    d11_tr.append(dt); g11_tr.append(gt); v11_tr.append(vt)
    d11_fd.append(df); g11_fd.append(gf); v11_fd.append(vf)

# IC à 90% calculés sur un second tirage indépendant
W1T_q11_ci, W2T_q11_ci = simulate_BM(N_ref)
ci11 = {g: {'tr': [], 'fd': []} for g in ['d','g','v']}
for Kv in K_vals:
    N2 = N_ref // 2
    w1h, w1a = W1T_q11_ci[:N2], -W1T_q11_ci[:N2]
    w2h, w2a = W2T_q11_ci[:N2], -W2T_q11_ci[:N2]
    def contrib_ci(w1, w2):
        s1, s2 = basket_assets(w1, w2)
        B = al*s1+be*s2; ind=(B>Kv).astype(float); dirac=(np.abs(B-Kv)<=xi_)/(2*xi_)
        d=np.exp(-r_*T_)*ind*al*s1/S10; g=np.exp(-r_*T_)*dirac*(al*s1/S10)**2
        v=np.exp(-r_*T_)*ind*al*s1*(w1-sig1*T_)
        return d,g,v
    dh,gh,vh=contrib_ci(w1h,w2h); da,ga,va=contrib_ci(w1a,w2a)
    for arr, key in [((dh+da)/2,'d'),((gh+ga)/2,'g'),((vh+va)/2,'v')]:
        ci11[key]['tr'].append(z90*np.std(arr,ddof=1)/np.sqrt(N2))
    def pf(s10v,sg1v):
        s1,s2=basket_assets(W1T_q11_ci,W2T_q11_ci,s10v,S20,sg1v)
        return np.exp(-r_*T_)*np.maximum(al*s1+be*s2-Kv,0)
    P0=pf(S10,sig1); Pp=pf(S10+eps_,sig1); Pm=pf(S10-eps_,sig1)
    Psp=pf(S10,sig1+eps_); Psm=pf(S10,sig1-eps_)
    ci11['d']['fd'].append(z90*np.std((Pp-Pm)/(2*eps_),ddof=1)/np.sqrt(N_ref))
    ci11['g']['fd'].append(z90*np.std((Pp-2*P0+Pm)/eps_**2,ddof=1)/np.sqrt(N_ref))
    ci11['v']['fd'].append(z90*np.std((Psp-Psm)/(2*eps_),ddof=1)/np.sqrt(N_ref))

# Delta → alpha (ITM, K→0), Delta → 0 (OTM, K grand) ; Gamma/Vega max à l'ATM
idx_atm = np.argmin(np.abs(K_vals - 1.0))
print("=" * 80)
print(f"Q11) Grecques à K≈{K_vals[idx_atm]:.2f} (ATM)")
for arr_tr, arr_fd, gname in [(d11_tr,d11_fd,'Delta1'),(g11_tr,g11_fd,'Gamma1'),(v11_tr,v11_fd,'Vega1')]:
    print(f"  {gname:7s}  Traj={arr_tr[idx_atm]:.5f}   DF={arr_fd[idx_atm]:.5f}")
print(f"  Delta -> 0 quand K grand (OTM), -> alpha={al} quand K->0 (ITM)")
print(f"  Gamma -> 0 aux extrêmes, max autour ATM")
print(f"  Vega  -> 0 aux extrêmes, max autour ATM")
print("=" * 80)

for arr_tr, arr_fd, ci_key, gname, fname11 in [
        (d11_tr, d11_fd, 'd', 'Delta1', 'delta'),
        (g11_tr, g11_fd, 'g', 'Gamma1', 'gamma'),
        (v11_tr, v11_fd, 'v', 'Vega1',  'vega')]:
    fig, ax = plt.subplots(figsize=(9,5))
    for arr, key, label, col in [(arr_tr,'tr','Trajectioriel (antith.)','darkorange'),
                                  (arr_fd,'fd','DF (var. communes)','steelblue')]:
        ms  = np.array(arr)
        cis = np.array(ci11[ci_key][key])
        ax.plot(K_vals, ms, label=label, color=col)
        ax.fill_between(K_vals, ms-cis, ms+cis, color=col, alpha=0.2)
    ax.set_title(f'Q11 — {gname} en fonction de K')
    ax.set_xlabel('K'); ax.set_ylabel(gname); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(f'figures/fig_q11_{fname11}.pdf'); plt.close()

# ============================================================
# Q12) Grecques en fonction de rho ∈ ]-1, 1[
# ============================================================
rho_vals = np.linspace(-0.99, 0.99, 40)
d12_tr, g12_tr, v12_tr = [], [], []
d12_fd, g12_fd, v12_fd = [], [], []

for rhov in rho_vals:
    tr = greeks_antithetic(N_ref, rho=rhov)
    fd = greeks_fd(N_ref, rho=rhov)
    d12_tr.append(np.mean(tr[0])); g12_tr.append(np.mean(tr[1])); v12_tr.append(np.mean(tr[2]))
    d12_fd.append(np.mean(fd[0])); g12_fd.append(np.mean(fd[1])); v12_fd.append(np.mean(fd[2]))

print("=" * 80)
print("Q12) Grecques en fonction de rho — valeurs aux extrêmes et au centre")
for arr, gname in [(d12_tr,'Delta1'),(g12_tr,'Gamma1'),(v12_tr,'Vega1')]:
    print(f"  {gname:7s}  rho=-0.99: {arr[0]:.5f}   rho=0: {arr[len(arr)//2]:.5f}   rho=+0.99: {arr[-1]:.5f}")
# Delta1 ne dépend que de S1 → peu sensible à rho
# Gamma1 décroît avec rho (moins de diversification → Dirac plus concentré)
# Vega1 croît avec rho (panier plus volatile → plus sensible à sigma1)
print("  -> Delta1 peu sensible à rho (ne dépend que de S1)")
print("  -> Gamma1 décroît avec rho : à rho=-1 panier peu volatile (diversif.) → distribution concentrée → forte courbure")
print("  -> Vega1  croît  avec rho  : à rho=+1 panier très volatile → prix très sensible à sigma")
print("=" * 80)

for arr_tr, arr_fd, gname in [
        (d12_tr, d12_fd, 'Delta1'),
        (g12_tr, g12_fd, 'Gamma1'),
        (v12_tr, v12_fd, 'Vega1')]:
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(rho_vals, arr_tr, label='Trajectioriel (antith.)', color='darkorange')
    ax.plot(rho_vals, arr_fd, label='DF (var. communes)', color='steelblue', linestyle='--')
    ax.set_title(f'Q12 — {gname} en fonction de rho')
    ax.set_xlabel('rho'); ax.set_ylabel(gname); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(f'figures/fig_q12_{gname.lower().replace("1","")}.pdf'); plt.close()


# ============================================================
# Q13) Parité Call-Put sur panier
# Call - Put = alpha*S1,0 + beta*S2,0 - K*e^{-rT}
# => Delta1_call - Delta1_put = alpha, Gamma1 et Vega1 identiques
# ============================================================
print("=" * 80)
print("Q13) Parité Call-Put sur panier")
print()
print("On calcule analytiquement :")
print("  e^{-rT}E[(alphaS1+betaS2-K)+] - e^{-rT}E[(K-alphaS1-betaS2)+]")
print("= e^{-rT}E[alphaS1+betaS2-K]")
print("= e^{-rT}[alpha*S1_0*e^{rT} + beta*S2_0*e^{rT} - K]")
print("= alpha*S1_0 + beta*S2_0 - K*e^{-rT}")
print()
print("Relations grecques Call/Put sur le même panier :")
print("  Delta1_call - Delta1_put = alpha")
print("  Gamma1_call - Gamma1_put = 0   (Gamma1_call = Gamma1_put)")
print("  Vega1_call  - Vega1_put  = 0   (Vega1_call  = Vega1_put)")

W1T_p, W2T_p = simulate_BM(N_ref)
S1p = S10 * np.exp((r_ - 0.5*sig1**2)*T_ + sig1*W1T_p)
S2p = S20 * np.exp((r_ - 0.5*sig2**2)*T_ + sig2*W2T_p)
Bp  = al*S1p + be*S2p
ind_call = (Bp > K_).astype(float)
ind_put  = (Bp < K_).astype(float)
dirac_p  = (np.abs(Bp - K_) <= xi_) / (2*xi_)

# Grecques call et put : mêmes trajectoires, indicatrices opposées
delta1_call_mc = np.exp(-r_*T_) * np.mean(ind_call * al * S1p / S10)
delta1_put_mc  = np.exp(-r_*T_) * np.mean(-ind_put  * al * S1p / S10)
gamma1_call_mc = np.exp(-r_*T_) * np.mean(dirac_p * (al*S1p/S10)**2)
gamma1_put_mc  = np.exp(-r_*T_) * np.mean(dirac_p * (al*S1p/S10)**2)  # Dirac identique
vega1_call_mc  = np.exp(-r_*T_) * np.mean(ind_call * al * S1p * (W1T_p - sig1*T_))
vega1_put_mc   = np.exp(-r_*T_) * np.mean(-ind_put  * al * S1p * (W1T_p - sig1*T_))

print()
P_call_mc = np.exp(-r_*T_) * np.mean(np.maximum(Bp - K_, 0))
P_put_mc  = np.exp(-r_*T_) * np.mean(np.maximum(K_ - Bp, 0))
parite_th = al*S10 + be*S20 - K_*np.exp(-r_*T_)
print(f"  Prix call panier (MC)         = {P_call_mc:.5f}")
print(f"  Prix put  panier (MC)         = {P_put_mc:.5f}")
print(f"  Call - Put (MC)               = {P_call_mc - P_put_mc:.5f}  (theorique = {parite_th:.5f})")
print()
print(f"  Delta1 call (MC)              = {delta1_call_mc:.5f}")
print(f"  Delta1 put  (MC)              = {delta1_put_mc:.5f}")
print(f"  Delta1_call - Delta1_put (MC) = {delta1_call_mc - delta1_put_mc:.5f}  (theorique = alpha = {al})")
print()
print(f"  Gamma1 call (MC)              = {gamma1_call_mc:.5f}")
print(f"  Gamma1 put  (MC)              = {gamma1_put_mc:.5f}")
print(f"  Gamma1_call - Gamma1_put (MC) = {gamma1_call_mc - gamma1_put_mc:.6f}  (theorique = 0)")
print()
print(f"  Vega1 call  (MC)              = {vega1_call_mc:.5f}")
print(f"  Vega1 put   (MC)              = {vega1_put_mc:.5f}")
print(f"  Vega1_call  - Vega1_put  (MC) = {vega1_call_mc  - vega1_put_mc:.6f}  (theorique = 0)")
print("=" * 80)


# ============================================================
# Q14) Couverture Delta quotidienne — paramètres importants
# P&L ≈ sum_k [0.5 * Gamma_k * (dS_k)² - Theta_k * dt]
# ============================================================
print("=" * 80)
print("Q14) Couverture Delta quotidienne — paramètres importants")
print()
print("Le P&L de couverture Delta sur [0,T] avec rebalancement discret quotidien")
print("vaut approximativement (par développement de Taylor au 2e ordre) :")
print()
print("  P&L ≈ sum_k [ 0.5 * Gamma_k * (dS_k)^2 - Theta_k * dt ]")
print()
print("Les paramètres déterminants sont :")
print("  - Gamma (Γ) : sensibilité du delta à S → ampleur de l'erreur de couverture")
print("  - Theta (Θ) : dépréciation temporelle de l'option")
print("  - Vega  (v) : risque résiduel si σ implicite ≠ σ réalisée")
print("  - Fréquence de rebalancement : quotidienne ici (dt = 1/252)")
print("  - Volatilité réalisée vs implicite : écart = source principale de P&L")
print()
print("Relation clé (Black-Scholes) :")
print("  Theta + 0.5 * sigma^2 * S^2 * Gamma = r * V")
print("  => un portefeuille Delta-neutre est aussi Gamma/Theta neutre en BS.")
print("=" * 80)
