import os
import numpy as np
import pandas as pd
import pymultinest
import corner
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM,LambdaCDM, FlatwCDM

# Ponemos las rutas a los archivos 
data_path = "/work1/labavanz/Ale/supernovas_multinest/Pantheon+SH0ES.dat"
cov_path = "/work1/labavanz/Ale/supernovas_multinest/Pantheon+SH0ES_STATONLY.cov"

output_folder = "/work1/labavanz/Ale/supernovas_multinest/chains_flat_LCDM/"
os.makedirs(output_folder, exist_ok=True)

# Leemos los datos 
data_original = pd.read_csv(data_path, sep=r"\s+", comment="#")

# y tambien la covarianza
with open(cov_path, "r") as f:
    N = int(f.readline())
    cov_values = np.loadtxt(f)

cov_original = cov_values.reshape((N, N))

# Retiramos SH0ES
mask = data_original["IS_CALIBRATOR"].values == 0
data = data_original[mask].reset_index(drop=True)

indices = np.where(mask)[0]
cov = cov_original[np.ix_(indices, indices)]

cov_inv = np.linalg.inv(cov)

#Variables
z = data["zHD"].values
mB = data["mB"].values
x1 = data["x1"].values
color = data["c"].values
bias = data["biasCor_m_b"].values

# Marcomos esta seccion para funciones generales

# Fijamos H0. En supernovas solas H0 y M están degenerados,
# por eso H0 se fija y M se deja como parámetro libre.

H0_fijo = 67.0

def mu_obs_tripp(alpha, beta, M):
    return mB + alpha*x1 - beta*color - M - bias

def calcular_AIC_BIC(samples, n_params, Ndatos):
    """
    Calcula AIC y BIC usando el máximo log-likelihood.

    AIC = -2 ln(Lmax) + 2k
    BIC = -2 ln(Lmax) + k ln(N)
    """

    loglikes = samples[:, -1]
    logL_max = np.max(loglikes)

    AIC = -2*logL_max + 2*n_params
    BIC = -2*logL_max + n_params*np.log(Ndatos)

    return logL_max, AIC, BIC

#Modelo 1, Flat LCDM, en esta definimos por los parametros 
# theta = [Omega_M, alpha, beta, M]
output_folder = "/work1/labavanz/Ale/supernovas_multinest/chains_flat_LCDM/"
os.makedirs(output_folder, exist_ok=True)

parameters = ["Omega_M", "alpha", "beta", "M"]
n_params = len(parameters)


def mu_modelo_flat_LCDM(Omega_M):

    cosmo = FlatLambdaCDM(
        H0=H0_fijo,
        Om0=Omega_M
    )

    return cosmo.distmod(z).value

#Definimos el prior para que funcione en el multinest
def prior(cube, ndim, nparams):

    cube[0] = 0.01 + cube[0]*(1.0 - 0.01)          # Omega_M
    cube[1] = 0.0 + cube[1]*(0.5 - 0.0)            # alpha
    cube[2] = 1.0 + cube[2]*(5.0 - 1.0)            # beta
    cube[3] = -25.0 + cube[3]*(-15.0 - (-25.0))    # M

# Ahora el Likelihood
def loglike(cube, ndim, nparams):

    Omega_M = cube[0]
    alpha = cube[1]
    beta = cube[2]
    M = cube[3]

    mu_obs = mu_obs_tripp(alpha, beta, M)
    mu_model = mu_modelo_flat_LCDM(Omega_M)

    delta = mu_obs - mu_model

    chi2 = delta.T @ cov_inv @ delta

    return -0.5*chi2

# Corremos el multinest para flat LCDM
pymultinest.run(
    LogLikelihood=loglike,
    Prior=prior,
    n_dims=n_params,
    outputfiles_basename=output_folder,
    resume=False,
    verbose=True,
    n_live_points=500,
    evidence_tolerance=0.5,
    sampling_efficiency=0.8
)

# Analiza el resultado del Multinest
analyzer = pymultinest.Analyzer(
    n_params=n_params,
    outputfiles_basename=output_folder
)

samples = analyzer.get_equal_weighted_posterior()
samples_params = samples[:, :-1]

print("\nRESULTADOS FLAT LCDM\n")

for i, name in enumerate(parameters):
    p16, p50, p84 = np.percentile(samples_params[:, i], [16, 50, 84])
    print(f"{name} = {p50:.4f} +{p84-p50:.4f} -{p50-p16:.4f}")
#Calculamos AIC y BIC para Flat LCDM
logL_flat, AIC_flat, BIC_flat = calcular_AIC_BIC(
    samples,
    n_params,
    len(z)
)

print("\nCRITERIOS DE INFORMACIÓN FLAT LCDM\n")
print(f"logL_max = {logL_flat:.4f}")
print(f"AIC = {AIC_flat:.4f}")
print(f"BIC = {BIC_flat:.4f}")

#Graficamos 
fig = corner.corner(
    samples_params,
    labels=parameters,
    show_titles=True
)

fig.savefig("/work1/labavanz/Ale/supernovas_multinest/corner_flat_LCDM.png")


# Modelo LCDM con curvatura, aqui utilizamos 
# theta = [Omega_M, Omega_Lambda, alpha, beta, M]
 
output_folder_curv = "/work1/labavanz/Ale/supernovas_multinest/chains_LCDM_curvatura/"
os.makedirs(output_folder_curv, exist_ok=True)

parameters_curv = ["Omega_M", "Omega_Lambda", "alpha", "beta", "M"]
n_params_curv = len(parameters_curv)

# En este modelo Omega_M y Omega_Lambda son libres.
    # La curvatura queda dada por:
    # Omega_k = 1 - Omega_M - Omega_Lambda


def mu_modelo_LCDM_curvatura(Omega_M, Omega_Lambda):
    cosmo = LambdaCDM(
        H0=H0_fijo,
        Om0=Omega_M,
        Ode0=Omega_Lambda
    )
    return cosmo.distmod(z).value

#Definimos ahora el Prior para LCDM curv
def prior_curv(cube, ndim, nparams):

    cube[0] = 0.01 + cube[0]*(1.5 - 0.01)          # Omega_M
    cube[1] = 0.0 + cube[1]*(2.0 - 0.0)            # Omega_Lambda
    cube[2] = 0.0 + cube[2]*(0.5 - 0.0)            # alpha
    cube[3] = 1.0 + cube[3]*(5.0 - 1.0)            # beta
    cube[4] = -25.0 + cube[4]*(-15.0 - (-25.0))    # M

#Igual definimos el Likelihood
def loglike_curv(cube, ndim, nparams):
    Omega_M = cube[0]
    Omega_Lambda = cube[1]
    alpha = cube[2]
    beta = cube[3]
    M = cube[4]

    mu_obs = mu_obs_tripp(alpha, beta, M)
    mu_model = mu_modelo_LCDM_curvatura(Omega_M, Omega_Lambda)

    delta = mu_obs - mu_model
    chi2 = delta.T @ cov_inv @ delta

    return -0.5*chi2

#Aplicamos Multinest para LCDM curv
pymultinest.run(
    LogLikelihood=loglike_curv,
    Prior=prior_curv,
    n_dims=n_params_curv,
    outputfiles_basename=output_folder_curv,
    resume=False,
    verbose=True,
    n_live_points=500,
    evidence_tolerance=0.5,
    sampling_efficiency=0.8
)

#Analizamos el resultadoo
analyzer_curv = pymultinest.Analyzer(
    n_params=n_params_curv,
    outputfiles_basename=output_folder_curv
)

samples_curv = analyzer_curv.get_equal_weighted_posterior()
samples_params_curv = samples_curv[:, :-1]

print("\nRESULTADOS LCDM CON CURVATURA\n")

for i, name in enumerate(parameters_curv):
    p16, p50, p84 = np.percentile(samples_params_curv[:, i], [16, 50, 84])
    print(f"{name} = {p50:.4f} +{p84-p50:.4f} -{p50-p16:.4f}")

#Calculamos Omega_K
Omega_k_samples = 1 - samples_params_curv[:, 0] - samples_params_curv[:, 1]
p16, p50, p84 = np.percentile(Omega_k_samples, [16, 50, 84])
print(f"Omega_k = {p50:.4f} +{p84-p50:.4f} -{p50-p16:.4f}")


#Ahora calculamos AIC y BIC
logL_curv, AIC_curv, BIC_curv = calcular_AIC_BIC(
    samples_curv,
    n_params_curv,
    len(z)
)

print("\nCRITERIOS DE INFORMACIÓN LCDM CON CURVATURA\n")
print(f"logL_max = {logL_curv:.4f}")
print(f"AIC = {AIC_curv:.4f}")
print(f"BIC = {BIC_curv:.4f}")

#Graficamos
fig = corner.corner(
    samples_params_curv,
    labels=parameters_curv,
    show_titles=True
)

fig.savefig("/work1/labavanz/Ale/supernovas_multinest/corner_LCDM_curvatura.png")

# Modelo wCDM sin curvatura, aqui aplicamos los paremetros 
# theta = [Omega_M, w, alpha, beta, M]

output_folder_w = "/work1/labavanz/Ale/supernovas_multinest/chains_wCDM/"
os.makedirs(output_folder_w, exist_ok=True)

parameters_w = ["Omega_M", "w", "alpha", "beta", "M"]
n_params_w = len(parameters_w)


def mu_modelo_wCDM(Omega_M, w):
    cosmo = FlatwCDM(
        H0=H0_fijo,
        Om0=Omega_M,
        w0=w
    )
    return cosmo.distmod(z).value

#Definimos su prior
def prior_w(cube, ndim, nparams):

    cube[0] = 0.01 + cube[0]*(1.0 - 0.01)          # Omega_M
    cube[1] = -3.0 + cube[1]*(0.0 - (-3.0))        # w
    cube[2] = 0.0 + cube[2]*(0.5 - 0.0)            # alpha
    cube[3] = 1.0 + cube[3]*(5.0 - 1.0)            # beta
    cube[4] = -25.0 + cube[4]*(-15.0 - (-25.0))    # M


# Igual su Likelihood
def loglike_w(cube, ndim, nparams):
    Omega_M = cube[0]
    w = cube[1]
    alpha = cube[2]
    beta = cube[3]
    M = cube[4]

    mu_obs = mu_obs_tripp(alpha, beta, M)
    mu_model = mu_modelo_wCDM(Omega_M, w)

    delta = mu_obs - mu_model
    chi2 = delta.T @ cov_inv @ delta

    return -0.5*chi2

#Corremos ahora su Multinest para wCDM
pymultinest.run(
    LogLikelihood=loglike_w,
    Prior=prior_w,
    n_dims=n_params_w,
    outputfiles_basename=output_folder_w,
    resume=False,
    verbose=True,
    n_live_points=500,
    evidence_tolerance=0.5,
    sampling_efficiency=0.8
)

#Analizamos ese resultado
analyzer_w = pymultinest.Analyzer(
    n_params=n_params_w,
    outputfiles_basename=output_folder_w
)

samples_w = analyzer_w.get_equal_weighted_posterior()
samples_params_w = samples_w[:, :-1]

print("\nRESULTADOS wCDM SIN CURVATURA\n")

for i, name in enumerate(parameters_w):
    p16, p50, p84 = np.percentile(samples_params_w[:, i], [16, 50, 84])
    print(f"{name} = {p50:.4f} +{p84-p50:.4f} -{p50-p16:.4f}")

#Calculamos su AIC y BIC
logL_w, AIC_w, BIC_w = calcular_AIC_BIC(
    samples_w,
    n_params_w,
    len(z)
)

print("\nCRITERIOS DE INFORMACIÓN wCDM\n")
print(f"logL_max = {logL_w:.4f}")
print(f"AIC = {AIC_w:.4f}")
print(f"BIC = {BIC_w:.4f}")

#Graficamos
fig = corner.corner(
    samples_params_w,
    labels=parameters_w,
    show_titles=True
)

fig.savefig("/work1/labavanz/Ale/supernovas_multinest/corner_wCDM.png")


# Creamos una tabla compartiva para AIC y BIC
# El mejor modelo es el que tiene menor AIC o menor BIC.

tabla = pd.DataFrame({
    "Modelo": [
        "Flat LCDM",
        "LCDM con curvatura",
        "Flat wCDM"
    ],

    "k": [
        n_params,
        n_params_curv,
        n_params_w
    ],

    "logL_max": [
        logL_flat,
        logL_curv,
        logL_w
    ],

    "AIC": [
        AIC_flat,
        AIC_curv,
        AIC_w
    ],

    "BIC": [
        BIC_flat,
        BIC_curv,
        BIC_w
    ]
})

tabla["Delta_AIC"] = tabla["AIC"] - tabla["AIC"].min()
tabla["Delta_BIC"] = tabla["BIC"] - tabla["BIC"].min()

print("\nCOMPARACIÓN FINAL DE MODELOS\n")
print(tabla)

tabla.to_csv(
    "/work1/labavanz/Ale/supernovas_multinest/comparacion_modelos_AIC_BIC.csv",
    index=False
)

print("\nTabla guardada en comparacion_modelos_AIC_BIC.csv")

