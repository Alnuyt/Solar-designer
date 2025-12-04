import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from excel_generator import get_catalog, generate_workbook_bytes

# ----------------------------------------------------
# CONFIG STREAMLIT
# ----------------------------------------------------
st.set_page_config(
    page_title="Dimensionneur Solaire Sigen",
    layout="wide",
)

# ----------------------------------------------------
# CATALOGUE
# ----------------------------------------------------
PANELS, INVERTERS, BATTERIES = get_catalog()
PANEL_IDS = [p[0] for p in PANELS]


# ----------------------------------------------------
# FONCTIONS CATALOGUE
# ----------------------------------------------------
def get_panel_elec(panel_id: str):
    for p in PANELS:
        if p[0] == panel_id:
            return {
                "id": p[0],
                "Pstc": float(p[1]),
                "Voc": float(p[2]),
                "Vmp": float(p[3]),
                "Isc": float(p[4]),
                "alpha_V": float(p[6]),  # %/°C
            }
    return None


def get_inverter_elec(inv_id: str):
    for inv in INVERTERS:
        # (ID, P_AC_nom, P_DC_max, V_MPP_min, V_MPP_max,
        #  V_DC_max, I_MPPT, Nb_MPPT, Type_reseau, Famille)
        if inv[0] == inv_id:
            return {
                "id": inv[0],
                "P_ac": float(inv[1]),
                "P_dc_max": float(inv[2]),
                "Vmpp_min": float(inv[3]),
                "Vmpp_max": float(inv[4]),
                "Vdc_max": float(inv[5]),
                "Impp_max": float(inv[6]),
                "nb_mppt": int(inv[7]),
                "type_reseau": inv[8],
                "famille": inv[9],
            }
    return None


# ----------------------------------------------------
# PROFILS CONSOMMATION / PRODUCTION
# ----------------------------------------------------
def monthly_pv_profile_kwh_kwp():
    """Profil mensuel PV Belgique (kWh/an/kWc)."""
    annual_kwh_kwp = 1034.0
    distribution = np.array([3.8, 5.1, 8.7, 11.5, 12.1, 11.8,
                             11.9, 10.8, 9.7, 7.0, 4.3, 3.3])
    return annual_kwh_kwp * distribution / 100.0


def monthly_consumption_profile(annual_kwh: float, profile: str):
    profiles = {
        "Standard":   [7, 7, 8, 9, 9, 9, 9, 9, 8, 8, 8, 9],
        "Hiver fort": [10,10,10, 9, 8, 7, 6, 6, 7, 8, 9,10],
        "Été fort":   [6, 6, 7, 8, 9,10,11,11,10, 8, 7, 7],
    }
    arr = np.array(profiles[profile], dtype=float)
    arr = arr / arr.sum()
    return annual_kwh * arr


def hourly_profile(profile_name: str):
    """Profil de consommation horaire (24 valeurs qui somment à 1)."""
    if profile_name == "Uniforme":
        return np.ones(24) / 24

    if profile_name == "Classique (matin + soir)":
        prof = np.array([
            0.02,0.02,0.02,0.02,0.02,
            0.04,0.06,0.08,0.06,0.03,
            0.02,0.02,0.02,0.02,0.03,
            0.04,0.06,0.08,0.07,0.04,
            0.02,0.01,0.01,0.01
        ])
        return prof / prof.sum()

    if profile_name == "Travail journée (soir fort)":
        prof = np.array([
            0.01,0.01,0.01,0.01,0.01,
            0.02,0.03,0.03,0.03,0.02,
            0.01,0.01,0.01,0.01,0.02,
            0.04,0.07,0.09,0.10,0.10,
            0.05,0.02,0.01,0.01
        ])
        return prof / prof.sum()

    if profile_name == "Télétravail":
        prof = np.array([
            0.02,0.02,0.03,0.03,0.03,
            0.04,0.05,0.06,0.06,0.06,
            0.05,0.05,0.05,0.05,0.05,
            0.05,0.05,0.06,0.06,0.06,
            0.05,0.03,0.02,0.02
        ])
        return prof / prof.sum()

    return np.ones(24) / 24


# ----------------------------------------------------
# OPTIMISATION DES STRINGS
# ----------------------------------------------------
def optimize_strings(
    N_tot: int,
    panel: dict,
    inverter: dict,
    T_min: float,
    T_max: float,
):
    """
    Optimisation automatique des strings :
    - calcule N_series, N_strings, N_used, P_dc, ratio DC/AC
    - 1 string max par MPPT (dans la pratique : soit 0 soit 1 string par MPPT)
    - vérifie Voc froid, Vmp chaud, courant MPPT
    """
    Voc = panel["Voc"]
    Vmp = panel["Vmp"]
    Isc = panel["Isc"]
    alpha_V = panel["alpha_V"] / 100.0  # %/°C -> 1/°C
    Pstc = panel["Pstc"]

    Vdc_max = inverter["Vdc_max"]
    Vmpp_min = inverter["Vmpp_min"]
    Vmpp_max = inverter["Vmpp_max"]
    Impp_max = inverter["Impp_max"]
    nb_mppt = inverter["nb_mppt"]
    P_ac = inverter["P_ac"]

    voc_factor_cold = (1 + alpha_V * (T_min - 25.0))
    vmp_factor_hot = (1 + alpha_V * (T_max - 25.0))

    if voc_factor_cold <= 0 or vmp_factor_hot <= 0:
        return None

    # Bornes sur le nombre de modules en série
    N_series_max = math.floor(Vdc_max / (Voc * voc_factor_cold))
    N_series_min = math.ceil(Vmpp_min / (Vmp * vmp_factor_hot))
    N_series_min = max(N_series_min, 6)  # au moins 6 modules / string

    if N_series_min > N_series_max:
        return None

    best = None
    best_score = -1e9

    for N_series in range(N_series_min, N_series_max + 1):
        Voc_cold = N_series * Voc * voc_factor_cold
        Vmp_hot = N_series * Vmp * vmp_factor_hot

        # Vérif tension
        if Voc_cold > Vdc_max:
            continue
        if not (Vmpp_min <= Vmp_hot <= Vmpp_max):
            continue

        # Nombre théorique de strings possibles
        N_strings_theo = N_tot // N_series
        if N_strings_theo < 1:
            continue

        # 1 string max par MPPT
        N_strings_max = min(N_strings_theo, nb_mppt)

        for N_strings in range(1, N_strings_max + 1):
            # On câble 1 string sur les N_strings premiers MPPT, les autres restent vides
            strings_per_mppt = [1 if i < N_strings else 0 for i in range(nb_mppt)]

            # Courant par MPPT : 1 seul string max -> Isc <= I_MPPT
            if Isc > Impp_max:
                continue

            N_used = N_series * N_strings
            P_dc = N_used * Pstc
            ratio_dc_ac = P_dc / P_ac

            # Critère principal : maximiser le nombre de modules utilisés.
            # Secondaire : utiliser un maximum de MPPT, puis favoriser
            # légèrement les strings plus longues.
            score = (
                1000 * N_used             # max de panneaux câblés
                - 10 * abs(nb_mppt - N_strings)  # on préfère utiliser tous les MPPT
                + N_series               # petite préférence pour des strings plus longues
            )

            if score > best_score:
                best_score = score
                best = {
                    "N_series": N_series,
                    "N_strings": N_strings,
                    "N_used": N_used,
                    "P_dc": P_dc,
                    "ratio_dc_ac": ratio_dc_ac,
                }

    return best


# ----------------------------------------------------
# CHOIX AUTOMATIQUE DU MEILLEUR ONDULEUR
# ----------------------------------------------------
def select_best_inverter(
    panel: dict,
    n_panels: int,
    grid_type: str,
    max_dc_ac: float,
    fam_pref: str | None,
    T_min: float,
    T_max: float,
):
    """
    Parcourt tous les onduleurs compatibles (type réseau + famille éventuelle),
    optimise les strings pour chacun, et choisit celui qui :
    - respecte les contraintes électriques (Voc, Vmp, I_MPPT)
    - respecte le ratio DC/AC <= max_dc_ac (slider utilisateur)
    - respecte P_dc <= P_DC_max
    - maximise la puissance DC installée
    """
    best = None
    best_score = -1e9

    for inv in INVERTERS:
        inv_id, p_ac, p_dc_max, vmin, vmax, vdcmax, imppt, nb_mppt, inv_type, inv_family = inv

        if inv_type != grid_type:
            continue
        if fam_pref is not None and inv_family != fam_pref:
            continue

        inv_elec = get_inverter_elec(inv_id)
        if inv_elec is None:
            continue

        opt = optimize_strings(
            N_tot=n_panels,
            panel=panel,
            inverter=inv_elec,
            T_min=T_min,
            T_max=T_max,
        )
        if opt is None:
            continue

        P_dc = opt["P_dc"]
        ratio = P_dc / p_ac

        # Respect P_DC_max et ratio utilisateur
        if P_dc > p_dc_max:
            continue
        if ratio > max_dc_ac:
            continue

        score = P_dc  # on maximise la puissance DC installée

        if score > best_score:
            best_score = score
            best = {
                "inv_id": inv_id,
                "opt": opt,
                "P_dc": P_dc,
                "ratio": ratio,
                "P_ac": p_ac,
            }

    return best


with st.sidebar:
    st.markdown("### 🔧 Paramètres généraux")

    # Sélection panneau
    panel_id = st.selectbox("Panneau", options=PANEL_IDS, index=0)
    n_modules = st.number_input("Nombre de panneaux", min_value=6, max_value=100, value=12)

    panel_elec = get_panel_elec(panel_id)
    if panel_elec is None:
        st.error("Panneau introuvable dans le catalogue.")
        st.stop()

    # Type réseau
    grid_type = st.selectbox("Type de réseau", options=["Mono", "Tri 3x230", "Tri 3x400"], index=0)

    # Mode Store / Hybride
    sigenstore_mode = st.selectbox(
        "Installation compatible SigenStore ?",
        options=["Auto", "Oui (Store)", "Non (Hybride)"],
        index=0,
    )
    if sigenstore_mode == "Oui (Store)":
        fam_pref = "Store"
    elif sigenstore_mode == "Non (Hybride)":
        fam_pref = "Hybride"
    else:
        fam_pref = None

    # Ratio DC/AC (contrainte uniquement dans le choix d'onduleur)
    max_dc_ac = st.slider(
        "Ratio DC/AC max",
        min_value=1.0,
        max_value=2.0,   # étendu pour permettre des surdimensionnements plus forts
        value=1.35,
        step=0.01,
    )

    # ----------------------------------------------------
    # 🔋 Batterie (important pour Excel)
    # ----------------------------------------------------
    battery_enabled = st.checkbox("Batterie", value=False)
    if battery_enabled:
        battery_kwh = st.slider("Capacité batterie (kWh)", 6.0, 50.0, 6.0, 0.5)
    else:
        battery_kwh = 0.0

    st.markdown("---")
    st.markdown("### Profil de consommation")

    annual_consumption = st.number_input("Conso annuelle (kWh)", 500, 20000, 3500, 100)
    consumption_profile = st.selectbox("Profil mensuel", ["Standard", "Hiver fort", "Été fort"], 0)

    hourly_profile_choice = st.selectbox(
        "Profil horaire",
        ["Uniforme", "Classique (matin + soir)", "Travail journée (soir fort)", "Télétravail"],
        index=1
    )

    month_for_hours = st.slider("Mois pour le profil horaire", 1, 12, 6)

    st.markdown("---")
    st.markdown("### Températures de calcul")
    t_min = st.number_input("Température min (°C)", -30, 10, -10)
    t_max = st.number_input("Température max (°C)", 30, 90, 70)

    st.markdown("---")
    st.markdown("### Choix de l’onduleur")

    best = select_best_inverter(
        panel=panel_elec,
        n_panels=int(n_modules),
        grid_type=grid_type,
        max_dc_ac=float(max_dc_ac),
        fam_pref=fam_pref,
        T_min=float(t_min),
        T_max=float(t_max),
    )

    if best is None:
        st.error("Aucun onduleur compatible trouvé avec cette configuration.")
        st.stop()

    auto_inv_id = best["inv_id"]

    compatible_inv = [
        inv[0] for inv in INVERTERS
        if inv[8] == grid_type and (fam_pref is None or inv[9] == fam_pref)
    ]

    inv_options = [f"(Auto) {auto_inv_id}"] + compatible_inv

    selected_inv_label = st.selectbox("Onduleur", inv_options, index=0)

    if selected_inv_label.startswith("(Auto)"):
        inverter_id = auto_inv_id
    else:
        inverter_id = selected_inv_label

# ----------------------------------------------------
# CALCULS PRINCIPAUX
# ----------------------------------------------------
# 1) Récupérer les caractéristiques de l’onduleur choisi
inv_elec = get_inverter_elec(inverter_id)
if inv_elec is None:
    st.error("Spécifications onduleur introuvables.")
    st.stop()

# 2) Optimiser les strings pour CET onduleur (auto ou manuel)
opt_result = optimize_strings(
    N_tot=int(n_modules),
    panel=panel_elec,
    inverter=inv_elec,
    T_min=float(t_min),
    T_max=float(t_max),
)

# Si impossible → arrêter proprement
if opt_result is None:
    st.error(
        f"Aucun câblage valide trouvé pour l'onduleur {inverter_id}. "
        "Essayez un autre modèle ou modifiez les températures."
    )
    st.stop()

# 3) Calculs finaux basés sur CET onduleur + cette optimisation
P_dc = opt_result["P_dc"]
ratio_dc_ac = opt_result["ratio_dc_ac"]
p_dc_kwp = P_dc / 1000.0

# Profils mensuels
pv_kwh_per_kwp = monthly_pv_profile_kwh_kwp()
pv_monthly = pv_kwh_per_kwp * p_dc_kwp

cons_monthly = monthly_consumption_profile(annual_consumption, consumption_profile)
autocons_monthly = np.minimum(pv_monthly, cons_monthly)

months_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                 "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

pv_year = float(pv_monthly.sum())
cons_year = float(annual_consumption)
autocons_year = float(autocons_monthly.sum())

taux_auto = (autocons_year / pv_year * 100) if pv_year > 0 else 0.0
taux_couv = (autocons_year / cons_year * 100) if cons_year > 0 else 0.0

# ----------------------------------------------------
# EN-TÊTE / METRICS
# ----------------------------------------------------
col_logo, col_title = st.columns([1, 3])

with col_logo:
    if os.path.exists("logo_horizon.png"):
        st.image("logo_horizon.png", use_column_width=True)

with col_title:
    st.title("Dimensionneur Solaire Sigen")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Puissance DC installée", f"{P_dc:.0f} Wc")
    st.metric("Panneaux câblés", f"{opt_result['N_used']} / {int(n_modules)}")

with col2:
    st.metric("Prod PV annuelle", f"{pv_year:.0f} kWh")
    st.metric("Conso annuelle", f"{cons_year:.0f} kWh")

with col3:
    st.metric("Taux autocons.", f"{taux_auto:.1f} %")
    st.metric("Taux couverture", f"{taux_couv:.1f} %")

with col4:
    st.metric("Onduleur choisi", inverter_id)
    st.metric("Ratio DC/AC", f"{ratio_dc_ac:.2f}")

if opt_result["N_used"] < int(n_modules):
    st.warning(
        f"{opt_result['N_used']} panneaux seulement peuvent être câblés proprement sur cet onduleur "
        f"(sur {int(n_modules)} demandés)."
    )

# ----------------------------------------------------
# PROFIL MENSUEL
# ----------------------------------------------------
st.markdown("## 📊 Production vs Consommation – Profil mensuel")

df_month = pd.DataFrame({
    "Mois": months_labels,
    "Consommation (kWh)": cons_monthly,
    "Production PV (kWh)": pv_monthly,
    "Autoconsommation (kWh)": autocons_monthly,
})

fig = px.bar(
    df_month,
    x="Mois",
    y=["Consommation (kWh)", "Production PV (kWh)"],
    barmode="group",
    labels={"value": "kWh", "variable": ""},
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df_month)

# ----------------------------------------------------
# PROFIL HORAIRE – JOUR TYPE
# ----------------------------------------------------
st.markdown("## 🕒 Profil horaire – jour type")

days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
idx = month_for_hours - 1

day_cons = cons_monthly[idx] / days_in_month[idx]
day_pv = pv_monthly[idx] / days_in_month[idx]

cons_frac = hourly_profile(hourly_profile_choice)
cons_hour = day_cons * cons_frac

pv_frac = np.array([
    0, 0, 0, 0, 0,
    0.01, 0.04, 0.07, 0.10, 0.13, 0.14, 0.14,
    0.13, 0.10, 0.07, 0.04, 0.02,
    0, 0, 0, 0, 0, 0, 0,
])
if pv_frac.sum() > 0:
    pv_frac = pv_frac / pv_frac.sum()
pv_hour = day_pv * pv_frac

autocons_hour = np.minimum(cons_hour, pv_hour)

df_hour = pd.DataFrame({
    "Heure": np.arange(24),
    "Consommation (kWh)": cons_hour,
    "Production PV (kWh)": pv_hour,
    "Autoconsommation (kWh)": autocons_hour,
})

fig2 = px.line(
    df_hour,
    x="Heure",
    y=["Consommation (kWh)", "Production PV (kWh)", "Autoconsommation (kWh)"],
    markers=True,
    labels={"value": "kWh", "variable": ""},
)
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(df_hour)

# ----------------------------------------------------
# EXPORT EXCEL
# ----------------------------------------------------
st.markdown("## 📥 Export Excel complet")

config = {
    "panel_id": panel_id,
    "n_modules": int(n_modules),
    "grid_type": grid_type,
    "battery_enabled": battery_enabled,
    "battery_kwh": float(battery_kwh),
    "max_dc_ac": float(max_dc_ac),
    "annual_consumption": float(annual_consumption),
    "consumption_profile": consumption_profile,
    "t_min": float(t_min),
    "t_max": float(t_max),
    "n_series": int(opt_result["N_series"]),
    "inverter_id": inverter_id,
}

if st.button("Générer l’Excel"):
    xlsx_bytes = generate_workbook_bytes(config)
    st.download_button(
        "Télécharger le fichier Excel",
        data=xlsx_bytes,
        file_name="Dimensionnement_Sigen_Complet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
