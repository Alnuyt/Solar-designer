import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from excel_generator import get_catalog, generate_workbook_bytes

# ----------------------------------------------------
# CONFIG
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


def get_panel_power(panel_id: str):
    for p in PANELS:
        if p[0] == panel_id:
            return p[1]
    return 0.0


def get_panel_elec(panel_id: str):
    for p in PANELS:
        if p[0] == panel_id:
            return {
                "id": p[0],
                "Pstc": float(p[1]),
                "Voc": float(p[2]),
                "Vmp": float(p[3]),
                "Isc": float(p[4]),
                "alpha_V": float(p[6]),
            }
    return None


def get_inverter_elec(inv_id: str):
    for inv in INVERTERS:
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


def get_recommended_inverter(p_dc_total, grid_type, max_dc_ac, famille=None):
    for inv in INVERTERS:
        (inv_id, p_ac, p_dc_max, vmin, vmax,
         vdcmax, imppt, mppts, inv_type, inv_family) = inv

        if inv_type != grid_type:
            continue
        if famille and famille != inv_family:
            continue
        if p_dc_total <= p_dc_max and (p_dc_total / p_ac) <= max_dc_ac:
            return inv_id
    return None


# ----------------------------------------------------
# PROFILS
# ----------------------------------------------------
def monthly_pv_profile_kwh_kwp():
    annual_kwh_kwp = 1034
    distribution = np.array([3.8, 5.1, 8.7, 11.5, 12.1,
                             11.8, 11.9, 10.8, 9.7, 7.0, 4.3, 3.3])
    return annual_kwh_kwp * distribution / 100


def monthly_consumption_profile(annual_kwh, profile):
    table = {
        "Standard":   [7,7,8,9,9,9,9,9,8,8,8,9],
        "Hiver fort": [10,10,10,9,8,7,6,6,7,8,9,10],
        "Été fort":   [6,6,7,8,9,10,11,11,10,8,7,7],
    }
    vals = np.array(table[profile], dtype=float)
    vals = vals / vals.sum()
    return annual_kwh * vals


def hourly_profile(profile_name):
    if profile_name == "Uniforme":
        return np.ones(24) / 24

    if profile_name == "Classique (matin + soir)":
        arr = np.array([
            0.02,0.02,0.02,0.02,0.02,
            0.04,0.06,0.08,0.06,0.03,
            0.02,0.02,0.02,0.02,0.03,
            0.04,0.06,0.08,0.07,0.04,
            0.02,0.01,0.01,0.01
        ])
        return arr / arr.sum()

    if profile_name == "Travail journée (soir fort)":
        arr = np.array([
            0.01,0.01,0.01,0.01,0.01,
            0.02,0.03,0.03,0.03,0.02,
            0.01,0.01,0.01,0.01,0.02,
            0.04,0.07,0.09,0.10,0.10,
            0.05,0.02,0.01,0.01
        ])
        return arr / arr.sum()

    if profile_name == "Télétravail":
        arr = np.array([
            0.02,0.02,0.03,0.03,0.03,
            0.04,0.05,0.06,0.06,0.06,
            0.05,0.05,0.05,0.05,0.05,
            0.05,0.05,0.06,0.06,0.06,
            0.05,0.03,0.02,0.02
        ])
        return arr / arr.sum()

    return np.ones(24) / 24


# ----------------------------------------------------
# OPTIMISATION STRINGS
# ----------------------------------------------------
def optimize_strings(N_tot, panel, inverter, T_min, T_max):
    Voc = panel["Voc"]
    Vmp = panel["Vmp"]
    Isc = panel["Isc"]
    alpha = panel["alpha_V"] / 100
    Pstc = panel["Pstc"]

    Vdc_max = inverter["Vdc_max"]
    Vmpp_min = inverter["Vmpp_min"]
    Vmpp_max = inverter["Vmpp_max"]
    Impp = inverter["Impp_max"]
    nb_mppt = inverter["nb_mppt"]
    P_ac = inverter["P_ac"]

    voc_fac = 1 + alpha * (T_min - 25)
    vmp_fac = 1 + alpha * (T_max - 25)

    Ns_min = max(6, math.ceil(Vmpp_min / (Vmp * vmp_fac)))
    Ns_max = math.floor(Vdc_max / (Voc * voc_fac))

    if Ns_min > Ns_max:
        return None

    best = None
    best_score = -1e9

    for Ns in range(Ns_min, Ns_max + 1):

        Voc_cold = Voc * Ns * voc_fac
        Vmp_hot = Vmp * Ns * vmp_fac

        if Voc_cold > Vdc_max:
            continue
        if not (Vmpp_min <= Vmp_hot <= Vmpp_max):
            continue

        N_strings_theo = N_tot // Ns
        if N_strings_theo < 1:
            continue

        max_strings = min(N_strings_theo, nb_mppt * 2)

        for S in range(1, max_strings + 1):

            base = S // nb_mppt
            rest = S % nb_mppt
            per_mppt = [base + (1 if i < rest else 0) for i in range(nb_mppt)]

            if any(s * Isc > Impp for s in per_mppt):
                continue

            N_used = Ns * S
            P_dc = N_used * Pstc
            ratio = P_dc / P_ac

            if not (1.05 <= ratio <= 1.35):
                continue

            imbalance = max(per_mppt) - min(per_mppt)
            score = -10 * abs(ratio - 1.25) + 0.02 * N_used - 5 * (S - 1) - 2 * imbalance

            if score > best_score:
                best_score = score
                best = {
                    "N_series": Ns,
                    "N_strings": S,
                    "N_used": N_used,
                    "P_dc": P_dc,
                    "ratio": ratio,
                }

    return best


# ----------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Paramètres PV")

    panel_id = st.selectbox("Panneau", PANEL_IDS)
    n_modules = st.number_input("Nombre de panneaux", min_value=6, max_value=100, value=12)

    grid_type = st.selectbox("Type réseau", ["Mono", "Tri 3x230", "Tri 3x400"])

    instore = st.selectbox("Compatible SigenStore ?", ["Auto", "Oui", "Non"])
    fam_pref = None if instore == "Auto" else ("Store" if instore == "Oui" else "Hybride")

    max_dc_ac = st.slider("Ratio DC/AC max", 1.0, 1.5, 1.30)

    battery_enabled = st.checkbox("Batterie ?", False)
    battery_kwh = st.slider("Capacité batterie (kWh)", 2.0, 20.0, 6.0) if battery_enabled else 0

    st.markdown("---")
    st.markdown("### Chauffage & Conso")

    annual_consumption = st.number_input("Consommation annuelle (kWh)", 500, 20000, 3500)
    consumption_profile = st.selectbox("Profil mensuel", ["Standard", "Hiver fort", "Été fort"])
    hourly_choice = st.selectbox("Profil horaire", ["Uniforme", "Classique (matin + soir)",
                                                    "Travail journée (soir fort)", "Télétravail"])

    month_for_hours = st.slider("Mois pour le profil horaire", 1, 12, 6)

    st.markdown("---")
    t_min = st.number_input("T° min (°C)", -30, 10, -10)
    t_max = st.number_input("T° max (°C)", 30, 90, 70)


# ----------------------------------------------------
# CALCULS
# ----------------------------------------------------
panel = get_panel_elec(panel_id)
p_dc_theo = panel["Pstc"] * n_modules

recommended = get_recommended_inverter(p_dc_theo, grid_type, max_dc_ac, fam_pref)

inverters_valid = [inv for inv in INVERTERS if inv[8] == grid_type and (fam_pref is None or inv[9] == fam_pref)]
if not inverters_valid:
    inverters_valid = [inv for inv in INVERTERS if inv[8] == grid_type]

inv_options = []
if recommended:
    inv_options.append("(Auto) " + recommended)
inv_options += [inv[0] for inv in inverters_valid]

chosen = st.sidebar.selectbox("Onduleur", inv_options)
inverter_id = recommended if chosen.startswith("(Auto)") else chosen
inv = get_inverter_elec(inverter_id)

opt = optimize_strings(n_modules, panel, inv, t_min, t_max)

if opt:
    P_dc = opt["P_dc"]
    ratio = opt["ratio"]
else:
    P_dc = p_dc_theo
    ratio = P_dc / inv["P_ac"]

p_kwp = P_dc / 1000

pv_month = monthly_pv_profile_kwh_kwp() * p_kwp
cons_month = monthly_consumption_profile(annual_consumption, consumption_profile)
autocons_month = np.minimum(pv_month, cons_month)

pv_year = pv_month.sum()
autocons_year = autocons_month.sum()


# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.title("Dimensionneur Solaire Sigen – Horizon Énergie")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Puissance DC théorique", f"{p_dc_theo:.0f} Wc")
    st.metric("Puissance DC câblée", f"{P_dc:.0f} Wc")

with col2:
    st.metric("Production annuelle", f"{pv_year:.0f} kWh")
    st.metric("Autoconsommation", f"{100*autocons_year/pv_year:.1f} %")

with col3:
    st.metric("Onduleur", inverter_id)
    st.metric("Ratio DC/AC", f"{ratio:.2f}")


# ----------------------------------------------------
# PROFIL MENSUEL
# ----------------------------------------------------
st.markdown("## 📊 Profil mensuel")

df_month = pd.DataFrame({
    "Mois": ["Jan","Fév","Mar","Avr","Mai","Juin",
             "Juil","Août","Sep","Oct","Nov","Déc"],
    "Consommation (kWh)": cons_month,
    "Production PV (kWh)": pv_month,
    "Autoconsommation (kWh)": autocons_month,
})

fig = px.bar(df_month, x="Mois", y=["Consommation (kWh)", "Production PV (kWh)"], barmode="group")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df_month)

# ----------------------------------------------------
# PROFIL JOURNALIER
# ----------------------------------------------------
st.markdown("## 🕒 Profil horaire du mois sélectionné")

days = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
idx = month_for_hours - 1

day_pv = pv_month[idx] / days[idx]
day_cons = cons_month[idx] / days[idx]

pv_profile_hour = np.array([
    0,0,0,0,0,0.01,0.04,0.07,0.10,0.13,0.14,0.14,
    0.13,0.10,0.07,0.04,0.02,0,0,0,0,0,0,0
])
pv_profile_hour /= pv_profile_hour.sum()
pv_hour = day_pv * pv_profile_hour

cons_hour = hourly_profile(hourly_choice) * day_cons
autocons_hour = np.minimum(cons_hour, pv_hour)

df_hour = pd.DataFrame({
    "Heure": np.arange(24),
    "Consommation (kWh)": cons_hour,
    "Production PV (kWh)": pv_hour,
    "Autoconsommation (kWh)": autocons_hour,
})

fig2 = px.line(df_hour, x="Heure", y=["Consommation (kWh)", "Production PV (kWh)",
                                      "Autoconsommation (kWh)"], markers=True)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(df_hour)

# ----------------------------------------------------
# EXPORT EXCEL
# ----------------------------------------------------
st.markdown("## 📥 Export Excel")

cfg = {
    "panel_id": panel_id,
    "n_modules": n_modules,
    "grid_type": grid_type,
    "battery_enabled": battery_enabled,
    "battery_kwh": battery_kwh,
    "max_dc_ac": max_dc_ac,
    "annual_consumption": annual_consumption,
    "consumption_profile": consumption_profile,
    "t_min": t_min,
    "t_max": t_max,
    "n_series": opt["N_series"] if opt else n_modules,
    "inverter_id": inverter_id,
}

if st.button("Générer Excel"):
    xlsx = generate_workbook_bytes(cfg)
    st.download_button(
        "Télécharger",
        data=xlsx,
        file_name="Dimensionnement_Sigen.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
