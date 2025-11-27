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
# CHARGER CATALOGUE
# ----------------------------------------------------
PANELS, INVERTERS, BATTERIES = get_catalog()

panel_ids = [p[0] for p in PANELS]


# ----------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------
def get_panel_power(panel_id: str) -> float:
    for p in PANELS:
        if p[0] == panel_id:
            return p[1]
    return 0.0


def get_recommended_inverter(p_dc_total, grid_type, max_dc_ac):
    """
    Retourne l’onduleur Sigen le plus adapté :
    - bon type réseau
    - ratio DC/AC <= limite
    """
    for inv in INVERTERS:
        inv_id, p_ac, p_dc_max, vmin, vmax, vdcmax, imppt, mppts, inv_type = inv

        if inv_type != grid_type:
            continue

        ratio = p_dc_total / p_ac
        if ratio <= max_dc_ac:
            return inv_id

    return None


def monthly_pv_profile_kwh_kwp():
    annual_kwh_kwp = 1034.0
    distribution = np.array([3.8, 5.1, 8.7, 11.5, 12.1, 11.8,
                             11.9, 10.8, 9.7, 7.0, 4.3, 3.3])
    return annual_kwh_kwp * distribution / 100.0


def monthly_consumption_profile(annual_kwh: float, profile: str):
    profiles = {
        "Standard": [7, 7, 8, 9, 9, 9, 9, 9, 8, 8, 8, 9],
        "Hiver fort": [10, 10, 10, 9, 8, 7, 6, 6, 7, 8, 9, 10],
        "Été fort": [6, 6, 7, 8, 9, 10, 11, 11, 10, 8, 7, 7],
    }
    arr = np.array(profiles[profile], dtype=float)
    arr = arr / arr.sum()
    return annual_kwh * arr


# ----------------------------------------------------
# INTERFACE – SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Paramètres")

    panel_id = st.selectbox("Panneau", options=panel_ids)
    n_modules = st.number_input("Nombre de panneaux", 1, 100, 12)

    grid_type = st.selectbox("Type de réseau", ["Mono", "Tri 3x400"])

    battery_enabled = st.checkbox("Batterie", False)
    battery_kwh = st.slider("Capacité batterie (kWh)", 2.0, 20.0, 6.0, 0.5) if battery_enabled else 0.0

    max_dc_ac = st.slider("Ratio DC/AC max", 1.0, 1.5, 1.30, 0.01)

    st.markdown("---")
    annual_consumption = st.number_input("Consommation annuelle (kWh)", 500, 20000, 3500, 100)
    consumption_profile = st.selectbox("Profil de conso", ["Standard", "Hiver fort", "Été fort"])

    st.markdown("---")
    t_min = st.number_input("Température min (°C)", -30, 10, -10)
    t_max = st.number_input("Température max (°C)", 30, 90, 70)
    n_series = st.number_input("Modules en série (string)", 1, 30, 10)

    st.markdown("---")
    month_for_hours = st.slider("Mois profil horaire", 1, 12, 6)

# ----------------------------------------------------
# CALCULS
# ----------------------------------------------------
p_stc = get_panel_power(panel_id)
p_dc_total = p_stc * n_modules  # W
p_dc_kwp = p_dc_total / 1000.0

recommended = get_recommended_inverter(p_dc_total, grid_type, max_dc_ac)

# Liste des onduleurs compatibles
available_inverters = ["(Auto) " + recommended] if recommended else []
available_inverters += [inv[0] for inv in INVERTERS if inv[8] == grid_type]

selected_raw = st.sidebar.selectbox("Onduleur recommandé", available_inverters)

if selected_raw.startswith("(Auto)"):
    inverter_id = recommended
else:
    inverter_id = selected_raw

# PV mensuel
pv_kwh_per_kwp = monthly_pv_profile_kwh_kwp()
pv_monthly = pv_kwh_per_kwp * p_dc_kwp

# Conso mensuelle
cons_monthly = monthly_consumption_profile(annual_consumption, consumption_profile)

# Autoconsommation
autocons_monthly = np.minimum(pv_monthly, cons_monthly)

months_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                 "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

# ----------------------------------------------------
# AFFICHAGE – Résumés
# ----------------------------------------------------
st.title("⚡ Dimensionneur Solaire Sigen – Horizon Énergie")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Puissance DC totale", f"{p_dc_total:.0f} Wc")
    st.metric("Puissance panneau", f"{p_stc:.0f} Wc")

with col2:
    st.metric("Conso annuelle", f"{annual_consumption:.0f} kWh")
    st.metric("Prod PV annuelle", f"{pv_monthly.sum():.0f} kWh")

with col3:
    taux_auto = autocons_monthly.sum() / pv_monthly.sum() * 100 if pv_monthly.sum() > 0 else 0
    taux_couv = autocons_monthly.sum() / annual_consumption * 100 if annual_consumption > 0 else 0

    st.metric("Taux autocons.", f"{taux_auto:.1f} %")
    st.metric("Taux couverture", f"{taux_couv:.1f} %")


# ----------------------------------------------------
# GRAPH MENSUEL
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
    color_discrete_sequence=px.colors.qualitative.Set2,
)
st.plotly_chart(fig, use_container_width=True)

df_month_display = df_month.copy().astype({
    "Consommation (kWh)": float,
    "Production PV (kWh)": float,
    "Autoconsommation (kWh)": float,
})

st.dataframe(df_month_display)


# ----------------------------------------------------
# PROFIL HORAIRE
# ----------------------------------------------------
st.markdown("## 🕒 Profil horaire – jour type")

days_in_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
idx = month_for_hours - 1

day_cons = cons_monthly[idx] / days_in_month[idx]
day_pv = pv_monthly[idx] / days_in_month[idx]

pv_frac = np.array([
    0,0,0,0,0,
    0.01,0.04,0.07,0.10,0.13,0.14,0.14,
    0.13,0.10,0.07,0.04,0.02,
    0,0,0,0,0,0,0
])
pv_frac = pv_frac / pv_frac.sum() if pv_frac.sum() > 0 else pv_frac

hours = np.arange(24)
cons_hour = np.full(24, day_cons / 24)
pv_hour = day_pv * pv_frac
autocons_hour = np.minimum(cons_hour, pv_hour)

df_hour = pd.DataFrame({
    "Heure": hours,
    "Consommation (kWh)": cons_hour,
    "Production PV (kWh)": pv_hour,
    "Autoconsommation (kWh)": autocons_hour,
})

fig2 = px.line(
    df_hour,
    x="Heure",
    y=["Consommation (kWh)", "Production PV (kWh)", "Autoconsommation (kWh)"],
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Set1,
)
st.plotly_chart(fig2, use_container_width=True)

st.table(df_hour.style.format("{:.3f}"))

st.dataframe(df_month_display)

# ----------------------------------------------------
# EXPORT EXCEL
# ----------------------------------------------------
st.markdown("## 📥 Export Excel complet")

config = {
    "panel_id": panel_id,
    "n_modules": int(n_modules),
    "grid_type": grid_type,
    "battery_enabled": battery_enabled,
    "battery_kwh": battery_kwh,
    "max_dc_ac": max_dc_ac,
    "annual_consumption": annual_consumption,
    "consumption_profile": consumption_profile,
    "t_min": t_min,
    "t_max": t_max,
    "n_series": int(n_series),
    "inverter_id": inverter_id,
}

if st.button("Générer l’Excel"):
    file_bytes = generate_workbook_bytes(config)
    st.download_button(
        "Télécharger Excel",
        data=file_bytes,
        file_name="Dimensionnement_Sigen_Complet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
