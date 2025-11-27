# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from excel_generator import get_catalog, generate_workbook_bytes


st.set_page_config(
    page_title="Dimensionneur Solaire Sigen",
    layout="wide",
)


# ===================== DATA DE BASE =====================

PANELS, INVERTERS, BATTERIES = get_catalog()

panel_ids = [p[0] for p in PANELS]
inv_ids = [i[0] for i in INVERTERS]


def get_panel_power(panel_id: str) -> float:
    for p in PANELS:
        if p[0] == panel_id:
            return p[1]
    return 0.0


def monthly_pv_profile_kwh_kwp():
    annual_kwh_kwp = 1034.0
    distribution = [3.8, 5.1, 8.7, 11.5, 12.1, 11.8, 11.9, 10.8, 9.7, 7.0, 4.3, 3.3]
    vals = [annual_kwh_kwp * p / 100.0 for p in distribution]
    return np.array(vals)


def monthly_consumption_profile(annual_kwh: float, profile: str):
    profiles = {
        "Standard": [7, 7, 8, 9, 9, 9, 9, 9, 8, 8, 8, 9],
        "Hiver fort": [10, 10, 10, 9, 8, 7, 6, 6, 7, 8, 9, 10],
        "Été fort": [6, 6, 7, 8, 9, 10, 11, 11, 10, 8, 7, 7],
    }
    perc = np.array(profiles[profile], dtype=float)
    perc = perc / perc.sum()  # normalisation légère
    return annual_kwh * perc


# ===================== INTERFACE =====================

st.title("⚡ Dimensionneur Solaire Sigen – Horizon Énergie")

with st.sidebar:
    st.markdown("### Paramètres généraux")

    panel_id = st.selectbox("Panneau", options=panel_ids, index=0)
    n_modules = st.number_input("Nombre de panneaux", min_value=1, max_value=100, value=12)
    grid_type = st.selectbox("Type de réseau", options=["Mono", "Tri 3x400"], index=0)

    battery_enabled = st.checkbox("Batterie", value=False)
    battery_kwh = st.slider("Capacité batterie (kWh)", 2.0, 20.0, 6.0, 0.5) if battery_enabled else 0.0

    max_dc_ac = st.slider("Ratio DC/AC max", 1.0, 1.5, 1.30, 0.01)

    st.markdown("---")
    annual_consumption = st.number_input("Consommation annuelle client (kWh)", min_value=500, max_value=20000, value=3500, step=100)
    consumption_profile = st.selectbox("Profil de consommation", ["Standard", "Hiver fort", "Été fort"], index=0)

    st.markdown("---")
    st.markdown("### Vérification strings")
    t_min = st.number_input("Température min (°C)", value=-10)
    t_max = st.number_input("Température max (°C)", value=70)
    n_series = st.number_input("Modules en série (string)", min_value=1, max_value=30, value=10)

    st.markdown("---")
    st.markdown("### Profil horaire")
    month_for_hours = st.slider("Mois pour le profil horaire", 1, 12, 6)


# ===================== CALCULS PRINCIPAUX =====================

p_stc = get_panel_power(panel_id)
p_dc_total = p_stc * n_modules  # W
p_dc_kwp = p_dc_total / 1000.0

# On prend un onduleur "cible" uniquement pour le ratio: ex. 5 kW ou choix manuel plus tard
# Pour l'UI, on peut simplement afficher un ratio typique pour 5 kW,
# en pratique la sélection d'onduleur détaillée reste dans l'Excel.
p_ac_ref = 5000.0
dc_ac_ratio = p_dc_total / p_ac_ref

months_labels = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Août","Sep","Oct","Nov","Déc"]
pv_kwh_per_kwp = monthly_pv_profile_kwh_kwp()
pv_monthly = pv_kwh_per_kwp * p_dc_kwp
cons_monthly = monthly_consumption_profile(annual_consumption, consumption_profile)
autocons_monthly = np.minimum(pv_monthly, cons_monthly)


# ===================== LAYOUT =====================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Puissance DC totale", f"{p_dc_total:.0f} Wc")
    st.metric("Puissance panneau", f"{p_stc:.0f} Wc")

with col2:
    st.metric("Conso annuelle", f"{annual_consumption:.0f} kWh")
    st.metric("Prod PV annuelle", f"{pv_monthly.sum():.0f} kWh")

with col3:
    st.metric("Taux d'autoconsommation", f"{(autocons_monthly.sum()/pv_monthly.sum()*100):.1f} %" if pv_monthly.sum() > 0 else "–")
    st.metric("Taux de couverture", f"{(autocons_monthly.sum()/annual_consumption*100):.1f} %" if annual_consumption > 0 else "–")

st.markdown("## Production vs Consommation – Profil mensuel")

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

st.markdown("### Autoconsommation mensuelle")
st.dataframe(df_month.style.format("{:.1f}"))


# ===================== PROFIL HORAIRE SIMPLIFIÉ =====================

st.markdown("## Profil horaire (jour type)")

days_in_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
idx_month = month_for_hours - 1
day_cons = cons_monthly[idx_month] / days_in_month[idx_month] if days_in_month[idx_month] > 0 else 0
day_pv = pv_monthly[idx_month] / days_in_month[idx_month] if days_in_month[idx_month] > 0 else 0

# Profil PV journalier (fraction par heure)
pv_frac = np.array([
    0, 0, 0, 0, 0,
    0.01, 0.04, 0.07, 0.10, 0.13, 0.14, 0.14,
    0.13, 0.10, 0.07, 0.04, 0.02,
    0, 0, 0, 0, 0, 0, 0,
])
pv_frac = pv_frac / pv_frac.sum() if pv_frac.sum() > 0 else pv_frac

hours = np.arange(24)
cons_hour = np.full(24, day_cons / 24.0)
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
    labels={"value": "kWh", "variable": ""},
)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(df_hour.style.format("{:.3f}"))


# ===================== EXPORT EXCEL =====================

st.markdown("## Export Excel complet")

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
    "n_series": int(n_series),
}

if st.button("Générer le fichier Excel complet"):
    xlsx_bytes = generate_workbook_bytes(config)
    st.download_button(
        "Télécharger le fichier Excel",
        data=xlsx_bytes,
        file_name="Dimensionnement_Sigen_Complet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
