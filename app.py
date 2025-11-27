import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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


def get_panel_power(panel_id: str) -> float:
    for p in PANELS:
        if p[0] == panel_id:
            return p[1]
    return 0.0


def get_panel_params(panel_id: str):
    """Retourne Voc, Vmp, alpha_V (%/°C) pour un panneau."""
    for p in PANELS:
        if p[0] == panel_id:
            # [id, Pstc, Voc, Vmp, Isc, Imp, alpha_V]
            return float(p[2]), float(p[3]), float(p[6])
    return None, None, None


def get_inverter_params(inv_id: str):
    """Retourne (Vdc_max, VMPP_min, VMPP_max, nb_mppt) pour un onduleur."""
    for inv in INVERTERS:
        # (id, P_AC, P_DC_max, V_MPP_min, V_MPP_max, V_DC_max, I_MPPT, Nb_MPPT, Type)
        if inv[0] == inv_id:
            return float(inv[5]), float(inv[3]), float(inv[4]), int(inv[7])
    return None, None, None, 2


def get_recommended_inverter(p_dc_total: float, grid_type: str, max_dc_ac: float):
    """
    Retourne le premier onduleur compatible avec :
    - Type de réseau
    - Ratio DC/AC <= max_dc_ac
    Sinon None.
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
    """Profil mensuel PV Belgique (kWh/an/kWc réparti par mois)."""
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


def get_hourly_profile(profile_name: str):
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


def make_string_diagram(panel_id, n_series, inverter_id, grid_type,
                        n_strings=1, nb_mppt=2):
    """
    Schéma interactif : 1 ou 2 strings -> MPPT -> onduleur.
    - Chaque panneau est un rectangle.
    - On visualise clairement chaque string et son MPPT.
    """
    fig = go.Figure()

    panel_w = 0.35
    panel_h = 0.5
    spacing_x = 0.45
    spacing_y = 1.2

    mppt_x = n_series * spacing_x + 1.2
    inverter_x = mppt_x + 2.2

    # Dessin des strings
    for s in range(n_strings):
        y_base = s * spacing_y
        for i in range(n_series):
            x = i * spacing_x
            fig.add_shape(
                type="rect",
                x0=x, y0=y_base, x1=x + panel_w, y1=y_base + panel_h,
                line=dict(color="black")
            )
            if i == 0:
                fig.add_annotation(
                    x=x + panel_w/2,
                    y=y_base + panel_h + 0.2,
                    text=f"String {s+1}",
                    showarrow=False,
                    font=dict(size=10, color="green"),
                )

        fig.add_annotation(
            x=(n_series * spacing_x)/2,
            y=y_base + panel_h/2,
            text=f"{n_series} x {panel_id}",
            showarrow=False,
            font=dict(size=8),
        )

    # MPPTs
    for m in range(nb_mppt):
        y_mppt = (m * spacing_y) + 0.1
        fig.add_shape(
            type="rect",
            x0=mppt_x, y0=y_mppt,
            x1=mppt_x + 1.2, y1=y_mppt + 0.7,
            line=dict(color="blue", width=2),
        )
        fig.add_annotation(
            x=mppt_x + 0.6,
            y=y_mppt + 0.35,
            text=f"MPPT {m+1}",
            showarrow=False,
            font=dict(size=10, color="blue"),
        )

    # Câblage strings -> MPPT
    for s in range(n_strings):
        mppt_index = s if s < nb_mppt else 0  # si plus de strings que de MPPT
        y_string = s * spacing_y + panel_h/2
        y_mppt = mppt_index * spacing_y + 0.35
        fig.add_annotation(
            x=n_series * spacing_x,
            y=y_string,
            ax=mppt_x,
            ay=y_mppt,
            arrowhead=3,
            arrowwidth=1.2,
        )

    # Onduleur
    height_box = max(n_strings, nb_mppt) * spacing_y + 0.5
    fig.add_shape(
        type="rect",
        x0=inverter_x, y0=-0.2,
        x1=inverter_x + 2.0, y1=height_box,
        line=dict(color="red", width=2),
    )
    label_inv = inverter_id if inverter_id else "Onduleur"
    fig.add_annotation(
        x=inverter_x + 1.0,
        y=height_box/2,
        text=f"{label_inv}\n{grid_type}",
        showarrow=False,
        font=dict(size=11, color="red"),
    )

    # Câblage MPPT -> Onduleur
    for m in range(nb_mppt):
        y_mppt = m * spacing_y + 0.35
        fig.add_annotation(
            x=mppt_x + 1.2,
            y=y_mppt,
            ax=inverter_x,
            ay=y_mppt,
            arrowhead=3,
            arrowwidth=1.2,
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=300 + (n_strings - 1) * 80,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )

    return fig


# ----------------------------------------------------
# SIDEBAR – INPUTS
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Paramètres généraux")

    panel_id = st.selectbox("Panneau", options=PANEL_IDS, index=0)
    n_modules = st.number_input("Nombre de panneaux", min_value=1, max_value=100, value=12)

    grid_type = st.selectbox("Type de réseau", options=["Mono", "Tri 3x400"], index=0)

    max_dc_ac = st.slider("Ratio DC/AC max", min_value=1.0, max_value=1.5, value=1.30, step=0.01)

    battery_enabled = st.checkbox("Batterie", value=False)
    if battery_enabled:
        battery_kwh = st.slider("Capacité batterie (kWh)", 2.0, 20.0, 6.0, 0.5)
    else:
        battery_kwh = 0.0

    st.markdown("---")
    st.markdown("### Profil de consommation")

    annual_consumption = st.number_input(
        "Consommation annuelle (kWh)",
        min_value=500,
        max_value=20000,
        value=3500,
        step=100,
    )
    consumption_profile = st.selectbox(
        "Profil mensuel",
        options=["Standard", "Hiver fort", "Été fort"],
        index=0,
    )
    horaire_profile = st.selectbox(
        "Profil horaire",
        options=["Uniforme", "Classique (matin + soir)", "Travail journée (soir fort)", "Télétravail"],
        index=1,
    )

    st.markdown("---")
    st.markdown("### Vérification électrique (strings)")

    t_min = st.number_input("Température min (°C)", min_value=-30, max_value=10, value=-10)
    t_max = st.number_input("Température max (°C)", min_value=30, max_value=90, value=70)

    n_series = st.number_input("Modules en série par string", min_value=1, max_value=30, value=10)
    n_strings = st.radio("Nombre de strings", options=[1, 2], index=0)

    st.markdown("---")
    month_for_hours = st.slider("Mois pour le profil horaire", min_value=1, max_value=12, value=6)


# ----------------------------------------------------
# CALCULS PRINCIPAUX
# ----------------------------------------------------
p_stc = get_panel_power(panel_id)
p_dc_total = p_stc * n_modules
p_dc_kwp = p_dc_total / 1000.0

recommended = get_recommended_inverter(p_dc_total, grid_type, max_dc_ac)

inv_options = []
if recommended is not None:
    inv_options.append(f"(Auto) {recommended}")
inv_options += [inv[0] for inv in INVERTERS if inv[8] == grid_type]
if not inv_options:
    inv_options = [inv[0] for inv in INVERTERS]

selected_inv_label = st.sidebar.selectbox("Onduleur", options=inv_options, index=0)
if selected_inv_label.startswith("(Auto) "):
    inverter_id = recommended
else:
    inverter_id = selected_inv_label

# Production PV mensuelle
pv_kwh_per_kwp = monthly_pv_profile_kwh_kwp()
pv_monthly = pv_kwh_per_kwp * p_dc_kwp

# Consommation mensuelle
cons_monthly = monthly_consumption_profile(annual_consumption, consumption_profile)

# Autoconsommation mensuelle
autocons_monthly = np.minimum(pv_monthly, cons_monthly)

months_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                 "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

pv_year = float(pv_monthly.sum())
cons_year = float(annual_consumption)
autocons_year = float(autocons_monthly.sum())

taux_auto = (autocons_year / pv_year * 100) if pv_year > 0 else 0.0
taux_couv = (autocons_year / cons_year * 100) if cons_year > 0 else 0.0

# ----------------------------------------------------
# HEADER AVEC LOGO
# ----------------------------------------------------
col_logo, col_title = st.columns([1, 3])

with col_logo:
    if os.path.exists("logo_horizon.png"):
        st.image("logo_horizon.png", use_column_width=True)

with col_title:
    st.title("Dimensionneur Solaire Sigen – Horizon Énergie")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Puissance DC totale", f"{p_dc_total:.0f} Wc")
    st.metric("Puissance panneau", f"{p_stc:.0f} Wc")

with col2:
    st.metric("Conso annuelle", f"{cons_year:.0f} kWh")
    st.metric("Prod PV annuelle", f"{pv_year:.0f} kWh")

with col3:
    st.metric("Taux autocons.", f"{taux_auto:.1f} %")
    st.metric("Taux couverture", f"{taux_couv:.1f} %")

with col4:
    st.metric("Onduleur", inverter_id if inverter_id else "Aucun")

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

cons_frac = get_hourly_profile(horaire_profile)
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

hours = np.arange(24)
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
    labels={"value": "kWh", "variable": ""},
)
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(df_hour)

# ----------------------------------------------------
# VÉRIFICATION STRING (Voc / Vmp)
# ----------------------------------------------------
st.markdown("## ⚡ Vérification du string (Voc froid / Vmp chaud)")

Voc_mod, Vmp_mod, alpha_V = get_panel_params(panel_id)
Vdc_max, VMPP_min, VMPP_max, nb_mppt = get_inverter_params(inverter_id) if inverter_id else (None, None, None, 2)

voc_string = None
vmp_string = None
voc_ok = None
vmp_ok = None

if Voc_mod is not None and Vdc_max is not None:
    alpha = alpha_V / 100.0  # %/°C → 1/°C
    voc_string = n_series * Voc_mod * (1 + alpha * (t_min - 25))
    vmp_string = n_series * Vmp_mod * (1 + alpha * (t_max - 25))

    voc_ok = voc_string <= Vdc_max
    vmp_ok = (vmp_string >= VMPP_min) and (vmp_string <= VMPP_max)

col_a, col_b, col_c = st.columns(3)
with col_a:
    if voc_string is not None:
        st.metric("Voc string (froid)", f"{voc_string:.1f} V")
        st.write(f"Limite onduleur Vdc_max = {Vdc_max:.0f} V")
with col_b:
    if vmp_string is not None:
        st.metric("Vmp string (chaud)", f"{vmp_string:.1f} V")
        st.write(f"Plage MPPT = {VMPP_min:.0f} – {VMPP_max:.0f} V")
with col_c:
    if voc_ok is not None and vmp_ok is not None:
        if voc_ok and vmp_ok:
            st.success("String conforme : tension OK à froid et à chaud.")
        else:
            if not voc_ok:
                st.error("⚠ Voc string à froid dépasse Vdc_max de l’onduleur.")
            if not vmp_ok:
                st.error("⚠ Vmp string à chaud en dehors de la plage MPPT.")

# ----------------------------------------------------
# SCHÉMA DU STRING
# ----------------------------------------------------
st.markdown("## 📐 Schéma du câblage strings → MPPT → onduleur")

fig_string = make_string_diagram(
    panel_id=panel_id,
    n_series=n_series,
    inverter_id=inverter_id,
    grid_type=grid_type,
    n_strings=n_strings,
    nb_mppt=nb_mppt if nb_mppt else 2,
)
st.plotly_chart(fig_string, use_container_width=True)

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
    "n_series": int(n_series),
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
