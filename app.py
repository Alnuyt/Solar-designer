import os
import math
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


def get_panel_elec(panel_id: str):
    """
    Retourne un dict avec les paramètres électriques du panneau :
    Voc, Vmp, Isc, alpha_V (%/°C), P_STC.
    """
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
    """
    Retourne un dict avec les paramètres électriques de l'onduleur :
    Vdc_max, Vmpp_min, Vmpp_max, Impp_max, nb_mppt, P_ac_nom.
    """
    for inv in INVERTERS:
        # (id, P_AC_nom, P_DC_max, V_MPP_min, V_MPP_max, V_DC_max, I_MPPT, Nb_MPPT, Type_réseau, Famille)
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
                "famille": inv[9]
            }
    return None


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


def optimize_strings(
    N_tot,
    panel,
    inverter,
    T_min,
    T_max,
    ratio_dc_ac_target=1.25,
    ratio_dc_ac_min=1.05,
    ratio_dc_ac_max=1.35,
):
    """
    Optimisation automatique des strings :
    - calcule N_series, N_strings, répartition par MPPT
    - Standard Sigen : max 2 strings / MPPT
    - vérifie Voc_froid, Vmp_chaud, courant MPPT, ratio DC/AC
    - N_series >= 6 pour éviter les strings trop courts
    """
    Voc = panel["Voc"]
    Vmp = panel["Vmp"]
    Isc = panel["Isc"]
    alpha_V = panel["alpha_V"] / 100.0   # %/°C -> 1/°C
    Pstc = panel["Pstc"]

    Vdc_max = inverter["Vdc_max"]
    Vmpp_min = inverter["Vmpp_min"]
    Vmpp_max = inverter["Vmpp_max"]
    Impp_max = inverter["Impp_max"]
    nb_mppt = inverter["nb_mppt"]
    P_ac = inverter["P_ac"]

    # --- 1) bornes sur N_series ---
    voc_factor_cold = (1 + alpha_V * (T_min - 25.0))
    vmp_factor_hot = (1 + alpha_V * (T_max - 25.0))

    if voc_factor_cold <= 0 or vmp_factor_hot <= 0:
        return None  # incohérent

    N_series_max = math.floor(Vdc_max / (Voc * voc_factor_cold))
    N_series_min = math.ceil(Vmpp_min / (Vmp * vmp_factor_hot))

    # On impose au moins 6 modules par string
    N_series_min = max(N_series_min, 6)

    if N_series_max < 1 or N_series_min > N_series_max:
        return None  # impossible

    best = None
    best_score = -1e9

    for N_series in range(N_series_min, N_series_max + 1):
        # tensions string
        Voc_cold = N_series * Voc * voc_factor_cold
        Vmp_hot = N_series * Vmp * vmp_factor_hot

        if Voc_cold > Vdc_max:
            continue
        if not (Vmpp_min <= Vmp_hot <= Vmpp_max):
            continue

        # max strings possibles d'après le nombre total de modules
        N_strings_theo = N_tot // N_series
        if N_strings_theo < 1:
            continue

        # Standard Sigen : max 2 strings / MPPT
        N_strings_max_mppt = nb_mppt * 2
        N_strings_max = min(N_strings_theo, N_strings_max_mppt)

        for N_strings in range(1, N_strings_max + 1):
            # répartition sur MPPT (équilibre)
            base = N_strings // nb_mppt
            rest = N_strings % nb_mppt
            strings_per_mppt = [base + (1 if i < rest else 0) for i in range(nb_mppt)]

            # check courant par MPPT
            ok_current = True
            for s in strings_per_mppt:
                I_total = s * Isc
                if I_total > Impp_max:
                    ok_current = False
                    break
            if not ok_current:
                continue

            # puissance DC réellement câblée
            N_used = N_series * N_strings
            P_dc = N_used * Pstc
            ratio_dc_ac = P_dc / P_ac

            if ratio_dc_ac < ratio_dc_ac_min or ratio_dc_ac > ratio_dc_ac_max:
                continue

            # score :
            # - ratio proche de la cible
            # - plus de panneaux utilisés
            # - moins de strings (on préfère 1 string de 8 plutôt que 2x4)
            # - strings équilibrés sur les MPPT
            penalty_ratio = abs(ratio_dc_ac - ratio_dc_ac_target)
            imbalance = max(strings_per_mppt) - min(strings_per_mppt)

            score = (
                -10 * penalty_ratio        # ratio proche cible
                + N_used * 0.02            # plus de panneaux câblés
                - 5 * (N_strings - 1)      # pénalise le fait d'avoir plusieurs strings
                - 2 * imbalance            # pénalise déséquilibre MPPT
                + 0.5 * N_series           # encourage des strings plus longs
            )

            if score > best_score:
                best_score = score
                best = {
                    "N_series": N_series,
                    "N_strings": N_strings,
                    "strings_per_mppt": strings_per_mppt,
                    "N_used": N_used,
                    "P_dc": P_dc,
                    "ratio_dc_ac": ratio_dc_ac,
                    "Voc_cold": Voc_cold,
                    "Vmp_hot": Vmp_hot,
                }

    return best


def make_string_diagram(panel_id, opt_result, inverter_id, grid_type, nb_mppt):
    """
    Schéma simple et lisible :
    - 1 bloc par string avec "String 1 : N x panneau"
    - 1 ou plusieurs MPPT
    - 1 bloc onduleur
    """
    if opt_result is None:
        return go.Figure()

    N_series = opt_result["N_series"]
    N_strings = opt_result["N_strings"]
    strings_per_mppt = opt_result["strings_per_mppt"]

    fig = go.Figure()

    x_string = 0.0
    x_mppt = 2.5
    x_inverter = 5.0

    y_step = 1.5
    n_rows = max(N_strings, nb_mppt)
    total_height = (n_rows - 1) * y_step
    y_center = total_height / 2

    # --- Strings ---
    for s in range(N_strings):
        y = y_center - s * y_step
        fig.add_shape(
            type="rect",
            x0=x_string, y0=y - 0.4,
            x1=x_string + 1.8, y1=y + 0.4,
            line=dict(color="green", width=2),
        )
        fig.add_annotation(
            x=x_string + 0.9,
            y=y,
            text=f"String {s+1}\n{N_series} x {panel_id}",
            showarrow=False,
            font=dict(size=10, color="green"),
        )

    # --- MPPT ---
    for m in range(nb_mppt):
        y = y_center - m * y_step
        fig.add_shape(
            type="rect",
            x0=x_mppt, y0=y - 0.4,
            x1=x_mppt + 1.5, y1=y + 0.4,
            line=dict(color="blue", width=2),
        )
        fig.add_annotation(
            x=x_mppt + 0.75,
            y=y,
            text=f"MPPT {m+1}",
            showarrow=False,
            font=dict(size=10, color="blue"),
        )

    # --- Onduleur ---
    label_inv = inverter_id if inverter_id else "Onduleur"
    fig.add_shape(
        type="rect",
        x0=x_inverter, y0=y_center - 1.0,
        x1=x_inverter + 2.0, y1=y_center + 1.0,
        line=dict(color="red", width=2),
    )
    fig.add_annotation(
        x=x_inverter + 1.0,
        y=y_center,
        text=f"{label_inv}\n{grid_type}",
        showarrow=False,
        font=dict(size=11, color="red"),
    )

    # --- Câblage Strings -> MPPT ---
    # affectation des strings aux MPPT dans l'ordre
    mppt_assign = []
    for m, count in enumerate(strings_per_mppt):
        mppt_assign.extend([m] * count)

    for s in range(N_strings):
        mppt_index = mppt_assign[s] if s < len(mppt_assign) else 0
        y_string = y_center - s * y_step
        y_mppt = y_center - mppt_index * y_step
        fig.add_annotation(
            x=x_string + 1.8,
            y=y_string,
            ax=x_mppt,
            ay=y_mppt,
            arrowhead=3,
            arrowwidth=1.2,
        )

    # --- Câblage MPPT -> Onduleur ---
    for m in range(nb_mppt):
        y_mppt = y_center - m * y_step
        fig.add_annotation(
            x=x_mppt + 1.5,
            y=y_mppt,
            ax=x_inverter,
            ay=y_center,
            arrowhead=3,
            arrowwidth=1.2,
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=280 + (n_rows - 1) * 40,
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
    # Minimum 6 panneaux
    n_modules = st.number_input("Nombre de panneaux", min_value=6, max_value=100, value=12)

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
    st.markdown("### Températures de calcul (tensions)")

    t_min = st.number_input("Température min (°C)", min_value=-30, max_value=10, value=-10)
    t_max = st.number_input("Température max (°C)", min_value=30, max_value=90, value=70)

    st.markdown("---")
    month_for_hours = st.slider("Mois pour le profil horaire", min_value=1, max_value=12, value=6)


# ----------------------------------------------------
# CALCULS PRINCIPAUX
# ----------------------------------------------------
p_stc = get_panel_power(panel_id)
p_dc_total_theoretical = p_stc * n_modules

# Sélection / recommandation onduleur
recommended = get_recommended_inverter(p_dc_total_theoretical, grid_type, max_dc_ac)

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

panel_elec = get_panel_elec(panel_id)
inv_elec = get_inverter_elec(inverter_id) if inverter_id else None

opt_result = None
nb_mppt = inv_elec["nb_mppt"] if inv_elec else 2

if (panel_elec is not None) and (inv_elec is not None):
    opt_result = optimize_strings(
        N_tot=int(n_modules),
        panel=panel_elec,
        inverter=inv_elec,
        T_min=float(t_min),
        T_max=float(t_max),
    )

if opt_result is not None:
    N_series = opt_result["N_series"]
    N_strings = opt_result["N_strings"]
    N_used = opt_result["N_used"]
    p_dc_total = opt_result["P_dc"]
    ratio_dc_ac = opt_result["ratio_dc_ac"]
else:
    # fallback théorique si optimisation impossible
    N_series = int(n_modules)
    N_strings = 1
    N_used = int(n_modules)
    p_dc_total = p_dc_total_theoretical
    ratio_dc_ac = p_dc_total / inv_elec["P_ac"] if inv_elec else 0.0

p_dc_kwp = p_dc_total / 1000.0

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
    st.metric("Puissance DC théorique", f"{p_dc_total_theoretical:.0f} Wc")
    st.metric("Puissance DC câblée", f"{p_dc_total:.0f} Wc")

with col2:
    st.metric("Conso annuelle", f"{cons_year:.0f} kWh")
    st.metric("Prod PV annuelle", f"{pv_year:.0f} kWh")

with col3:
    st.metric("Taux autocons.", f"{taux_auto:.1f} %")
    st.metric("Taux couverture", f"{taux_couv:.1f} %")

with col4:
    st.metric("Onduleur", inverter_id if inverter_id else "Aucun")
    st.metric("Ratio DC/AC", f"{ratio_dc_ac:.2f}" if ratio_dc_ac > 0 else "-")

if opt_result is not None and N_used < n_modules:
    st.warning(
        f"Configuration optimale utilise {N_used} panneaux sur {n_modules} disponibles. "
        "Certains panneaux ne peuvent pas être câblés proprement dans les limites électriques."
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
    color_discrete_sequence=["#E74C3C", "#F1C40F"]   # rouge / jaune
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
    color_discrete_sequence=["#E74C3C", "#F1C40F", "#2ECC71"]  # rouge / jaune / vert
)
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(df_hour)

# ----------------------------------------------------
# (Vérification électrique affichée sans gros titre "Vérification du string")
# ----------------------------------------------------
voc_string = None
vmp_string = None
voc_ok = None
vmp_ok = None

if opt_result is not None and inv_elec is not None:
    voc_string = opt_result["Voc_cold"]
    vmp_string = opt_result["Vmp_hot"]
    Vdc_max = inv_elec["Vdc_max"]
    Vmpp_min = inv_elec["Vmpp_min"]
    Vmpp_max = inv_elec["Vmpp_max"]

    voc_ok = voc_string <= Vdc_max
    vmp_ok = (vmp_string >= Vmpp_min) and (vmp_string <= Vmpp_max)

# ----------------------------------------------------
# SCHÉMA DU STRING
# ----------------------------------------------------
st.markdown("## 📐 Schéma du câblage strings → MPPT → onduleur")

fig_string = make_string_diagram(
    panel_id=panel_id,
    opt_result=opt_result,
    inverter_id=inverter_id,
    grid_type=grid_type,
    nb_mppt=nb_mppt,
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
    # N_series auto pour cohérence avec l’onglet Strings d’Excel
    "n_series": int(N_series),
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
