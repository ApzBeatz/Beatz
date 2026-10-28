import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="BEATZ IA - Fijas Champions", page_icon="⚽", layout="centered")

st.title("🤖 BEATZ IA: Las Fijas de la Champions League")
st.write("Tu sistema inteligente automatizado para calcular pronósticos y detectar valor.")

# 1. Base de datos histórica y reciente
data_champions = {
    'Local': [
        'Real Madrid', 'Manchester City', 'Bayern Munich', 'Barcelona', 
        'Paris Saint-Germain', 'Inter Milan', 'Liverpool', 'Arsenal',
        'Atletico Madrid', 'Borussia Dortmund', 'Bayer Leverkusen', 'Juventus',
        'Real Madrid', 'Manchester City', 'Barcelona', 'Bayern Munich'
    ],
    'Visitante': [
        'Borussia Dortmund', 'Inter Milan', 'Barcelona', 'Paris Saint-Germain',
        'Manchester City', 'Real Madrid', 'Bayer Leverkusen', 'Atletico Madrid',
        'Liverpool', 'Arsenal', 'Juventus', 'Bayern Munich',
        'AC Milan', 'Sporting CP', 'Bayer Leverkusen', 'Aston Villa'
    ],
    'GolesLocal': [2, 1, 3, 2, 1, 2, 3, 2, 1, 2, 1, 1, 3, 2, 2, 1],
    'GolesVisitante': [1, 1, 2, 2, 2, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1]
}

df_champions = pd.DataFrame(data_champions)

goles_favor_local_promedio = df_champions['GolesLocal'].mean()
goles_favor_visitante_promedio = df_champions['GolesVisitante'].mean()

def calcular_fuerza_champions(df):
    ataque_local = df.groupby('Local')['GolesLocal'].mean() / goles_favor_local_promedio
    defensa_local = df.groupby('Local')['GolesVisitante'].mean() / goles_favor_visitante_promedio
    return ataque_local.to_dict(), defensa_local.to_dict()

ataque_dict, defensa_dict = calcular_fuerza_champions(df_champions)
equipos = sorted(list(set(df_champions['Local'].tolist() + df_champions['Visitante'].tolist())))

# Interfaz web
st.sidebar.header("⚙️ Seleccionar Partido")
equipo_local = st.sidebar.selectbox("Equipo Local", equipos, index=0)
equipo_visitante = st.sidebar.selectbox("Equipo Visitante", equipos, index=1)

if equipo_local == equipo_visitante:
    st.sidebar.error("⚠️ El local y el visitante no pueden ser el mismo.")
else:
    if st.sidebar.button("🚀 Analizar con BEATZ"):
        atk_loc = ataque_dict.get(equipo_local, 1.0)
        def_vis = defensa_dict.get(equipo_visitante, 1.0)
        atk_vis = ataque_dict.get(equipo_visitante, 1.0)
        def_loc = defensa_dict.get(equipo_local, 1.0)
        
        lambda_local = atk_loc * def_vis * goles_favor_local_promedio
        lambda_visitante = atk_vis * def_loc * goles_favor_visitante_promedio
        
        matriz_prob = np.zeros((7, 7))
        for g_loc in range(7):
            for g_vis in range(7):
                prob = poisson.pmf(g_loc, lambda_local) * poisson.pmf(g_vis, lambda_visitante)
                matriz_prob[g_loc, g_vis] = prob
                
        prob_local = np.sum(np.tril(matriz_prob, -1))
        prob_empate = np.sum(np.diag(matriz_prob))
        prob_visita = np.sum(np.triu(matriz_prob, 1))
        
        max_idx = np.argmax(matriz_prob)
        g_l, g_v = np.unravel_index(max_idx, matriz_prob.shape)
        
        st.subheader(f"🏟️ Pronóstico: {equipo_local} vs {equipo_visitante}")
        
        col1, col2 = st.columns(2)
        col1.metric(label=f"xG ({equipo_local})", value=f"{lambda_local:.2f}")
        col2.metric(label=f"xG ({equipo_visitante})", value=f"{lambda_visitante:.2f}")
        
        st.markdown("---")
        st.markdown("### 📊 Probabilidades (1X2)")
        st.progress(float(prob_local), text=f"Victoria {equipo_local}: {prob_local * 100:.1f}%")
        st.progress(float(prob_empate), text=f"Empate: {prob_empate * 100:.1f}%")
        st.progress(float(prob_visita), text=f"Victoria {equipo_visitante}: {prob_visita * 100:.1f}%")
        
        # Detector de Fijas en la App
        if prob_local >= 0.55:
            st.success(f"🔥 **LA FIJA DE BEATZ:** Gana {equipo_local} (Confianza Alta ⭐⭐⭐)")
        elif prob_visita >= 0.55:
            st.success(f"🔥 **LA FIJA DE BEATZ:** Gana {equipo_visitante} (Confianza Alta ⭐⭐⭐)")
        elif (lambda_local + lambda_visitante) > 3.0:
            st.warning(f"⚡ **LA FIJA DE BEATZ:** Más de 2.5 Goles esperados (Partido Abierto ⚽)")
        else:
            st.info(f"⚖️ Partido muy cerrado. Analizar en vivo.")
            
        st.markdown(f"🎯 **Marcador más probable:** {equipo_local} **{g_l} - {g_v}** {equipo_visitante}")
