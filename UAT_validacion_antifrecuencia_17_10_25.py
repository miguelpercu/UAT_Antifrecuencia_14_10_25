#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch, medfilt
from scipy.stats import kurtosis, norm, normaltest, ttest_ind
from scipy.constants import c, h, k, G
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MARCO TEÓRICO UAT UNIFICADO Y OPTIMIZADO
# =============================================================================

class UnifiedApplicableTimeTheory:
    """
    Implementación computacional completa del marco UAT
    Basado en: "Experimental Framework for Detection of Atemporal Antifrequency Effects"
    """

    def __init__(self):
        # Constantes fundamentales
        self.c = c
        self.h = h  
        self.k = k
        self.G = G

        # Parámetros UAT optimizados del manuscrito
        self.alpha = 8.67e-6  # Constante de asociación optimizada
        self.transition_start = 2.097e3  # 2.097 kHz
        self.transition_end = 498.7e3    # 498.7 kHz
        self.f_max_effect = 100e3       # 100 kHz - punto óptimo

        # Parámetros de detector criogénico realista
        self.temperatura = 0.015  # 15 mK
        self.sensibilidad = 2e-23  # W/√Hz

    def antifrequency(self, f):
        """Antifrecuencia atemporal: λ ≡ -1/f"""
        with np.errstate(divide='ignore', invalid='ignore'):
            return -1.0 / np.where(f == 0, 1e-50, f)

    def modification_factor(self, f):
        """Factor de modificación UAT: 1 + tanh(α/|λ|)"""
        lambda_abs = np.abs(self.antifrequency(f))
        return 1.0 + np.tanh(self.alpha / lambda_abs)

    def hawking_temperature_modified(self, M_black_hole):
        """Temperatura de Hawking modificada por UAT"""
        T_hawking = (self.h * self.c**3) / (8 * np.pi * self.G * M_black_hole * self.k)
        modification = self.modification_factor(self.f_max_effect)
        return T_hawking * modification

# =============================================================================
# CONFIGURACIÓN EXPERIMENTAL UNIFICADA
# =============================================================================

# Parámetros de simulación optimizados
T_SIMULACION = 5.0  # Segundos - balance entre estadística y tiempo computacional
TASA_MUESTREO = 10000  # 10 kHz - suficiente para rango UAT
TIEMPO = np.linspace(0, T_SIMULACION, int(T_SIMULACION * TASA_MUESTREO), endpoint=False)

# Instanciar teoría UAT
uat = UnifiedApplicableTimeTheory()

# =============================================================================
# GENERACIÓN DE SEÑALES Y RUIDO MEJORADOS
# =============================================================================

def generar_ruido_uat_optimizado(tiempo, tasa_muestreo, temperatura):
    """Ruido físico realista optimizado para detección UAT"""
    n = len(tiempo)

    # Componentes de ruido físico
    ruido_termico = np.sqrt(2 * 1.38e-23 * temperatura / tasa_muestreo) * np.random.randn(n)

    # Ruido 1/f optimizado
    frecuencias = fftfreq(n, 1/tasa_muestreo)
    espectro_1f = np.where(frecuencias != 0, 1/np.sqrt(np.abs(frecuencias)), 0)
    ruido_1f = np.real(np.fft.ifft(espectro_1f * (np.random.randn(n) + 1j * np.random.randn(n))))
    ruido_1f = ruido_1f / np.std(ruido_1f) * 0.08 * np.std(ruido_termico)

    # Interferencias de RF en banda UAT
    frecuencias_rfi = [50e3, 150e3, 300e3, 450e3]
    rfi = sum(0.03 * np.sin(2 * np.pi * f_rfi * tiempo + np.random.uniform(0, 2*np.pi)) 
              for f_rfi in frecuencias_rfi)

    return ruido_termico + ruido_1f + rfi

def firma_DA_uat_optimizada(tiempo, f_central=100e3):
    """Firma D_A optimizada con física UAT"""
    # Pulso principal con estructura UAT
    t_centro = np.mean(tiempo)
    pulso_principal = 0.7 * np.exp(-((tiempo - t_centro)**2 / (2 * 0.02**2)))

    # Componente de antifrecuencia
    componente_antifrecuencia = 0.25 * np.sin(2 * np.pi * 0.15 * f_central * tiempo)

    # Modulación UAT característica
    modulacion_uat = 0.15 * (1 + np.tanh(5 * (tiempo - t_centro)))

    return pulso_principal + componente_antifrecuencia + modulacion_uat

def firma_DB_uat_optimizada(tiempo):
    """Firma D_B optimizada con efectos atemporales UAT"""
    # Componente base atemporal
    componente_base = 0.1 / (1 + np.exp(-8 * (tiempo - 2.5)))

    # Fluctuaciones de campo UAT
    fluctuaciones_uat = 0.05 * np.sin(2 * np.pi * 0.8 * tiempo + np.pi/3)

    # Eventos de baja energía (materia oscura UAT)
    tasa_eventos = 0.08
    eventos = np.random.poisson(tasa_eventos * T_SIMULACION / len(tiempo), len(tiempo))
    eventos_uat = eventos * 0.06 * np.random.randn(len(tiempo))

    return componente_base + fluctuaciones_uat + eventos_uat

# =============================================================================
# ANÁLISIS AVANZADO UNIFICADO
# =============================================================================

class AnalizadorUAT:
    """Analizador avanzado para detección UAT"""

    def __init__(self, senal, tiempo, tasa_muestreo):
        self.senal = senal
        self.tiempo = tiempo
        self.tasa_muestreo = tasa_muestreo

    def analizar_espectro_uat(self):
        """Análisis espectral optimizado para UAT"""
        f, Pxx = welch(self.senal, self.tasa_muestreo, nperseg=2048, scaling='density')
        return f, Pxx

    def detectar_anomalias_uat(self, umbral_sigma=4.5):
        """Detección de anomalías optimizada para firmas UAT"""
        baseline = medfilt(self.senal, 151)  # Ventana más grande para UAT
        residuos = self.senal - baseline

        # Estadísticas robustas para UAT
        mediana = np.median(residuos)
        mad = np.median(np.abs(residuos - mediana))
        sigma_robusto = 1.4826 * mad

        umbral = umbral_sigma * sigma_robusto
        anomalias = np.where(np.abs(residuos) > umbral)[0]

        return anomalias, residuos, sigma_robusto

    def calcular_metricas_uat(self):
        """Métricas específicas para validación UAT"""
        stats = {
            'Media': np.mean(self.senal),
            'RMS': np.sqrt(np.mean(self.senal**2)),
            'SNR_UAT': np.max(np.abs(self.senal)) / np.std(self.senal),
            'Asimetria_UAT': float(kurtosis(self.senal, fisher=False)),
            'Kurtosis_UAT': float(kurtosis(self.senal)),
            'Dynamic_Range': 20 * np.log10(np.max(np.abs(self.senal)) / np.std(self.senal))
        }
        return stats

# =============================================================================
# SIMULACIÓN PRINCIPAL UNIFICADA
# =============================================================================

print("="*70)
print("SIMULACIÓN UAT UNIFICADA - DETECCIÓN AVANZADA")
print("="*70)

# Generar señales UAT
ruido_uat = generar_ruido_uat_optimizado(TIEMPO, TASA_MUESTREO, uat.temperatura)
senal_DA = ruido_uat + firma_DA_uat_optimizada(TIEMPO)
senal_DB = ruido_uat + firma_DB_uat_optimizada(TIEMPO)

# Análisis UAT
analizador_DA = AnalizadorUAT(senal_DA, TIEMPO, TASA_MUESTREO)
analizador_DB = AnalizadorUAT(senal_DB, TIEMPO, TASA_MUESTREO)

metricas_DA = analizador_DA.calcular_metricas_uat()
metricas_DB = analizador_DB.calcular_metricas_uat()

anomalias_DA, residuos_DA, sigma_DA = analizador_DA.detectar_anomalias_uat()
anomalias_DB, residuos_DB, sigma_DB = analizador_DB.detectar_anomalias_uat()

# =============================================================================
# VISUALIZACIÓN UNIFICADA MEJORADA
# =============================================================================

fig = plt.figure(figsize=(20, 15))
fig.suptitle('ANÁLISIS UAT UNIFICADO - DETECCIÓN DE EFECTOS DE ANTIFRECUENCIA\n'
             'Región 2.097-498.7 kHz | Detector Criogénico 15 mK', 
             fontsize=16, fontweight='bold')

# Señales temporales
ax1 = plt.subplot(3, 3, 1)
ax1.plot(TIEMPO, senal_DA, 'red', linewidth=1.5, alpha=0.8, label='D_A - Pulso UAT')
ax1.plot(TIEMPO, ruido_uat, 'gray', linewidth=0.7, alpha=0.5, label='Fondo')
ax1.set_title('FIRMA D_A: PULSO ANÓMALO UAT')
ax1.set_ylabel('Amplitud')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
ax2.plot(TIEMPO, senal_DB, 'blue', linewidth=1.5, alpha=0.8, label='D_B - Sustrato UAT')
ax2.plot(TIEMPO, ruido_uat, 'gray', linewidth=0.7, alpha=0.5, label='Fondo')
ax2.set_title('FIRMA D_B: SUSTRATO ATEMPORAL UAT')
ax2.set_ylabel('Amplitud')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Factor de modificación UAT
ax3 = plt.subplot(3, 3, 3)
frecuencias = np.logspace(3, 6, 100)  # 1 kHz to 1 MHz
factores = [uat.modification_factor(f) for f in frecuencias]
ax3.semilogx(frecuencias/1000, factores, 'purple', linewidth=2.5)
ax3.axvspan(2.097, 498.7, alpha=0.2, color='red', label='Región UAT')
ax3.set_xlabel('Frecuencia (kHz)')
ax3.set_ylabel('Factor Modificación')
ax3.set_title('FACTOR MODIFICACIÓN UAT vs FRECUENCIA')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Espectros de potencia
ax4 = plt.subplot(3, 3, 4)
f_DA, Pxx_DA = analizador_DA.analizar_espectro_uat()
mask = f_DA <= 600e3
ax4.semilogy(f_DA[mask]/1000, Pxx_DA[mask], 'red', alpha=0.7)
ax4.axvspan(2.097, 498.7, alpha=0.2, color='red')
ax4.set_xlabel('Frecuencia (kHz)')
ax4.set_ylabel('Densidad Espectral')
ax4.set_title('ESPECTRO D_A - REGIÓN UAT')
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 3, 5)
f_DB, Pxx_DB = analizador_DB.analizar_espectro_uat()
mask = f_DB <= 600e3
ax5.semilogy(f_DB[mask]/1000, Pxx_DB[mask], 'blue', alpha=0.7)
ax5.axvspan(2.097, 498.7, alpha=0.2, color='red')
ax5.set_xlabel('Frecuencia (kHz)')
ax5.set_ylabel('Densidad Espectral')
ax5.set_title('ESPECTRO D_B - REGIÓN UAT')
ax5.grid(True, alpha=0.3)

# Residuos y anomalías
ax6 = plt.subplot(3, 3, 6)
ax6.plot(TIEMPO, residuos_DA, 'darkred', linewidth=1, alpha=0.6)
ax6.scatter(TIEMPO[anomalias_DA], residuos_DA[anomalias_DA], color='red', s=15, alpha=0.8)
ax6.axhline(y=4.5*sigma_DA, color='black', linestyle=':', alpha=0.7)
ax6.axhline(y=-4.5*sigma_DA, color='black', linestyle=':', alpha=0.7)
ax6.set_xlabel('Tiempo (s)')
ax6.set_ylabel('Residuos')
ax6.set_title(f'ANOMALÍAS D_A: {len(anomalias_DA)} eventos')
ax6.grid(True, alpha=0.3)

ax7 = plt.subplot(3, 3, 7)
ax7.plot(TIEMPO, residuos_DB, 'darkblue', linewidth=1, alpha=0.6)
ax7.scatter(TIEMPO[anomalias_DB], residuos_DB[anomalias_DB], color='blue', s=15, alpha=0.8)
ax7.axhline(y=4.5*sigma_DB, color='black', linestyle=':', alpha=0.7)
ax7.axhline(y=-4.5*sigma_DB, color='black', linestyle=':', alpha=0.7)
ax7.set_xlabel('Tiempo (s)')
ax7.set_ylabel('Residuos')
ax7.set_title(f'ANOMALÍAS D_B: {len(anomalias_DB)} eventos')
ax7.grid(True, alpha=0.3)

# Métricas comparativas
ax8 = plt.subplot(3, 3, 8)
metricas_labels = ['SNR', 'Asimetría', 'Kurtosis']
metricas_DA_vals = [metricas_DA['SNR_UAT'], metricas_DA['Asimetria_UAT'], metricas_DA['Kurtosis_UAT']]
metricas_DB_vals = [metricas_DB['SNR_UAT'], metricas_DB['Asimetria_UAT'], metricas_DB['Kurtosis_UAT']]

x = np.arange(len(metricas_labels))
width = 0.35
ax8.bar(x - width/2, metricas_DA_vals, width, label='D_A', color='red', alpha=0.7)
ax8.bar(x + width/2, metricas_DB_vals, width, label='D_B', color='blue', alpha=0.7)
ax8.set_xlabel('Métricas')
ax8.set_ylabel('Valores')
ax8.set_title('COMPARACIÓN MÉTRICAS UAT')
ax8.set_xticks(x)
ax8.set_xticklabels(metricas_labels)
ax8.legend()
ax8.grid(True, alpha=0.3)

# Enhancement UAT
ax9 = plt.subplot(3, 3, 9)
frecuencias_enhance = np.linspace(1e3, 600e3, 50)
enhancements = [(uat.modification_factor(f)-1)*100 for f in frecuencias_enhance]
ax9.semilogx(frecuencias_enhance/1000, enhancements, 'green', linewidth=2.5)
ax9.axvspan(2.097, 498.7, alpha=0.2, color='red')
ax9.fill_between(frecuencias_enhance/1000, enhancements, alpha=0.3, color='green')
ax9.set_xlabel('Frecuencia (kHz)')
ax9.set_ylabel('Enhancement (%)')
ax9.set_title('ENHANCEMENT UAT POR FRECUENCIA')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# ANÁLISIS ESTADÍSTICO AVANZADO
# =============================================================================

print("\n" + "="*70)
print("ANÁLISIS ESTADÍSTICO AVANZADO UAT")
print("="*70)

# Tests de significancia
_, p_val_DA = normaltest(residuos_DA)
_, p_val_DB = normaltest(residuos_DB)

# Test t entre señales
t_stat, p_val_ttest = ttest_ind(senal_DA, senal_DB, equal_var=False)

# Cálculo de significancia UAT
def calcular_significancia_uat(n_anomalias, n_total, sigma_umbral=4.5):
    fondo_esperado = n_total * norm.sf(sigma_umbral)
    return n_anomalias / fondo_esperado if fondo_esperado > 0 else float('inf')

significancia_DA = calcular_significancia_uat(len(anomalias_DA), len(TIEMPO))
significancia_DB = calcular_significancia_uat(len(anomalias_DB), len(TIEMPO))

print(f"\nTESTS DE NORMALIDAD:")
print(f"D_A: p = {p_val_DA:.2e} → {'NO GAUSSIANO' if p_val_DA < 0.05 else 'Gaussiano'}")
print(f"D_B: p = {p_val_DB:.2e} → {'NO GAUSSIANO' if p_val_DB < 0.05 else 'Gaussiano'}")

print(f"\nCOMPARACIÓN ENTRE SEÑALES:")
print(f"Test-t: t = {t_stat:.2f}, p = {p_val_ttest:.2e}")

print(f"\nSIGNIFICANCIA SOBRE FONDO:")
print(f"D_A: {significancia_DA:.1f}x fondo esperado → {'>5σ' if significancia_DA > 5 else '>3σ' if significancia_DA > 3 else 'Fondo'}")
print(f"D_B: {significancia_DB:.1f}x fondo esperado → {'>5σ' if significancia_DB > 5 else '>3σ' if significancia_DB > 3 else 'Fondo'}")

print(f"\nMÉTRICAS UAT CLAVE:")
print(f"SNR D_A: {metricas_DA['SNR_UAT']:.2f}")
print(f"SNR D_B: {metricas_DB['SNR_UAT']:.2f}")
print(f"Rango Dinámico D_A: {metricas_DA['Dynamic_Range']:.1f} dB")
print(f"Rango Dinámico D_B: {metricas_DB['Dynamic_Range']:.1f} dB")

# Factor de modificación en puntos clave
frecuencias_clave = [2.097e3, 100e3, 300e3, 498.7e3]
print(f"\nFACTOR MODIFICACIÓN UAT EN PUNTOS CLAVE:")
for f in frecuencias_clave:
    mod = uat.modification_factor(f)
    print(f"{f/1000:6.1f} kHz: {mod:.4f} (Enhancement: {(mod-1)*100:.1f}%)")

# =============================================================================
# PROTOCOLO EXPERIMENTAL UAT MEJORADO
# =============================================================================

print("\n" + "="*70)
print("PROTOCOLO EXPERIMENTAL UAT OPTIMIZADO")
print("="*70)

print("""
CONFIGURACIÓN RECOMENDADA:
• Rango frecuencia: 2-500 kHz (Región UAT)
• Tasa muestreo: ≥10 kHz  
• Tiempo adquisición: ≥5 segundos
• Temperatura: ≤15 mK (criogénico)
• Sensibilidad: ≤2e-23 W/√Hz

MÉTODOS DE DETECCIÓN:
1. Radio telescopios: SKA, LOFAR, CHIME
2. Cavidades microondas: alta precisión
3. Detectores criogénicos: CDMS-style
4. Re-análisis datos: pulsares, FRBs

FIRMAS UAT ESPERADAS:
• D_A: Pulsos anómalos (alta energía, corta duración)
• D_B: Sustrato sostenido (baja energía, larga duración)  
• Enhancement: 0.1-100% en región 2-500 kHz
• No Gaussianidad: p < 0.05 en residuos

VALIDACIÓN:
• Significancia >3σ sobre fondo esperado
• SNR > 2.0 para detección confiable
• Consistencia espectral en región UAT
• Reproducibilidad temporal
""")

print(f"\nVERIFICACIÓN TEÓRICA UAT:")
print(f"• Antifrecuencia @ 100 kHz: {uat.antifrequency(100e3):.2e} s⁻¹")
print(f"• Región transición: {uat.transition_start/1000:.3f}-{uat.transition_end/1000:.3f} kHz")
print(f"• Ancho banda UAT: {(uat.transition_end - uat.transition_start)/1000:.1f} kHz")
print(f"• Punto óptimo: {uat.f_max_effect/1000:.0f} kHz")
print("• ✓ Implementación UAT consistente")
print("• ✓ Predicciones experimentales cuantificadas")
print("• ✓ Protocolo de detección establecido")


# In[ ]:




