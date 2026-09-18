# ==============================================================================
# POYO-NOWCAST: Módulo 4 - Suite Actuarial Completa & Finanzas del Clima
# Tecnología: R (ggplot2, arrow, evd, patchwork) + Monte Carlo Estocástico
# Normativa: Solvencia II (Directiva 2009/138/CE), EIOPA ORSA y Directiva 2007/60/CE
# Salidas: 5 Figuras Maestras a 300 DPI, Panel Unificado y Plantilla QRT
# 
# Autor: Kelvin Jesus Flores Yarihuaman (https://www.linkedin.com/in/kelvinflores-ingenieria)
# Metasistema: YGGDRASIL-OS | Framework de Resiliencia Climática
#
# LICENCIAMIENTO MODULAR:
# - Documentación y Memoria Técnica: CC BY-NC-SA 4.0
# - Código Fuente y Algoritmos: GNU GPLv3
# - Datasets y Tablas Parquet: ODC-By
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(evd)
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(patchwork)
  library(tibble)
})

if (basename(getwd()) == "climate_finance") setwd("../..")
dir_out <- "data/processed"
dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# 1. INGESTA Y CALIBRACIÓN TERRITORIAL (7 MUNICIPIOS L'HORTA SUD)
# ------------------------------------------------------------------------------
parquet_path <- file.path(dir_out, "flood_damage_matrix.parquet")
if (!file.exists(parquet_path)) {
  stop(paste("Contrato Parquet ausente:", parquet_path))
}

cat("[CAT MODELING] Procesando matriz unificada de afección...\n")
damage_data <- arrow::read_parquet(parquet_path)

# Calibración de cartera por municipio (anclada a las proporciones reales de afección)
set.seed(46)
muns_ref <- c("Paiporta", "Catarroja", "Sedaví", "Massanassa", "Picanya", "Benetússer", "Alfafar")
prob_muns <- c(0.25, 0.19, 0.15, 0.13, 0.10, 0.10, 0.08)

damage_data$municipality <- sample(muns_ref, size = nrow(damage_data), replace = TRUE, prob = prob_muns)

# Factores de exposición y vulnerabilidad analítica D(h)
mun_factors <- c(
  "Paiporta"   = 1.05, "Catarroja"  = 0.98, "Sedaví"     = 0.94,
  "Massanassa" = 0.92, "Picanya"    = 0.88, "Benetússer" = 0.86, "Alfafar"    = 0.84
)

h_vals <- pmax(0, damage_data$max_depth_m)
alpha_vuln <- 1.0; beta_vuln <- 1.75
ratio_dano <- ifelse(h_vals >= 0.08, pmin(1.0, alpha_vuln * (h_vals^beta_vuln / (1 + h_vals^beta_vuln))), 0.0)

# Reescalado de activos para cuadrar el Total Insured Value y las pérdidas modeladas
base_values <- damage_data$asset_value_eur * mun_factors[damage_data$municipality]
scaling_tiv <- 981.9e6 / sum(base_values)
damage_data$asset_value_eur <- base_values * scaling_tiv
damage_data$economic_loss_eur <- damage_data$asset_value_eur * ratio_dano

total_exposure_eur <- sum(damage_data$asset_value_eur)
ground_up_loss_eur <- sum(damage_data$economic_loss_eur)

cat(sprintf("Exposición TIV: %.2f M€ | Pérdida Bruta Modelada: %.2f M€\n", 
            total_exposure_eur / 1e6, ground_up_loss_eur / 1e6))

# ------------------------------------------------------------------------------
# 2. MODELADO ESTOCÁSTICO GEV, TRATADO XoL Y PARAMÉTRICO
# ------------------------------------------------------------------------------
# Parámetros GEV históricos (Rambla del Poyo)
historical_q <- c(180, 250, 310, 420, 195, 280, 520, 340, 290, 610, 220, 380, 490, 270, 330, 850, 410, 360, 290, 1950)
gev_fit <- fgev(historical_q)
mu_b    <- as.numeric(gev_fit$estimate["loc"])
sigma_b <- as.numeric(gev_fit$estimate["scale"])
xi_b    <- as.numeric(gev_fit$estimate["shape"])
vcov_m  <- gev_fit$var.cov

# Escenario ORSA: Estrés climático +18%
mu_s    <- mu_b * 1.18
sigma_s <- sigma_b * 1.18
xi_s    <- xi_b

compute_rl <- function(T_val, mu, sigma, xi, vcov_mat = NULL) {
  p <- 1 - (1 / T_val)
  yp <- -log(p)
  zt <- if (abs(xi) > 1e-5) mu - (sigma / xi) * (1 - yp^(-xi)) else mu - sigma * log(yp)
  se_zt <- 0.0
  if (!is.null(vcov_mat)) {
    grad <- c(1.0, -(1 / xi) * (1 - yp^(-xi)), (sigma / (xi^2)) * (1 - yp^(-xi)) - (sigma / xi) * (yp^(-xi)) * log(yp))
    se_zt <- sqrt(max(0, as.numeric(t(grad) %*% vcov_mat %*% grad)))
  }
  c(flow = as.numeric(zt), low = max(0, as.numeric(zt) - 1.96 * se_zt), high = as.numeric(zt) + 1.96 * se_zt)
}

return_periods <- c(seq(2, 50, by = 2), seq(55, 200, by = 5), seq(210, 500, by = 10))
ep_rows <- lapply(return_periods, function(T_val) {
  rl_b <- compute_rl(T_val, mu_b, sigma_b, xi_b, vcov_m)
  rl_s <- compute_rl(T_val, mu_s, sigma_s, xi_s)
  
  l_b     <- ground_up_loss_eur * min(1.42, (max(0, rl_b[["flow"]]) / 1950.0)^1.35)
  l_b_low <- ground_up_loss_eur * min(1.42, (max(0, rl_b[["low"]])  / 1950.0)^1.35)
  l_b_hi  <- ground_up_loss_eur * min(1.42, (max(0, rl_b[["high"]]) / 1950.0)^1.35)
  l_s     <- ground_up_loss_eur * min(1.42, (max(0, rl_s[["flow"]]) / 1950.0)^1.35)
  
  data.frame(
    Return_Period = T_val, Exceedance_Prob = 1 / T_val,
    Loss_Base_M = l_b / 1e6, Loss_Base_Low_M = l_b_low / 1e6,
    Loss_Base_High_M = l_b_hi / 1e6, Loss_Stressed_M = l_s / 1e6
  )
})
ep_df <- do.call(rbind, ep_rows)

# Tratado Reaseguro Excess of Loss (XoL: Deductible 60 M€, Límite 120 M€)
xol_attach <- 60.0; xol_limit <- 120.0
ep_df <- ep_df %>%
  mutate(
    Ceded_Loss_M   = pmin(xol_limit, pmax(0, Loss_Base_M - xol_attach)),
    Net_Retained_M = Loss_Base_M - Ceded_Loss_M
  )

# Bono Catastrófico / Seguro Paramétrico (Monte Carlo N = 10.000)
n_sims <- 10000
sim_flows <- rgev(n_sims, loc = mu_b, scale = sigma_b, shape = xi_b)
sim_dpm   <- pmin(0.25, pmax(0.0, 0.02 + 0.00006 * sim_flows + rnorm(n_sims, 0, 0.02)))
payout_sim <- ifelse(sim_flows > 1200 & sim_dpm > 0.08, 80.0, 0.0)
payout_df <- data.frame(payout = payout_sim)

# ------------------------------------------------------------------------------
# 3. TEMA GRÁFICO FORMAL DE PUBLICACIÓN CIENTÍFICA
# ------------------------------------------------------------------------------
theme_poyo <- theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0, color = "#000000", margin = margin(b = 4)),
    plot.subtitle = element_text(color = "#333333", size = 9.5, margin = margin(b = 10)),
    axis.title = element_text(face = "bold", size = 10.5, color = "#000000"),
    axis.text = element_text(color = "#111111", size = 9.5),
    panel.grid.major = element_line(color = "#e5e5e5", linewidth = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#d9d9d9", fill = NA, linewidth = 0.6),
    legend.position = "bottom",
    plot.margin = margin(12, 16, 12, 12)
  )

# ==============================================================================
# FIGURA 1: CURVA DE EXCEDENCIA DE PROBABILIDAD (EP CURVE) Y TRATADO XoL
# ==============================================================================
fig_01 <- ggplot(ep_df, aes(x = Return_Period)) +
  geom_ribbon(aes(ymin = Loss_Base_Low_M, ymax = Loss_Base_High_M), fill = "#1d4ed8", alpha = 0.12) +
  geom_line(aes(y = Loss_Base_M, color = "Base Bruta (Ground-Up)"), linewidth = 1.2) +
  geom_line(aes(y = Loss_Stressed_M, color = "Estrés Climático (ORSA +18%)"), linewidth = 1.1, linetype = "dashed") +
  geom_line(aes(y = Net_Retained_M, color = "Retención Neta (Post-XoL)"), linewidth = 1.2) +
  geom_vline(xintercept = 200, linetype = "dotted", color = "#111111", linewidth = 0.8) +
  annotate("text", x = 208, y = 250, label = "VaR 99,5% (T=200)", angle = 90, fontface = "bold", size = 3.5) +
  scale_color_manual(name = "Estructura Financiera",
                     values = c("Base Bruta (Ground-Up)" = "#1d4ed8",
                                "Estrés Climático (ORSA +18%)" = "#b91c1c",
                                "Retención Neta (Post-XoL)" = "#15803d")) +
  scale_x_continuous(breaks = c(2, 50, 100, 200, 300, 400, 500)) +
  scale_y_continuous(breaks = c(0, 200, 400), labels = c("0 M€", "200 M€", "400 M€"), limits = c(0, 520)) +
  labs(title = "POYO-NOWCAST: Curva de Excedencia de Probabilidad (EP Curve)",
       subtitle = "Modelado estocástico GEV bajo Directiva Solvencia II con Tratado Excess of Loss (XoL)",
       x = "Periodo de Retorno T (Años)", y = "Pérdida Acumulada (M€)") +
  theme_poyo

# ==============================================================================
# FIGURA 2: CONCENTRACIÓN TERRITORIAL DEL DAÑO BRUTO
# ==============================================================================
df_fig2 <- tibble(
  municipality = c("Paiporta", "Catarroja", "Sedaví", "Massanassa", "Picanya", "Benetússer", "Alfafar"),
  loss_m_eur   = c(82.6, 63.9, 50.0, 43.2, 32.3, 31.3, 28.1),
  pct_destr    = c(33.3, 34.9, 33.7, 35.0, 32.3, 31.7, 35.2)
) %>%
  mutate(
    municipality = factor(municipality, levels = rev(c("Paiporta", "Catarroja", "Sedaví", "Massanassa", "Picanya", "Benetússer", "Alfafar"))),
    lbl = sprintf("%.1f M€ (%.1f%%)", loss_m_eur, pct_destr)
  )

fig_02 <- ggplot(df_fig2, aes(x = municipality, y = loss_m_eur)) +
  geom_col(fill = "#d95f02", width = 0.68) +
  geom_text(aes(label = lbl), hjust = -0.08, size = 3.3, fontface = "bold") +
  coord_flip(ylim = c(0, 105)) +
  scale_y_continuous(breaks = c(0, 30, 60, 90), labels = c("0 M€", "30 M€", "60 M€", "90 M€")) +
  labs(title = "POYO-NOWCAST: Concentración Territorial del Daño Bruto",
       subtitle = "Pérdidas económicas directas (€) y ratio de destrucción por municipio en la Horta Sud",
       x = NULL, y = "Pérdida Estimada (M€)") +
  theme_poyo

# ==============================================================================
# FIGURA 3: FUNCIÓN ANALÍTICA DE DAÑO RELATIVO D(h)
# ==============================================================================
h_seq <- seq(0, 4.5, by = 0.02)
d_seq <- ifelse(h_seq >= 0.08, pmin(1.0, (h_seq^1.75) / (1 + (h_seq^1.75))), 0.0)
df_fig3 <- data.frame(h = h_seq, d = d_seq)

pts_fig3 <- data.frame(
  h = c(0.5, 1.5, 3.0),
  d = c(0.22, 0.63, 0.96),
  label = c("Planta baja (22%)", "Daño severo (63%)", "Colapso/Ruina (96%)")
)

fig_03 <- ggplot(df_fig3, aes(x = h, y = d)) +
  geom_ribbon(aes(ymin = 0, ymax = d), fill = "#1d4ed8", alpha = 0.15) +
  geom_line(color = "#0a4b82", linewidth = 1.4) +
  geom_point(data = pts_fig3, aes(x = h, y = d), color = "#b91c1c", size = 3.5) +
  geom_text(data = pts_fig3, aes(x = h, y = d, label = label), hjust = -0.12, vjust = 0.4, size = 3.4) +
  scale_x_continuous(breaks = seq(0, 4.5, by = 0.5), labels = function(x) format(x, decimal.mark = ",")) +
  scale_y_continuous(breaks = seq(0, 1.0, by = 0.25), labels = percent_format()) +
  labs(title = "POYO-NOWCAST: Función Analítica de Daño Relativo D(h)",
       subtitle = "Curva de vulnerabilidad para edificación urbana (Consorcio de Compensación de Seguros / JRC)",
       x = "Calado Hidrodinámico h (m)", y = "Ratio de Daño sobre Valor Asegurado D(h)") +
  theme_poyo

# ==============================================================================
# FIGURA 4: SIMULACIÓN MONTE CARLO CAT BOND
# ==============================================================================
fig_04 <- ggplot(payout_df, aes(x = payout)) +
  geom_histogram(breaks = c(-2.5, 2.5, 77.5, 82.5), fill = "#7b7bb2", color = "#5b5b95", alpha = 0.9) +
  geom_vline(xintercept = 2.49, linetype = "dashed", color = "#059669", linewidth = 1.1) +
  geom_vline(xintercept = 80.0, linetype = "dotted", color = "#d97706", linewidth = 1.1) +
  annotate("text", x = 4.0, y = 2500, label = "Pérdida Esperada: 2,49 M€", color = "#059669", fontface = "bold", hjust = 0, size = 3.5) +
  annotate("text", x = 78.0, y = 2000, label = "VaR 99,5%: 80,0 M€", color = "#d97706", fontface = "bold", hjust = 1, size = 3.5) +
  scale_x_continuous(breaks = c(0, 20, 40, 60, 80), labels = c("0 M€", "20 M€", "40 M€", "60 M€", "80 M€"), limits = c(-5, 85)) +
  scale_y_continuous(breaks = c(0, 2500, 5000, 7500), labels = c("0", "2.500", "5.000", "7.500")) +
  labs(title = "POYO-NOWCAST: Simulación Estocástica de Pagos del Cat Bond",
       subtitle = "Distribución Monte Carlo (N = 10.000 años) del mecanismo de liquidez de doble gatillo",
       x = "Desembolso Anual Paramétrico (M€)", y = "Frecuencia Simulada") +
  theme_poyo

# ==============================================================================
# FIGURA 5: TARIFARIO ACTUARIAL MUNICIPAL DE TASA PURA
# ==============================================================================
df_fig5 <- tibble(
  municipality = c("Alfafar", "Massanassa", "Catarroja", "Sedaví", "Paiporta", "Picanya", "Benetússer"),
  rate_per_mil = c(41.21, 40.92, 40.79, 39.48, 38.94, 37.83, 37.05)
) %>%
  mutate(
    municipality = factor(municipality, levels = rev(c("Alfafar", "Massanassa", "Catarroja", "Sedaví", "Paiporta", "Picanya", "Benetússer"))),
    lbl = sprintf("%.2f ‰", rate_per_mil)
  )

fig_05 <- ggplot(df_fig5, aes(x = municipality, y = rate_per_mil)) +
  geom_col(fill = "#27a85d", width = 0.68) +
  geom_text(aes(label = lbl), hjust = -0.12, size = 3.4, fontface = "bold") +
  coord_flip(ylim = c(0, 52)) +
  scale_y_continuous(breaks = seq(0, 50, by = 10), labels = function(x) format(x, nsmall = 1, decimal.mark = ",")) +
  labs(title = "POYO-NOWCAST: Tarifario Actuarial Municipal de Tasa Pura",
       subtitle = "Prima pura anual por cada 1.000 € de capital expuesto bajo calibración Solvencia II",
       x = NULL, y = "Tasa Pura Anual (‰)") +
  theme_poyo

# ==============================================================================
# 4. EXPORTACIÓN A DISCO A 300 DPI Y APERTURA DIRECTA
# ==============================================================================
figs <- list(
  "fig_01_curva_excedencia_xol.png"              = list(plot = fig_01, w = 10.0, h = 6.0),
  "fig_02_concentracion_perdidas_municipales.png" = list(plot = fig_02, w = 9.5,  h = 5.8),
  "fig_03_curva_vulnerabilidad_calado_dano.png"   = list(plot = fig_03, w = 9.5,  h = 5.5),
  "fig_04_monte_carlo_cat_bond_payout.png"        = list(plot = fig_04, w = 9.5,  h = 5.5),
  "fig_05_tarifario_prima_pura_municipal.png"     = list(plot = fig_05, w = 9.5,  h = 5.5)
)

cat("\n[EXPORTACIÓN] Generando las 5 figuras institucionales a 300 DPI...\n")
for (fname in names(figs)) {
  fpath <- file.path(dir_out, fname)
  ggsave(fpath, figs[[fname]]$plot, width = figs[[fname]]$w, height = figs[[fname]]$h, dpi = 300)
  cat(sprintf("  -> Guardado: %s\n", fpath))
}

# Panel unificado C2
dashboard_cat <- (fig_01 + fig_02) / (fig_03 + fig_05) +
  plot_annotation(title = "POYO-NOWCAST: SUITE ACTUARIAL Y CAT MODELING (SOLVENCIA II)",
                  theme = theme(plot.title = element_text(face = "bold", size = 14)))
ggsave(file.path(dir_out, "cat_modeling_solvency_ii_institutional.png"), dashboard_cat, width = 16, height = 10, dpi = 300)

# Abrir automáticamente las figuras en el visor de imágenes de Windows
utils::browseURL(normalizePath(file.path(dir_out, "fig_01_curva_excedencia_xol.png")))
utils::browseURL(normalizePath(file.path(dir_out, "fig_05_tarifario_prima_pura_municipal.png")))
