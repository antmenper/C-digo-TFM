set.seed(42)

# Librerías 
library(quantmod)    # descarga de series financieras desde Yahoo Finance
library(ggplot2)
library(cowplot)
library(vars)        # modelo VAR frecuentistas
library(bvartools)   # modelo BVAR con prior de Minnesota
library(urca)        # tests de raíz unitaria (KPSS, ADF) y cointegración (Johansen)
library(tseries)     # test ADF y PP adicionales
library(forecast)    # auto.arima y forecast (fallback)
library(broom)       # tidy() para resultados de regresiones 
library(tidyverse)   
library(lubridate)   # manejo de fechas
library(zoo)         # objetos de series temporales irregulares
library(xts)         # series temporales extendidas
library(tictoc)      # medición de tiempos de ejecución
library(scales)      # formato de ejes en ggplot2
library(recipes)     # preprocesamiento (normalización)
library(keras)       # interfaz R para Keras / TensorFlow
py_require_legacy_keras()


# Datos 
# Todos los tickers "DIVISA=X" expresan la misma convención directa (USD por
# unidad de divisa extranjera), por lo que NO es necesario invertir ninguna
# serie. Las series JPY y CNY tienen valores pequeños pero son consistentes
# internamente con el resto.

getSymbols("EUR=X", src = "yahoo", from = "2020-01-01", to = "2024-12-31")
getSymbols("GBP=X", src = "yahoo", from = "2020-01-01", to = "2024-12-31")
getSymbols("JPY=X", src = "yahoo", from = "2020-01-01", to = "2024-12-31")
getSymbols("CHF=X", src = "yahoo", from = "2020-01-01", to = "2024-12-31")
getSymbols("CNY=X", src = "yahoo", from = "2020-01-01", to = "2024-12-31")

# Extraer series de cierre como vectores numéricos 

euro <- `EUR=X`$`EUR=X.Close`
gbp  <- `GBP=X`$`GBP=X.Close`
jpy  <- `JPY=X`$`JPY=X.Close`
chf  <- `CHF=X`$`CHF=X.Close`
cny  <- `CNY=X`$`CNY=X.Close`


# Gráficos exploratorios de las series originales

# Gráficos individuales de cada una de las divisas obtenidas del forex
# Tanto JPY como CNY se obtienen al contrario que el resto de divisas
chartSeries(`EUR=X`, theme = chartTheme("white"), name = "EUR/USD 2020–2024")
chartSeries(`GBP=X`, theme = chartTheme("white"), name = "GBP/USD 2020–2024")
chartSeries(`JPY=X`, theme = chartTheme("white"), name = "USD/JPY 2020–2024")
chartSeries(`CHF=X`, theme = chartTheme("white"), name = "CHF/USD 2020–2024")
chartSeries(`CNY=X`, theme = chartTheme("white"), name = "USD/CNY 2020–2024")

# Data frame de precios (VAR y BVAR) 

precios_ts <- data.frame(
  EUR = as.numeric(coredata(euro)),
  GBP = as.numeric(coredata(gbp)),
  JPY = 1 / as.numeric(coredata(jpy)),   # USD por 1 JPY
  CHF = as.numeric(coredata(chf)),
  CNY = 1 / as.numeric(coredata(cny))    # USD por 1 CNY
) |> na.omit()
nrow(precios_ts)
# Vector con los nombres de las divisas (usado en todos los bucles posteriores)

pares <- colnames(precios_ts)

# Data frame con fechas (LSTM y gráficos) 

fechas_comunes <- as.Date(index(euro))

fx_niveles <- data.frame(
  index = fechas_comunes,
  EUR   = as.numeric(coredata(euro)),
  GBP   = as.numeric(coredata(gbp)),
  JPY   = 1 / as.numeric(coredata(jpy)),
  CHF   = as.numeric(coredata(chf)),
  CNY   = 1 / as.numeric(coredata(cny))
) |> tidyr::drop_na()
nrow(fx_niveles)

# Análisis exploratorio y descriptivos 

summary(precios_ts)

# Tests de estacionariedad en niveles 

# Test KPSS  (H0: estacionaria  vs  H1: raíz unitaria)

lapply(precios_ts, function(x) summary(ur.kpss(x, type = "mu")))

# H0: Datos estacionarios -> Se rechaza

# Test ADF  (H0: raíz unitaria  vs  H1: estacionaria)

lapply(precios_ts, function(x) summary(ur.df(x, type = "drift", selectlags = "AIC")))
# No rechazo H0 -> Se debe integrar.

# Test Phillips-Perron  (robusto a heteroscedasticidad. H0: raíz untaria)

lapply(precios_ts, function(x) tseries::pp.test(x))
# No se puede rechazar H0 -> confirma I(1)

# Verificación en primeras diferencias 

precios_diff <- as.data.frame(apply(precios_ts, 2, diff))
summary(precios_diff)
# Test KPSS en primeras diferencias 

lapply(precios_diff, function(x) summary(ur.kpss(x, type = "mu")))


# Test ADF en primeras diferencias 
lapply(precios_diff, function(x) summary(ur.df(x, type = "drift", selectlags = "AIC")))

# Test Phillips-Perron 

lapply(precios_diff, function(x) tseries::pp.test(x))
       
# Test de cointegración de Johansen 

johansen_trace <- ca.jo(
  precios_ts,
  type  = "trace",
  ecdet = "const",
  K     = 5           # número de lags (consistente con lag.max del VAR)
)
summary(johansen_trace)

# Test de Johansen (máximo valor propio) 
johansen_eigen <- ca.jo(
  precios_ts,
  type  = "eigen",
  ecdet = "const",
  K     = 5
)
summary(johansen_eigen)

# Conclusión: evidencia de cointegración débil o espuria.
# Los rechazos de la traza están impulsados por acumulación de señales
# marginales, no por una relación de largo plazo genuina y robusta.
# Se estima el VAR en primeras diferencias.


# Métricas de evaluación 
# Calcula MAE, RMSE y MAPE para un horizonte y divisa dados.
# Recibe el objeto resultado (lista con $predicciones y $reales).

calcular_metricas <- function(resultado, h_eval = 1, divisa = 1) {
  
  pred <- resultado$predicciones[, h_eval, divisa]
  real <- resultado$reales[,       h_eval, divisa]
  e    <- real - pred
  
  mae   <- mean(abs(e))
  rmse  <- sqrt(mean(e^2))
  mape  <- mean(abs(e / real)) * 100
  
  if (!is.null(resultado$ultimos_niveles)) {
    pred_rw <- resultado$ultimos_niveles[, divisa]
    rmse_rw <- sqrt(mean((real - pred_rw)^2))
  } 
  
  tibble(
    Horizonte = h_eval,
    MAE       = round(mae,   6),
    RMSE      = round(rmse,  6),
    MAPE      = round(mape,  4)
  )
}

# Métricas del paseo aleatorio 
calcular_metricas_rw <- function(resultado, h_eval = 1, divisa = 1, nombre_par) {
  
  real    <- resultado$reales[,          h_eval, divisa]
  pred_rw <- resultado$ultimos_niveles[, divisa]
  e       <- real - pred_rw
  
  tibble(
    Horizonte = h_eval,
    MAE       = round(mean(abs(e)),              6),
    RMSE      = round(sqrt(mean(e^2)),            6),
    MAPE      = round(mean(abs(e / real)) * 100,  4),
    Par       = nombre_par,
    Modelo    = "RW"
  )
}

# Test de Diebold-Mariano 
# Contrasta si la diferencia en precisión entre dos modelos es estadísticamente
# significativa. H0: pérdida esperada igual (ninguno es superior).
# Usa pérdida cuadrática (power = 2) por defecto.
dm_test <- function(resultado_a, resultado_b,
                    h_eval  = 1,
                    divisa  = 1,
                    power   = 2) {
  
  pred_a <- resultado_a$predicciones[, h_eval, divisa]
  pred_b <- resultado_b$predicciones[, h_eval, divisa]
  real   <- resultado_a$reales[,       h_eval, divisa]
  
  e_a <- real - pred_a
  e_b <- real - pred_b
  
  d   <- abs(e_a)^power - abs(e_b)^power          # diferencia de pérdidas
  T   <- length(d)
  d_bar <- mean(d)
  
  # Varianza de larga memoria (Newey-West con h-1 lags)
  h_lag <- max(1, h_eval - 1)
  gamma0 <- var(d)
  gammas  <- sapply(1:h_lag, function(k) mean((d - d_bar)[(k + 1):T] *
                                                (d - d_bar)[1:(T - k)]))
  v_d    <- (gamma0 + 2 * sum(gammas)) / T
  dm_stat <- d_bar / sqrt(max(v_d, .Machine$double.eps))
  
  p_val <- 2 * pnorm(-abs(dm_stat))
  
  tibble(
    DM_stat  = round(dm_stat, 4),
    p_valor  = round(p_val,   4),
    sig      = case_when(
      p_val < 0.01 ~ "***",
      p_val < 0.05 ~ "**",
      p_val < 0.10 ~ "*",
      TRUE         ~ ""
    )
  )
}



# Modelo VAR
# Protocolo rolling con ventana fija W = 252 días (≈ 1 año bursátil).
# En cada origen i:
#   1. Se diferencia la ventana de entrenamiento → serie I(0)
#   2. Se selecciona el lag óptimo por AIC (lag.max = 5)
#   3. Se estima VAR(p) y se verifica estabilidad (módulos de raíces < 1)
#   4. Si el VAR es inestable → fallback a ARIMA univariante por divisa
#   5. Las predicciones en diferencias se acumulan para volver a niveles

var_rolling <- function(serie, W, h = 1, expandible = FALSE) {
  
  serie_mat      <- as.matrix(serie)
  colnames_serie <- colnames(serie_mat)
  serie <- matrix(
    as.numeric(serie_mat),
    nrow     = nrow(serie_mat),
    ncol     = ncol(serie_mat),
    dimnames = list(NULL, colnames_serie)
  )
  
  N          <- nrow(serie)
  n_col      <- ncol(serie)
  n_origenes <- N - W - h + 1
  if (n_origenes < 1) stop("W + h > longitud de la serie")
  
  predicciones    <- array(NA_real_, dim = c(n_origenes, h, n_col))
  reales          <- array(NA_real_, dim = c(n_origenes, h, n_col))
  ultimos_niveles <- matrix(NA_real_, nrow = n_origenes, ncol = n_col)
  modelos_info    <- vector("list", n_origenes)
  
  cat(sprintf("VAR rolling — %d orígenes de predicción (W=%d, h=%d)...\n",
              n_origenes, W, h))
  pb <- txtProgressBar(min = 0, max = n_origenes, style = 3)
  
  for (i in seq_len(n_origenes)) {
    
    inicio_train       <- if (expandible) 1L else i
    fin_train          <- W + i - 1L
    ventana_train      <- serie[inicio_train:fin_train, ]
    ventana_train_diff <- apply(ventana_train, 2, diff)
    ultimo_nivel       <- ventana_train[nrow(ventana_train), ]
    ultimos_niveles[i, ] <- ultimo_nivel
    
    modelo <- tryCatch({
      lag_optimo <- max(1L, as.integer(
        VARselect(ventana_train_diff, lag.max = 5, type = "const")$selection["AIC(n)"]
      ))
      mod <- VAR(ventana_train_diff, p = lag_optimo, type = "const")
      
      # Verificar estabilidad: todos los módulos de raíces < 1
      if (any(roots(mod) >= 1)) NULL else mod
      
    }, error = function(e) NULL)
    
    pred_niveles <- matrix(NA_real_, nrow = h, ncol = n_col)
    
    if (!is.null(modelo) && inherits(modelo, "varest")) {
      
      pred_var <- predict(modelo, n.ahead = h)
      for (j in seq_len(n_col)) {
        pred_diff_j       <- pred_var$fcst[[j]][, "fcst"]
        pred_niveles[, j] <- ultimo_nivel[j] + cumsum(pred_diff_j)
      }
      modelos_info[[i]] <- list(lag = modelo$p, AIC = AIC(modelo), tipo = "VAR")
      
    } else {
      
      # Fallback: ARIMA univariante por divisa
      for (j in seq_len(n_col)) {
        mod_arima <- tryCatch(
          auto.arima(
            ventana_train_diff[, j],
            max.p         = 5,
            max.q         = 2,
            max.d         = 0,    # ya es I(0), no diferenciar de nuevo
            stepwise      = TRUE,
            approximation = TRUE
          ),
          error = function(e) NULL
        )
        if (!is.null(mod_arima)) {
          pred_diff_j       <- as.numeric(forecast(mod_arima, h = h)$mean)
          pred_niveles[, j] <- ultimo_nivel[j] + cumsum(pred_diff_j)
        }
      }
      modelos_info[[i]] <- list(lag = NA, AIC = NA, tipo = "ARIMA_fallback")
    }
    
    predicciones[i, , ] <- pred_niveles
    reales[i, , ]       <- serie[(fin_train + 1):(fin_train + h), ]
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  list(
    predicciones    = predicciones,
    reales          = reales,
    ultimos_niveles = ultimos_niveles,
    idx_reales      = (W + 1):(N - h + 1),
    modelos_info    = modelos_info,
    config          = list(W = W, h = h, expandible = expandible)
  )
}

# Ejecución del VAR rolling 
W <- 252
H <- 5

tic()
resultado_var <- var_rolling(serie = precios_ts, W = W, h = H, expandible = FALSE)
toc()

# Tabla de métricas VAR 
tabla_metricas_var <- map_dfr(seq_along(pares), function(j) {
  map_dfr(1:H, function(h_eval) {
    calcular_metricas(resultado_var, h_eval = h_eval, divisa = j) |>
      mutate(Par = pares[j])
  })
}) |> select(Par, Horizonte, MAE, RMSE, MAPE)

print(tabla_metricas_var)

# Tabla comparativa VAR vs RW 
tabla_rw <- map_dfr(seq_along(pares), function(j) {
  map_dfr(1:H, function(h_eval) {
    calcular_metricas_rw(resultado_var, h_eval = h_eval,
                         divisa = j, nombre_par = pares[j])
  })
})

tabla_var_rw <- bind_rows(
  tabla_metricas_var |> mutate(Modelo = "VAR"),
  tabla_rw
) |>
  select(Par, Horizonte, Modelo, MAE, RMSE, MAPE) |>
  arrange(Par, Horizonte, Modelo)

print(tabla_var_rw)

# Sensibilidad a la ventana W (VAR) 
ventanas <- c(126, 189, 252, 378)

cat("\n── Sensibilidad VAR a la ventana W ──\n")
resultados_W <- map(ventanas, function(w) {
  cat("Ejecutando W =", w, "\n")
  var_rolling(serie = precios_ts, W = w, h = H, expandible = FALSE)  # corregido: var_rolling
})

tabla_sensibilidad_W <- map2_dfr(resultados_W, ventanas, function(res, w) {
  map_dfr(seq_along(pares), function(j) {
    calcular_metricas(res, h_eval = 1, divisa = j) |>
      mutate(Par = pares[j], W = w)
  })
}) |> arrange(Par, W)

print(tabla_sensibilidad_W)

# MODELO BVAR 

# Prior de Minnesota (Litterman, 1986):
#   κ0: varianza del prior sobre el primer lag propio  (= 0.04)
#   κ1: decaimiento por lag                            (= 0.5)
#   κ2: varianza cruzada relativa                      (= 0.5)
#   κ3: prior sobre la constante                       (= 5)
#
# Inferencia posterior mediante Gibbs sampler (iter = 5000, burnin = 2500).
# Predicción: media de los draws predictivos a posteriori.
#
# CORRECCIÓN: la estructura del array pred_bvar$fcst es [draws, h, variables].
#             Se usa apply(..., 2, mean) sobre la dimensión h correcta.
grid_kappas <- expand.grid(
  kappa0 = c(0.01, 0.04, 0.10),
  kappa1 = c(0.1,  0.3,  0.5),
  kappa2 = c(0.3,  0.5,  1.0)
)
rf_bvar <- function(serie, W, h = 1, expandible = FALSE,
                    p_lag      = 1,
                    iterations = 500,
                    burnin     = 200,
                    paso       = 5,         # submuestreo: 1 de cada 'paso' orígenes
                    kappa0 = kappa0,
                    kappa1 = kappa1,
                    kappa2 = kappa2,
                    kappa3 = 5) {
  
  serie      <- as.matrix(serie)
  N          <- nrow(serie)
  n_col      <- ncol(serie)
  n_origenes <- N - W - h + 1
  if (n_origenes < 1) stop("W + h > longitud de la serie")
  
  # Submuestreo de orígenes: 1 de cada 'paso' (coste computacional del BVAR)
  idx_eval <- seq(1L, n_origenes, by = paso)
  n_eval   <- length(idx_eval)
  
  predicciones    <- array(NA_real_, dim = c(n_eval, h, n_col))
  reales          <- array(NA_real_, dim = c(n_eval, h, n_col))
  ultimos_niveles <- matrix(NA_real_, nrow = n_eval, ncol = n_col)
  modelos_info    <- vector("list", n_eval)
  
  cat(sprintf("BVAR rolling — %d de %d orígenes evaluados (paso=%d, W=%d, h=%d, iterations=%d, burnin=%d)...\n",
              n_eval, n_origenes, paso, W, h, iterations, burnin))
  pb <- txtProgressBar(min = 0, max = n_eval, style = 3)
  
  for (idx in seq_len(n_eval)) {
    
    i <- idx_eval[idx]   # índice de origen real dentro del esquema rolling completo
    
    inicio_train       <- if (expandible) 1L else i
    fin_train          <- W + i - 1L
    ventana_train      <- serie[inicio_train:fin_train, ]
    ventana_train_diff <- apply(ventana_train, 2, diff)
    
    # gen_var() exige un objeto de clase 'ts'; apply() devuelve matriz simple
    nombres_col <- colnames(ventana_train_diff)
    ventana_train_diff <- ts(ventana_train_diff, frequency = 1)
    colnames(ventana_train_diff) <- nombres_col
    
    ultimo_nivel          <- ventana_train[nrow(ventana_train), ]
    ultimos_niveles[idx, ] <- ultimo_nivel
    
    pred_niveles <- matrix(NA_real_, nrow = h, ncol = n_col)
    
    modelo_bvar <- tryCatch({
      bvar_data <- gen_var(ventana_train_diff,
                           p             = p_lag,
                           deterministic = "const",
                           iterations    = iterations,
                           burnin        = burnin)
      
      bvar_data <- add_priors(bvar_data,
                              coef = list(
                                minnesota = list(kappa0 = kappa0,
                                                 kappa1 = kappa1,
                                                 kappa2 = kappa2,
                                                 kappa3 = kappa3)
                              ))
      
      draw_posterior(bvar_data)
      
    }, error = function(e) NULL)
    
    if (!is.null(modelo_bvar)) {
      
      pred_bvar <- predict(modelo_bvar, n.ahead = h)
      
      # pred_bvar$fcst es una LISTA nombrada por variable; cada elemento es
      # una matriz (h x 3) con columnas "2.5%", "50%", "97.5%".
      # Se usa la mediana posterior ("50%") como predicción puntual.
      for (j in seq_len(n_col)) {
        var_j       <- nombres_col[j]
        pred_diff_j <- pred_bvar$fcst[[var_j]][, "50%"]
        pred_niveles[, j] <- ultimo_nivel[j] + cumsum(pred_diff_j)
      }
      modelos_info[[idx]] <- list(lag = p_lag, iterations = iterations,
                                  burnin = burnin, tipo = "BVAR")
      
    } else {
      
      # Fallback: AR(1) univariante por divisa
      for (j in seq_len(n_col)) {
        mod_arima <- tryCatch(
          forecast::Arima(ventana_train_diff[, j], order = c(1, 0, 0)),
          error = function(e) NULL
        )
        if (!is.null(mod_arima)) {
          pred_diff_j       <- as.numeric(forecast(mod_arima, h = h)$mean)
          pred_niveles[, j] <- ultimo_nivel[j] + cumsum(pred_diff_j)
        }
      }
      modelos_info[[idx]] <- list(lag = NA, iterations = NA, burnin = NA,
                                  tipo = "AR1_fallback")
    }
    
    predicciones[idx, , ] <- pred_niveles
    reales[idx, , ]       <- serie[(fin_train + 1):(fin_train + h), ]
    setTxtProgressBar(pb, idx)
  }
  close(pb)
  
  # Resumen de cuántas ventanas usaron BVAR real vs. fallback AR(1)
  tipos <- sapply(modelos_info, function(x) x$tipo)
  n_bvar     <- sum(tipos == "BVAR")
  n_fallback <- sum(tipos == "AR1_fallback")
  cat(sprintf("\nVentanas con BVAR real: %d / %d (%.1f%%)\n",
              n_bvar, n_eval, 100 * n_bvar / n_eval))
  cat(sprintf("Ventanas con fallback AR(1): %d / %d (%.1f%%)\n",
              n_fallback, n_eval, 100 * n_fallback / n_eval))
  
  list(
    predicciones    = predicciones,
    reales          = reales,
    ultimos_niveles = ultimos_niveles,
    idx_eval        = idx_eval,                    # índices de origen (1..n_origenes) evaluados
    idx_reales      = (W + 1):(N - h + 1),
    modelos_info    = modelos_info,
    config          = list(W = W, h = h, expandible = expandible,
                           p = p_lag, iterations = iterations, burnin = burnin,
                           paso = paso, n_origenes_total = n_origenes, n_eval = n_eval)
  )
}

# Ejecución del BVAR rolling (submuestreado: 1 de cada 5 orígenes → 210 de 1049)
tic()
resultado_bvar <- rf_bvar(
  serie      = precios_ts,
  W          = W,
  h          = H,
  p_lag      = 1,
  kappa0     = 0.04,
  kappa1     = 0.5,
  kappa2     = 0.5,
  kappa3     = 5,
  iterations = 500,
  burnin     = 200,
  paso       = 5
)
toc()

# Tabla de métricas BVAR (sobre los 210 orígenes evaluados)
tabla_metricas_bvar <- map_dfr(seq_along(pares), function(j) {
  map_dfr(1:H, function(h_eval) {
    calcular_metricas(resultado_bvar, h_eval = h_eval, divisa = j) |>
      mutate(Par = pares[j])
  })
}) |> select(Par, Horizonte, MAE, RMSE, MAPE)

print(tabla_metricas_bvar)


# ── Comparación BVAR vs VAR vs RW en la SUBMUESTRA de 210 orígenes ──────────
# El BVAR se evalúa en 1 de cada 5 orígenes (210 de 1049) por coste
# computacional del muestreador de Gibbs.
# Para una comparación válida, VAR y RW se recalculan sobre los MISMOS
# 210 orígenes a partir de los resultados ya disponibles.


idx_eval_bvar <- resultado_bvar$idx_eval   # índices de origen (1..1049) evaluados

resultado_var_210 <- list(
  predicciones    = resultado_var$predicciones[idx_eval_bvar, , , drop = FALSE],
  reales          = resultado_var$reales[idx_eval_bvar, , , drop = FALSE],
  ultimos_niveles = resultado_var$ultimos_niveles[idx_eval_bvar, , drop = FALSE]
)

# Tabla de métricas VAR sobre la submuestra de 210
tabla_metricas_var_210 <- map_dfr(seq_along(pares), function(j) {
  map_dfr(1:H, function(h_eval) {
    calcular_metricas(resultado_var_210, h_eval = h_eval, divisa = j) |>
      mutate(Par = pares[j])
  })
}) |> select(Par, Horizonte, MAE, RMSE, MAPE)

# Tabla de métricas RW sobre la submuestra de 210
tabla_rw_210 <- map_dfr(seq_along(pares), function(j) {
  map_dfr(1:H, function(h_eval) {
    calcular_metricas_rw(resultado_var_210, h_eval = h_eval,
                         divisa = j, nombre_par = pares[j])
  })
})

# Tabla comparativa BVAR vs VAR vs RW (misma submuestra de 210 orígenes)
tabla_comparativa_210 <- bind_rows(
  tabla_metricas_bvar    |> mutate(Modelo = "BVAR"),
  tabla_metricas_var_210 |> mutate(Modelo = "VAR"),
  tabla_rw_210           |> select(Par, Horizonte, MAE, RMSE, MAPE, Modelo)
) |>
  select(Par, Horizonte, Modelo, MAE, RMSE, MAPE) |>
  arrange(Par, Horizonte, Modelo)

cat(sprintf("\n── Comparación BVAR vs VAR vs RW (submuestra de %d orígenes) ──\n",
            length(idx_eval_bvar)))
print(tabla_comparativa_210, n = 75)

view(tabla_metricas_bvar)
view(tabla_comparativa_210)
view(tabla_var_rw)

# MODELO LSTM 
vars_lstm <- pares
# Diferenciación
fx_diff <- fx_niveles |>
  dplyr::mutate(dplyr::across(dplyr::all_of(vars_lstm), ~ . - dplyr::lag(.))) |>
  tidyr::drop_na()

# Parámetros de la red
n_timesteps   <- 256     # solo 5 dias de historia
n_predictions <- H
n_features    <- length(vars_lstm)
batch_size    <- 64 


n_total    <- nrow(fx_diff)
n_train    <- round(n_total * 2 / 3)   # corte train/test
n_trn_puro <- round(n_train * 0.80)    # 80% para train, 20% para val interno

df_trn_puro  <- fx_diff[1:n_trn_puro, ]
df_trainval  <- fx_diff[1:n_train, ]   # train + val juntos
df_tst       <- fx_diff[(n_train + 1):n_total, ]

# Normalización (center + scale sobre train) 
# Evita data leakage de la validacion al calcular medias y desviaciones.
rec_obj <- recipes::recipe(
  EUR + GBP + JPY + CHF + CNY ~ index,
  data = df_trn_puro
) |>
  recipes::step_center(EUR, GBP, JPY, CHF, CNY) |>
  recipes::step_scale(EUR,  GBP, JPY, CHF, CNY) |>
  recipes::prep()

center_history <- rec_obj$steps[[1]]$means
scale_history  <- rec_obj$steps[[2]]$sds

# Aplicar la misma transformacion a los tres bloques y a toda la serie
df_trainval_proc <- recipes::bake(rec_obj, df_trainval)
fx_diff_proc     <- recipes::bake(rec_obj, fx_diff)

# Construcción de tensores
build_matrix <- function(serie_v, ancho) {
  n <- length(serie_v) - ancho + 1
  t(sapply(1:n, function(i) serie_v[i:(i + ancho - 1)]))
}

build_tensors <- function(df_proc, vars_v, n_ts, n_pred) {
  ventana <- n_ts + n_pred
  mats    <- lapply(vars_v, function(v) build_matrix(dplyr::pull(df_proc, v), ventana))
  n_obs   <- nrow(mats[[1]])
  
  X <- array(
    data = do.call(cbind, lapply(mats, function(m) m[, 1:n_ts])),
    dim  = c(n_obs, n_ts, length(vars_v))
  )
  y <- array(
    data = do.call(cbind, lapply(mats, function(m)
      m[, (n_ts + 1):(n_ts + n_pred)])),
    dim  = c(n_obs, n_pred, length(vars_v))
  )
  list(X = X, y = y)
}

recortar <- function(arr, bs) {
  n <- (dim(arr)[1] %/% bs) * bs
  if (n == 0) stop("Bloque demasiado pequeño para el batch_size indicado")
  arr[1:n, , ]
}

tensores_tv <- build_tensors(df_trainval_proc, vars_lstm, n_timesteps, n_predictions)

tensores_tv_X <- recortar(tensores_tv$X, batch_size)
tensores_tv_y <- recortar(tensores_tv$y, batch_size)

n_total_ventanas <- dim(tensores_tv_X)[1]
n_val_ventanas   <- round(n_total_ventanas * 0.20 / batch_size) * batch_size
n_trn_ventanas   <- n_total_ventanas - n_val_ventanas

X_tr  <- tensores_tv_X[1:n_trn_ventanas, , ]
y_tr  <- tensores_tv_y[1:n_trn_ventanas, , ]
X_val <- tensores_tv_X[(n_trn_ventanas + 1):n_total_ventanas, , ]
y_val <- tensores_tv_y[(n_trn_ventanas + 1):n_total_ventanas, , ]
# Arquitectura encoder-decoder 
inputs <- layer_input(shape = c(n_timesteps, n_features))

outputs <- inputs |>
  layer_lstm(
    units             = 8,      # solo 8 neuronas
    dropout           = 0.2,
    recurrent_dropout = 0.0,
    return_sequences  = FALSE
  ) |>
  layer_dense(units = n_predictions * n_features) |>
  layer_reshape(target_shape = c(n_predictions, n_features))

model_mv <- keras_model(inputs = inputs, outputs = outputs)
summary(model_mv)

model_mv |> compile(
  loss      = "mae",
  optimizer = optimizer_adam(learning_rate = 0.0005),
  metrics   = list("mean_absolute_error")
)

tic()
history_mv <- model_mv |> fit(
  x               = X_tr,
  y               = y_tr,
  validation_data = list(X_val, y_val),
  batch_size      = batch_size,
  epochs          = 100,
  callbacks       = list(
    callback_early_stopping(
      monitor              = "val_loss",
      patience             = 15,
      restore_best_weights = TRUE,
      min_delta            = 1e-5
    )
  )
)
toc()

plot(history_mv, metrics = "loss")
# Entrenamiento 
tic()
history_mv <- model_mv |> fit(
  x               = X_tr,
  y               = y_tr,
  validation_data = list(X_val, y_val),
  batch_size      = batch_size,
  epochs          = 100,
  callbacks       = list(
    callback_early_stopping(
      monitor              = "val_loss",
      patience             = 15,
      restore_best_weights = TRUE,
      min_delta            = 1e-5
    ),
    callback_reduce_lr_on_plateau(
      monitor  = "val_loss",
      factor   = 0.5,
      patience = 7,
      min_lr   = 1e-6
    )
  )
)
toc()

plot(history_mv, metrics = "loss")

# Curva de pérdida durante el entrenamiento
plot(history_mv, metrics = "loss")

# Evaluación rolling sobre test
# Los índices se basan en FECHAS (más robusto que índices numéricos frente a NAs).

desnorm <- function(z, var) {
  z * scale_history[[var]] + center_history[[var]]
}

reconstruir_niveles <- function(y_base, deltas) {
  cumsum(c(y_base, deltas))[-1]
}

n_test_proc <- nrow(fx_diff_proc) - n_train
n_ventanas  <- n_test_proc - n_predictions

resultados_rolling <- vector("list", n_ventanas)

cat(sprintf("LSTM evaluación rolling — %d ventanas...\n", n_ventanas))

for (i in seq_len(n_ventanas)) {
  
  idx_x_ini <- n_train - n_timesteps + i
  idx_x_fin <- n_train - 1 + i
  idx_y_ini <- idx_x_fin + 1
  idx_y_fin <- idx_x_fin + n_predictions
  
  # Verificar que los índices están dentro de rango 
  if (idx_y_fin > nrow(fx_diff_proc)) break
  
  X_window <- fx_diff_proc[idx_x_ini:idx_x_fin, vars_lstm] |>
    as.matrix() |>
    array(dim = c(1, n_timesteps, n_features))
  
  pred_z <- predict(model_mv, X_window, batch_size = 1)
  real_z <- fx_diff_proc[idx_y_ini:idx_y_fin, vars_lstm] |> as.matrix()
  
  # Nivel base: última fila de niveles ANTES de la ventana de predicción.
  # fx_diff fila k corresponde a fx_niveles fila k+1 → ajuste +1
  idx_nivel_base <- idx_x_fin + 1
  
  # Guardar usando la fecha del nivel base como ancla (robusto a NAs)
  fecha_base <- fx_niveles$index[idx_nivel_base]
  
  filas_h <- lapply(seq_along(vars_lstm), function(j) {
    v <- vars_lstm[j]
    
    pred_delta <- desnorm(pred_z[1, , j], v)
    real_delta <- desnorm(real_z[,  j],  v)
    y_base     <- fx_niveles[[v]][idx_nivel_base]
    
    pred_niv <- reconstruir_niveles(y_base, pred_delta)
    real_niv <- reconstruir_niveles(y_base, real_delta)
    
    tibble(
      ventana = i,
      fecha   = fecha_base + 1:n_predictions,   # días siguientes a la base
      h       = 1:n_predictions,
      divisa  = v,
      real    = real_niv,
      pred    = pred_niv
    )
  })
  
  resultados_rolling[[i]] <- dplyr::bind_rows(filas_h)
}

resultados_df <- dplyr::bind_rows(resultados_rolling)

# Métricas LSTM 
metricas_lstm <- resultados_df |>
  dplyr::group_by(Par = divisa, Horizonte = h) |>
  dplyr::summarise(
    MAE  = round(mean(abs(real - pred)),              6),
    RMSE = round(sqrt(mean((real - pred)^2)),          6),
    MAPE = round(mean(abs((real - pred) / real)) * 100, 4),
    .groups = "drop"
  ) |>
  dplyr::arrange(Par, Horizonte)

# Para cada horizonte h, el RMSE_RW escala con sqrt(h) aproximadamente
# pero lo más riguroso es usar el RMSE_RW del VAR (mismo benchmark, misma muestra)
# Se extrae directamente de tabla_rw que ya está calculada

rmse_rw_referencia <- tabla_rw |>
  dplyr::select(Par, Horizonte, RMSE_RW = RMSE)

metricas_lstm <- metricas_lstm |>
  dplyr::left_join(rmse_rw_referencia, by = c("Par", "Horizonte")) |>
  dplyr::select(-RMSE_RW)

print(metricas_lstm, n = 25)
# EVALUACIÓN COMPARATIVA GLOBAL

# VAR, RW y LSTM sobre sus respectivos esquemas completos (1049 / test LSTM).
# La columna n_orig documenta el tamaño muestral usado para cada RMSE/MAE/MAPE.
n_orig_bvar <- resultado_bvar$config$n_eval
n_orig_var  <- nrow(precios_ts) - resultado_var$config$W - H + 1

tabla_comparativa_final <- bind_rows(
  tabla_metricas_var  |> mutate(Modelo = "VAR",  n_orig = n_orig_var),
  tabla_metricas_bvar |> mutate(Modelo = "BVAR", n_orig = n_orig_bvar),
  metricas_lstm       |> mutate(Modelo = "LSTM", n_orig = n_ventanas),
  tabla_rw            |> select(Par, Horizonte, MAE, RMSE, MAPE, Modelo) |>
    mutate(n_orig = n_orig_var)
) |>
  select(Par, Horizonte, Modelo, MAE, RMSE, MAPE, n_orig) |>
  arrange(Par, Horizonte, Modelo)

print(tabla_comparativa_final)
cat("\nNOTA: n_orig indica el número de orígenes de predicción usados para\n")
cat("calcular cada fila. BVAR se evalúa sobre una submuestra (ver sección 6.6).\n")

# Identificar el mejor modelo por Par, Horizonte y métrica
mejor_por_metrica <- tabla_comparativa_final |>
  group_by(Par, Horizonte) |>
  summarise(
    mejor_MAE  = Modelo[which.min(MAE)],
    mejor_RMSE = Modelo[which.min(RMSE)],
    mejor_MAPE = Modelo[which.min(MAPE)],
    .groups = "drop"
  )

print(mejor_por_metrica)

#  Test de Diebold-Mariano (VAR vs RW, BVAR vs RW, LSTM vs RW)

resultado_rw <- resultado_var 
for (j in seq_along(pares)) {
  for (h_eval in 1:H) {
    resultado_rw$predicciones[, h_eval, j] <- resultado_var$ultimos_niveles[, j]
  }
}

# RW análogo, pero restringido a los 210 orígenes evaluados por el BVAR
resultado_rw_210 <- resultado_var_210  #210 x h x n_col
for (j in seq_along(pares)) {
  for (h_eval in 1:H) {
    resultado_rw_210$predicciones[, h_eval, j] <- resultado_var_210$ultimos_niveles[, j]
  }
}


tabla_dm <- map_dfr(seq_along(pares), function(j) {
  map_dfr(1:H, function(h_eval) {
    
    dm_var  <- dm_test(resultado_var,  resultado_rw,     h_eval = h_eval, divisa = j)
    dm_bvar <- dm_test(resultado_bvar, resultado_rw_210, h_eval = h_eval, divisa = j)
    
    # Para LSTM se construye resultado compatible
    pred_lstm_h <- metricas_lstm |>
      filter(Par == pares[j], Horizonte == h_eval) |>
      pull(RMSE)   # solo para referencia; DM completo requiere el vector de errores
    
    tibble(
      Par       = pares[j],
      Horizonte = h_eval,
      DM_VAR    = dm_var$DM_stat,
      p_VAR     = dm_var$p_valor,
      sig_VAR   = dm_var$sig,
      DM_BVAR   = dm_bvar$DM_stat,
      p_BVAR    = dm_bvar$p_valor,
      sig_BVAR  = dm_bvar$sig
    )
  })
})
view(tabla_dm)
print(tabla_dm,n=25)
cat("\nNOTA: DM_VAR se calcula sobre 1049 orígenes; DM_BVAR sobre los 210\n")
cat("orígenes submuestreados evaluados por el BVAR (ver sección 6.6).\n")

print(tabla_comparativa_final, n = 100)
print(mejor_por_metrica, n = 25)
print(tabla_dm, n = 25)



# VISUALIZACIONES

#  Series en niveles (2020–2024)
fx_niveles |>
  pivot_longer(cols = all_of(vars_lstm), names_to = "Divisa", values_to = "Precio") |>
  ggplot(aes(x = index, y = Precio)) +
  geom_line(color = "#2C5F8A", linewidth = 0.5) +
  facet_wrap(~ Divisa, scales = "free_y", ncol = 2) +
  labs(
    title    = "Tipos de cambio frente al USD (2020–2024)",
    subtitle = "Cierre diario — convención: USD por unidad de divisa",
    x = NULL, y = "Precio de cierre"
  ) +
  theme_minimal(base_size = 11) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

# Series en primeras diferencias
as.data.frame(apply(precios_ts, 2, diff)) |>
  mutate(t = seq_len(n())) |>
  pivot_longer(cols = all_of(pares), names_to = "Divisa", values_to = "Delta") |>
  ggplot(aes(x = t, y = Delta)) +
  geom_line(color = "#555555", linewidth = 0.4, alpha = 0.7) +
  facet_wrap(~ Divisa, scales = "free_y", ncol = 2) +
  labs(
    title    = "Primeras diferencias de los tipos de cambio",
    subtitle = "Δs_t = s_t − s_{t−1}",
    x = "Día", y = "Δ Precio"
  ) +
  theme_minimal(base_size = 11)

# MAPE por horizonte: todos los modelos 
tabla_comparativa_final |>
  ggplot(aes(x = Horizonte, y = MAPE, color = Modelo, linetype = Modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ Par, scales = "free_y", ncol = 3) +
  scale_color_manual(values = c(
    RW   = "#888888",
    VAR  = "#2CA25F",
    BVAR = "#E87722",
    LSTM = "#0070C0"
  )) +
  labs(
    title    = "MAPE por horizonte de predicción: RW vs VAR vs BVAR vs LSTM",
    subtitle = "Evaluación rolling — niveles originales (h = 1, …, 5 días)",
    x        = "Horizonte h (días)",
    y        = "MAPE (%)",
    color    = "Modelo",
    linetype = "Modelo"
  ) +
  theme_minimal(base_size = 11)

# RMSE por horizonte: todos los modelos 
tabla_comparativa_final |>
  ggplot(aes(x = Horizonte, y = RMSE, color = Modelo, linetype = Modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ Par, scales = "free_y", ncol = 3) +
  scale_color_manual(values = c(
    RW   = "#888888",
    VAR  = "#2CA25F",
    BVAR = "#E87722",
    LSTM = "#0070C0"
  )) +
  labs(
    title    = "RMSE por horizonte de predicción: RW vs VAR vs BVAR vs LSTM",
    subtitle = "Evaluación rolling — niveles originales (h = 1, …, 5 días)",
    x        = "Horizonte h (días)",
    y        = "RMSE",
    color    = "Modelo",
    linetype = "Modelo"
  ) +
  theme_minimal(base_size = 11)

# Predicciones vs valores reales LSTM (h = 1)
resultados_df |>
  filter(h == 1) |>
  ggplot(aes(x = fecha)) +
  geom_line(aes(y = real), color = "black", linewidth = 0.5) +
  geom_line(aes(y = pred), color = "#0070C0", linewidth = 0.5, alpha = 0.8) +
  facet_wrap(~ divisa, scales = "free_y", ncol = 2) +
  labs(
    title    = "LSTM — Predicciones vs valores reales (h = 1)",
    subtitle = "Negro: real | Azul: predicción LSTM | Evaluación rolling",
    x = NULL, y = "Precio de cierre (USD)"
  ) +
  theme_minimal(base_size = 11)

# Curva de pérdida LSTM
plot(history_mv, metrics = "loss",
     main = "Curva de pérdida (MAE) — Entrenamiento vs Validación")


# RESUMEN EJECUTIVO DE RESULTADOS


cat("\n\n══════════════════════════════════════════════\n")
cat("  RESUMEN EJECUTIVO — Tabla comparativa final\n")
cat("══════════════════════════════════════════════\n\n")
print(tabla_comparativa_final, n = Inf)

cat("\n── Mejor modelo por Par, Horizonte y métrica MAPE ──\n")
print(mejor_por_metrica, n = Inf)

cat("\n── Porcentaje de casos en que cada modelo supera al RW (RMSE) ──\n")
tabla_comparativa_final |>
  filter(Modelo != "RW") |>
  left_join(
    tabla_comparativa_final |>
      filter(Modelo == "RW") |>
      select(Par, Horizonte, RMSE_RW = RMSE),
    by = c("Par", "Horizonte")
  ) |>
  group_by(Modelo) |>
  summarise(
    pct_supera_RW = round(mean(RMSE < RMSE_RW) * 100, 1),
    .groups = "drop"
  ) |>
  print()

cat("\n── Test de Diebold-Mariano (vs RW) ──\n")
print(tabla_dm, n = Inf)

# ============================================================================
#  COMPARACIÓN UNIVARIANTE vs. MULTIVARIANTE — VAR y BVAR
#  VAR  → AR(p) por divisa con selección AIC (lag.max = 5)
#  BVAR → AR(1) bayesiano con prior de Minnesota univariante
#  Mismo protocolo rolling que Final.R (W = 252, H = 5)
# ============================================================================

# AR(p) rolling univariante (equivalente univariante del VAR)

ar_rolling_uni <- function(serie_vec, nombre_par, W, h = 1) {
  
  N          <- length(serie_vec)
  n_origenes <- N - W - h + 1
  if (n_origenes < 1) stop("W + h > longitud de la serie")
  
  predicciones    <- matrix(NA_real_, nrow = n_origenes, ncol = h)
  reales          <- matrix(NA_real_, nrow = n_origenes, ncol = h)
  ultimos_niveles <- numeric(n_origenes)
  
  for (i in seq_len(n_origenes)) {
    
    fin_train          <- W + i - 1L
    ventana_train      <- serie_vec[i:fin_train]          # ventana fija
    ventana_train_diff <- diff(ventana_train)              # I(0)
    ultimo_nivel       <- ventana_train[length(ventana_train)]
    ultimos_niveles[i] <- ultimo_nivel
    
    # Selección de lag por AIC (igual que VARselect en el VAR multivariante)
    lag_optimo <- tryCatch({
      sel <- ar(ventana_train_diff, aic = TRUE, order.max = 5,
                method = "ols", demean = FALSE)
      max(1L, sel$order)
    }, error = function(e) 1L)
    
    mod <- tryCatch(
      forecast::Arima(ventana_train_diff,
                      order         = c(lag_optimo, 0, 0),
                      include.mean  = TRUE),
      error = function(e) NULL
    )
    
    if (!is.null(mod)) {
      pred_diff        <- as.numeric(forecast::forecast(mod, h = h)$mean)
      predicciones[i, ] <- ultimo_nivel + cumsum(pred_diff)
    }
    
    reales[i, ] <- serie_vec[(fin_train + 1):(fin_train + h)]
  }
  
  # Calcular métricas por horizonte
  map_dfr(1:h, function(h_eval) {
    e <- reales[, h_eval] - predicciones[, h_eval]
    tibble(
      Par       = nombre_par,
      Horizonte = h_eval,
      MAE       = round(mean(abs(e),         na.rm = TRUE), 6),
      RMSE      = round(sqrt(mean(e^2,       na.rm = TRUE)), 6),
      MAPE      = round(mean(abs(e / reales[, h_eval]),
                             na.rm = TRUE) * 100, 4),
      Modelo    = "AR_uni"
    )
  })
}

# AR(1) bayesiano rolling univariante (equivalente univariante del BVAR)

bvar_uni_rolling <- function(serie_vec, nombre_par, W, h = 1,
                             iterations = 500,
                             burnin     = 200,
                             paso       = 5,    # mismo submuestreo que el BVAR multivariante
                             kappa0 = 0.04,
                             kappa1 = 0.5,
                             kappa2 = 0.5,
                             kappa3 = 5) {
  
  N          <- length(serie_vec)
  n_origenes <- N - W - h + 1
  if (n_origenes < 1) stop("W + h > longitud de la serie")
  
  # Mismo submuestreo de orígenes que rf_bvar(), para comparabilidad directa
  idx_eval <- seq(1L, n_origenes, by = paso)
  n_eval   <- length(idx_eval)
  
  predicciones    <- matrix(NA_real_, nrow = n_eval, ncol = h)
  reales          <- matrix(NA_real_, nrow = n_eval, ncol = h)
  tipos_modelo    <- character(n_eval)
  
  pb <- txtProgressBar(min = 0, max = n_eval, style = 3)
  
  for (idx in seq_len(n_eval)) {
    
    i <- idx_eval[idx]
    
    fin_train          <- W + i - 1L
    ventana_train      <- serie_vec[i:fin_train]
    ventana_train_diff <- diff(ventana_train)
    ultimo_nivel       <- ventana_train[length(ventana_train)]
    
    # Convertir a serie ts de una columna; gen_var() exige clase 'ts'
    ventana_mat <- matrix(ventana_train_diff, ncol = 1,
                          dimnames = list(NULL, nombre_par))
    ventana_mat <- ts(ventana_mat, frequency = 1)
    colnames(ventana_mat) <- nombre_par
    
    modelo_bvar_uni <- tryCatch({
      bvar_data <- bvartools::gen_var(ventana_mat,
                                      p             = 1,
                                      deterministic = "const",
                                      iterations    = iterations,
                                      burnin        = burnin)
      
      bvar_data <- bvartools::add_priors(bvar_data,
                                         coef = list(
                                           minnesota = list(kappa0 = kappa0,
                                                            kappa1 = kappa1,
                                                            kappa2 = kappa2,
                                                            kappa3 = kappa3)
                                         ))
      
      bvartools::draw_posterior(bvar_data)
    }, error = function(e) NULL)
    
    if (!is.null(modelo_bvar_uni)) {
      pred_bvar_uni <- predict(modelo_bvar_uni, n.ahead = h)
      # pred_bvar_uni$fcst es una lista nombrada por variable; cada elemento
      # es una matriz (h x 3) con columnas "2.5%", "50%", "97.5%".
      pred_diff         <- pred_bvar_uni$fcst[[nombre_par]][, "50%"]
      predicciones[idx, ] <- ultimo_nivel + cumsum(pred_diff)
      tipos_modelo[idx]   <- "BVAR"
      
    } else {
      # Fallback AR(1) frecuentista
      mod_ar <- tryCatch(
        forecast::Arima(ventana_train_diff, order = c(1, 0, 0),
                        include.mean = TRUE),
        error = function(e) NULL
      )
      if (!is.null(mod_ar)) {
        pred_diff           <- as.numeric(forecast::forecast(mod_ar, h = h)$mean)
        predicciones[idx, ] <- ultimo_nivel + cumsum(pred_diff)
      }
      tipos_modelo[idx] <- "AR1_fallback"
    }
    
    reales[idx, ] <- serie_vec[(fin_train + 1):(fin_train + h)]
    setTxtProgressBar(pb, idx)
  }
  close(pb)
  
  n_bvar     <- sum(tipos_modelo == "BVAR")
  n_fallback <- sum(tipos_modelo == "AR1_fallback")
  cat(sprintf("\n  [%s] Ventanas con BVAR real: %d / %d (%.1f%%)\n",
              nombre_par, n_bvar, n_eval, 100 * n_bvar / n_eval))
  cat(sprintf("  [%s] Ventanas con fallback AR(1): %d / %d (%.1f%%)\n",
              nombre_par, n_fallback, n_eval, 100 * n_fallback / n_eval))
  
  map_dfr(1:h, function(h_eval) {
    e <- reales[, h_eval] - predicciones[, h_eval]
    tibble(
      Par       = nombre_par,
      Horizonte = h_eval,
      MAE       = round(mean(abs(e),         na.rm = TRUE), 6),
      RMSE      = round(sqrt(mean(e^2,       na.rm = TRUE)), 6),
      MAPE      = round(mean(abs(e / reales[, h_eval]),
                             na.rm = TRUE) * 100, 4),
      Modelo    = "BAR_uni"   # Bayesian AR univariante
    )
  })
}


# Ejecución

W <- 252
H <- 5

# AR univariante rápido
cat("── AR(p) univariante rolling ──\n")
tic()
tabla_ar_uni <- map_dfr(pares, function(par) {
  cat(sprintf("  %s\n", par))
  ar_rolling_uni(
    serie_vec   = precios_ts[[par]],
    nombre_par  = par,
    W           = W,
    h           = H
  )
})
toc()

# BAR univariante lento 
cat("\n── BAR(1) univariante rolling ──\n")
tic()
tabla_bar_uni <- map_dfr(pares, function(par) {
  cat(sprintf("\n  %s\n", par))
  bvar_uni_rolling(
    serie_vec  = precios_ts[[par]],
    nombre_par = par,
    W          = W,
    h          = H,
    kappa0     = 0.04,
    kappa1     = 0.5,
    kappa2     = 0.5,
    kappa3     = 5,
    iterations = 500,
    burnin     = 200,
    paso       = 5
  )
})
toc()


# Tablas comparativas
# VAR multivariante vs AR univariante
comparacion_var <- bind_rows(
  tabla_metricas_var |> mutate(Modelo = "VAR_multi"),
  tabla_ar_uni
) |>
  select(Par, Horizonte, Modelo, MAE, RMSE, MAPE) |>
  arrange(Par, Horizonte, Modelo)

print(comparacion_var, n = 50)

# BVAR multivariante vs BAR univariante
# Ambos evaluados sobre los mismos 210 orígenes submuestreados (paso=5),
comparacion_bvar <- bind_rows(
  tabla_metricas_bvar |> mutate(Modelo = "BVAR_multi"),
  tabla_bar_uni
) |>
  select(Par, Horizonte, Modelo, MAE, RMSE, MAPE) |>
  arrange(Par, Horizonte, Modelo)

cat(sprintf("\n── BVAR_multi vs BAR_uni (n = %d orígenes submuestreados) ──\n",
            resultado_bvar$config$n_eval))
print(comparacion_bvar, n = 50)


#Ganancia relativa
ganancia_var <- comparacion_var |>
  select(Par, Horizonte, Modelo, RMSE) |>
  pivot_wider(names_from = Modelo, values_from = RMSE) |>
  mutate(
    ganancia_pct = round((AR_uni - VAR_multi) / AR_uni * 100, 2)
    # positivo  VAR multivariante mejor
  ) |>
  arrange(Par, Horizonte)

cat("\n── Ganancia relativa VAR_multi vs AR_uni (RMSE) ──\n")
print(ganancia_var, n = 25)

ganancia_bvar <- comparacion_bvar |>
  select(Par, Horizonte, Modelo, RMSE) |>
  pivot_wider(names_from = Modelo, values_from = RMSE) |>
  mutate(
    ganancia_pct = round((BAR_uni - BVAR_multi) / BAR_uni * 100, 2)
    # positivo BVAR multivariante mejor
  ) |>
  arrange(Par, Horizonte)

cat("\n── Ganancia relativa BVAR_multi vs BAR_uni (RMSE) ──\n")
print(ganancia_bvar, n = 25)


#mVisualizaciones
comparacion_var |>
  ggplot(aes(x = Horizonte, y = RMSE, color = Modelo, linetype = Modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ Par, scales = "free_y", ncol = 3) +
  scale_color_manual(values = c(VAR_multi = "#2CA25F", AR_uni = "#99D8C9")) +
  labs(
    title    = "VAR multivariante vs. AR univariante — RMSE por horizonte",
    subtitle = "Verde oscuro: 5 divisas | Verde claro: divisa individual",
    x        = "Horizonte h (días)", y = "RMSE",
    color = "Modelo", linetype = "Modelo"
  ) +
  theme_minimal(base_size = 11)

comparacion_bvar |>
  ggplot(aes(x = Horizonte, y = RMSE, color = Modelo, linetype = Modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ Par, scales = "free_y", ncol = 3) +
  scale_color_manual(values = c(BVAR_multi = "#E87722", BAR_uni = "#FEC89A")) +
  labs(
    title    = "BVAR multivariante vs. BAR univariante — RMSE por horizonte",
    subtitle = "Naranja oscuro: 5 divisas | Naranja claro: divisa individual",
    x        = "Horizonte h (días)", y = "RMSE",
    color = "Modelo", linetype = "Modelo"
  ) +
  theme_minimal(base_size = 11)



# ============================================================================
#  LSTM UNIVARIANTE — Comparación individual vs. multivariante
#  Se entrena un LSTM separado por cada divisa y se evalúa con rolling window
# ============================================================================

lstm_univariante <- function(var_name,
                             fx_niveles,
                             fx_diff_proc,
                             center_history,
                             scale_history,
                             n_train,
                             n_timesteps   = 256,
                             n_predictions = 5,
                             batch_size    = 64,
                             epochs        = 100,
                             patience      = 15) {
  
  # Tensores de entrenamiento (solo la divisa seleccionada) 
  n_features_uni <- 1L
  
  build_matrix_uni <- function(serie_v, ancho) {
    n <- length(serie_v) - ancho + 1
    t(sapply(1:n, function(i) serie_v[i:(i + ancho - 1)]))
  }
  
  # Extraer la serie procesada de la divisa concreta
  serie_proc <- fx_diff_proc[[var_name]]
  
  ventana <- n_timesteps + n_predictions
  mat     <- build_matrix_uni(serie_proc[1:n_train], ventana)
  n_obs   <- nrow(mat)
  
  X_all <- array(mat[, 1:n_timesteps],
                 dim = c(n_obs, n_timesteps, n_features_uni))
  y_all <- array(mat[, (n_timesteps + 1):(n_timesteps + n_predictions)],
                 dim = c(n_obs, n_predictions, n_features_uni))
  
  # Recortar al múltiplo de batch_size
  n_recorte <- (n_obs %/% batch_size) * batch_size
  X_all <- X_all[1:n_recorte, , , drop = FALSE]
  y_all <- y_all[1:n_recorte, , , drop = FALSE]
  
  # Split train / val interno (80/20)
  n_val <- round(n_recorte * 0.20 / batch_size) * batch_size
  n_trn <- n_recorte - n_val
  
  X_tr  <- X_all[1:n_trn, , , drop = FALSE]
  y_tr  <- y_all[1:n_trn, , , drop = FALSE]
  X_val <- X_all[(n_trn + 1):n_recorte, , , drop = FALSE]
  y_val <- y_all[(n_trn + 1):n_recorte, , , drop = FALSE]
  
  # Arquitectura (idéntica al multivariante, pero n_features = 1)
  inputs_u <- layer_input(shape = c(n_timesteps, n_features_uni))
  
  outputs_u <- inputs_u |>
    layer_lstm(
      units             = 8,
      dropout           = 0.2,
      recurrent_dropout = 0.0,
      return_sequences  = FALSE
    ) |>
    layer_dense(units = n_predictions * n_features_uni) |>
    layer_reshape(target_shape = c(n_predictions, n_features_uni))
  
  model_uni <- keras_model(inputs = inputs_u, outputs = outputs_u)
  
  model_uni |> compile(
    loss      = "mae",
    optimizer = optimizer_adam(learning_rate = 0.0005),
    metrics   = list("mean_absolute_error")
  )
  
  # Entrenamiento
  cat(sprintf("\n── Entrenando LSTM univariante: %s ──\n", var_name))
  
  model_uni |> fit(
    x               = X_tr,
    y               = y_tr,
    validation_data = list(X_val, y_val),
    batch_size      = batch_size,
    epochs          = epochs,
    verbose         = 0,          # suprimir output por epoch
    callbacks       = list(
      callback_early_stopping(
        monitor              = "val_loss",
        patience             = patience,
        restore_best_weights = TRUE,
        min_delta            = 1e-5
      ),
      callback_reduce_lr_on_plateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 7,
        min_lr   = 1e-6
      )
    )
  )
  
  # Evaluación rolling 
  n_test_proc <- nrow(fx_diff_proc) - n_train
  n_ventanas  <- n_test_proc - n_predictions
  
  desnorm_uni <- function(z) z * scale_history[[var_name]] + center_history[[var_name]]
  
  reconstruir <- function(y_base, deltas) cumsum(c(y_base, deltas))[-1]
  
  resultados <- vector("list", n_ventanas)
  
  for (i in seq_len(n_ventanas)) {
    
    idx_x_ini <- n_train - n_timesteps + i
    idx_x_fin <- n_train - 1 + i
    idx_y_ini <- idx_x_fin + 1
    idx_y_fin <- idx_x_fin + n_predictions
    
    if (idx_y_fin > nrow(fx_diff_proc)) break
    
    # Tensor de entrada (1, n_timesteps, 1)
    X_window <- array(
      fx_diff_proc[[var_name]][idx_x_ini:idx_x_fin],
      dim = c(1, n_timesteps, 1)
    )
    
    pred_z <- predict(model_uni, X_window, batch_size = 1, verbose = 0)
    real_z <- fx_diff_proc[[var_name]][idx_y_ini:idx_y_fin]
    
    idx_nivel_base <- idx_x_fin + 1
    y_base         <- fx_niveles[[var_name]][idx_nivel_base]
    
    pred_niv <- reconstruir(y_base, desnorm_uni(pred_z[1, , 1]))
    real_niv <- reconstruir(y_base, desnorm_uni(real_z))
    
    resultados[[i]] <- tibble(
      ventana = i,
      h       = 1:n_predictions,
      divisa  = var_name,
      real    = real_niv,
      pred    = pred_niv
    )
  }
  
  bind_rows(resultados)
}

# Ejecutar para las cinco divisas
set.seed(42)

vars_lstm <- c("EUR", "GBP", "JPY", "CHF", "CNY")

resultados_uni <- map(vars_lstm, function(v) {
  lstm_univariante(
    var_name       = v,
    fx_niveles     = fx_niveles,
    fx_diff_proc   = fx_diff_proc,    # ya calculado en Final.R
    center_history = center_history,
    scale_history  = scale_history,
    n_train        = n_train,
    n_timesteps    = 256,
    n_predictions  = H,
    batch_size     = 64
  )
}) |> bind_rows()

# Métricas LSTM univariante
metricas_lstm_uni <- resultados_uni |>
  group_by(Par = divisa, Horizonte = h) |>
  summarise(
    MAE  = round(mean(abs(real - pred)),               6),
    RMSE = round(sqrt(mean((real - pred)^2)),           6),
    MAPE = round(mean(abs((real - pred) / real)) * 100, 4),
    .groups = "drop"
  ) |>
  arrange(Par, Horizonte)

# Tabla comparativa: univariante vs. multivariante 

comparacion_uni_multi <- bind_rows(
  metricas_lstm_uni |> mutate(Modelo = "LSTM_uni"),
  metricas_lstm     |> mutate(Modelo = "LSTM_multi")
) |>
  select(Par, Horizonte, Modelo, MAE, RMSE, MAPE) |>
  arrange(Par, Horizonte, Modelo)

print(comparacion_uni_multi, n = 50)

# Ganancia relativa del multivariante sobre el univariante
comparacion_uni_multi |>
  select(Par, Horizonte, Modelo, RMSE) |>
  pivot_wider(names_from = Modelo, values_from = RMSE) |>
  mutate(
    ganancia_pct = round((LSTM_uni - LSTM_multi) / LSTM_uni * 100, 2)
    # positivo -> el multivariante es mejor (RMSE menor)
  ) |>
  arrange(Par, Horizonte) |>
  print(n = 25)

# Visualización: RMSE por horizonte, uni vs. multi 
comparacion_uni_multi |>
  ggplot(aes(x = Horizonte, y = RMSE, color = Modelo, linetype = Modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ Par, scales = "free_y", ncol = 3) +
  scale_color_manual(values = c(
    LSTM_uni   = "#0070C0",
    LSTM_multi = "#C00000"
  )) +
  labs(
    title    = "LSTM univariante vs. multivariante — RMSE por horizonte",
    subtitle = "Rojo: 5 divisas simultáneas | Azul: divisa individual",
    x        = "Horizonte h (días)",
    y        = "RMSE",
    color    = "Modelo",
    linetype = "Modelo"
  ) +
  theme_minimal(base_size = 11)

resultados_df |>
  filter(h == 1) |>
  ggplot(aes(x = fecha)) +
  geom_line(aes(y = real), color = "black", linewidth = 0.5) +
  geom_line(aes(y = pred), color = "#0070C0", linewidth = 0.5, alpha = 0.8) +
  facet_wrap(~ divisa, scales = "free_y", ncol = 2)
plot(history_mv, metrics = "loss")

fx_niveles |>
  pivot_longer(cols = all_of(vars_lstm), names_to = "Divisa", values_to = "Precio") |>
  ggplot(aes(x = index, y = Precio)) +
  geom_line(color = "#2C5F8A", linewidth = 0.5) +
  facet_wrap(~ Divisa, scales = "free_y", ncol = 2) +
  labs(
    title    = "Tipos de cambio frente al USD (2020–2024)",
    subtitle = "Cierre diario — convención: USD por unidad de divisa extranjera",
    x = NULL, y = "Precio de cierre"
  ) +
  theme_minimal(base_size = 11) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

# Tabla resumen global uni vs. multi 

tabla_resumen_uni_multi <- bind_rows(
  tabla_ar_uni        |> mutate(Especificación = "Univariante"),
  tabla_metricas_var  |> mutate(Modelo = "VAR",  Especificación = "Multivariante"),
  tabla_bar_uni       |> mutate(Especificación = "Univariante"),
  tabla_metricas_bvar |> mutate(Modelo = "BVAR", Especificación = "Multivariante"),
  metricas_lstm_uni   |> mutate(Modelo = "LSTM", Especificación = "Univariante"),
  metricas_lstm       |> mutate(Modelo = "LSTM", Especificación = "Multivariante")
) |>
  select(Modelo, Especificación, Par, Horizonte, RMSE) |>
  arrange(Modelo, Par, Horizonte, Especificación)

print(tabla_resumen_uni_multi, n = 60)
cat("\nNOTA: BVAR y BAR_uni se evalúan sobre 210 orígenes submuestreados;\n")
cat("VAR, AR_uni y LSTM sobre sus esquemas de evaluación completos.\n")
cat("La comparación BVAR_multi vs BAR_uni es internamente consistente (mismos\n")
cat("210 orígenes para ambos), pero no es directamente comparable en magnitud\n")
cat("absoluta con las filas de VAR/AR_uni (1049 orígenes). Ver sección 6.6.\n")
