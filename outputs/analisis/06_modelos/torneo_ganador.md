# Resultado del ranking de nowcast

**Objetivo principal:** anticipacion contemporanea del cambio mensual.

- **Modelo adoptado:** Random Forest
- Rango promedio por R2 en desarrollo: 1.50
- R2 mediano en desarrollo: -0.097
- Ranking calculado sobre 4 productos con muestras suficientes: Papa, Arroz, Aceite, Huevos
- Excluidos del ranking por no alcanzar los mínimos muestrales: Cebolla, Zanahoria
- Mínimos por producto para rankear: 60 observaciones de entrenamiento y 20 de evaluación
- Holdout: 3 de 6 productos con R2 > 0; 3 con acierto > 55 %
- Compuerta: superada

### Holdout por producto

| Producto | R2 | Acierto | n |
|---|---:|---:|---:|
| Papa | 0.270 | 69.5 % | 60 |
| Cebolla | 0.218 | 64.4 % | 60 |
| Zanahoria | 0.080 | 52.9 % | 52 |
| Arroz | -0.842 | 41.9 % | 31 |
| Aceite | -0.103 | 50.0 % | 24 |
| Huevos | -0.013 | 55.3 % | 38 |

## Justificacion

La ventana de desarrollo es 2018-2020 (36 meses calendario) y se usa para evaluar/rankear; cada modelo se entrena con las observaciones disponibles antes de 2018. Solo se rankean productos con al menos 60 observaciones de entrenamiento y 20 de evaluación. El R2 es la metrica principal del nowcast porque el MAE sobre cambios favorece predecir cero. El holdout 2021-01 a 2026-10 se uso una sola vez; noviembre queda fuera porque no se verifico su fecha de cierre/publicacion.

Por que no gana LightGBM si tiene mayor acierto en el holdout? LightGBM acierta el signo en 62.2 % de los meses, frente a 58.0 % de Random Forest, pero el acierto es una lectura binaria: trata igual un error de 0,1 % que uno de 5 %. El ganador se eligio en desarrollo con R2, donde Random Forest tiene rango promedio 1.50 y acierto medio 46.9 %, contra 4.00 y 42.8 % de LightGBM. En el holdout Random Forest tiene menor MAE que LightGBM en 5 de 6 productos y R2 positivo en papa, cebolla y zanahoria; LightGBM tiene R2 muy negativo en arroz y zanahoria. Cambiar el ganador segun el holdout seria seleccionar sobre el conjunto de prueba. La matriz de confusion se reporta como diagnostico complementario, no como criterio de seleccion.

## Pronostico a un mes: holdout

Cambio porcentual del MAE frente a la caminata aleatoria: positivo = mejora; negativo = empeora.

| Producto | N | OLS | Ridge | Random Forest | LightGBM |
|---|---:|---:|---:|---:|---:|
| Papa | 62 | +14.5 % | +12.7 % | -74.2 % | -85.6 % |
| Cebolla | 62 | +8.3 % | +9.9 % | -2.8 % | -12.5 % |
| Zanahoria | 53 | -4.7 % | -7.7 % | -62.0 % | -59.8 % |
| Arroz | 36 | +3.9 % | +7.7 % | -282.0 % | -353.2 % |
| Aceite | 24 | -37.8 % | -36.6 % | -303.1 % | -306.8 % |
| Huevos | 38 | -3.6 % | -3.3 % | -146.4 % | -142.3 % |

Ridge y OLS mejoran en Papa, Cebolla, Arroz; ambos empeoran en Zanahoria, Aceite, Huevos. Random Forest y LightGBM empeoran frente al baseline en los seis productos. La referencia de Lima aporta valor contemporaneo, pero no ofrece un pronostico general a un mes.
