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

## Pronostico a un mes: holdout

Cambio porcentual del MAE frente a la caminata aleatoria: positivo = mejora; negativo = empeora.

| Producto | N | OLS | Ridge | Random Forest | LightGBM |
|---|---:|---:|---:|---:|---:|
| Papa | 62 | +14.0 % | +12.2 % | -65.7 % | -58.8 % |
| Cebolla | 62 | +8.2 % | +9.7 % | -8.4 % | -14.1 % |
| Zanahoria | 53 | +0.7 % | -1.9 % | -65.0 % | -73.0 % |
| Arroz | 36 | +6.3 % | +9.9 % | -282.8 % | -351.0 % |
| Aceite | 24 | -33.9 % | -33.1 % | -303.3 % | -307.6 % |
| Huevos | 38 | +6.7 % | +6.2 % | -148.0 % | -142.4 % |

Ridge y OLS mejoran en papa, arroz, cebolla y huevos; OLS mejora marginalmente en zanahoria (+0,7 %), mientras Ridge empeora (−1,9 %); ambos empeoran en aceite. Random Forest y LightGBM empeoran frente al baseline en los seis productos. La referencia de Lima aporta valor contemporaneo, pero no ofrece un pronostico general a un mes.
