# Resultado del ranking de estimación contemporánea

**Objetivo principal:** anticipacion contemporanea del cambio mensual.

- **Modelo adoptado:** Random Forest
- Rango promedio por R² en validación: 2.00
- R² mediano en validación: -0.097
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

La ventana de validación es 2018-2020 (36 meses calendario) y se usa para evaluar/rankear; cada modelo se entrena con las observaciones disponibles de 2012 a 2017. Solo se rankean productos con al menos 60 observaciones de entrenamiento y 20 de evaluación. El R² es la métrica principal de la estimación contemporánea porque el MAE sobre cambios favorece predecir cero. La prueba final 2021-01 a 2026-10 se usó una sola vez; noviembre queda fuera porque no se verificó su fecha de cierre/publicacion.

¿Por qué no gana LightGBM si tiene mayor acierto en la prueba final? LightGBM acierta el signo en 62.2 % de los meses, frente a 58.0 % de Random Forest, pero el acierto es una lectura binaria: trata igual un error de 0,1 % que uno de 5 %. El ganador se eligió en validación con R², donde Random Forest tiene rango promedio 2.00 y acierto medio 46.9 %, contra 5.75 y 42.8 % de LightGBM. En la prueba final Random Forest tiene menor MAE que LightGBM en 5 de 6 productos y R² positivo en papa, cebolla y zanahoria; LightGBM tiene R² muy negativo en arroz y zanahoria. Cambiar el ganador según la prueba final sería seleccionar sobre el conjunto de prueba. La matriz de confusión se reporta como diagnóstico complementario, no como criterio de selección.

## Pronóstico a un mes: prueba final

Cambio porcentual del MAE frente a la caminata aleatoria: positivo = mejora; negativo = empeora.

| Producto | N | OLS | Ridge | Random Forest | LightGBM |
|---|---:|---:|---:|---:|---:|
| Papa | 62 | +14.5 % | +12.7 % | -74.2 % | -85.6 % |
| Cebolla | 62 | +8.3 % | +9.9 % | -2.8 % | -12.5 % |
| Zanahoria | 53 | -4.7 % | -7.7 % | -62.0 % | -59.8 % |
| Arroz | 36 | +3.9 % | +7.7 % | -282.0 % | -353.2 % |
| Aceite | 24 | -37.8 % | -36.6 % | -303.1 % | -306.8 % |
| Huevos | 38 | -3.6 % | -3.3 % | -146.4 % | -142.3 % |

Ridge y OLS mejoran en Papa, Cebolla, Arroz; ambos empeoran en Zanahoria, Aceite, Huevos. Random Forest y LightGBM empeoran frente al baseline en los seis productos. La referencia de Lima aporta valor contemporáneo, pero no ofrece un pronóstico general a un mes.
