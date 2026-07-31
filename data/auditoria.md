```
============================================================================================
AUDITORIA NUMERICA — Bad IP Detector (BID)
Archivo: BID_dataset.csv   |   Filas: 970   |   Columnas: 15
Columna score: abuseip_score   |   Columna objetivo: is_malicious
============================================================================================


VERIFICACION                                          DECLARADO         OBSERVADO                     VEREDICTO
-----------------------------------------------------------------------------------------------------------------
Total de registros                                    970               970                           PASS
Registros benignos                                    780               780                           PASS
Registros maliciosos                                  190               190                           PASS
Media de abuseip_score                                19.54             0.1237                        FAIL
Desv. estandar de abuseip_score                       39.63             1.7599                        FAIL
% global con score = 0                                75.0%             98.97% (960/970)              FAIL
Benignas con score > 0 (falsos positivos)             6                 6                             PASS
Maliciosas con score = 0 (escenario APT)              186/190           186/190                       PASS
Maliciosas con score en [99.0, 100.0]                 -                 0/190 (0.00%)                 INFO
    -> El articulo afirma que el segmento malicioso salta a valores maximos.
Coherencia: media compatible con n=186 en cero        <= 1.03           0.1237                        PASS
    -> Si 186/190 maliciosas estuvieran en 0 y solo 6 benignas fueran > 0, la media global no podria superar ese techo aritmetico.
Registros en score ~100 implicados por la media       -                 ~1 de 970 (0.12%)             INFO
    -> Estimacion bajo distribucion bimodal 0/100.
--- Origenes alternativos del n = 186 ---             -                                               INFO
Maliciosas con score nulo/NaN                         -                 0                             INFO
Maliciosas con score = 0 o nulo                       -                 186                           INFO
Maliciosas con score < 1                              -                 186                           INFO
Maliciosas con score < 50                             -                 190                           INFO
P(maliciosa) base                                     19.58%            19.59%                        PASS
P(maliciosa | Data Center/Hosting)                    30.65%            30.49% (n=551)                PASS
Denominador implicado por recall CatBoost             -                 83.87% x 186 = 156.00         INFO
    -> Un entero exacto confirma que el 186 se calculo sobre datos reales.
Denominador implicado por recall Baseline             -                 85.48% x 186 = 158.99         INFO
    -> Un entero exacto confirma que el 186 se calculo sobre datos reales.
-----------------------------------------------------------------------------------------------------------------
Total verificaciones: 11 | FAIL: 3

LECTURA: cada FAIL es una cifra del articulo que el dataset no respalda.
Corrige el texto para que coincida con la columna 'OBSERVADO'.
```
