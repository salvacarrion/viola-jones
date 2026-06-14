# Viola-Jones: Real Workflow

Guía rápida del ciclo end-to-end que se usa realmente: preparar datos, entrenar una cascade base, iterar con curación (score + filtro + reservoir de hard-negs raw), evaluar, diagnosticar, ajustar y detectar. Lectura ~5 min, ejecución mínima ~30 min (cold start, recipe 19×19 rápida) o ~12-40 h (recipe iteración completa).

Para el "por qué" detrás de cada flag y los fallos que motivaron cada fix, ver [FINDINGS.md](FINDINGS.md). Para resultados numéricos por experimento, [RESULTS.md](RESULTS.md).

---

## Pipeline en una vista

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ prepare_data.py │───▶│ data/<res>_<src>/   │───▶│ main.py train       │
│ (HF → NPY)      │    │   train_pos, val,   │    │ (cold start)        │
└─────────────────┘    │   neg_seed, pool    │    └──────────┬──────────┘
                       └─────────────────────┘               │
                                                             ▼
                                                  weights/<res>/baseline.pkl
                                                             │
            ┌────────────────────────────────────────────────┤
            ▼                                                ▼
┌─────────────────────┐  oracle  ┌─────────────────────┐    ┌──────────────────┐
│ score_faces.py      │◀────────│ mine_hard_negs_raw  │    │ test + diagnose  │
│ (rank positives)    │   (cbcl │ (stream from HF)    │    │ + tune_thresholds│
└──────────┬──────────┘   _v2)  └──────────┬──────────┘    └──────────────────┘
           │                                │
           ▼                                ▼
  data/<dir>/face_scores      weights/<res>/<stem>__vhardneg_raw.npy
  + score_samples.png
           │                                │
           └──────────────┬─────────────────┘
                          ▼
       ┌──────────────────────────────────────┐
       │ main.py train                        │
       │   --drop-low-score-pos FRAC          │
       │   --very-hard-neg-pool <vhard.npy>   │
       └──────────────────┬───────────────────┘
                          ▼
              weights/<res>/<run_v3>.pkl
                          │
                          ▼
              detect + tune + iterate
```

---

## 1. Preparar datos

Una vez por combinación `(resolución, source de positivos)`. Salida bajo `data/<res>_<src>/` con `train_pos.npy`, `val_pos.npy`, `caltech_pool.npy`, `neg_seed.npy`, `test_pos.npy`, `test_neg.npy`, `manifest.json`.

**Recipe 19×19 rápido (CBCL):**

```bash
python tools/prepare_data.py \
    --face-source cbcl \
    --neg-source mixed \
    --resolution 19 \
    --augment --jitter 1 \
    --benchmark cbcl --val-size 500 \
    --pool-size 50000000 \
    --out-dir data/19_cbcl
```

**Recipe con CelebA alineado (más caras, alignment frontal validada):**

```bash
python tools/prepare_data.py \
    --face-source celeba_aligned --n-faces 5000 \
    --neg-source mixed --resolution 19 \
    --augment --jitter 1 \
    --benchmark cbcl --val-size 500 \
    --pool-size 50000000 \
    --out-dir data/19_celeba_aligned
```

**Recipe mixed (CelebA aligned + CBCL):**

```bash
python tools/prepare_data.py \
    --face-source celeba_aligned+cbcl --n-faces 5000 \
    --neg-source mixed --resolution 19 \
    --augment --jitter 1 \
    --benchmark cbcl --val-size 500 \
    --pool-size 50000000 \
    --out-dir data/19_celeba_aligned+cbcl
```

Inspección visual rápida de un dataset preparado (sanity check de buckets):

```bash
python tools/inspect_dataset.py --out-dir samples/19_celeba_aligned
```

---

## 2. Entrenar baseline (cold start)

Sin oracle aún, primera cascade base sobre el dataset preparado. Es la que después usaremos como oracle para puntuar y minar.

```bash
python main.py train \
    --data-dir data/19_cbcl \
    --max-stages 20 \
    --max-wcs-per-stage 400 \
    --target-stage-fpr 0.5 \
    --min-cascade-recall 0.95 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 \
    --min-stage-negatives 1000
```

Hiperparámetros explicados en [FINDINGS.md §Adaptive cascade](FINDINGS.md#7-adaptive-cascade--calibrated-fpr--recall). El checkpoint vivo (`weights/19/cvj_weights_<ts>.pkl`) se sobrescribe tras cada stage, si se interrumpe en hora 5/15 el archivo sigue siendo una cascade utilizable parcial.

Cuando termine, renombra a algo legible:

```bash
mv weights/19/cvj_weights_<ts>.pkl weights/19/cbcl__19_v1.pkl
```

---

## 3. Test + diagnose + tune (siempre tras entrenar)

Tres pasos baratos (<1 min cada uno) que se ejecutan **siempre** sobre un modelo nuevo. Sin ellos no sabes si el modelo es decente.

```bash
# Test sobre el benchmark CBCL (472 caras / 23K no-caras)
python main.py test \
    --data-dir data/19_cbcl \
    --weights-path weights/19/cbcl__19_v1.pkl

# Per-stage pass rates + score distributions
python tools/diagnose_cascade.py \
    --weights weights/19/cbcl__19_v1.pkl \
    --data-dir data/19_cbcl

# Greedy F1 sweep sobre per-stage thresholds (no reentrena)
python tools/tune_thresholds.py \
    --weights weights/19/cbcl__19_v1.pkl \
    --data-dir data/19_cbcl \
    --objective f1
# → escribe cbcl__19_v1_tuned.pkl

# Test tras tuning
python main.py test \
    --data-dir data/19_cbcl \
    --weights-path weights/19/cbcl__19_v1_tuned.pkl
```

El tuned F1 es el que reportas. La diferencia raw↔tuned (~5-7 pp en 19×19) mide cuánto deja el calibration on-the-fly sobre la mesa frente a una optimización global posterior.

**Alternativas de objetivo del tuner:**

- `--objective f1`: para benchmark / comparativas (default).
- `--objective recall-at-spec --min-spec 0.97`: para "encontrar todas las caras posibles" con spec mínima dada (uso real de detección).

---

## 4. Curación de positivos (score + visualización + filtro)

Esto es lo que se hace cuando un dataset tiene ruido de alineación (típicamente CelebA, incluso tras "alignment"). Usa el baseline CBCL como oracle frozen para rankear cada cara de un dataset distinto.

**4.1. Score positivos** (~3 min para 20K caras):

```bash
python tools/score_faces.py \
    --weights weights/19/cbcl__19_v1.pkl \
    --data-dir data/19_celeba_aligned \
    --save-samples 16
```

Produce:
- `data/19_celeba_aligned/face_scores.npy`: float32 por cara, suma de márgenes per-stage (no short-circuit).
- `data/19_celeba_aligned/score_samples.png`: grid con 8 bandas de percentiles (p00-p05 worst, ..., p95-p100 best), borde verde si la cascade pasa la cara y rojo si la rechaza, score amarillo por cara.

Mira el PNG. Decide la fracción del bottom a descartar según lo que veas:
- Si el bottom 10-20% son crops claramente no-canónicos (ojos descentrados, mucha frente o cuello, perfiles parciales) → drop 0.20.
- Si el bottom 20-30% sigue siendo "feo" → drop 0.30.
- Si las primeras bandas son caras razonables → drop 0.10 o no filtres en absoluto.

El output del comando imprime también:
- `passed/rejected`: % exacto que la cascade acepta/rechaza con `classify()` (criterio determinista: distinto del signo del score).
- Histograma de percentiles del score continuo (ranking signal).

**4.2. Mining de very-hard negatives sobre raw HF** (~2-6h, depende de la fuerza del oracle):

```bash
python tools/mine_hard_negatives_raw.py \
    --weights weights/19/cbcl__19_v1.pkl \
    --target 15000 \
    --budget 500000000 \
    --patches-per-image 200 \
    --out weights/19/cbcl__19_v1__vhardneg_raw.npy
```

Streamea patches random desde `ds["negatives"]` (Caltech raw del HF dataset) hasta llegar a `--target` o agotar `--budget`. Sin pool intermedio en disco; multi-pass cuando es necesario. Si la salida es menor que el target no es problema, el trainer la usa como reservoir de top-up.

Diferencia clave vs el legacy [tools/mine_hard_negatives.py](tools/mine_hard_negatives.py): aquel mina del `caltech_pool.npy` finito (~50M patches); este streamea raw sin esa cota. Usa el nuevo cuando el cascade es fuerte y el pool finito no rinde.

---

## 5. Re-entrenar con curación

Misma config que el baseline pero con los dos flags nuevos:

```bash
python main.py train \
    --data-dir data/19_celeba_aligned \
    --max-stages 20 \
    --max-wcs-per-stage 400 \
    --target-stage-fpr 0.5 \
    --min-cascade-recall 0.95 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 \
    --min-stage-negatives 1000 \
    --drop-low-score-pos 0.20 \
    --very-hard-neg-pool weights/19/cbcl__19_v1__vhardneg_raw.npy
```

Lo que cambia internamente:
- `--drop-low-score-pos 0.20` descarta el bottom 20% de `train_pos` ordenado por `face_scores.npy`. Cache de features se rota a `xf_pos__drop0.20.npy` para no contaminar el cache full-set.
- `--very-hard-neg-pool` carga el reservoir y lo usa **sólo** como top-up cuando seed + caltech mining se queda corto en una stage (típicamente stages 12+).

Tras entrenar: renombrar, test, diagnose, tune, test del tuned, mismo ciclo del §3.

---

## 6. Extender stages (resume)

Si una cascade capó stages con `final_fpr > target_stage_fpr` (capacity ceiling, ver [FINDINGS.md](FINDINGS.md#1919-capacity-ceiling-jitter-saturates-the-cascade)), se puede resumir con un target relajado:

```bash
python main.py train \
    --data-dir data/19_cbcl \
    --resume-from weights/19/cbcl__19_v1.pkl \
    --max-stages 20 \
    --max-wcs-per-stage 800 \
    --target-stage-fpr 0.65 \
    --min-cascade-recall 0.95 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 \
    --min-stage-negatives 1000
```

Si en el resume una stage satura (FPR > nuevo target) y quieres descartarla antes de continuar:

```bash
python tools/truncate_checkpoint.py \
    --weights weights/19/cvj_weights_<ts>.pkl \
    --keep-stages 12 \
    --out weights/19/cbcl__19_v1.1.pkl
```

---

## 7. Detect sobre imágenes reales

```bash
python main.py detect \
    --detect-images images/people.png images/judybats.jpg \
    --detect-output images/outputs/cbcl__19_v1 \
    --weights-path weights/19/cbcl__19_v1.pkl \
    --nms-threshold 0.2 \
    --nms-metric hybrid \
    --detect-min-face 30 \
    --detect-scale 1.3
```

**Recomendaciones por experiencia:**

- **Usa el modelo raw para detección**, no el `_tuned.pkl`. El tuned está optimizado para F1 sobre patches del benchmark; en sliding-window con prior negativo abrumador, los thresholds relajados meten más FPs.
- **`--nms-threshold 0.2`** + **`--nms-metric hybrid`** (default): fusiona duplicados anidados (mismo rostro detectado a escalas 1×, 1.5×, 2×). Ver [FINDINGS.md §6](FINDINGS.md#6-hybrid-nms-for-multi-scale-duplicates).
- **`--detect-min-face 30`** para retratos / caras grandes: elimina FPs pequeños de fondo. Para fotos de clase o multitudes, dejar None o bajar a 15.
- **`--detect-scale 1.3`**: menos niveles de pirámide que el default 1.25, ~30% más rápido con pérdida marginal de recall.
- **`--detect-min-score 0.1-0.5`**: descarta detections con cumulative margin bajo (probablemente FPs).

---

## 8. Gotchas habituales

- **Cache de features stale**: cambiar `--face-source` sin borrar `data/<res>_<src>/_cache/` reutiliza features viejos con `train_pos` nuevo (mismatch silencioso). Solución: `rm -rf data/<res>_<src>/_cache/` antes de reentrenar con datos distintos. El filtro `--drop-low-score-pos` ya rota el cache automáticamente.
- **Stale `cvj_weights_*.pkl`**: scripts de extension (`scripts/run_19_extend.sh`) refusan arrancar si hay un `cvj_weights_*.pkl` en `weights/<res>/` sin renombrar. Limpia o renombra antes.
- **`auto-pick` de pesos**: si omites `--weights-path` en test/detect, se usa el `.pkl` más reciente por mtime bajo `weights/`. Útil para iterar; peligroso si tienes varios runs paralelos.
- **`--resume-from` valida resolución**: el checkpoint resume sólo si `clf.base_width == res` del data-dir; mismatch falla limpio.
- **`min_stage_negatives 1000`**: si el mining devuelve menos, la cascade para con un mensaje claro. Es el síntoma típico de pool agotado en stages tardías: solución: pre-minar very-hard reservoir con [tools/mine_hard_negatives_raw.py](tools/mine_hard_negatives_raw.py) y pasar via `--very-hard-neg-pool`.

---

## 9. Recipe end-to-end completa (copy-paste)

Para un experimento "celeba_aligned mejorado con oracle CBCL":

```bash
# 1. Prepare data (~10-15 min)
python tools/prepare_data.py --face-source cbcl --neg-source mixed \
    --resolution 19 --augment --jitter 1 --benchmark cbcl --val-size 500 \
    --pool-size 50000000 --out-dir data/19_cbcl
python tools/prepare_data.py --face-source celeba_aligned --n-faces 5000 \
    --neg-source mixed --resolution 19 --augment --jitter 1 \
    --benchmark cbcl --val-size 500 --pool-size 50000000 \
    --out-dir data/19_celeba_aligned

# 2. Train CBCL baseline (oracle) (~6h)
python main.py train --data-dir data/19_cbcl --max-stages 20 \
    --max-wcs-per-stage 400 --target-stage-fpr 0.5 \
    --min-cascade-recall 0.95 --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 --min-stage-negatives 1000
mv weights/19/cvj_weights_*.pkl weights/19/cbcl__19_v1.pkl

# 3. Score CelebA positives + visualize
python tools/score_faces.py --weights weights/19/cbcl__19_v1.pkl \
    --data-dir data/19_celeba_aligned --save-samples 16
# → revisar data/19_celeba_aligned/score_samples.png antes de fijar FRAC

# 4. Pre-mine very-hard negs desde HF raw (~2-6h)
python tools/mine_hard_negatives_raw.py \
    --weights weights/19/cbcl__19_v1.pkl --target 15000 \
    --out weights/19/cbcl__19_v1__vhardneg_raw.npy

# 5. Train CelebA con curación (~5-10h)
python main.py train --data-dir data/19_celeba_aligned --max-stages 20 \
    --max-wcs-per-stage 400 --target-stage-fpr 0.5 \
    --min-cascade-recall 0.95 --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 --min-stage-negatives 1000 \
    --drop-low-score-pos 0.20 \
    --very-hard-neg-pool weights/19/cbcl__19_v1__vhardneg_raw.npy
mv weights/19/cvj_weights_*.pkl weights/19/celeba_aligned__19_v3.pkl

# 6. Evaluate + tune + detect
python main.py test --data-dir data/19_celeba_aligned \
    --weights-path weights/19/celeba_aligned__19_v3.pkl
python tools/diagnose_cascade.py \
    --weights weights/19/celeba_aligned__19_v3.pkl \
    --data-dir data/19_celeba_aligned
python tools/tune_thresholds.py \
    --weights weights/19/celeba_aligned__19_v3.pkl \
    --data-dir data/19_celeba_aligned --objective f1
python main.py test --data-dir data/19_celeba_aligned \
    --weights-path weights/19/celeba_aligned__19_v3_tuned.pkl
python main.py detect \
    --weights-path weights/19/celeba_aligned__19_v3.pkl \
    --detect-images images/people.png images/judybats.jpg \
    --detect-output images/outputs/celeba_aligned__19_v3
```
