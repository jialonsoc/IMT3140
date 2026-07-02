# Validacion cruzada de modelos avanzados

Ejecutar desde la raiz del proyecto:

```powershell
python nuevo_intento\cross_validation\run_validation.py
```

La configuracion predeterminada usa 5 folds externos, 3 folds internos, 8
configuraciones aleatorias por fold y semilla 42. Se puede modificar desde CLI:

```powershell
python nuevo_intento\cross_validation\run_validation.py --folds 5 --inner-folds 3 --n-iter 8 --seed 42
```

El split externo e interno usa `StratifiedGroupKFold` por paciente. En los
experimentos mixtos, la validacion siempre contiene solo ventanas reales y las
series sinteticas se agregan al entrenamiento solo cuando su `source_record`
pertenece al subfold de entrenamiento.

Salidas:

- `data/nuevo_intento/cross_validation_fold_metrics.csv`
- `data/nuevo_intento/cross_validation_summary.csv`
- `data/nuevo_intento/cross_validation_details.json`
