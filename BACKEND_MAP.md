# BACKEND_MAP — карта бэкенда для защиты

Быстрый справочник по архитектуре «Анализатора»: где что лежит, какой алгоритм где работает. Использовать на защите для оперативной навигации по коду.

---

## 1. ПРАВИЛА (rule-based рекомендация типа задачи)

**Файл:** `backend/app/services/task_recommender.py`

Главная функция: `apply_rules()` (≈ строки 158–498). Структура — дерево решений из 5 веток.

### Список правил

**Ветка 1 — Без target (строки 191–230)**
- Условие: `target_column is None` или `target_kind is None`.
- Рекомендация: `CLUSTERING`. Если `n_cols > 10`, добавляется `DIMENSIONALITY_REDUCTION`.
- Confidence: 0.8–0.9 (зависит от размерности).

**Ветка 2 — Numeric-discrete bridge (строки 232–242)**
- Условие: профайлер отдал `categorical`, но все значения парсятся как числа и `3 ≤ n_unique ≤ 20`.
- Действие: перенаправляет в ветку 3 как `regression` (не меняет код, а лечит симптом расхождения профайлера).

**Ветка 3 — Numeric target (`_apply_numeric_target_rules`, строки 275–367)**
- `n_unique ≤ 2` → `BINARY_CLASSIFICATION` (confidence 0.95).
- `3 ≤ n_unique ≤ 10` → `MULTICLASS_CLASSIFICATION` (0.5, `requires_ml=True`).
- `11 ≤ n_unique ≤ 20`:
  - `|skewness| < 1.0` → `MULTICLASS_CLASSIFICATION` (0.7).
  - иначе → `REGRESSION` (0.5, `requires_ml=True`).
- `n_unique > 20` → `REGRESSION` (0.95).

**Ветка 4 — Categorical target (`_apply_categorical_target_rules`, строки 370–498)**
- `n_classes < 2` → `NOT_READY` (0.99).
- `n_classes = 2`:
  - `imbalance_ratio > 10.0` → `BINARY_CLASSIFICATION` (0.9, с пометкой о дисбалансе).
  - иначе → `BINARY_CLASSIFICATION` (0.95).
- `3 ≤ n_classes ≤ 20`:
  - `min_class_size < 50` → `MULTICLASS_CLASSIFICATION` (0.85).
  - иначе → `MULTICLASS_CLASSIFICATION` (0.95).
- `n_classes > 20`:
  - `cardinality_ratio > 0.5` → `NOT_READY` (0.8, ID-подобная колонка).
  - иначе → `MULTICLASS_CLASSIFICATION` (0.6, `requires_ml=True`).

**Ветка 5 — Critical quality flags (строки 501–518)**
- Дополняет `applied_rules` без смены `task_type_code`.
- Критические коды: `TARGET_MISSING`, `LEAKAGE_SUSPICION`, `SMALL_DATASET`.

### Параметры и пороги

- `RULES_CONFIDENCE_THRESHOLD = 0.7` — порог, ниже которого решение передаётся ML-классификатору.
- `NUMERIC_BRIDGE_MIN_UNIQUE = 3`, `NUMERIC_BRIDGE_MAX_UNIQUE = 20`.
- `CRITICAL_QUALITY_FLAGS` — словарь критических флагов и их сообщений.

### Конфигурация

Правила НЕ параметризованы через YAML/JSON — пороги жёстко зашиты в код. Менять только редактированием `task_recommender.py`. Справочник правил качества (12 штук) лежит в БД (таблица `quality_rules`), сидится из `backend/seeds/seed_quality_rules.py`.

---

## 2. ML-МОДЕЛЬ (мета-классификатор, второй слой)

### Файлы

- **Обученная модель:** `backend/ml/models/meta_classifier.pkl` (`RandomForestClassifier`).
- **Scaler:** `backend/ml/models/scaler.pkl` (`StandardScaler`).
- **Отчёт о метриках:** `backend/ml/models/meta_classifier_report.json`.
- **Скрипт обучения:** `backend/ml/train_meta_classifier.py`.
- **Извлечение признаков:** `backend/ml/feature_vector.py`.

### Признаковое пространство (`feature_vector.py`)

**Размерность:** 16 скалярных признаков, паддятся до 128 для pgvector-embedding.

Канонический порядок (`CANONICAL_FEATURE_ORDER`):

1. `n_rows`
2. `n_cols`
3. `memory_mb`
4. `__features_to_rows_ratio__` (производное: `n_cols / max(n_rows, 1)`)
5. `total_missing_pct`
6. `max_col_missing_pct`
7. `duplicate_rows_pct`
8. `mean_skewness`
9. `mean_kurtosis`
10. `normality_test_pvalue`
11. `outliers_pct`
12. `target_n_unique`
13. `max_abs_correlation`
14. `target_correlation_max`
15. `target_mutual_information_max`
16. `target_mutual_information_mean`

**Padding:** 128 − 16 = 112 нулей. Нужны для совпадения размерности с pgvector-каталогом (`external_datasets.embedding Vector(128)`) — единый формат для поиска похожих.

**Намеренно исключены** (структурный leakage): `target_imbalance_ratio`, `target_class_entropy`, `target_skewness` — они None для регрессии и работают как индикатор класса.

### Обучающие наборы

- `backend/ml/data/real_set.json` — реальная часть (sklearn, UCI, GitHub).
- `backend/ml/data/synthetic_set.json` — синтетическая часть (`make_classification`, `make_regression`).
- Сборка: `make build-real-set`, `make build-synthetic-set`.

Из `meta_classifier_report.json` (по последнему обучению):
- реальных записей: 27;
- синтетических: 150;
- итого после фильтрации CLUSTERING: 177;
- распределение по классам: `BINARY=35`, `MULTICLASS=59`, `REGRESSION=83`.

### Алгоритм и гиперпараметры

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
)
```

**Кросс-валидация:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

### Метрики (из отчёта)

- Accuracy: mean ≈ 0.977, std ≈ 0.021.
- F1-macro: mean ≈ 0.976.
- F1-weighted: mean ≈ 0.977.

Топ-3 feature importance: `target_n_unique` (≈ 0.43), `max_abs_correlation` (≈ 0.13), `normality_test_pvalue` (≈ 0.09).

### Запуск

```bash
make train-meta
# фактически: docker compose exec backend python -m ml.train_meta_classifier
```

---

## 3. ПРОФАЙЛЕР (выявление качества данных)

**Файл:** `backend/app/services/profiler.py`

Главная функция: `compute_meta_features(df, target_col)`. Возвращает словарь ~30 meta-features.

### Проверки качества

Отдельный сервис: `backend/app/services/quality_checker.py`. Применяет правила из БД (таблица `quality_rules`, 12 штук). Каждое правило: код, описание, severity (`info|warning|critical`), threshold.

Состав проверок (по справочнику):

| Код | Что детектирует | Базис | Severity |
|---|---|---|---|
| `MISSING_VALUES_HIGH` | колонки с долей пропусков выше порога | threshold | warning |
| `MISSING_VALUES_TOTAL_HIGH` | общая доля пропусков | threshold | warning |
| `DUPLICATE_ROWS_HIGH` | доля полных дубликатов | threshold | warning |
| `OUTLIERS_HIGH` | доля выбросов по IQR | threshold | info |
| `HIGH_CARDINALITY_CATEGORICAL` | категориальные с card-ratio > 0.5 | эвристика | warning |
| `LOW_VARIANCE_NUMERIC` | `std/|mean| < 0.01` | статистика | info |
| `LOW_VARIANCE_CATEGORICAL` | нормированная энтропия < 0.1 | информационный критерий | info |
| `SMALL_DATASET` | `n_rows < 50` | threshold | critical |
| `TARGET_MISSING` | target указан, но пропусков много | threshold | critical |
| `LEAKAGE_SUSPICION` | `|corr(feature, target)| ≈ 1` | корреляционная эвристика | critical |
| `IMBALANCE_HIGH` | `imbalance_ratio > 10` | threshold | warning |
| `NON_NORMAL_DISTRIBUTION` | p-value теста нормальности < α | стат. тест | info |

### KS-тест (нормальность для больших выборок)

**Где вызывается:** `check_normality()` в `profiler.py` (≈ строки 138–192).

Логика:
- `n ≤ 5000` → Шапиро–Уилк (`stats.shapiro`).
- `n > 5000` → Колмогоров–Смирнов (`stats.kstest(values, "norm", args=(mean, std))`).

Альфа: 0.05. Результат уходит в `normality_test_pvalue` (медиана по числовым колонкам) и `normality_test_method` (`shapiro|ks|mixed|None`).

Зачем KS на больших выборках: Шапиро–Уилк ограничен `n ≤ 5000` в scipy. KS работает на любой `n` и сравнивает эмпирическую функцию распределения с N(μ̂, σ̂²).

### Прочие важные константы

- `SAMPLING_THRESHOLD = 50_000` — выше этого числа строк включается семплинг.
- `SAMPLE_SIZE = 50_000` — целевой размер выборки.
- `SHAPIRO_MAX_N = 5_000`.
- `RANDOM_STATE = 42`.

---

## 4. BASELINE (обучение базовых моделей)

**Основной файл:** `backend/app/services/baseline_trainer.py`
**Оркестратор (async-обёртка):** `backend/app/services/baseline_orchestrator.py`

### Какие модели и почему

- **Classification (binary / multiclass):**
  - `LogisticRegression(max_iter=200, class_weight="balanced")` — линейный baseline, быстрый, интерпретируемый.
  - `RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1)` — ансамблевый baseline.
- **Regression:**
  - `Ridge` — линейный baseline с L2-регуляризацией.
  - `RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1)`.
- **Clustering / NOT_READY:** заглушка (`models=[]`, объяснение в `note`).

Гиперпараметры жёстко зафиксированы — цель не выжать максимум, а получить воспроизводимую нижнюю оценку за 5–15 секунд.

### Метрики и CV

- Classification: `StratifiedKFold(n_splits=5, shuffle=True)`, метрики: `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (binary); `accuracy`, `f1_macro`, `f1_weighted` (multiclass).
- Regression: `KFold(n_splits=5, shuffle=True)`, метрики: `mae`, `rmse` (через `sqrt(neg_mean_squared_error)`), `r2`.

Агрегация: mean ± std по фолдам (функция `_aggregate_cv_results`).

### Препроцессинг (`_preprocess`)

1. Удаление строк с NaN в target.
2. Семплинг до 5000 строк (стратифицированный для классификации).
3. Отделение y, удаление target + `leakage_cols` + datetime + high-card категориальных.
4. Импутация: медиана для числовых, мода для категориальных.
5. One-hot для категориальных с `cardinality_ratio < 0.1`.
6. `StandardScaler` для числовых.

### Вердикт (надёжность сигнала)

В `baseline_trainer.py` вердикт «Уверенный / Умеренный / …» НЕ формируется — функция возвращает словарь метрик. Текстовая интерпретация формируется на фронте (`AnalysisResult.tsx`) либо в шаблоне отчёта (`report_service.py` + Jinja-шаблон). Пороги: см. шаблон отчёта.

### Async-оркестратор

`run_baseline_async(analysis_id, session_factory)`:
1. Открывает свежую сессию БД.
2. Загружает Analysis, Dataset, meta_features, leakage_cols.
3. Переводит `baseline_status='running'`.
4. Запускает `asyncio.to_thread(train_baseline_from_df, ...)` — CPU-bound в thread pool.
5. Записывает результат и `baseline_status='done'`.
6. На исключении: `baseline_status='failed'`, `baseline_error=str(exc)[:500]`.

---

## 5. ПОХОЖИЕ ДАТАСЕТЫ

**Файл:** `backend/app/services/dataset_matcher.py`

### Метрика близости

- **Основная:** косинус (оператор pgvector `<=>`).
- **Доп. опции для исследования:** евклид (`<->`), манхэттен (`<+>`).
- Whitelist операторов: словарь `OPERATORS` в начале модуля (защита от SQL-инъекции через имя метрики).

### Размерность

Embedding — 128-мерный вектор (`EMBEDDING_DIM = 128` в `feature_vector.py`). Первые 16 значений — отскейленные канонические признаки; остальные 112 — нули-padding.

### Каталог

- **Таблица:** `external_datasets` (модель: `backend/app/models/external_dataset.py`).
- **Ключевые поля:** `embedding Vector(128)`, `task_type_code`, `meta_features JSONB`, `title`, `source_url`, `description`.
- **Индекс:** HNSW по `embedding` (`external_datasets_embedding_idx`).
- **Сидинг:** `make seed-catalog` → `backend/seeds/seed_external_datasets.py` из `real_set.json`.

### Количество записей в каталоге

В каталог попадают только записи из `real_set.json` (синтетика — нет).
По текущему отчёту обучения: **27 реальных датасетов**.

Проверить вживую:

```bash
docker compose exec postgres psql -U analyzer -d analyzer -c "SELECT COUNT(*) FROM external_datasets;"
```

### Алгоритм поиска

SQL вида:
```sql
SELECT id, title, meta_features, embedding <=> :query AS distance
FROM external_datasets
WHERE task_type_code = :task_type    -- если фильтр задан
ORDER BY embedding <=> :query
LIMIT :k;
```

Top-K (по умолчанию 5) уходит на фронт с `distance` в каждом элементе.

---

## 6. PDF-ОТЧЁТ

**Файл:** `backend/app/services/report_service.py`

### Библиотека

**WeasyPrint** — HTML+CSS → PDF в python-процессе. Шаблоны Jinja2 лежат в `backend/app/templates/` (главный — `report.html`).

Почему WeasyPrint, а не wkhtmltopdf / reportlab:
- работает в Python без subprocess'а;
- понимает современный CSS (flex, grid) и встраиваемые `data:image/png;base64,...` графики;
- стабильнее wkhtmltopdf, который зависит от системного WebKit и заброшен.

### Сборка контекста

`_build_context()` собирает словарь для Jinja:
- `title`, `generated_at`, `report_id`;
- `user`, `dataset`, `analysis`;
- `summary` — `n_rows`, `n_cols`, `missing_pct`, `dtype_counts`;
- `target_info`;
- `quality_flags`;
- `task_recommendation`;
- `baseline` — метрики, feature importance;
- `similar_datasets` — топ-5;
- `charts` — PNG-байты (вставляются через base64).

PNG-чарты рендерит `backend/app/services/chart_renderer.py` (matplotlib без GUI-бэкенда).

### Хранение

- Путь: `data/reports/{user_id}/{report_id}.pdf`.
- Запись атомарная: сначала `.pdf.tmp`, потом `os.replace`.
- Метаданные (статус, путь, ошибка) — в таблице `reports`.
- Статусы: `pending → running → success | failed`.
- Кеш: явного кеша нет — повторный POST создаёт новую запись и новый файл.

---

## 7. БАЗА ДАННЫХ

**Модели:** `backend/app/models/`

| Файл | Таблица | Назначение |
|---|---|---|
| `user.py` | `users` | Пользователи (email/username/password_hash/role) |
| `dataset.py` | `datasets` | Загруженные датасеты (метаданные + storage_path + file_hash) |
| `analysis.py` | `analyses` | Запуски анализа (статус, target_column, временные метки) |
| `analysis_result.py` | `analysis_results` | Результат: meta_features, embedding(128), task_recommendation, baseline |
| `quality_rule.py` | `quality_rules` | Справочник 12 правил качества (код, severity, threshold) |
| `quality_flag.py` | `quality_flags` | Сработавшие флаги для конкретного анализа |
| `external_dataset.py` | `external_datasets` | Каталог эталонных датасетов с pgvector-embedding |
| `report.py` | `reports` | PDF-отчёты (статус, file_path, error_message) |

Ключевые связи: `Dataset 1:N Analysis`, `Analysis 1:1 AnalysisResult`, `Analysis 1:N QualityFlag`, `Analysis 1:N Report`. FK везде с `ON DELETE CASCADE`.

Миграции: `backend/alembic/versions/`. Применить: `make migrate`.

---

## 8. ФРОНТ — карта страниц

**Корневой файл:** `frontend/src/App.tsx`

| Путь | Компонент | Файл | Защита |
|---|---|---|---|
| `/` | `Landing` | `frontend/src/pages/Landing.tsx` | публичный |
| `/login` | `Login` | `frontend/src/pages/Login.tsx` | публичный |
| `/register` | `Register` | `frontend/src/pages/Register.tsx` | публичный |
| `/` (для авторизованного) | `Dashboard` | `frontend/src/pages/Dashboard.tsx` | `RequireAuth` |
| `/upload` | `Upload` | `frontend/src/pages/Upload.tsx` | `RequireAuth` |
| `/profile` | `Profile` | `frontend/src/pages/Profile.tsx` | `RequireAuth` |
| `/history` | `History` (lazy) | `frontend/src/pages/History.tsx` | `RequireAuth` |
| `/analyses/:id` | `AnalysisResult` (lazy) | `frontend/src/pages/AnalysisResult.tsx` | `RequireAuth` |
| `/admin` | `Admin` (lazy) | `frontend/src/pages/Admin.tsx` | `RequireAdmin` |

### Lazy-загрузка

- `AnalysisResult` — содержит Plotly (~3 МБ), грузится по требованию.
- `History`, `Admin` — для единого паттерна и сокращения main-bundle.

### Слои защиты

- `RequireAuth` — проверка наличия токена и user в `authStore`; redirect на `/login` с `state.flash`.
- `RequireAdmin` — обёрнут поверх `RequireAuth`, дополнительно проверяет `user.role === "admin"`.

### Стор и API

- Глобальное состояние: `frontend/src/store/authStore.ts` (Zustand).
- API-клиенты: `frontend/src/api/` (auth, datasets, analyses, reports, admin).
