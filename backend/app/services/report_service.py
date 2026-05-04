"""
Оркестратор генерации PDF-отчёта по результатам анализа.

Связывает Jinja2-шаблон, matplotlib (через `chart_renderer`), WeasyPrint и БД.
Запускается через FastAPI BackgroundTasks (Phase 4) — здесь определена
синхронная функция `generate_report`, которую BackgroundTasks выполнит в
worker-потоке. Сессия БД создаётся внутри функции через `SessionLocal()`,
а не передаётся через `Depends` (HTTP-сессия уже закрыта к моменту запуска
фон-задачи).

См. .knowledge/stack/why-weasyprint.md (выбор движка PDF) и
.knowledge/architecture/charts.md (политика согласования PNG-графиков
между matplotlib для PDF и Plotly для UI).

Архитектура: четыре функции, разделённые ради тестируемости.
- `_build_context`        — собирает dict для Jinja2 из БД-сущностей.
- `_render_html`          — Jinja2 → HTML-строка. Не требует WeasyPrint.
- `_render_pdf_bytes`     — WeasyPrint HTML → PDF-байты.
- `generate_report`       — оркестратор: открывает сессию, делегирует трём.
"""
from __future__ import annotations

import base64
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from weasyprint import HTML

from app.config import settings
from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult
from app.models.dataset import Dataset
from app.models.quality_flag import QualityFlag
from app.models.quality_rule import QualityRule
from app.models.report import Report
from app.models.user import User
from app.services import chart_renderer
from app.services.dataset_matcher import find_similar_datasets

logger = logging.getLogger(__name__)

# Жёсткий лимит длины поля error в записи Report — синхронизирован с
# String(1000) в моделе. Защита от переполнения колонки при очень длинном
# stacktrace.
ERROR_MAX_LEN = 1000

# Лимиты на количество графиков в PDF — план Phase 3 фиксирует 10 числовых
# и 5 категориальных. Выше получается громоздко на A4.
MAX_NUMERIC_DISTRIBUTIONS = 10
MAX_CATEGORICAL_DISTRIBUTIONS = 5

# Топ-K похожих датасетов для секции «Похожие». Совпадает с фронт-секцией
# `SimilarDatasetsCard` (5 карточек).
SIMILAR_TOP_K = 5

# Топ-K фичей для секции «Важность признаков» в baseline. Согласовано
# с baseline_trainer (хранит топ-10 в feature_importance).
FEATURE_IMPORTANCE_TOP = 10

# Корень шаблонов. Папка лежит в пакете app, поэтому путь резолвится через
# __file__, не через cwd — иначе тесты падали бы в зависимости от рабочей
# директории.
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "report"


def _create_environment(templates_dir: Path = TEMPLATES_DIR) -> Environment:
    """
    Создаёт настроенный Jinja2-Environment.

    autoescape=True — защита от XSS, если в `dataset.original_filename` или
    в подписях колонок проскользнут HTML-метасимволы (на этапе ввода
    у нас санитизации нет).

    Фильтр `b64` нужен для встраивания PNG-графиков в `<img src="data:...">`
    через base64. Альтернатива — собирать data-URI в Python, но тогда
    шаблон становится непрозрачным.
    """
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["b64"] = lambda raw: base64.b64encode(raw).decode("ascii")
    return env


# Singleton — Jinja2 Environment кешируется между вызовами.
_env: Environment = _create_environment()


def _build_chart_data(meta: dict, target_kind: str | None) -> dict:
    """
    Собирает PNG-байты графиков по meta_features.

    Источник данных — секция `meta.distributions` (pre-binned гистограммы и
    счётчики категорий из профайлера). Для PDF используем те же бины, что
    Plotly на фронте — визуальная согласованность UI и отчёта.

    Returns:
        dict с ключами `numeric_histograms`, `categorical_bars`,
        `correlation_heatmap`, `target_chart`. Отсутствующие графики —
        пустой список или `None`.
    """
    distributions = meta.get("distributions") or {}

    numeric_dists = distributions.get("numeric") or {}
    numeric_histograms = []
    for col_name, dist in list(numeric_dists.items())[:MAX_NUMERIC_DISTRIBUTIONS]:
        bin_edges = dist.get("bin_edges")
        counts = dist.get("counts")
        if not bin_edges or not counts:
            continue
        png = chart_renderer.render_distribution_from_bins(bin_edges, counts, col_name)
        numeric_histograms.append({"col_name": col_name, "png": png})

    categorical_dists = distributions.get("categorical") or {}
    categorical_bars = []
    for col_name, dist in list(categorical_dists.items())[:MAX_CATEGORICAL_DISTRIBUTIONS]:
        categories = dist.get("categories") or []
        counts = dist.get("counts") or []
        other_count = int(dist.get("other_count") or 0)
        counts_dict = {str(cat): int(cnt) for cat, cnt in zip(categories, counts)}
        if other_count > 0:
            counts_dict[f"Прочее ({other_count})"] = other_count
        if not counts_dict:
            continue
        png = chart_renderer.render_categorical_bar(counts_dict, col_name)
        categorical_bars.append({"col_name": col_name, "png": png})

    correlation_png: bytes | None = None
    correlation_matrix = meta.get("correlation_matrix") or {}
    labels = list(correlation_matrix.keys())
    if len(labels) >= 2:
        # correlation_matrix хранится как dict[row_label][col_label] → float.
        # Преобразуем в плотный list[list[float]] перед передачей в renderer.
        # NaN остаются как None (jsonb_safe прошёл при записи) — заменяем на 0.0,
        # чтобы numpy.asarray не падал при float(None).
        matrix = [
            [float(correlation_matrix[row].get(col) or 0.0) for col in labels]
            for row in labels
        ]
        correlation_png = chart_renderer.render_correlation_heatmap(matrix, labels)

    target_chart: bytes | None = None
    if target_kind == "categorical":
        counts = meta.get("target_value_counts") or {}
        if counts:
            target_chart = chart_renderer.render_target_classification(
                {str(k): int(v) for k, v in counts.items()}
            )
    elif target_kind == "regression":
        # Для регрессии профайлер не сохраняет сырой массив target — есть
        # только агрегаты (skewness, missing). Поэтому секцию пропускаем,
        # это согласованное решение Phase 3 (вопрос 11).
        target_chart = None

    return {
        "numeric_histograms": numeric_histograms,
        "categorical_bars": categorical_bars,
        "correlation_heatmap": correlation_png,
        "target_chart": target_chart,
    }


def _build_baseline_view(baseline: dict | None) -> dict | None:
    """
    Готовит секцию baseline для шаблона.

    Шаблон ожидает дополнительные ключи поверх сырого `baseline`:
    - `metric_names` — упорядоченный список имён метрик (для заголовков таблицы).
    - `feature_importance_top` — список (feature, importance), отсортированный
      по убыванию importance, ограниченный TOP_K.
    - `feature_importance_max` — максимум, чтобы пересчитать ширину бара в %.

    Возвращает `None` если baseline пустой или ещё не считался.
    """
    if not baseline or not baseline.get("models"):
        return None

    metrics = baseline.get("metrics") or {}
    metric_names: list[str] = []
    if metrics:
        first_model = next(iter(metrics.values()))
        metric_names = list(first_model.keys())

    importance = baseline.get("feature_importance") or {}
    sorted_importance = sorted(
        importance.items(), key=lambda kv: kv[1], reverse=True
    )[:FEATURE_IMPORTANCE_TOP]
    max_importance = sorted_importance[0][1] if sorted_importance else 1.0

    return {
        **baseline,
        "metric_names": metric_names,
        "feature_importance_top": sorted_importance,
        "feature_importance_max": max_importance or 1.0,
    }


def _resolve_quality_flags(db: Session, flags: list[QualityFlag]) -> list[dict]:
    """
    Превращает ORM-флаги в словари для шаблона, подгрузив code+severity из rules.

    Один SELECT по списку rule_id (а не lazy-load на каждый флаг) — иначе
    словишь N+1.
    """
    if not flags:
        return []
    rule_ids = {flag.rule_id for flag in flags}
    rules = {
        r.id: r for r in db.scalars(
            select(QualityRule).where(QualityRule.id.in_(rule_ids))
        )
    }
    out: list[dict] = []
    for flag in flags:
        rule = rules.get(flag.rule_id)
        column_name = (flag.context or {}).get("column_name")
        out.append({
            "code": rule.code if rule else "UNKNOWN",
            "severity": rule.severity if rule else "info",
            "rule_name": rule.name if rule else "—",
            "message": flag.message,
            "column_name": column_name,
        })
    return out


def _build_similar_view(db: Session, embedding: list[float] | None) -> list[dict]:
    """
    Возвращает топ-5 похожих датасетов в форме для шаблона. Пустой список
    при отсутствии embedding или ошибке поиска (best-effort).
    """
    if not embedding:
        return []
    try:
        rows = find_similar_datasets(
            db, embedding, top_k=SIMILAR_TOP_K, metric="cosine"
        )
    except Exception:  # noqa: BLE001 — секция опциональна, не валим отчёт
        logger.exception("find_similar_datasets failed; section will be skipped")
        return []
    return [
        {
            "title": row.title,
            "task_type_code": row.task_type_code,
            "source": row.source,
            # При cosine distance ∈ [0, 2]; similarity = 1 - distance / 2,
            # чтобы выводить «похожесть» как привычный процент в [0, 1].
            "similarity": max(0.0, min(1.0, 1.0 - getattr(row, "distance", 0.0) / 2.0)),
        }
        for row in rows
    ]


def _build_context(
    *,
    report: Report,
    analysis: Analysis,
    dataset: Dataset,
    user: User,
    result: AnalysisResult,
    quality_flags: list[dict],
    similar_datasets: list[dict],
) -> dict:
    """
    Собирает context для Jinja2 из подготовленных сущностей.

    Все «опциональные» секции получают явные пустые дефолты (`baseline=None`,
    `similar_datasets=[]`) — шаблон проверяет их через `{% if %}` без
    `is defined`. Контракт ключей зафиксирован, при отсутствии данных
    шаблон не падает и не показывает «битых» секций.
    """
    meta = result.meta_features or {}
    target_kind = meta.get("target_kind")

    summary = {
        "n_rows": meta.get("n_rows", dataset.n_rows or 0),
        "n_cols": meta.get("n_cols", dataset.n_cols or 0),
        "total_missing_pct": float(meta.get("total_missing_pct") or 0.0),
        "max_col_missing_pct": float(meta.get("max_col_missing_pct") or 0.0),
        "dtype_counts": meta.get("dtype_counts") or {},
    }

    target_info = {
        "kind": target_kind,
        "n_unique": meta.get("target_n_unique"),
        "imbalance_ratio": meta.get("target_imbalance_ratio"),
        "class_entropy": meta.get("target_class_entropy"),
        "skewness": meta.get("target_skewness"),
    }

    charts = _build_chart_data(meta, target_kind)
    baseline_view = _build_baseline_view(result.baseline)

    return {
        "title": f"Отчёт по анализу датасета «{dataset.original_filename}»",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "report_id": str(report.id),
        "user": {"username": user.username},
        "dataset": {
            "original_filename": dataset.original_filename,
            "format": dataset.format,
        },
        "analysis": {
            "id": str(analysis.id),
            "target_column": analysis.target_column,
            "started_at": analysis.started_at.isoformat() if analysis.started_at else None,
            "finished_at": analysis.finished_at.isoformat() if analysis.finished_at else None,
        },
        "summary": summary,
        "target_info": target_info,
        "quality_flags": quality_flags,
        "task_recommendation": result.task_recommendation,
        "similar_datasets": similar_datasets,
        "baseline": baseline_view,
        "charts": charts,
    }


def _render_html(context: dict, env: Environment = _env) -> str:
    """Jinja2 → HTML-строка. Не требует WeasyPrint, поэтому быстро в тестах."""
    template = env.get_template("report.html")
    return template.render(**context)


def _render_pdf_bytes(html: str, base_url: Path = TEMPLATES_DIR) -> bytes:
    """
    HTML-строка → PDF-байты через WeasyPrint.

    `base_url` нужен для разрешения относительных URL в HTML (например,
    `<link rel="stylesheet" href="report.css">` ищется относительно этого
    пути). Без base_url WeasyPrint попытается резолвить через cwd — это
    нестабильно в фон-задачах.
    """
    return HTML(string=html, base_url=str(base_url)).write_pdf()


def generate_report(
    report_id: uuid.UUID,
    session_factory: sessionmaker,
) -> None:
    """
    Полный цикл генерации PDF-отчёта.

    Алгоритм:
    1. Открыть свежую БД-сессию из фабрики.
    2. Получить Report, поставить status='running', commit.
    3. Подгрузить связанные сущности (Analysis, Dataset, User, AnalysisResult,
       quality_flags). При отсутствии meta_features в AnalysisResult —
       это явная ошибка данных, помечаем status='failed'.
    4. Собрать context, отрендерить HTML, отрендерить PDF.
    5. Сохранить PDF на диск в `{REPORTS_DIR}/{user_id}/{report_id}.pdf`.
    6. Обновить Report: status='success', file_path, file_size_bytes, commit.
    7. На любом исключении — rollback, status='failed', error[:1000], commit.

    Args:
        report_id: UUID записи Report (создана в API-эндпоинте Phase 4).
        session_factory: SessionLocal — фабрика для открытия новой сессии.
    """
    db = session_factory()
    report: Report | None = None
    try:
        report = db.get(Report, report_id)
        if report is None:
            logger.error("generate_report: report %s not found", report_id)
            return

        # 1) pending → running, отдельный commit ради polling-видимости.
        report.status = "running"
        db.commit()

        # 2) Подгружаем связанные сущности.
        analysis = db.get(Analysis, report.analysis_id)
        if analysis is None:
            raise RuntimeError("Связанный анализ не найден")
        dataset = analysis.dataset
        user = db.get(User, report.user_id)
        if user is None:
            raise RuntimeError("Связанный пользователь не найден")
        result = db.get(AnalysisResult, analysis.id)
        if result is None or not (result.meta_features or {}):
            raise RuntimeError("Результаты анализа отсутствуют (meta_features missing)")

        flags_orm = list(
            db.scalars(
                select(QualityFlag).where(QualityFlag.analysis_id == analysis.id)
            )
        )
        quality_flags = _resolve_quality_flags(db, flags_orm)
        # pgvector хранит embedding как numpy.ndarray — приводим к list[float],
        # иначе `if not embedding` внутри _build_similar_view упирается в
        # «The truth value of an array with more than one element is ambiguous».
        embedding_list: list[float] | None = (
            [float(v) for v in result.embedding]
            if result.embedding is not None
            else None
        )
        similar_datasets = _build_similar_view(db, embedding_list)

        # 3) Контекст → HTML → PDF.
        context = _build_context(
            report=report,
            analysis=analysis,
            dataset=dataset,
            user=user,
            result=result,
            quality_flags=quality_flags,
            similar_datasets=similar_datasets,
        )
        html = _render_html(context)
        pdf_bytes = _render_pdf_bytes(html)

        # 4) Сохранение файла через tmp+rename для транзакционности.
        # Сначала пишем в *.pdf.tmp, затем атомарно переименовываем в *.pdf —
        # это гарантирует, что на диске не появится «полу-PDF» при сбое
        # записи. os.replace выполняется ДО db.commit финального статуса:
        # если rename упадёт (нет места на диске и т.п.), исключение пойдёт
        # в общий except и статус станет failed, а БД не будет рассинхронна
        # с файлом-orphan.
        reports_root = Path(settings.REPORTS_DIR)
        user_dir = reports_root / str(user.id)
        user_dir.mkdir(parents=True, exist_ok=True)
        relative_path = f"{user.id}/{report.id}.pdf"
        final_path = reports_root / relative_path
        tmp_path = final_path.with_suffix(".pdf.tmp")
        tmp_path.write_bytes(pdf_bytes)
        os.replace(tmp_path, final_path)

        # 5) Финальный статус.
        report.status = "success"
        report.file_path = relative_path
        report.file_size_bytes = len(pdf_bytes)
        db.commit()
        logger.info(
            "Report %s generated, size=%d bytes", report.id, len(pdf_bytes)
        )

    except Exception as exc:  # noqa: BLE001 — финализируем status=failed
        logger.exception("generate_report failed for report_id=%s", report_id)
        db.rollback()
        # Чистим .tmp если он успел появиться, чтобы не оставлять orphan'ов
        # на диске. Финального .pdf после rename'а здесь по построению нет —
        # либо rename прошёл и мы вышли из try выше, либо упал и этой
        # ветке tmp ещё лежит на месте.
        try:
            tmp_candidate = (
                Path(settings.REPORTS_DIR)
                / str(report.user_id)
                / f"{report.id}.pdf.tmp"
            ) if report is not None else None
            if tmp_candidate is not None and tmp_candidate.exists():
                tmp_candidate.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to clean tmp file after error")

        if report is None:
            return
        try:
            # Перечитываем запись после rollback — атрибуты сессии могут
            # быть сброшены.
            report = db.get(Report, report_id)
            if report is None:
                return
            report.status = "failed"
            report.error = str(exc)[:ERROR_MAX_LEN]
            db.commit()
        except Exception:
            logger.exception(
                "Failed to record failure status for report_id=%s", report_id
            )
            db.rollback()
    finally:
        db.close()
