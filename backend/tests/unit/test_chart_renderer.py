"""
Unit-тесты серверного рендера графиков для PDF-отчётов.

Покрывают пять публичных функций chart_renderer'а + два инфраструктурных
требования: устойчивость к кириллице в подписях и отсутствие утечек
matplotlib-фигур при батч-рендере.

Проверка валидности PNG идёт по сигнатуре первых 8 байтов (b"\\x89PNG\\r\\n\\x1a\\n").
Логика свёртки категорий и отбора топ-K корреляций тестируется через
helper'ы напрямую — это надёжнее, чем парсить готовый PNG.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from app.services.chart_renderer import (
    HEATMAP_MAX_LABELS,
    _prepare_categorical_top_n,
    _select_top_correlations,
    render_categorical_bar,
    render_correlation_heatmap,
    render_distribution_histogram,
    render_target_classification,
    render_target_regression,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_distribution_histogram_returns_valid_png() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(loc=30.0, scale=10.0, size=200).tolist()
    png = render_distribution_histogram(values, "age")
    assert isinstance(png, bytes)
    assert png[:8] == PNG_SIGNATURE
    assert len(png) > 1024  # реальный PNG крупнее минимально-валидного


def test_categorical_bar_top_n_with_other_bucket() -> None:
    """20 категорий + max=15 → 16 баров (15 топ + «Прочее»), сумма прочих сходится."""
    counts = {f"cat{i:02d}": i for i in range(1, 21)}
    labels, values = _prepare_categorical_top_n(counts, max_categories=15)

    assert len(labels) == 16
    assert len(values) == 16
    # Топ-15 — категории с самыми большими счётчиками: cat20..cat06
    expected_top = [f"cat{n:02d}" for n in range(20, 5, -1)]
    assert labels[:15] == expected_top
    # Прочее = сумма cat01..cat05 = 1+2+3+4+5 = 15
    assert labels[-1] == "Прочее (15)"
    assert values[-1] == 15

    # PNG также генерируется без ошибок.
    png = render_categorical_bar(counts, "category_col", max_categories=15)
    assert png[:8] == PNG_SIGNATURE


def test_correlation_heatmap_returns_valid_png() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(100, 5))
    matrix = np.corrcoef(base, rowvar=False).tolist()
    labels = [f"f{i}" for i in range(5)]
    png = render_correlation_heatmap(matrix, labels)
    assert png[:8] == PNG_SIGNATURE


def test_correlation_heatmap_truncates_large_matrix() -> None:
    """30×30 → внутри отрисовки используются 20×20, остальное обрезано."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 30))
    matrix = np.corrcoef(base, rowvar=False)
    labels = [f"f{i:02d}" for i in range(30)]

    sub_matrix, sub_labels, indices = _select_top_correlations(
        matrix, labels, HEATMAP_MAX_LABELS
    )
    assert sub_matrix.shape == (HEATMAP_MAX_LABELS, HEATMAP_MAX_LABELS)
    assert len(sub_labels) == HEATMAP_MAX_LABELS
    assert len(indices) == HEATMAP_MAX_LABELS

    # Полный рендер без падений.
    png = render_correlation_heatmap(matrix.tolist(), labels)
    assert png[:8] == PNG_SIGNATURE


def test_correlation_heatmap_top20_selection_correctness() -> None:
    """
    25×25: 20 колонок сильно скоррелированы (общий фактор), 5 — независимы.
    Ожидаем, что отобранные 20 колонок — именно «сильные».
    """
    rng = np.random.default_rng(2)
    n_rows = 500
    factor = rng.normal(size=n_rows)
    strong = np.column_stack([
        factor + 0.05 * rng.normal(size=n_rows) for _ in range(20)
    ])
    weak = rng.normal(size=(n_rows, 5))
    data = np.column_stack([strong, weak])
    labels = [f"strong_{i:02d}" for i in range(20)] + [
        f"weak_{i}" for i in range(5)
    ]
    matrix = np.corrcoef(data, rowvar=False)

    _, sub_labels, indices = _select_top_correlations(
        matrix, labels, HEATMAP_MAX_LABELS
    )
    selected_indices = set(indices)
    expected_strong = set(range(20))
    assert selected_indices == expected_strong, (
        f"В топ-20 попали лишние/потерялись сильные: {sub_labels}"
    )


def test_target_classification_returns_valid_png() -> None:
    counts = {"setosa": 50, "versicolor": 50, "virginica": 50}
    png = render_target_classification(counts)
    assert png[:8] == PNG_SIGNATURE


def test_target_regression_returns_valid_png() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(loc=100.0, scale=15.0, size=300).tolist()
    png = render_target_regression(values)
    assert png[:8] == PNG_SIGNATURE


def test_cyrillic_label_does_not_crash() -> None:
    """Кириллица в col_name отрисовывается без UnicodeEncodeError или квадратиков."""
    rng = np.random.default_rng(4)
    values = rng.normal(size=100).tolist()
    png = render_distribution_histogram(values, "Возраст пассажира")
    assert png[:8] == PNG_SIGNATURE
    # Косвенный sanity-check: при отсутствии шрифта matplotlib часто пишет
    # warning-glyphs и заметно урезает изображение. Размер > 2 KB говорит,
    # что на холсте нашлась реальная подпись.
    assert len(png) > 2048


def test_no_figure_leaks_after_batch_render() -> None:
    """50 рендеров не оставляют незакрытых matplotlib-фигур в глобальном реестре."""
    rng = np.random.default_rng(5)
    before = set(plt.get_fignums())
    for _ in range(50):
        values = rng.normal(size=100).tolist()
        render_distribution_histogram(values, "x")
    after = set(plt.get_fignums())
    assert after == before, f"Утечка matplotlib-figure: было {before}, стало {after}"
