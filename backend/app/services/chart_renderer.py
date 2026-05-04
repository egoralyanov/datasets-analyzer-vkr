"""
Серверный рендер графиков для PDF-отчётов через matplotlib.

Каждая функция возвращает PNG-байты, которые orchestrator (Phase 3) встраивает
в HTML-шаблон через `<img src="data:image/png;base64,...">`. WeasyPrint
не выполняет JavaScript, поэтому Plotly-графики UI здесь не подходят —
используется matplotlib в Agg-backend, без X-сервера.

См. .knowledge/architecture/charts.md, раздел «На бэке (matplotlib)» —
архитектурное обоснование двух движков рендера и принцип «один источник
данных, два рендера».

Цветовая палитра жёстко синхронизирована с фронтом
(frontend/src/components/analysis/Distributions.tsx). Точка синхронизации —
константы в начале модуля. При смене цвета в одном месте обязательно
поправить второе, иначе UI и PDF будут визуально расходиться.

Шрифт DejaVu Sans гарантированно установлен в backend Dockerfile через
пакет fonts-dejavu (Спринт 0). Кириллические подписи отрисовываются как
есть, без замены на квадратики.
"""
from __future__ import annotations

import io

import matplotlib

# Agg-backend выбирается ДО импорта pyplot — иначе matplotlib попытается
# подгрузить tkinter/Qt и упадёт в headless-контейнере.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (импорт после matplotlib.use)
import numpy as np  # noqa: E402

# Цвета взяты из фронта (Distributions.tsx) — Tailwind blue-600 / emerald-500 /
# violet-600. Diverging RdBu_r соответствует Plotly RdBu reversed для heatmap.
COLOR_NUMERIC = "#2563eb"      # blue-600 — гистограммы числовых признаков
COLOR_CATEGORICAL = "#10b981"  # emerald-500 — bar chart категориальных
COLOR_TARGET = "#7c3aed"       # violet-600 — распределение target
CMAP_CORRELATION = "RdBu_r"    # diverging colormap для матрицы корреляций

# Глобальные настройки matplotlib. font.size=11 совпадает с PLOT_LAYOUT_BASE
# в фронте; font.family="DejaVu Sans" гарантирует кириллицу.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

# Если категорий в bar chart больше — топ-N + «Прочее» (в один сводный бар).
DEFAULT_MAX_CATEGORIES = 15

# Если число колонок в корреляционной матрице больше — отбираем топ-K
# по сумме абсолютных корреляций (без диагонали). Метки и значения в PDF
# при K > 20 уже нечитаемы.
HEATMAP_MAX_LABELS = 20

# Порог, при котором в ячейках heatmap печатаем числовые значения корреляций.
# При большем числе колонок — только цвет, чтобы не было каши из текста.
HEATMAP_ANNOT_THRESHOLD = 10


def _figure_to_png_bytes(fig: plt.Figure) -> bytes:
    """
    Сохраняет matplotlib Figure в PNG-байты и закрывает её.

    Закрытие фигуры обязательно — matplotlib держит каждую figure в
    глобальном реестре, и без plt.close() при батч-рендере отчётов
    (десятки графиков) растёт RSS контейнера. Возврат сделан через
    try/finally чтобы не утечь даже на исключении в savefig.
    """
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    finally:
        plt.close(fig)
    return buffer.getvalue()


def render_distribution_histogram(values: list[float], col_name: str) -> bytes:
    """
    Гистограмма распределения числового признака (30 бинов).

    Используется для секции «Распределения» PDF-отчёта по топ-10
    числовым колонкам датасета.

    Args:
        values: одномерный массив числовых значений (NaN допустимы — будут
            проигнорированы matplotlib через np.histogram).
        col_name: имя столбца для подписи оси X и заголовка.

    Returns:
        PNG-изображение в виде байтов.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    ax.hist(array, bins=30, color=COLOR_NUMERIC, edgecolor="white")
    ax.set_xlabel(col_name)
    ax.set_ylabel("Частота")
    ax.set_title(f"Распределение признака «{col_name}»")
    return _figure_to_png_bytes(fig)


def _prepare_categorical_top_n(
    counts: dict[str, int], max_categories: int
) -> tuple[list[str], list[int]]:
    """
    Готовит метки и значения для bar chart с фолбэком в категорию «Прочее».

    Если входных категорий не больше `max_categories` — возвращает их все
    в порядке убывания. Иначе оставляет топ-N и добавляет финальный пункт
    «Прочее (X)» с суммой свёрнутых категорий.

    Вынесен из `render_categorical_bar` отдельно, чтобы тест мог проверить
    логику без анализа PNG-выхода.
    """
    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_items) > max_categories:
        top = sorted_items[:max_categories]
        other_total = sum(count for _, count in sorted_items[max_categories:])
        labels = [str(k) for k, _ in top] + [f"Прочее ({other_total})"]
        values = [v for _, v in top] + [other_total]
    else:
        labels = [str(k) for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
    return labels, values


def render_categorical_bar(
    counts: dict[str, int],
    col_name: str,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> bytes:
    """
    Bar chart распределения категориального признака.

    При числе категорий больше `max_categories` оставляет топ-N по
    убыванию количества и сворачивает остаток в один бар «Прочее (X)»,
    где X — сумма частот свёрнутых категорий. Это сохраняет визуальную
    плотность графика на A4 без потери информации о минорных категориях.

    Args:
        counts: словарь {категория: количество}.
        col_name: имя столбца для подписи и заголовка.
        max_categories: верхний предел отрисованных категорий (без «Прочее»).

    Returns:
        PNG-изображение в виде байтов.
    """
    labels, values = _prepare_categorical_top_n(counts, max_categories)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=COLOR_CATEGORICAL, edgecolor="white")
    ax.set_xlabel("Категория")
    ax.set_ylabel("Количество")
    ax.set_title(f"Распределение признака «{col_name}»")
    fig.autofmt_xdate(rotation=30)  # автоматический наклон длинных подписей
    return _figure_to_png_bytes(fig)


def _select_top_correlations(
    matrix: np.ndarray, labels: list[str], top_k: int
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Возвращает подматрицу топ-K колонок по сумме |корреляций| без диагонали.

    Метод: для каждой колонки i считаем S_i = Σ |corr_ij| − |corr_ii|.
    Сортируем колонки по S_i убыванию, берём первые K. Это идентично
    выборке «наиболее связанных переменных» — именно они интересны
    в отчёте, тогда как почти-независимые колонки только засоряют график.

    Args:
        matrix: квадратная корреляционная матрица NxN.
        labels: метки длиной N.
        top_k: сколько колонок оставить.

    Returns:
        (подматрица KxK, метки длины K, индексы выбранных колонок в исходной
        матрице — нужны тестам для верификации правильности отбора).
    """
    abs_matrix = np.abs(matrix)
    diagonal = np.abs(np.diag(matrix))
    weights = abs_matrix.sum(axis=1) - diagonal
    top_indices = np.argsort(weights)[::-1][:top_k]
    # Сортируем индексы по возрастанию, чтобы порядок колонок в подматрице
    # соответствовал исходному (визуально предсказуемо).
    top_indices = np.sort(top_indices)
    sub_matrix = matrix[np.ix_(top_indices, top_indices)]
    sub_labels = [labels[i] for i in top_indices]
    return sub_matrix, sub_labels, list(top_indices)


def render_correlation_heatmap(
    matrix: list[list[float]], labels: list[str]
) -> bytes:
    """
    Тепловая карта парных корреляций числовых признаков.

    При числе колонок > HEATMAP_MAX_LABELS (=20) отбирает топ-20 по
    суммарной абсолютной корреляции через `_select_top_correlations` —
    в подпись графика добавляется «топ-20 из N». При N ≤ 20 рисует
    матрицу как есть. При N ≤ HEATMAP_ANNOT_THRESHOLD (=10) дополнительно
    печатает значения корреляций в ячейках.

    Args:
        matrix: квадратный список списков NxN с парными корреляциями
            в диапазоне [-1, 1].
        labels: имена колонок длиной N.

    Returns:
        PNG-изображение в виде байтов.
    """
    full_matrix = np.asarray(matrix, dtype=float)
    n_total = len(labels)

    if n_total > HEATMAP_MAX_LABELS:
        plot_matrix, plot_labels, _ = _select_top_correlations(
            full_matrix, labels, HEATMAP_MAX_LABELS
        )
        title = (
            f"Корреляционная матрица — топ-{HEATMAP_MAX_LABELS} "
            f"из {n_total} колонок"
        )
    else:
        plot_matrix = full_matrix
        plot_labels = labels
        title = "Корреляционная матрица"

    n_plot = len(plot_labels)
    side = min(7.0, n_plot * 0.4 + 2.5)
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(
        plot_matrix, cmap=CMAP_CORRELATION, vmin=-1.0, vmax=1.0, aspect="equal"
    )
    ax.set_xticks(range(n_plot))
    ax.set_yticks(range(n_plot))
    ax.set_xticklabels(plot_labels, rotation=45, ha="right")
    ax.set_yticklabels(plot_labels)

    if n_plot <= HEATMAP_ANNOT_THRESHOLD:
        for i in range(n_plot):
            for j in range(n_plot):
                value = plot_matrix[i, j]
                # Контрастный текст: тёмные ячейки → белый, светлые → чёрный.
                text_color = "white" if abs(value) > 0.6 else "black"
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", color=text_color, fontsize=9,
                )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="r")
    ax.set_title(title)
    return _figure_to_png_bytes(fig)


def render_target_classification(counts: dict[str, int]) -> bytes:
    """
    Bar chart распределения классов целевой переменной.

    Используется для задач классификации, где target дискретный. Для
    регрессии нужна гистограмма — см. `render_target_regression`.

    Args:
        counts: словарь {класс: количество} из meta_features.target_value_counts.

    Returns:
        PNG-изображение в виде байтов.
    """
    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [str(k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(labels, values, color=COLOR_TARGET, edgecolor="white")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество")
    ax.set_title("Распределение целевой переменной")
    fig.autofmt_xdate(rotation=30)
    return _figure_to_png_bytes(fig)


def render_target_regression(values: list[float]) -> bytes:
    """
    Гистограмма распределения непрерывной целевой переменной.

    Используется для задач регрессии. Для классификации — bar chart
    счётчиков классов, см. `render_target_classification`.

    Args:
        values: одномерный массив числовых значений target.

    Returns:
        PNG-изображение в виде байтов.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    ax.hist(array, bins=30, color=COLOR_TARGET, edgecolor="white")
    ax.set_xlabel("Значение target")
    ax.set_ylabel("Частота")
    ax.set_title("Распределение целевой переменной")
    return _figure_to_png_bytes(fig)
