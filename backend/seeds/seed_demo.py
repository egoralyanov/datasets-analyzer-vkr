"""
Подготовка демонстрационного окружения для предзащиты.

Создаёт:
- основной пользовательский аккаунт egor (для дашборда)
- административный аккаунт egoradm (для админ-панели)
- N пользователей-массовки (по умолчанию 30) — чтобы админка показывала
  пагинированный список и сводку с правдоподобными цифрами

Заливает датасеты из sklearn для egor (Iris, California Housing,
Breast Cancer, Diabetes) и запускает по ним полный цикл анализа
(profiler → quality → recommender → embedding). Для части анализов
обучает baseline и генерирует PDF-отчёты.

Для массовки случайно раздаёт по 1–2 датасета и по 0–1 анализа, чтобы
колонки «датасеты» и «анализы» в админ-панели не были одинаковыми.

Запуск:

    docker compose exec backend python -m seeds.seed_demo

Скрипт идемпотентен по username/email: если пользователь egor или egoradm
уже есть — переиспользует. Однако массовку он не дедуплицирует и при
повторном запуске создаст ещё 30 user_NNN-аккаунтов.
"""
from __future__ import annotations

import hashlib
import io
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_iris,
)

from app.config import settings
from app.core.db import SessionLocal
from app.core.security import hash_password
from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult
from app.models.dataset import Dataset
from app.models.report import Report
from app.models.user import User
from app.repositories import (
    analysis_repo,
    dataset_repo,
    report_repo,
    user_repo,
)
from app.services.analysis_service import run_analysis
from app.services.baseline_trainer import train_baseline_from_df
from app.services.baseline_orchestrator import _resolve_leakage_columns
from app.services.dataset_service import read_dataset_full
from app.services.report_service import generate_report
from app.utils.jsonb import jsonb_safe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seed_demo")

random.seed(20260514)
np.random.seed(20260514)

DEMO_PASSWORD = "egor1234"
MASSOVKA_PASSWORD = "demo1234"
MASSOVKA_COUNT = 30


# ----- helpers --------------------------------------------------------------


def _save_csv_for_user(
    df: pd.DataFrame, user_id: uuid.UUID
) -> tuple[str, int, str]:
    """
    Сохраняет DataFrame как CSV в storage пользователя.

    Эмулирует поведение file_storage.save_uploaded_file + compute_file_sha256:
    кладёт файл под UUID-именем, возвращает абсолютный путь, размер и SHA-256.
    """
    user_dir = Path(settings.DATASETS_DIR) / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    storage_uuid = uuid.uuid4()
    storage_path = user_dir / f"{storage_uuid}.csv"

    # Сериализуем в bytes сразу, чтобы посчитать хэш без повторного чтения.
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    payload = buf.getvalue()
    storage_path.write_bytes(payload)

    file_hash = hashlib.sha256(payload).hexdigest()
    return str(storage_path), len(payload), file_hash


def _make_iris_df() -> pd.DataFrame:
    bunch = load_iris(as_frame=True)
    df = bunch.frame.copy()
    df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        },
        inplace=True,
    )
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df["species"] = df["target"].map(species_map)
    df.drop(columns=["target"], inplace=True)
    return df


def _make_california_df() -> pd.DataFrame:
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame.copy()
    # Сэмплируем 5000 строк — иначе профайлер на 20640 строках идёт дольше
    # 60 секунд и тяжелит демо-сид.
    return df.sample(n=5000, random_state=42).reset_index(drop=True)


def _make_breast_cancer_df() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df.rename(columns={"target": "diagnosis"}, inplace=True)
    return df


def _make_diabetes_df() -> pd.DataFrame:
    bunch = load_diabetes(as_frame=True)
    df = bunch.frame.copy()
    return df


def _make_small_synthetic_df(seed: int) -> pd.DataFrame:
    """
    Маленький синтетический датасет для массовки — чтобы анализ был быстрым,
    но при этом записи в таблицах datasets/analyses/analysis_results
    выглядели правдоподобно.
    """
    rng = np.random.default_rng(seed)
    n = 200
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(5, 2, n),
            "feature_c": rng.uniform(0, 10, n),
            "category": rng.choice(["alpha", "beta", "gamma"], n),
        }
    )
    df["target"] = (df["feature_a"] + df["feature_b"] * 0.5 > rng.normal(2, 1, n)).astype(int)
    return df


# Каталог датасетов для egor: (имя файла, фабрика DataFrame, target_column).
EGOR_DATASETS = [
    ("iris.csv", _make_iris_df, "species"),
    ("california_housing.csv", _make_california_df, "MedHouseVal"),
    ("breast_cancer.csv", _make_breast_cancer_df, "diagnosis"),
    ("diabetes.csv", _make_diabetes_df, "target"),
]


def _get_or_create_user(
    db,
    *,
    email: str,
    username: str,
    password: str,
    role: str = "user",
) -> User:
    """Идемпотентное создание: если есть — переиспользуем и нормализуем роль."""
    existing = user_repo.get_user_by_email(db, email)
    if existing is None:
        existing = user_repo.get_user_by_username(db, username)
    if existing is not None:
        if existing.role != role:
            existing.role = role
            db.commit()
        return existing
    user = user_repo.create_user(
        db,
        email=email,
        username=username,
        password_hash=hash_password(password),
    )
    # user_repo.create_user не принимает role — сохраняем дефолт 'user' в БД.
    # Если запрошена иная роль, выставляем её отдельным апдейтом.
    if user.role != role:
        user.role = role
        db.commit()
        db.refresh(user)
    return user


def _create_dataset_for_user(
    db,
    *,
    user: User,
    df: pd.DataFrame,
    original_filename: str,
) -> Dataset:
    storage_path, size, file_hash = _save_csv_for_user(df, user.id)
    # Если у пользователя уже есть датасет с таким хэшем — переиспользуем,
    # не плодим дубликатов (соответствует поведению upload endpoint'а).
    existing = dataset_repo.find_by_user_and_hash(db, user.id, file_hash)
    if existing is not None:
        Path(storage_path).unlink(missing_ok=True)
        return existing
    return dataset_repo.create_dataset(
        db,
        user_id=user.id,
        original_filename=original_filename,
        storage_path=storage_path,
        file_size_bytes=size,
        file_hash=file_hash,
        fmt="csv",
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
    )


def _run_analysis_sync(db, dataset: Dataset, target_column: str | None) -> Analysis:
    """
    Запускает run_analysis синхронно вместо BackgroundTask — для seed-скрипта
    нам не нужна асинхронность, наоборот хочется дождаться завершения.
    """
    analysis = analysis_repo.create_analysis(
        db,
        dataset_id=dataset.id,
        user_id=dataset.user_id,
        target_column=target_column,
    )
    # run_analysis открывает свою сессию через SessionLocal — наша сессия
    # тут не используется. После завершения нужно обновить локальный объект.
    run_analysis(analysis.id, SessionLocal)
    db.expire(analysis)
    db.refresh(analysis)
    return analysis


def _train_baseline_sync(db, analysis: Analysis) -> None:
    """
    Синхронный аналог run_baseline_async для сид-скрипта.

    Шаги те же, что в orchestrator'е, минус asyncio.to_thread (нам не нужен
    event loop). На ошибке записываем baseline_status='failed' с короткой
    ошибкой и продолжаем с другими записями.
    """
    ar = db.get(AnalysisResult, analysis.id)
    if ar is None or analysis.status != "done":
        return
    task_rec = ar.task_recommendation or {}
    task_type = str(task_rec.get("task_type_code") or "NOT_READY")
    dataset = analysis.dataset
    df = read_dataset_full(Path(dataset.storage_path), dataset.format)
    leakage_cols = _resolve_leakage_columns(db, analysis.id)
    ar.baseline_status = "running"
    ar.baseline_error = None
    db.commit()
    try:
        result = train_baseline_from_df(
            df,
            ar.meta_features or {},
            leakage_cols,
            analysis.target_column or "",
            task_type,
        )
        ar = db.get(AnalysisResult, analysis.id)
        if ar is None:
            return
        ar.baseline = jsonb_safe(result)
        ar.baseline_status = "done"
        ar.baseline_error = None
        db.commit()
    except Exception as exc:  # noqa: BLE001 — best-effort в сид-скрипте
        logger.exception("Baseline training failed for analysis %s", analysis.id)
        db.rollback()
        ar = db.get(AnalysisResult, analysis.id)
        if ar is not None:
            ar.baseline_status = "failed"
            ar.baseline_error = str(exc)[:500]
            db.commit()


def _generate_report_sync(db, analysis: Analysis) -> Report:
    report = report_repo.create_report(
        db,
        analysis_id=analysis.id,
        user_id=analysis.user_id,
    )
    generate_report(report.id, SessionLocal)
    db.expire(report)
    db.refresh(report)
    return report


# ----- main -----------------------------------------------------------------


def seed_main_users(db) -> tuple[User, User]:
    egor = _get_or_create_user(
        db, email="egor@example.com", username="egor", password=DEMO_PASSWORD
    )
    egoradm = _get_or_create_user(
        db,
        email="egoradm@example.com",
        username="egoradm",
        password=DEMO_PASSWORD,
        role="admin",
    )
    logger.info("Main users ready: egor=%s, egoradm=%s", egor.id, egoradm.id)
    return egor, egoradm


def seed_massovka_users(db, count: int = MASSOVKA_COUNT) -> list[User]:
    users: list[User] = []
    password_hash = hash_password(MASSOVKA_PASSWORD)
    for i in range(1, count + 1):
        username = f"user_{i:03d}"
        email = f"{username}@example.com"
        existing = user_repo.get_user_by_username(db, username)
        if existing is not None:
            users.append(existing)
            continue
        user = User(email=email, username=username, password_hash=password_hash)
        db.add(user)
        users.append(user)
    db.commit()
    for u in users:
        db.refresh(u)
    logger.info("Massovka users ready: %d total", len(users))
    return users


def seed_egor_activity(db, egor: User) -> None:
    """Полный цикл: 4 датасета, 4 анализа, 2 baseline, 2 PDF."""
    logger.info("Seeding egor activity (4 datasets)...")
    datasets_and_targets: list[tuple[Dataset, str | None]] = []
    for filename, factory, target in EGOR_DATASETS:
        df = factory()
        ds = _create_dataset_for_user(
            db, user=egor, df=df, original_filename=filename
        )
        datasets_and_targets.append((ds, target))
        logger.info(
            "  uploaded %s (n_rows=%d, n_cols=%d) → dataset_id=%s",
            filename,
            ds.n_rows,
            ds.n_cols,
            ds.id,
        )

    analyses: list[Analysis] = []
    for ds, target in datasets_and_targets:
        logger.info("  running analysis for %s (target=%s)...", ds.original_filename, target)
        analysis = _run_analysis_sync(db, ds, target)
        analyses.append(analysis)
        logger.info(
            "    → status=%s, analysis_id=%s",
            analysis.status,
            analysis.id,
        )

    # baseline — для первых двух done-анализов
    baseline_done = 0
    for analysis in analyses:
        if analysis.status == "done" and baseline_done < 2:
            logger.info("  training baseline for analysis %s...", analysis.id)
            _train_baseline_sync(db, analysis)
            baseline_done += 1

    # PDF — для первых трёх done-анализов
    reports_done = 0
    for analysis in analyses:
        if analysis.status == "done" and reports_done < 3:
            logger.info("  generating PDF for analysis %s...", analysis.id)
            report = _generate_report_sync(db, analysis)
            logger.info("    → report.status=%s", report.status)
            reports_done += 1

    logger.info(
        "egor activity done: %d datasets, %d analyses (done=%d), %d baselines, %d reports",
        len(datasets_and_targets),
        len(analyses),
        sum(1 for a in analyses if a.status == "done"),
        baseline_done,
        reports_done,
    )


def seed_massovka_activity(db, users: list[User]) -> dict[str, int]:
    """Раздаёт массовке датасеты и анализы, чтобы админ-панель выглядела живо."""
    # ~12 пользователей получают по 1–2 датасета, 6 из них — анализ, 2 — PDF.
    dataset_owners = random.sample(users, k=min(12, len(users)))
    logger.info(
        "Seeding massovka activity: %d users get datasets...", len(dataset_owners)
    )

    counts = {"datasets": 0, "analyses": 0, "reports": 0}
    analyses_so_far: list[Analysis] = []

    for i, user in enumerate(dataset_owners):
        n_datasets = random.choice([1, 1, 2])  # bias к 1
        for k in range(n_datasets):
            df = _make_small_synthetic_df(seed=hash((str(user.id), k)) & 0xFFFFFFFF)
            ds = _create_dataset_for_user(
                db,
                user=user,
                df=df,
                original_filename=f"synthetic_{k+1}.csv",
            )
            counts["datasets"] += 1
            # Первые 6 пользователей в выборке получают анализ по первому датасету
            if i < 6 and k == 0:
                analysis = _run_analysis_sync(db, ds, "target")
                counts["analyses"] += 1
                analyses_so_far.append(analysis)

    # PDF — для двух первых done-анализов из массовки
    pdfs_done = 0
    for analysis in analyses_so_far:
        if analysis.status == "done" and pdfs_done < 2:
            _generate_report_sync(db, analysis)
            counts["reports"] += 1
            pdfs_done += 1

    logger.info(
        "Massovka activity done: %d datasets, %d analyses, %d reports",
        counts["datasets"],
        counts["analyses"],
        counts["reports"],
    )
    return counts


def main() -> int:
    started = datetime.now(timezone.utc)
    db = SessionLocal()
    try:
        egor, egoradm = seed_main_users(db)
        massovka = seed_massovka_users(db)
        seed_egor_activity(db, egor)
        try:
            seed_massovka_activity(db, massovka)
        except Exception:
            # Массовка — best-effort. egor + egoradm уже на месте, основной
            # сценарий защиты не сломаем.
            logger.exception("Massovka seeding failed — leaving as-is")
    finally:
        db.close()

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    logger.info("seed_demo completed in %.1f seconds", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
