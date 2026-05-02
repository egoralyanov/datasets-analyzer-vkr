import { Link } from "react-router-dom";
import { Upload, BarChart3, FileText } from "lucide-react";
import { useAuthStore } from "../store/authStore";

export function Landing() {
  const user = useAuthStore((s) => s.user);

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <section className="text-center">
        <h1 className="text-4xl font-semibold text-slate-900">Анализатор</h1>
        <p className="mt-3 text-slate-600 max-w-2xl mx-auto">
          Интеллектуальная система анализа наборов данных для решения задач
          машинного обучения. Загрузите CSV или XLSX — получите рекомендации по
          типу задачи, проверки качества данных и подбор похожих датасетов.
        </p>

        <div className="mt-8 flex items-center justify-center gap-3">
          {user ? (
            <button
              type="button"
              disabled
              title="Скоро"
              className="inline-flex items-center gap-2 rounded-md bg-slate-300 px-5 py-2.5 text-sm font-medium text-slate-600 cursor-not-allowed"
            >
              <Upload className="h-4 w-4" />
              Загрузить датасет (скоро)
            </button>
          ) : (
            <>
              <Link
                to="/register"
                className="inline-flex items-center rounded-md bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700"
              >
                Создать аккаунт
              </Link>
              <Link
                to="/login"
                className="inline-flex items-center rounded-md border border-slate-300 bg-white px-5 py-2.5 text-sm font-medium text-slate-900 hover:bg-slate-50"
              >
                Войти
              </Link>
            </>
          )}
        </div>
      </section>

      <section className="mt-16 grid gap-6 sm:grid-cols-3">
        <FeatureCard
          icon={<Upload className="h-5 w-5 text-blue-600" />}
          title="Загрузка датасета"
          text="CSV или XLSX до 100 МБ. Автоопределение кодировки и разделителя."
        />
        <FeatureCard
          icon={<BarChart3 className="h-5 w-5 text-blue-600" />}
          title="Профиль и качество"
          text="Метрики, типы колонок, флаги качества — пропуски, дубли, дисбаланс."
        />
        <FeatureCard
          icon={<FileText className="h-5 w-5 text-blue-600" />}
          title="Рекомендация задачи"
          text="Тип ML-задачи, метрики, baseline и подбор похожих наборов."
        />
      </section>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  text,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center gap-2">
        {icon}
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      </div>
      <p className="mt-2 text-sm text-slate-600">{text}</p>
    </div>
  );
}
