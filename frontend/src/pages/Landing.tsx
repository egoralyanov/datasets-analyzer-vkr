// Главная страница.
//
// Стиль (Sprint 5, Phase 3.3): scientific/archive — серифный hero, под ним
// статичный «препарат» — уменьшённый кусок страницы анализа на эталонном
// датасете Iris, чтобы гость сразу увидел, что именно генерирует система.
// Никакого API, никакой интерактивности — данные захардкожены ниже.
//
// Авторизованный пользователь обслуживается отдельной веткой (см. Phase 3.4) —
// hero сворачивается, вместо демо появляется dashboard.
import { Link } from "react-router-dom";
import { useAuthStore } from "../store/authStore";

export function Landing() {
  const user = useAuthStore((s) => s.user);

  if (user) {
    return <AuthenticatedStub />;
  }

  return <GuestLanding />;
}

// Phase 3.3: временная заглушка для авторизованного состояния — будет
// переписана в Phase 3.4 как полноценный dashboard. Сохраняет привычный
// сценарий «есть кнопка → перейти в загрузку», чтобы не блокировать рабочий
// поток между шагами.
function AuthenticatedStub() {
  return (
    <div className="mx-auto max-w-[1100px] px-8 py-16 lg:px-16">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        АНАЛИЗАТОР · ВЫ ВОШЛИ
      </p>
      <h1 className="mt-3 font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900">
        Готовы к анализу
      </h1>
      <div className="mt-6">
        <Link
          to="/upload"
          className="inline-flex items-center border border-ink-700 bg-ink-700 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
        >
          ЗАГРУЗИТЬ ДАТАСЕТ
        </Link>
      </div>
    </div>
  );
}

function GuestLanding() {
  return (
    <div className="mx-auto max-w-[1100px] px-8 py-16 lg:px-16">
      {/* Hero */}
      <section className="border-b border-paper-300 pb-12">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
          ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА · МГТУ ИМ. Н.Э. БАУМАНА · ИУ5
        </p>
        <h1 className="mt-3 max-w-3xl font-serif text-[3rem] font-bold leading-[1.05] tracking-tight text-paper-900">
          Анализатор
        </h1>
        <p className="mt-4 max-w-2xl font-serif text-[1.0625rem] leading-relaxed text-paper-600">
          Интеллектуальная система анализа наборов данных для решения задач
          машинного обучения. Загрузите CSV или XLSX — получите рекомендации по
          типу задачи, проверки качества данных и подбор похожих датасетов.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link
            to="/register"
            className="inline-flex items-center border border-ink-700 bg-ink-700 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
          >
            СОЗДАТЬ АККАУНТ
          </Link>
          <Link
            to="/login"
            className="inline-flex items-center border border-ink-700 bg-paper-50 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-ink-700 transition-colors hover:bg-ink-700 hover:text-paper-50"
          >
            ВОЙТИ
          </Link>
        </div>
      </section>

      {/* Demo specimen */}
      <section className="mt-12">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
          ПРИМЕР АНАЛИЗА · ДАТАСЕТ IRIS · ДЕМОНСТРАЦИЯ
        </p>
        <div className="mt-4 border border-paper-300 bg-paper-50 px-8 py-10 lg:px-12">
          <IrisDemo />
        </div>
      </section>
    </div>
  );
}

// Статичный «срез» страницы анализа на Iris. Данные захардкожены —
// никаких API-вызовов, никакого state, не имитирует polling или interactive.
// Соответствует визуальному языку реальной AnalysisResult.tsx.
function IrisDemo() {
  return (
    <>
      <div>
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
          ОТЧЁТ ОБ АНАЛИЗЕ ДАТАСЕТА
        </p>
        <h2 className="mt-2 font-serif text-[1.75rem] font-bold leading-tight tracking-tight text-paper-900">
          Iris.csv
        </h2>
        <dl className="mt-4 grid grid-cols-1 gap-x-8 gap-y-1 border-y border-paper-300 py-3 sm:grid-cols-[10rem_1fr_10rem_1fr]">
          <SpecimenField label="ID анализа" value="01b3-iris-demo" />
          <SpecimenField label="Статус" value="DONE" />
          <SpecimenField label="Завершён" value="2026-05-06, 12:00" />
          <SpecimenField label="Target" value="«species»" />
        </dl>
      </div>

      <div className="mt-10 space-y-12">
        <DemoSection number={1} title="Профайл данных">
          <div className="border border-paper-300 bg-paper-50">
            <dl className="divide-y divide-paper-200">
              <DemoRow label="Имя файла" value="Iris.csv" />
              <DemoRow label="Строк × столбцов" value="150 × 5" />
              <DemoRow label="Размер в памяти" value="0.01 МБ" />
              <DemoRow label="Типы колонок" value="float64 4 · object 1" />
              <DemoRow label="Пропусков (всего)" value="0.00%" />
              <DemoRow label="Дубликатов строк" value="0.00%" />
              <DemoRow
                label="Целевая переменная"
                value={
                  <>
                    <span className="font-mono">«species»</span>
                    <span className="ml-2 text-paper-500">— категориальная</span>
                  </>
                }
                mono={false}
              />
            </dl>
          </div>
        </DemoSection>

        <DemoSection number={4} title="Рекомендация типа задачи">
          <blockquote className="border-l-[3px] border-ink-700 pl-6">
            <p className="font-serif text-[2rem] font-semibold leading-tight tracking-tight text-ink-900">
              Многоклассовая классификация
            </p>
            <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 font-sans text-xs uppercase tracking-wider">
              <span className="text-paper-500">
                CONFIDENCE{" "}
                <span className="ml-1 font-mono normal-case tracking-normal text-success-700">
                  0.950
                </span>
                <span className="ml-1 font-mono normal-case tracking-normal text-paper-500">
                  (95%)
                </span>
              </span>
              <span className="text-paper-500">
                SOURCE{" "}
                <span className="ml-1 font-mono normal-case tracking-normal text-ink-700">
                  rules
                </span>
              </span>
            </div>
          </blockquote>
        </DemoSection>
      </div>
    </>
  );
}

function DemoSection({
  number,
  title,
  children,
}: {
  number: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <header className="mb-5 border-b border-paper-300 pb-2">
        <h3 className="flex items-baseline gap-3 font-serif text-[1.375rem] font-semibold leading-snug tracking-tight text-paper-900">
          <span className="font-sans text-base font-medium text-paper-400">
            §{number}
          </span>
          {title}
        </h3>
      </header>
      {children}
    </section>
  );
}

function DemoRow({
  label,
  value,
  mono = true,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="grid grid-cols-[12rem_1fr] gap-6 px-5 py-3">
      <dt className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd className={`text-sm text-paper-800 ${mono ? "font-mono" : "font-sans"}`}>
        {value}
      </dd>
    </div>
  );
}

function SpecimenField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-3 py-1">
      <dt className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd className="font-mono text-xs text-paper-800">{value}</dd>
    </div>
  );
}
