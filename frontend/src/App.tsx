import { ServerStatus } from "./components/ServerStatus";

export default function App() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center px-4">
        <h1 className="text-4xl font-bold text-blue-600">Анализатор</h1>
        <p className="mt-4 text-gray-600 max-w-2xl">
          Интеллектуальная система анализа наборов данных для решения задач
          машинного обучения
        </p>
      </div>
      <div className="fixed bottom-4 right-4">
        <ServerStatus />
      </div>
    </div>
  );
}
