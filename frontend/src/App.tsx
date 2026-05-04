import { Suspense, lazy } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { Header } from "./components/layout/Header";
import { ServerStatus } from "./components/ServerStatus";
import { RequireAuth } from "./components/layout/RequireAuth";
import { RequireAdmin } from "./components/layout/RequireAdmin";
import { Landing } from "./pages/Landing";
import { Login } from "./pages/Login";
import { Register } from "./pages/Register";
import { Profile } from "./pages/Profile";
import { Upload } from "./pages/Upload";

// AnalysisResult тащит за собой Plotly (~3 МБ) — выносим в отдельный chunk,
// чтобы main bundle оставался лёгким и не падал на init из-за тяжёлой
// зависимости. Plotly грузится только когда пользователь открыл /analyses/:id.
const AnalysisResult = lazy(() =>
  import("./pages/AnalysisResult").then((m) => ({ default: m.AnalysisResult })),
);

// History — лёгкая страница со списком; lazy ради единого паттерна
// code-splitting (как у AnalysisResult), main bundle от неё не растёт.
const History = lazy(() => import("./pages/History"));

// Admin: защищённая страница, грузим только когда открыта.
const Admin = lazy(() => import("./pages/Admin"));

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
        <Header />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route element={<RequireAuth />}>
              <Route path="/upload" element={<Upload />} />
              <Route path="/profile" element={<Profile />} />
              <Route
                path="/history"
                element={
                  <Suspense fallback={<RouteSpinner />}>
                    <History />
                  </Suspense>
                }
              />
              <Route
                path="/analyses/:id"
                element={
                  <Suspense fallback={<RouteSpinner />}>
                    <AnalysisResult />
                  </Suspense>
                }
              />
              <Route element={<RequireAdmin />}>
                <Route
                  path="/admin"
                  element={
                    <Suspense fallback={<RouteSpinner />}>
                      <Admin />
                    </Suspense>
                  }
                />
              </Route>
            </Route>
          </Routes>
        </main>
        <div className="fixed bottom-4 right-4">
          <ServerStatus />
        </div>
      </div>
    </BrowserRouter>
  );
}

function RouteSpinner() {
  return (
    <div className="flex h-[60vh] items-center justify-center text-slate-500">
      <Loader2 className="h-6 w-6 animate-spin" />
    </div>
  );
}
