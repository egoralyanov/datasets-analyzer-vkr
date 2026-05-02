import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Header } from "./components/layout/Header";
import { ServerStatus } from "./components/ServerStatus";
import { RequireAuth } from "./components/layout/RequireAuth";
import { Landing } from "./pages/Landing";
import { Login } from "./pages/Login";
import { Register } from "./pages/Register";
import { Profile } from "./pages/Profile";
import { Upload } from "./pages/Upload";
import { AnalysisResult } from "./pages/AnalysisResult";

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
              <Route path="/analyses/:id" element={<AnalysisResult />} />
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
