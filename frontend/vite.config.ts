import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // plotly.js-dist-min — CommonJS-модуль (UMD-бандл). Pre-bundling в Vite
  // гарантирует корректный ESM-интероп при использовании в нашей обёртке
  // PlotlyChart (см. components/analysis/PlotlyChart.tsx).
  optimizeDeps: {
    include: ['plotly.js-dist-min'],
  },
})
