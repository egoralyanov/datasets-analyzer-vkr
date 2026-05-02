import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // react-plotly.js и plotly.js-dist-min — CommonJS-модули. Vite по умолчанию
  // не пре-бандлит их в ESM-обёртку для production-сборки rolldown'ом, и
  // default-импорт получает объект `{default: ...}` вместо самой функции.
  // Принудительный pre-bundling решает проблему ESM/CJS interop.
  optimizeDeps: {
    include: ['react-plotly.js', 'plotly.js-dist-min'],
  },
})
