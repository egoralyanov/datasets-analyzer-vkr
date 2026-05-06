/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: {
          50: "#FAF8F4",
          100: "#F2EEE5",
          200: "#E5E0D5",
          300: "#D2CCBE",
          400: "#A89F8E",
          500: "#6E6657",
          600: "#4F4A40",
          700: "#3A362F",
          800: "#28251F",
          900: "#1A1814",
        },
        ink: {
          50: "#EEF0F5",
          100: "#D6DBE6",
          200: "#A9B2C7",
          300: "#6F7A95",
          500: "#3A4566",
          700: "#1F2A44",
          800: "#161E33",
          900: "#0E1424",
        },
        critical: {
          50: "#F7EEEC",
          500: "#A53A2A",
          700: "#7A2A1E",
        },
        warning: {
          50: "#F7F0E1",
          500: "#B07D2A",
          700: "#7E5618",
        },
        info: {
          50: "#EBEEF6",
          500: "#4A5780",
          700: "#2E3858",
        },
        success: {
          50: "#EDF1E8",
          500: "#5A7A55",
          700: "#3D5639",
        },
      },
      fontFamily: {
        serif: [
          '"IBM Plex Serif"',
          '"Source Serif 4"',
          "Georgia",
          '"Times New Roman"',
          "serif",
        ],
        sans: [
          '"IBM Plex Sans"',
          '"Source Sans 3"',
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
        mono: [
          '"IBM Plex Mono"',
          '"JetBrains Mono"',
          '"SFMono-Regular"',
          "Consolas",
          "monospace",
        ],
      },
      borderRadius: {
        none: "0",
        xs: "2px",
        sm: "3px",
      },
      boxShadow: {
        none: "none",
        overlay: "0 2px 8px rgba(28, 25, 31, 0.08)",
      },
      letterSpacing: {
        wider: "0.04em",
      },
    },
  },
  plugins: [],
};
