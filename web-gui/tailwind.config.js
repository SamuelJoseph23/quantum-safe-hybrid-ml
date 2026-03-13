/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
      extend: {
        colors: {
            // Cyber-Security Palette
            'cyber-bg': '#0f172a', // slate-900
            'cyber-dark': '#020617', // slate-950
            'cyber-cyan': '#06b6d4', // cyan-500
            'cyber-green': '#10b981', // emerald-500
            'cyber-red': '#ef4444', // red-500
        },
        fontFamily: {
            sans: ['"Rajdhani"', 'sans-serif'],
            mono: ['"JetBrains Mono"', 'monospace'],
            display: ['"Orbitron"', 'sans-serif'],
        },
        boxShadow: {
            'neon': '0 0 5px theme("colors.cyber-cyan"), 0 0 20px theme("colors.cyber-cyan")',
            'neon-red': '0 0 5px theme("colors.cyber-red"), 0 0 20px theme("colors.cyber-red")',
        }
      },
    },
    plugins: [],
  }
