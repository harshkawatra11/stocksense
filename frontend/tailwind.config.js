/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: { primary: '#0d0f14', card: '#1a1d24', hover: '#21242e' },
        border: '#2a2d37',
        text: { primary: '#e8eaed', secondary: '#9aa0ab' },
        green: '#00d4aa',
        red: '#ff4757',
        yellow: '#ffa502',
        blue: '#3d85c8',
        accent: '#5c6bc0',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
    },
  },
  plugins: [],
}
