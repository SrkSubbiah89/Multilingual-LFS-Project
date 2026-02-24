/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        arabic: ["Noto Sans Arabic", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};
