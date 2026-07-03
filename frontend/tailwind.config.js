/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101418",
        panel: "#f7f9fb",
        line: "#d7dee8",
        snapdragon: "#c61d38",
        npu: "#0f8b8d",
        cloud: "#4169e1",
        caution: "#b7791f"
      }
    },
  },
  plugins: [],
};
