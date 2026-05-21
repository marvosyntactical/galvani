import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves project sites under https://<user>.github.io/<repo>/
// so all asset URLs need that prefix when building for production.
// In dev (`npm run dev`) base stays "/" so localhost works as-is.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === "build" ? "/galvani/" : "/",
  server: {
    port: 5173,
    host: "127.0.0.1",
  },
}));
