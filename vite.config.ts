import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      name: "AnimaleseVoiceMix",
      formats: ["es"],
      fileName: () => "animalese-voice-mix.js",
    },
    target: "es2022",
  },
});
