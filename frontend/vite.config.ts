import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/vllm-metrics': {
        target: 'http://10.184.94.22:30069',
        changeOrigin: true,
        rewrite: () => '/metrics',
      },
    }
  }
})