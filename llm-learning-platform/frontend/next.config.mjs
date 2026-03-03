/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  distDir: 'dist',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
  env: {
    NEXT_PUBLIC_API_URL: 'https://047b-154-161-146-65.ngrok-free.app',
    NEXT_PUBLIC_WS_URL: 'wss://047b-154-161-146-65.ngrok-free.app',
  },
};

export default nextConfig;
