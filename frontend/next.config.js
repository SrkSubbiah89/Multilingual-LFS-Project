/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Dev server is accessed through public tunnels (ngrok/Cloudflare) for
  // pilot testing, not just localhost — Next.js 14's cross-origin dev
  // warning otherwise flags every request. Currently non-blocking (just a
  // warning), but explicit here since it's exactly this project's real
  // deployment pattern, not a hypothetical future need.
  allowedDevOrigins: [
    "*.ngrok-free.dev",
    "*.ngrok-free.app",
    "*.trycloudflare.com",
    "192.168.0.163",
  ],
};

module.exports = nextConfig;
