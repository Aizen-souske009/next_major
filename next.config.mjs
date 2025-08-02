/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
        // either domains OR remotePatterns — domains is simpler
        domains: [
            'avatars.githubusercontent.com', // GitHub avatars
            'lh3.googleusercontent.com',     // Google profile photos
            'localhost',                     // Local uploaded images
            'images.unsplash.com',          // Test images
        ],
        remotePatterns: [
            {
                protocol: 'http',
                hostname: 'localhost',
                port: '3000',
                pathname: '/uploads/**',
            },
        ],
    },
    reactStrictMode: true,
};

export default nextConfig;
