'use client';

import { signIn } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { FcGoogle } from 'react-icons/fc';
import { FaGithub } from 'react-icons/fa';
import Link from 'next/link';

export default function LoginPage() {
    return (
        <main className="flex min-h-screen items-center justify-center bg-gradient-to-b from-black via-gray-900 to-black text-white px-4">
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl shadow-xl p-10 max-w-md w-full text-center">
                <h1 className="text-3xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-pink-500 to-blue-500">
                    Sign In
                </h1>
                <p className="text-gray-300 mb-8">
                    Welcome to <span className="font-semibold text-purple-400">DeepTrace AI</span> – Choose your login method
                </p>

                {/* Google Login */}
                <Button
                    onClick={() => signIn('google', { callbackUrl: '/dashboard' })}
                    className="w-full flex items-center justify-center gap-3 bg-white text-black hover:bg-gray-200 transition-all duration-200 mb-4"
                >
                    <FcGoogle size={22} />
                    Sign in with Google
                </Button>

                {/* GitHub Login */}
                <Button
                    onClick={() => signIn('github', { callbackUrl: '/dashboard' })}
                    className="w-full flex items-center justify-center gap-3 bg-gray-900 hover:bg-gray-800 transition-all duration-200"
                >
                    <FaGithub size={22} />
                    Sign in with GitHub
                </Button>

                {/* Back to home */}
                <p className="mt-6 text-sm text-gray-400">
                    Want to know more?{' '}
                    <Link href="/" className="text-purple-400 hover:underline">
                        Go to Landing Page
                    </Link>
                </p>
            </div>
        </main>
    );
}
