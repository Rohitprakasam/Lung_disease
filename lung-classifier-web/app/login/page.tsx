"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { apiFetch } from '@/lib/api';

export default function LoginPage() {
    const router = useRouter();
    const [email, setEmail] = useState('dr.meena@cityhospital.org');
    const [password, setPassword] = useState('DemoPass123!');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const res = await apiFetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.error || 'Login failed');
            }

            // Persist JWT token and user profile
            sessionStorage.setItem('token', data.token);
            sessionStorage.setItem('doctor', JSON.stringify(data.doctor));
            localStorage.setItem('token', data.token);

            router.push('/patients');
        } catch (err: any) {
            setError(err.message || 'Failed to authenticate');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#121212] text-cyan-400 font-sans flex flex-col items-center justify-center p-4">
            <div className="absolute top-4 left-4">
                <Link href="/" className="text-gray-400 hover:text-white transition-colors">
                    &larr; Back to Home
                </Link>
            </div>

            <div className="bg-gray-900 border-2 border-cyan-500 rounded-xl p-8 shadow-[0_0_30px_rgba(0,225,255,0.3)] w-full max-w-md">
                <div className="text-center mb-6">
                    <h1 className="text-3xl font-bold text-white mb-2">Clinical Portal Login</h1>
                    <p className="text-sm text-gray-400">Universal Scan Memory &amp; Diagnostics</p>
                </div>

                <form onSubmit={handleLogin} className="flex flex-col gap-4">
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Clinical Email</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            className="w-full bg-gray-800 text-white border border-gray-600 rounded p-2.5 focus:border-cyan-400 outline-none"
                        />
                    </div>

                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className="w-full bg-gray-800 text-white border border-gray-600 rounded p-2.5 focus:border-cyan-400 outline-none"
                        />
                    </div>

                    <div className="mt-2 text-xs text-gray-400 bg-gray-800/60 p-2 rounded border border-gray-700">
                        <p className="font-semibold text-cyan-300">Quick Demo Accounts:</p>
                        <p>&bull; <span className="text-gray-200">dr.meena@cityhospital.org</span> (City Hospital)</p>
                        <p>&bull; <span className="text-gray-200">dr.sarah@greenvalley.org</span> (Green Valley Clinic)</p>
                        <p>&bull; <span className="text-gray-200">admin@cityhospital.org</span> (Audit Admin)</p>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="mt-4 py-3 bg-gradient-to-r from-cyan-400 to-green-400 text-black font-bold rounded-lg hover:scale-105 transition-transform"
                    >
                        {loading ? 'Authenticating...' : 'Sign In'}
                    </button>
                </form>

                {error && (
                    <div className="mt-4 p-3 bg-red-950/60 border border-red-500 rounded text-red-400 text-sm text-center">
                        {error}
                    </div>
                )}
            </div>
        </div>
    );
}

