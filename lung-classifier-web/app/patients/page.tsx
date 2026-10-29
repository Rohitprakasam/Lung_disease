"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { apiFetch } from '@/lib/api';

interface PatientItem {
    id: number;
    uid: string;
    name: string;
    birth_year?: number;
    sex: string;
}

export default function PatientsPage() {
    const router = useRouter();
    const [doctor, setDoctor] = useState<any>(null);
    const [patients, setPatients] = useState<PatientItem[]>([]);
    const [uidQuery, setUidQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [showRegister, setShowRegister] = useState(false);
    const [newPatient, setNewPatient] = useState({ name: '', birth_year: 1990, sex: 'Male', phone: '+91' });
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const storedDoc = sessionStorage.getItem('doctor');
        if (!storedDoc) {
            router.push('/login');
            return;
        }
        setDoctor(JSON.parse(storedDoc));
        fetchRecentPatients();
    }, [router]);

    const fetchRecentPatients = async () => {
        setLoading(true);
        try {
            const res = await apiFetch('/api/patients');
            if (res.ok) {
                const data = await res.json();
                setPatients(data.patients || []);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!uidQuery.trim()) {
            fetchRecentPatients();
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const res = await apiFetch(`/api/patients?uid=${encodeURIComponent(uidQuery.trim())}`);
            if (res.ok) {
                const data = await res.json();
                setPatients(data.patients || []);
                if (data.patients.length === 0) {
                    setError(`No patient found matching UID "${uidQuery}". Check character verification failed or record does not exist.`);
                }
            }
        } catch (err: any) {
            setError(err.message || 'Error searching patient');
        } finally {
            setLoading(false);
        }
    };

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const res = await apiFetch('/api/patients', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newPatient),
            });
            if (res.ok) {
                const created = await res.json();
                setShowRegister(false);
                router.push(`/patients/${created.uid}`);
            } else {
                const errData = await res.json();
                setError(errData.error || 'Failed to register patient');
            }
        } catch (err: any) {
            setError(err.message);
        }
    };

    return (
        <div className="min-h-screen bg-[#121212] text-cyan-400 font-sans p-6 max-w-5xl mx-auto">
            {/* Top Navigation Bar */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-gray-800 pb-4 mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-white">Universal Scan Memory Portal</h1>
                    {doctor && (
                        <p className="text-sm text-gray-400">
                            Logged in: <span className="text-cyan-300 font-semibold">{doctor.name}</span> ({doctor.center.name} - {doctor.center.code})
                        </p>
                    )}
                </div>
                <div className="flex gap-3">
                    <Link href="/class" className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-green-500 text-black font-bold rounded-lg text-sm hover:scale-105 transition-transform">
                        + New Diagnostic Scan
                    </Link>
                    <Link href="/phone-sim" className="px-3 py-2 bg-gray-800 text-cyan-300 border border-cyan-800 rounded-lg text-sm hover:bg-gray-700">
                        📱 Phone Simulator
                    </Link>
                    <button
                        onClick={() => { sessionStorage.clear(); router.push('/login'); }}
                        className="px-3 py-2 bg-red-950/60 border border-red-700 text-red-400 rounded-lg text-sm"
                    >
                        Logout
                    </button>
                </div>
            </div>

            {/* Search Bar & Register CTA */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <form onSubmit={handleSearch} className="md:col-span-2 flex gap-2">
                    <input
                        type="text"
                        placeholder="Search by Patient UID (e.g. PT-7K3M-9Q2D)"
                        value={uidQuery}
                        onChange={(e) => setUidQuery(e.target.value)}
                        className="flex-1 bg-gray-900 border border-cyan-700 rounded-lg p-3 text-white focus:border-cyan-400 outline-none"
                    />
                    <button type="submit" className="px-6 py-3 bg-cyan-600 text-black font-bold rounded-lg hover:bg-cyan-500">
                        Search
                    </button>
                </form>
                <button
                    onClick={() => setShowRegister(!showRegister)}
                    className="py-3 bg-gray-800 border border-gray-600 text-white rounded-lg hover:border-cyan-400"
                >
                    {showRegister ? 'Cancel Registration' : '+ Register New Patient'}
                </button>
            </div>

            {/* Quick Demo Pre-filled Patient Buttons */}
            <div className="mb-6 p-3 bg-gray-900/80 border border-gray-800 rounded-lg text-xs flex flex-wrap items-center gap-2">
                <span className="text-gray-400">Demo Patient Quick Load:</span>
                <button onClick={() => { setUidQuery('PT-7K3M-9Q2D'); router.push('/patients/PT-7K3M-9Q2D'); }} className="px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-cyan-300 rounded border border-cyan-800">
                    Ramesh Kumar (PT-7K3M-9Q2D) - 3 Centers
                </button>
                <button onClick={() => { setUidQuery('PT-4H8N-2K9B'); router.push('/patients/PT-4H8N-2K9B'); }} className="px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-cyan-300 rounded border border-cyan-800">
                    Anita Sharma (PT-4H8N-2K9B)
                </button>
            </div>

            {/* Registration Modal Form */}
            {showRegister && (
                <form onSubmit={handleRegister} className="mb-6 p-6 bg-gray-900 border border-cyan-500 rounded-xl space-y-4">
                    <h2 className="text-lg font-bold text-white">Register Patient</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <input
                            type="text"
                            placeholder="Full Name"
                            value={newPatient.name}
                            onChange={(e) => setNewPatient({ ...newPatient, name: e.target.value })}
                            required
                            className="bg-gray-800 p-2.5 rounded text-white border border-gray-700"
                        />
                        <input
                            type="number"
                            placeholder="Birth Year"
                            value={newPatient.birth_year}
                            onChange={(e) => setNewPatient({ ...newPatient, birth_year: parseInt(e.target.value) })}
                            required
                            className="bg-gray-800 p-2.5 rounded text-white border border-gray-700"
                        />
                        <select
                            value={newPatient.sex}
                            onChange={(e) => setNewPatient({ ...newPatient, sex: e.target.value })}
                            className="bg-gray-800 p-2.5 rounded text-white border border-gray-700"
                        >
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Other">Other</option>
                        </select>
                        <input
                            type="text"
                            placeholder="Mobile Phone (for OTP consent)"
                            value={newPatient.phone}
                            onChange={(e) => setNewPatient({ ...newPatient, phone: e.target.value })}
                            required
                            className="bg-gray-800 p-2.5 rounded text-white border border-gray-700"
                        />
                    </div>
                    <button type="submit" className="py-2.5 px-6 bg-gradient-to-r from-cyan-400 to-green-400 text-black font-bold rounded-lg">
                        Submit &amp; Generate Patient UID
                    </button>
                </form>
            )}

            {error && (
                <div className="mb-4 p-3 bg-red-950/60 border border-red-500 rounded text-red-400 text-sm">
                    {error}
                </div>
            )}

            {/* Patients Directory Grid */}
            <div className="space-y-3">
                <h2 className="text-sm uppercase tracking-wider text-gray-500 font-semibold">
                    {uidQuery ? 'Search Result' : 'Recent Center Patients'}
                </h2>

                {loading ? (
                    <p className="text-gray-400 py-6 text-center">Loading patient records...</p>
                ) : patients.length === 0 ? (
                    <div className="p-8 bg-gray-900 rounded-lg text-center text-gray-400 border border-gray-800">
                        No patients found. Enter an exact Patient UID to look up longitudinal scans across hospitals.
                    </div>
                ) : (
                    patients.map((p) => (
                        <div
                            key={p.uid}
                            className="p-4 bg-gray-900 border border-gray-800 rounded-lg hover:border-cyan-500 transition-colors flex flex-col md:flex-row justify-between items-start md:items-center gap-3"
                        >
                            <div>
                                <div className="flex items-center gap-3">
                                    <h3 className="text-lg font-bold text-white">{p.name}</h3>
                                    <span className="px-2.5 py-0.5 bg-cyan-950 border border-cyan-500 text-cyan-300 text-xs font-mono rounded">
                                        {p.uid}
                                    </span>
                                </div>
                                <p className="text-xs text-gray-400 mt-1">
                                    Sex: {p.sex} | Born: {p.birth_year ?? 'N/A'}
                                </p>
                            </div>
                            <Link
                                href={`/patients/${p.uid}`}
                                className="px-4 py-2 bg-cyan-500 text-black font-bold text-sm rounded hover:bg-cyan-400 transition-colors"
                            >
                                Open Scan Timeline &rarr;
                            </Link>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

