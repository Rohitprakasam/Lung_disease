"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { apiFetch } from '@/lib/api';

interface OtpMessage {
    phone: string;
    otp: string;
    consent_id: number;
    patient_uid: string;
    doctor_name: string;
    center_name: string;
    requested_at: string;
}

export default function PhoneSimPage() {
    const [messages, setMessages] = useState<OtpMessage[]>([]);
    const [loading, setLoading] = useState(false);
    const [revoking, setRevoking] = useState<number | null>(null);
    const [revokeStatus, setRevokeStatus] = useState<string | null>(null);

    const fetchInbox = async () => {
        setLoading(true);
        try {
            const res = await apiFetch('/api/demo/otp-inbox');
            if (res.ok) {
                const data = await res.json();
                setMessages(data.messages || []);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchInbox();
        const interval = setInterval(fetchInbox, 3000);
        return () => clearInterval(interval);
    }, []);

    const handleRevoke = async (consentId: number) => {
        setRevoking(consentId);
        setRevokeStatus(null);
        try {
            const res = await apiFetch(`/api/consents/${consentId}/revoke`, {
                method: 'POST',
            });
            const data = await res.json();
            if (res.ok) {
                setRevokeStatus(`Consent #${consentId} revoked successfully.`);
                fetchInbox();
            } else {
                setRevokeStatus(data.error || 'Failed to revoke consent');
            }
        } catch (err: any) {
            setRevokeStatus(err.message);
        } finally {
            setRevoking(null);
        }
    };

    return (
        <div className="min-h-screen bg-[#121212] text-cyan-400 font-sans p-6 flex flex-col items-center justify-center">
            <div className="absolute top-4 left-4">
                <Link href="/patients" className="text-gray-400 hover:text-white transition-colors text-sm">
                    &larr; Back to Patients Directory
                </Link>
            </div>

            {/* Mobile Phone Device Mockup */}
            <div className="w-full max-w-sm bg-black border-4 border-gray-700 rounded-3xl p-4 shadow-[0_0_40px_rgba(0,0,0,0.8)] relative">
                {/* Speaker Notch */}
                <div className="w-24 h-4 bg-gray-800 rounded-full mx-auto mb-4" />

                <div className="flex justify-between items-center border-b border-gray-800 pb-2 mb-4 text-xs text-gray-400">
                    <span className="font-bold text-white">📱 Patient Phone Simulator</span>
                    <button onClick={fetchInbox} className="text-cyan-400 hover:underline">
                        Refresh
                    </button>
                </div>

                {revokeStatus && (
                    <div className="mb-3 p-2 bg-yellow-950/60 border border-yellow-500 rounded text-yellow-300 text-[11px] text-center">
                        {revokeStatus}
                    </div>
                )}

                <div className="space-y-3 min-h-[380px] max-h-[480px] overflow-y-auto pr-1">
                    {loading && messages.length === 0 ? (
                        <p className="text-gray-500 text-center text-xs py-10">Listening for clinical OTP notifications...</p>
                    ) : messages.length === 0 ? (
                        <div className="text-center py-12 text-gray-500 text-xs">
                            <p className="text-2xl mb-2">💬</p>
                            No incoming messages. When a doctor requests cross-center access, the OTP will appear here in real time.
                        </div>
                    ) : (
                        messages.slice().reverse().map((msg, idx) => (
                            <div key={idx} className="p-3 bg-gray-900 border border-cyan-800 rounded-xl space-y-2 text-xs">
                                <div className="flex justify-between items-center text-[10px] text-gray-400">
                                    <span>From: MOHFW-CONSENT</span>
                                    <span>{new Date(msg.requested_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                </div>

                                <p className="text-gray-300">
                                    <strong className="text-white">{msg.doctor_name}</strong> from <span className="text-cyan-300">{msg.center_name}</span> requests 30-day access to your longitudinal chest X-rays.
                                </p>

                                <div className="p-2 bg-black/60 border border-cyan-500 rounded text-center">
                                    <span className="text-[10px] text-gray-400 block">YOUR 6-DIGIT CONSENT OTP</span>
                                    <span className="text-2xl font-mono font-bold text-green-400 tracking-widest">{msg.otp}</span>
                                </div>

                                <div className="pt-1 flex justify-between items-center">
                                    <span className="text-[10px] text-gray-500">Patient: {msg.patient_uid}</span>
                                    <button
                                        onClick={() => handleRevoke(msg.consent_id)}
                                        disabled={revoking === msg.consent_id}
                                        className="px-2 py-1 bg-red-950/80 hover:bg-red-900 border border-red-700 text-red-300 rounded text-[10px]"
                                    >
                                        {revoking === msg.consent_id ? 'Revoking...' : 'Revoke Consent'}
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Home Indicator bar */}
                <div className="w-32 h-1 bg-gray-600 rounded-full mx-auto mt-4" />
            </div>
        </div>
    );
}

