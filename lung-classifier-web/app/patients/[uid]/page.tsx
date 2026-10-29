"use client";

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { apiFetch, API_BASE_URL } from '@/lib/api';

interface ScanItem {
    id: number;
    public_id: string;
    taken_at: string;
    center_id: number;
    center_name: string;
    center_code: string;
    classifier_label: string;
    classifier_confidence: number;
    probabilities: Record<string, number>;
    twin_index?: number;
    twin_status?: string;
    doctor_notes?: string;
    doctor_confirmed_label?: string;
}

export default function PatientTimelinePage() {
    const params = useParams();
    const router = useRouter();
    const uid = params.uid as string;

    const [patient, setPatient] = useState<any>(null);
    const [scans, setScans] = useState<ScanItem[]>([]);
    const [hiddenCount, setHiddenCount] = useState<number>(0);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Consent modal state
    const [showConsentModal, setShowConsentModal] = useState(false);
    const [consentNotice, setConsentNotice] = useState<any>(null);
    const [consentId, setConsentId] = useState<number | null>(null);
    const [otp, setOtp] = useState('');
    const [consentLoading, setConsentLoading] = useState(false);
    const [consentSuccess, setConsentSuccess] = useState<string | null>(null);

    // Compare Slider State
    const [sliderVal, setSliderVal] = useState(50);
    const [compareScanA, setCompareScanA] = useState<string | null>(null);
    const [compareScanB, setCompareScanB] = useState<string | null>(null);

    useEffect(() => {
        loadTimeline();
    }, [uid]);

    const loadTimeline = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await apiFetch(`/api/patients/${uid}/timeline`);
            if (!res.ok) {
                if (res.status === 401) {
                    router.push('/login');
                    return;
                }
                const errData = await res.json();
                throw new Error(errData.error || 'Failed to load timeline');
            }
            const data = await res.json();
            setPatient(data.patient);
            setScans(data.scans || []);
            setHiddenCount(data.hidden_other_center_scans || 0);

            if (data.scans && data.scans.length >= 2) {
                setCompareScanA(data.scans[data.scans.length - 1].public_id); // Oldest
                setCompareScanB(data.scans[0].public_id); // Newest
            } else if (data.scans && data.scans.length === 1) {
                setCompareScanA(data.scans[0].public_id);
                setCompareScanB(data.scans[0].public_id);
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleRequestConsent = async () => {
        setConsentLoading(true);
        setError(null);
        try {
            const res = await apiFetch(`/api/patients/${uid}/consent/request`, {
                method: 'POST',
            });
            const data = await res.json();
            if (res.ok) {
                setConsentId(data.consent_id);
                setConsentNotice(data.notice);
                setShowConsentModal(true);
            } else {
                setError(data.error || 'Consent request failed');
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setConsentLoading(false);
        }
    };

    const handleVerifyOtp = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!consentId || !otp) return;

        setConsentLoading(true);
        try {
            const res = await apiFetch(`/api/patients/${uid}/consent/verify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ consent_id: consentId, otp: otp.trim() }),
            });
            const data = await res.json();
            if (res.ok) {
                setConsentSuccess(data.message);
                setShowConsentModal(false);
                setOtp('');
                loadTimeline(); // Reload to fetch newly unlocked cross-center scans
            } else {
                setError(data.error || 'OTP verification failed');
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setConsentLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-[#121212] text-cyan-400 flex items-center justify-center">
                <p className="animate-pulse">Loading Universal Scan Memory...</p>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#121212] text-cyan-400 font-sans p-6 max-w-6xl mx-auto">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-gray-800 pb-4 mb-6">
                <div>
                    <Link href="/patients" className="text-xs text-gray-400 hover:text-white mb-2 inline-block">
                        &larr; Back to Patients Directory
                    </Link>
                    <div className="flex items-center gap-3">
                        <h1 className="text-3xl font-bold text-white">{patient?.name}</h1>
                        <span className="px-3 py-1 bg-cyan-950 border border-cyan-500 font-mono text-cyan-300 text-sm rounded">
                            {patient?.uid}
                        </span>
                        <span className={`px-2.5 py-0.5 text-xs rounded-full border ${patient?.has_active_consent ? 'bg-green-950/80 border-green-500 text-green-300' : 'bg-gray-800 border-gray-600 text-gray-400'}`}>
                            {patient?.has_active_consent ? '🟢 All Connected Centers Authorized (30 Days)' : '🟡 Own Center Access Only'}
                        </span>
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                        Sex: {patient?.sex} | Birth Year: {patient?.birth_year ?? 'N/A'} | Phone: {patient?.phone ?? 'N/A'}
                    </p>
                </div>

                <div className="flex gap-3">
                    <Link
                        href={`/class?patient=${uid}`}
                        className="px-4 py-2 bg-gradient-to-r from-cyan-400 to-green-400 text-black font-bold rounded-lg text-sm hover:scale-105 transition-transform"
                    >
                        + Upload Scan for {patient?.name?.split(' ')[0]}
                    </Link>
                </div>
            </div>

            {/* Cross-center consent alert banner */}
            {hiddenCount > 0 && (
                <div className="mb-6 p-4 bg-yellow-950/40 border border-yellow-600 rounded-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                        <h3 className="text-sm font-bold text-yellow-300">
                            🔒 {hiddenCount} Historical Scans Available at External Clinical Centers
                        </h3>
                        <p className="text-xs text-yellow-200/80 mt-0.5">
                            Under the DPDP Act 2023, patient consent via 6-digit OTP is required to unlock longitudinal records from other medical centers.
                        </p>
                    </div>
                    <button
                        onClick={handleRequestConsent}
                        disabled={consentLoading}
                        className="px-4 py-2 bg-yellow-500 hover:bg-yellow-400 text-black font-bold text-xs rounded-lg shrink-0"
                    >
                        {consentLoading ? 'Requesting...' : 'Request Access (Send OTP)'}
                    </button>
                </div>
            )}

            {consentSuccess && (
                <div className="mb-6 p-3 bg-green-950/60 border border-green-500 rounded text-green-300 text-sm text-center font-semibold">
                    {consentSuccess}
                </div>
            )}

            {error && (
                <div className="mb-6 p-3 bg-red-950/60 border border-red-500 rounded text-red-400 text-sm">
                    {error}
                </div>
            )}

            {/* Trend Chart (Inline SVG Polyline) */}
            {scans.length >= 2 && (
                <div className="mb-8 p-5 bg-gray-900 border border-gray-800 rounded-xl">
                    <div className="flex justify-between items-center mb-3">
                        <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                            Longitudinal Abnormality Trend (Healthy Twin Index)
                        </h2>
                        <span className="text-xs text-gray-400">Baseline healthy cutoff: &le; 20</span>
                    </div>

                    <div className="h-32 w-full flex items-end relative border-b border-gray-700 pb-2">
                        <svg className="w-full h-full overflow-visible">
                            {/* Guideline at cutoff 20% */}
                            <line x1="0" y1="80%" x2="100%" y2="80%" stroke="#374151" strokeDasharray="4" />
                            {/* SVG Trend Polyline */}
                            <polyline
                                fill="none"
                                stroke="#00e1ff"
                                strokeWidth="3"
                                points={scans.map((s, idx) => {
                                    const x = (idx / (scans.length - 1)) * 95 + 2.5;
                                    const y = 100 - (s.twin_index ?? 20);
                                    return `${x}%,${y}%`;
                                }).join(' ')}
                            />
                            {/* Data points */}
                            {scans.map((s, idx) => {
                                const x = (idx / (scans.length - 1)) * 95 + 2.5;
                                const y = 100 - (s.twin_index ?? 20);
                                return (
                                    <circle
                                        key={s.public_id}
                                        cx={`${x}%`}
                                        cy={`${y}%`}
                                        r="5"
                                        fill={s.classifier_label === 'NORMAL' ? '#4ade80' : '#f87171'}
                                        stroke="#121212"
                                        strokeWidth="2"
                                    />
                                );
                            })}
                        </svg>
                    </div>
                    <div className="flex justify-between text-[11px] text-gray-500 mt-2">
                        {scans.map((s) => (
                            <span key={s.public_id} className="text-center">
                                {new Date(s.taken_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                <br />
                                <span className="text-cyan-400 font-bold">{s.twin_index ?? '-'}/100</span>
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Interactive Compare Slider */}
            {scans.length >= 2 && compareScanA && compareScanB && (
                <div className="mb-8 p-5 bg-gray-900 border border-cyan-800 rounded-xl">
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 mb-4">
                        <div>
                            <h2 className="text-lg font-bold text-white">Visual Temporal Progression Slider</h2>
                            <p className="text-xs text-gray-400">Scrub the slider horizontally to compare anatomical changes between clinical visits</p>
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                            <span className="text-gray-400">Comparing:</span>
                            <span className="font-semibold text-cyan-300">Scan A vs Scan B</span>
                        </div>
                    </div>

                    <div className="relative w-full max-w-lg mx-auto h-72 rounded-lg overflow-hidden border-2 border-cyan-500 bg-black">
                        {/* Background Scan (Scan A) */}
                        <img
                            src={`${API_BASE_URL}/api/scans/${compareScanA}/image/enhanced`}
                            alt="Scan A"
                            className="absolute inset-0 w-full h-full object-contain"
                        />
                        {/* Foreground Clipped Scan (Scan B) */}
                        <img
                            src={`${API_BASE_URL}/api/scans/${compareScanB}/image/enhanced`}
                            alt="Scan B"
                            className="absolute inset-0 w-full h-full object-contain"
                            style={{ clipPath: `inset(0 ${100 - sliderVal}% 0 0)` }}
                        />
                        {/* Divider Line */}
                        <div
                            className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 pointer-events-none shadow-[0_0_10px_#00e1ff]"
                            style={{ left: `${sliderVal}%` }}
                        />
                    </div>

                    <div className="max-w-lg mx-auto mt-4">
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={sliderVal}
                            onChange={(e) => setSliderVal(parseInt(e.target.value))}
                            className="w-full accent-cyan-400 cursor-ew-resize"
                        />
                        <div className="flex justify-between text-xs text-gray-400 mt-1">
                            <span>&larr; Earlier Scan</span>
                            <span className="text-cyan-300 font-bold">{sliderVal}% Split</span>
                            <span>Later Scan &rarr;</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Scans Timeline Cards */}
            <div className="space-y-4">
                <h2 className="text-sm uppercase tracking-wider text-gray-500 font-semibold">
                    Longitudinal CXR Diagnostic Visits ({scans.length})
                </h2>

                {scans.length === 0 ? (
                    <div className="p-8 bg-gray-900 border border-gray-800 rounded-lg text-center text-gray-400">
                        No authorized scans on record for this patient. Click "Upload Scan" above to add the initial baseline radiograph.
                    </div>
                ) : (
                    scans.map((scan) => (
                        <div
                            key={scan.public_id}
                            className="p-5 bg-gray-900 border border-gray-800 rounded-xl hover:border-cyan-500 transition-colors flex flex-col md:flex-row gap-5 items-start"
                        >
                            {/* Encrypted Thumbnail loaded securely */}
                            <img
                                src={`${API_BASE_URL}/api/scans/${scan.public_id}/image/enhanced`}
                                alt="CXR Scan"
                                className="w-28 h-28 object-contain rounded bg-black border border-gray-700 shrink-0"
                            />

                            <div className="flex-1">
                                <div className="flex flex-wrap items-center gap-2 mb-1">
                                    <span className="text-xs px-2 py-0.5 bg-gray-800 text-gray-300 rounded border border-gray-700">
                                        🏥 {scan.center_name} ({scan.center_code})
                                    </span>
                                    <span className="text-xs text-gray-500">
                                        {new Date(scan.taken_at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                                    </span>
                                    {scan.doctor_confirmed_label && (
                                        <span className="text-xs px-2 py-0.5 bg-green-950 text-green-300 rounded border border-green-600">
                                            ✓ Doctor Confirmed: {scan.doctor_confirmed_label}
                                        </span>
                                    )}
                                </div>

                                <div className="flex items-baseline gap-3 mt-2">
                                    <h3 className={`text-xl font-bold ${scan.classifier_label === 'NORMAL' ? 'text-green-400' : 'text-red-400'}`}>
                                        {scan.classifier_label}
                                    </h3>
                                    <span className="text-sm text-gray-400">
                                        {(scan.classifier_confidence * 100).toFixed(1)}% Confidence
                                    </span>
                                    {scan.twin_index !== undefined && (
                                        <span className="text-xs font-semibold px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-700">
                                            Abnormality Index: {scan.twin_index}/100
                                        </span>
                                    )}
                                </div>

                                {scan.doctor_notes && (
                                    <p className="text-xs text-gray-300 mt-2 bg-gray-800/50 p-2 rounded border border-gray-750">
                                        Clinical Notes: {scan.doctor_notes}
                                    </p>
                                )}
                            </div>

                            <div className="flex flex-col gap-2 shrink-0">
                                <a
                                    href={`${API_BASE_URL}/api/scans/${scan.public_id}/report`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-cyan-300 border border-cyan-800 rounded text-xs text-center font-semibold"
                                >
                                    📄 PDF Report
                                </a>
                                <a
                                    href={`${API_BASE_URL}/api/scans/${scan.public_id}/image/triptych`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 rounded text-xs text-center"
                                >
                                    🖼️ Healthy Twin
                                </a>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Consent Request Modal */}
            {showConsentModal && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
                    <div className="bg-gray-900 border-2 border-cyan-500 rounded-xl p-6 max-w-md w-full shadow-[0_0_30px_rgba(0,225,255,0.4)]">
                        <h2 className="text-xl font-bold text-white mb-2">Patient Consent Authorization</h2>
                        <p className="text-xs text-gray-400 mb-4">DPDP Act 2023 Digital Healthcare Consent</p>

                        {consentNotice && (
                            <div className="space-y-2 text-xs bg-gray-800/60 p-3 rounded-lg border border-gray-700 mb-4 text-gray-300">
                                <p><span className="text-cyan-300 font-semibold">Requesting Doctor:</span> {consentNotice.requesting_doctor}</p>
                                <p><span className="text-cyan-300 font-semibold">Medical Center:</span> {consentNotice.requesting_center}</p>
                                <p><span className="text-cyan-300 font-semibold">Scope:</span> {consentNotice.scope}</p>
                                <p><span className="text-cyan-300 font-semibold">Access Duration:</span> {consentNotice.duration}</p>
                            </div>
                        )}

                        <form onSubmit={handleVerifyOtp} className="space-y-4">
                            <div>
                                <label className="block text-xs text-gray-400 mb-1">
                                    Enter 6-Digit OTP (Check Patient Phone Simulator)
                                </label>
                                <input
                                    type="text"
                                    maxLength={6}
                                    placeholder="e.g. 542109"
                                    value={otp}
                                    onChange={(e) => setOtp(e.target.value)}
                                    required
                                    className="w-full text-center tracking-widest text-2xl font-mono bg-gray-800 border border-cyan-400 rounded p-2 text-white outline-none"
                                />
                            </div>

                            <div className="flex gap-2">
                                <button
                                    type="submit"
                                    disabled={consentLoading || otp.length < 6}
                                    className="flex-1 py-2.5 bg-gradient-to-r from-cyan-400 to-green-400 text-black font-bold text-sm rounded-lg hover:scale-105 transition-transform"
                                >
                                    {consentLoading ? 'Verifying...' : 'Verify & Grant Access'}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setShowConsentModal(false)}
                                    className="px-4 py-2.5 bg-gray-800 text-gray-400 text-sm rounded-lg hover:bg-gray-700"
                                >
                                    Cancel
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}

