"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { apiFetch } from '@/lib/api';

interface AuditRow {
    id: number;
    at: string;
    action: string;
    actor_doctor_id?: number;
    patient_id?: number;
    scan_id?: number;
    ip?: string;
    details?: string;
}

export default function AdminAuditPage() {
    const router = useRouter();
    const [logs, setLogs] = useState<AuditRow[]>([]);
    const [actionFilter, setActionFilter] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchLogs = async () => {
        setLoading(true);
        setError(null);
        try {
            const url = actionFilter ? `/api/audit?action=${encodeURIComponent(actionFilter)}` : '/api/audit';
            const res = await apiFetch(url);
            if (!res.ok) {
                if (res.status === 403) {
                    throw new Error('Access denied. Administrator privileges required to inspect audit trail.');
                }
                const errData = await res.json();
                throw new Error(errData.error || 'Failed to fetch audit records');
            }
            const data = await res.json();
            setLogs(data.logs || []);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLogs();
    }, [actionFilter]);

    return (
        <div className="min-h-screen bg-[#121212] text-cyan-400 font-sans p-6 max-w-6xl mx-auto">
            <div className="flex justify-between items-center border-b border-gray-800 pb-4 mb-6">
                <div>
                    <Link href="/patients" className="text-xs text-gray-400 hover:text-white mb-1 inline-block">
                        &larr; Back to Patients Directory
                    </Link>
                    <h1 className="text-2xl font-bold text-white">Immutable Clinical Audit Trail</h1>
                    <p className="text-xs text-gray-400">DPDP Act 2023 Access Logs &amp; Integrity Verification</p>
                </div>

                <div className="flex items-center gap-2">
                    <select
                        value={actionFilter}
                        onChange={(e) => setActionFilter(e.target.value)}
                        className="bg-gray-800 text-xs text-white border border-gray-700 rounded p-2 outline-none"
                    >
                        <option value="">All Actions</option>
                        <option value="login">login</option>
                        <option value="timeline.view">timeline.view</option>
                        <option value="scan.image.view">scan.image.view</option>
                        <option value="consent.request">consent.request</option>
                        <option value="consent.grant">consent.grant</option>
                        <option value="consent.revoke">consent.revoke</option>
                        <option value="access.denied">access.denied</option>
                    </select>
                    <button
                        onClick={fetchLogs}
                        className="px-3 py-2 bg-cyan-600 text-black text-xs font-bold rounded hover:bg-cyan-500"
                    >
                        Refresh
                    </button>
                </div>
            </div>

            {error && (
                <div className="mb-6 p-4 bg-red-950/60 border border-red-500 rounded-lg text-red-300 text-sm">
                    {error}
                </div>
            )}

            <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                <table className="w-full text-left text-xs">
                    <thead className="bg-gray-800 text-gray-300 border-b border-gray-700 uppercase">
                        <tr>
                            <th className="p-3">Timestamp (UTC)</th>
                            <th className="p-3">Action</th>
                            <th className="p-3">Doctor ID</th>
                            <th className="p-3">Patient ID</th>
                            <th className="p-3">Scan ID</th>
                            <th className="p-3">IP Address</th>
                            <th className="p-3">Metadata</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800 font-mono text-gray-400">
                        {loading ? (
                            <tr>
                                <td colSpan={7} className="p-6 text-center text-gray-500">
                                    Loading audit trail...
                                </td>
                            </tr>
                        ) : logs.length === 0 ? (
                            <tr>
                                <td colSpan={7} className="p-6 text-center text-gray-500">
                                    No audit entries found.
                                </td>
                            </tr>
                        ) : (
                            logs.map((log) => (
                                <tr key={log.id} className="hover:bg-gray-850">
                                    <td className="p-3 text-gray-300">
                                        {new Date(log.at).toLocaleString()}
                                    </td>
                                    <td className="p-3">
                                        <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${log.action === 'access.denied' ? 'bg-red-950 text-red-400 border border-red-700' : (log.action.startsWith('consent') ? 'bg-green-950 text-green-300 border border-green-700' : 'bg-gray-800 text-cyan-300')}`}>
                                            {log.action}
                                        </span>
                                    </td>
                                    <td className="p-3">{log.actor_doctor_id ?? '-'}</td>
                                    <td className="p-3">{log.patient_id ?? '-'}</td>
                                    <td className="p-3">{log.scan_id ?? '-'}</td>
                                    <td className="p-3">{log.ip ?? '-'}</td>
                                    <td className="p-3 text-[10px] text-gray-500 max-w-xs truncate">
                                        {log.details ?? '-'}
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

