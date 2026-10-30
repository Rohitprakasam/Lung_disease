"use client";

import { useState } from 'react';
import Link from 'next/link';
import { apiFetch } from '@/lib/api';

interface TwinData {
    abnormality_index: number;
    is_abnormal: boolean;
    status: string;
    message: string;
    images: {
        original: string;
        twin: string;
        diff: string;
        triptych: string;
    };
}

export default function ClassifyPage() {
    const [file, setFile] = useState<File | null>(null);
    const [patientDetails, setPatientDetails] = useState({ id: '', name: '', age: '', history: '' });
    const [prediction, setPrediction] = useState<string | null>(null);
    const [confidence, setConfidence] = useState<number | null>(null);
    const [probabilities, setProbabilities] = useState<Record<string, number> | null>(null);
    const [twin, setTwin] = useState<TwinData | null>(null);
    const [reportUrl, setReportUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            setFile(selectedFile);
            setImagePreview(URL.createObjectURL(selectedFile));
            setPrediction(null);
            setConfidence(null);
            setProbabilities(null);
            setTwin(null);
            setReportUrl(null);
            setError(null);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;

        setLoading(true);
        setError(null);
        setReportUrl(null);

        const formData = new FormData();
        formData.append('image', file);
        formData.append('id', patientDetails.id);
        formData.append('name', patientDetails.name);
        formData.append('age', patientDetails.age);
        formData.append('history', patientDetails.history);

        try {
            const res = await apiFetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            if (res.status === 422) {
                const errData = await res.json();
                setError(errData.error || 'This does not look like a frontal chest X-ray.');
                return;
            }

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error || 'Failed to fetch prediction');
            }

            const data = await res.json();
            setPrediction(data.prediction);
            setConfidence(data.confidence ?? null);
            setProbabilities(data.probabilities ?? null);
            setTwin(data.twin ?? null);
            setReportUrl(data.report_url);
        } catch (err: any) {
            setError(err.message || 'Error processing image. Ensure backend is running.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#121212] text-cyan-400 font-sans flex flex-col items-center justify-center p-4 py-12">

            <div className="absolute top-4 left-4 z-50">
                <Link href="/" className="text-gray-400 hover:text-white transition-colors">
                    &larr; Back to Home
                </Link>
            </div>

            <h1 className="text-4xl font-bold mb-8 text-center animate-pulse drop-shadow-[0_0_10px_rgba(0,225,255,0.8)]">
                Lung Disease Classification &amp; Healthy Twin
            </h1>

            <div className="bg-gray-900 border-2 border-cyan-500 rounded-xl p-8 shadow-[0_0_30px_rgba(0,225,255,0.3)] w-full max-w-2xl relative z-10">
                <form onSubmit={handleSubmit} className="flex flex-col gap-6">

                    {/* Patient Details Form */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <input
                            type="text"
                            placeholder="Patient ID / UID"
                            className="w-full bg-gray-800 text-white border border-gray-600 rounded p-2 focus:border-cyan-400 outline-none"
                            value={patientDetails.id}
                            onChange={(e) => setPatientDetails({ ...patientDetails, id: e.target.value })}
                            required
                        />
                        <input
                            type="text"
                            placeholder="Patient Name"
                            className="w-full bg-gray-800 text-white border border-gray-600 rounded p-2 focus:border-cyan-400 outline-none"
                            value={patientDetails.name}
                            onChange={(e) => setPatientDetails({ ...patientDetails, name: e.target.value })}
                            required
                        />
                        <input
                            type="number"
                            placeholder="Age"
                            className="w-full bg-gray-800 text-white border border-gray-600 rounded p-2 focus:border-cyan-400 outline-none"
                            value={patientDetails.age}
                            onChange={(e) => setPatientDetails({ ...patientDetails, age: e.target.value })}
                            required
                        />
                        <textarea
                            placeholder="Medical History (Optional)"
                            className="w-full bg-gray-800 text-white border border-gray-600 rounded p-2 focus:border-cyan-400 outline-none h-10 resize-none"
                            value={patientDetails.history}
                            onChange={(e) => setPatientDetails({ ...patientDetails, history: e.target.value })}
                        />
                    </div>

                    <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-600 rounded-lg p-6 hover:border-cyan-400 transition-colors cursor-pointer relative"
                        onClick={() => document.getElementById('fileInput')?.click()}>

                        {imagePreview ? (
                            <img src={imagePreview} alt="Preview" className="max-h-64 object-contain rounded" />
                        ) : (
                            <div className="text-center">
                                <p className="text-gray-400 mb-2">Click to Upload Chest X-Ray</p>
                                <span className="text-4xl">📁</span>
                            </div>
                        )}

                        <input
                            id="fileInput"
                            type="file"
                            accept="image/png, image/jpeg, image/jpg"
                            onChange={handleFileChange}
                            className="hidden"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={!file || loading}
                        className={`
                py-3 px-6 rounded-lg font-bold text-black transition-all transform hover:scale-105
                ${!file || loading
                                ? 'bg-gray-600 cursor-not-allowed'
                                : 'bg-gradient-to-r from-cyan-400 to-green-400 hover:shadow-[0_0_20px_#56ff62]'
                            }
            `}
                    >
                        {loading ? 'Analyzing with AI & Healthy Twin...' : 'Predict Disease'}
                    </button>
                </form>

                {error && (
                    <div className="mt-4 p-3 bg-red-950/60 border border-red-500 rounded text-red-400 text-center font-semibold">
                        {error}
                    </div>
                )}

                {prediction && (
                    <div className="mt-6 text-center animate-fade-in-up border-t border-gray-800 pt-6">
                        <h2 className="text-xl text-gray-300 mb-1">Diagnostic Prediction:</h2>
                        <p className="text-3xl font-extrabold text-green-400 drop-shadow-[0_0_10px_rgba(86,255,98,0.8)]">
                            {prediction} {confidence !== null && `(${(confidence * 100).toFixed(1)}%)`}
                        </p>

                        {/* Probabilities breakdown */}
                        {probabilities && (
                            <div className="mt-4 space-y-2 text-left max-w-md mx-auto">
                                <p className="text-xs text-gray-400 font-semibold mb-1">Probability Distribution:</p>
                                {Object.entries(probabilities).map(([cls, prob]) => (
                                    <div key={cls} className="flex items-center gap-2 text-xs">
                                        <span className="w-28 text-gray-300">{cls}</span>
                                        <div className="flex-1 bg-gray-800 rounded-full h-3 overflow-hidden border border-gray-700">
                                            <div
                                                className={`h-full ${cls === prediction ? 'bg-cyan-400' : 'bg-gray-600'}`}
                                                style={{ width: `${Math.round(prob * 100)}%` }}
                                            />
                                        </div>
                                        <span className="w-12 text-right text-gray-400">{(prob * 100).toFixed(1)}%</span>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Healthy Twin 3 Panels */}
                        {twin && (
                            <div className="mt-6 p-4 bg-gray-950 border border-cyan-800 rounded-lg text-left">
                                <div className="flex items-center justify-between mb-3">
                                    <h3 className="text-lg font-bold text-cyan-300">Healthy Twin Comparative Analysis</h3>
                                    <span className="px-3 py-1 bg-cyan-950 border border-cyan-500 text-xs font-semibold rounded-full">
                                        Abnormality: {twin.abnormality_index}/100
                                    </span>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                                    <div className="text-center">
                                        <p className="text-xs text-gray-400 mb-1">Original Input</p>
                                        <img src={twin.images.original} alt="Original" className="rounded border border-gray-700 w-full h-36 object-contain bg-black" />
                                    </div>
                                    <div className="text-center">
                                        <p className="text-xs text-gray-400 mb-1">Healthy Twin Reconstruction</p>
                                        <img src={twin.images.twin} alt="Healthy Twin" className="rounded border border-gray-700 w-full h-36 object-contain bg-black" />
                                    </div>
                                    <div className="text-center">
                                        <p className="text-xs text-gray-400 mb-1">Anomaly Heatmap</p>
                                        <img src={twin.images.diff} alt="Difference Heatmap" className="rounded border border-gray-700 w-full h-36 object-contain bg-black" />
                                    </div>
                                </div>
                            </div>
                        )}

                        {reportUrl && (
                            <a
                                href={reportUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-block mt-5 px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-green-500 text-black font-bold rounded-lg shadow-[0_0_15px_rgba(0,225,255,0.4)] hover:scale-105 transition-transform"
                            >
                                📄 Download Detailed Medical Report (PDF)
                            </a>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
