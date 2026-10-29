export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

export async function apiFetch(endpoint: string, options: RequestInit = {}): Promise<Response> {
    const url = endpoint.startsWith("http")
        ? endpoint
        : `${API_BASE_URL}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

    const headers = new Headers(options.headers || {});
    
    // Attach JWT token if stored in sessionStorage/localStorage
    if (typeof window !== "undefined") {
        const token = sessionStorage.getItem("token") || localStorage.getItem("token");
        if (token && !headers.has("Authorization")) {
            headers.set("Authorization", `Bearer ${token}`);
        }
    }

    return await fetch(url, {
        ...options,
        headers,
    });
}

