import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Users, Clock, ImageOff } from "lucide-react";
import { BACKEND_URL } from "@/lib/config";

interface PeakCrowdData {
  peakCount: number | null;
  timestamp: number | null;
  frameId: number | null;
  imageUrl: string | null;
  crowdState: string | null;
  riskZone: string | null;
  jobId: string | null;
}

type Status =
  | "loading"
  | "ready"
  | "empty"
  | "error"
  | "signed-out"
  | "demo";

const getAccessToken = () =>
  localStorage.getItem("accessToken") ||
  localStorage.getItem("access_token") ||
  localStorage.getItem("authToken");

const formatTimestamp = (seconds: number | null) => {
  if (seconds === null || Number.isNaN(seconds)) return "N/A";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};

const formatLabel = (value: string | null) =>
  value ? value.replace(/_/g, " ") : null;

/**
 * Stat card showing the busiest moment (peak crowd) from the latest
 * crowd-monitoring analysis. Data comes from GET /api/crowd/latest.
 */
export default function PeakCrowdCard() {
  const [status, setStatus] = useState<Status>("loading");
  const [data, setData] = useState<PeakCrowdData | null>(null);
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    const token = getAccessToken();
    if (!token) {
      setStatus("signed-out");
      return;
    }
    // The built-in demo login uses a placeholder token the backend rejects.
    if (token.startsWith("demo_")) {
      setStatus("demo");
      return;
    }

    const controller = new AbortController();

    fetch(`${BACKEND_URL}/api/crowd/latest`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    })
      .then((res) => {
        if (res.status === 404) {
          setStatus("empty");
          return null;
        }
        if (res.status === 401 || res.status === 403) {
          setStatus("signed-out");
          return null;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json) => {
        if (!json) return;
        const summary = json.summary ?? {};
        const peak = summary.peak_crowd_frame ?? {};
        const rawPeak = json.crowd?.peak_crowd_frame ?? {};
        const count = peak.person_count ?? summary.peak_person_count ?? null;

        if (count === null || count === undefined) {
          setStatus("empty");
          return;
        }

        setData({
          peakCount: count,
          timestamp: peak.timestamp ?? null,
          frameId: peak.frame_id ?? null,
          imageUrl:
            rawPeak.annotated_frame_path ??
            rawPeak.people_annotated_frame_path ??
            null,
          crowdState: summary.crowd_state ?? null,
          riskZone: summary.highest_risk_zone ?? null,
          jobId: json.job_id ?? null,
        });
        setStatus("ready");
      })
      .catch((err) => {
        if (err?.name !== "AbortError") setStatus("error");
      });

    return () => controller.abort();
  }, []);

  return (
    <Card data-testid="peak-crowd-card">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <p className="text-sm text-gray-600">Peak Crowd</p>

            {status === "loading" && (
              <p className="text-2xl font-bold text-gray-400">Loading...</p>
            )}

            {status === "ready" && data && (
              <>
                <p className="text-3xl font-bold text-gray-900">
                  {data.peakCount!.toLocaleString()}
                  <span className="ml-1 text-base font-medium text-gray-500">
                    people
                  </span>
                </p>
                <div className="mt-1 flex items-center gap-1 text-xs text-gray-500">
                  <Clock className="h-3 w-3" />
                  <span>
                    at {formatTimestamp(data.timestamp)}
                    {data.frameId !== null && ` (frame ${data.frameId})`}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {data.crowdState && (
                    <Badge variant="secondary" className="capitalize">
                      {formatLabel(data.crowdState)}
                    </Badge>
                  )}
                  {data.riskZone && (
                    <Badge variant="outline">Risk zone {data.riskZone}</Badge>
                  )}
                </div>
              </>
            )}

            {status === "empty" && (
              <>
                <p className="text-2xl font-bold text-gray-400">--</p>
                <p className="text-xs text-gray-500">
                  No crowd analysis yet. Upload a video to see the peak.
                </p>
              </>
            )}

            {status === "signed-out" && (
              <>
                <p className="text-2xl font-bold text-gray-400">--</p>
                <p className="text-xs text-gray-500">
                  Session expired or signed out. Sign in again to see peak crowd data.
                </p>
              </>
            )}

            {status === "demo" && (
              <>
                <p className="text-2xl font-bold text-gray-400">--</p>
                <p className="text-xs text-gray-500">
                  Demo account has no live data. Sign in with a registered
                  account to see peak crowd.
                </p>
              </>
            )}

            {status === "error" && (
              <>
                <p className="text-2xl font-bold text-gray-400">--</p>
                <p className="text-xs text-red-600">
                  Could not load peak crowd data.
                </p>
              </>
            )}
          </div>

          {status === "ready" && data?.imageUrl && !imageFailed ? (
            <img
              src={data.imageUrl}
              alt="Frame with the highest crowd count"
              className="h-20 w-32 rounded-md border object-cover"
              onError={() => setImageFailed(true)}
            />
          ) : status === "ready" ? (
            <div className="flex h-20 w-32 items-center justify-center rounded-md border bg-gray-50">
              <ImageOff className="h-6 w-6 text-gray-400" />
            </div>
          ) : (
            <Users className="h-8 w-8 text-blue-500" />
          )}
        </div>
      </CardContent>
    </Card>
  );
}
