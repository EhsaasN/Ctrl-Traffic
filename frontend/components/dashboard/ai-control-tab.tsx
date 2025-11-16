// ai-control-tab.tsx
import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Circle, ArrowRight, Clock } from 'lucide-react';
import type { Junction, Direction, AIControlState } from '@/types/junction';
import { generateAISuggestions } from '@/utils/ai-mock';
import { IntersectionVisualization } from './intersection-visualization';

interface AIControlTabProps {
  junction: Junction | null;
  uploadedImages: Record<Direction, string>;
  onImageUpload: (direction: Direction, file: File) => void;
  onApplyAISuggestions?: (junctionId: string, ordered: Array<{ signal_num: number; green_time: number; score?: number; label?: string }>) => void;
}

const GREEN_TO_LABEL: Record<number, string> = {
  120: 'emergency',
  80: 'high-traffic',
  60: 'medium-traffic',
  30: 'low-traffic',
  0: 'no-traffic'
};

const CLASS_PRIORITY: Record<string, number> = {
  'emergency': 5,
  'high-traffic': 4,
  'medium-traffic': 3,
  'low-traffic': 2,
  'no-traffic': 1
};

// **CORRECTED MAPPING TO MATCH UI BUTTON LABELS**
const DIR_TO_NUM: Record<Direction, number> = {
  north: 1, // #1 in your UI
  east: 2,  // #2 in your UI
  west: 3,  // #3 in your UI (note: west was #3 in the visualization)
  south: 4  // #4 in your UI
};

export function AIControlTab({ junction, uploadedImages, onImageUpload, onApplyAISuggestions }: AIControlTabProps) {
  const [aiState, setAiState] = useState<AIControlState | null>(null);
  const [busy, setBusy] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [orderedQueue, setOrderedQueue] = useState<Array<{ signal_num: number; green_time: number; score: number; label: string }>>([]);

  useEffect(() => {
    if (junction) {
      setAiState(generateAISuggestions(junction.id));
    }
  }, [junction]);

  async function dataUrlToFile(dataUrl: string, filename: string) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new File([blob], filename, { type: blob.type || 'image/jpeg' });
  }

  function extractScoreFromPayload(payload: any): number {
    if (!payload) return 0;
    const keys = ['score', 'top_prob', 'top_score', 'prob', 'probability', 'vehicle_prob'];
    for (const k of keys) {
      if (payload[k] != null) {
        const v = Number(payload[k]);
        if (!Number.isNaN(v)) return Math.max(0, Math.min(1, v));
      }
    }
    if (payload?.traffic_probabilities && typeof payload.traffic_probabilities === 'object') {
      const vals = Object.values(payload.traffic_probabilities).map((x: any) => Number(x)).filter((n: number) => !Number.isNaN(n));
      if (vals.length) return Math.max(0, Math.min(1, Math.max(...vals)));
    }
    if (payload?.predictions && typeof payload.predictions === 'object') {
      const vals = Object.values(payload.predictions).map((x: any) => Number(x)).filter((n: number) => !Number.isNaN(n));
      if (vals.length) return Math.max(0, Math.min(1, Math.max(...vals)));
    }
    if (payload?.vehicle_prob != null) {
      const v = Number(payload.vehicle_prob);
      if (!Number.isNaN(v)) return Math.max(0, Math.min(1, v));
    }
    return 0;
  }

  const handleSubmitToAI = async () => {
    setError(null);
    setLastResult(null);
    setOrderedQueue([]);

    if (!junction) {
      setError('No junction selected');
      return;
    }

    const directions: Direction[] = ['north', 'east', 'south', 'west'];

    const toSubmit = directions.map((d) => ({ dir: d, dataUrl: uploadedImages?.[d] })).filter((x) => !!x.dataUrl);

    if (toSubmit.length === 0) {
      setError('No uploaded images found for this junction. Upload images first.');
      return;
    }

    setBusy(true);
    try {
      const responses: Array<{ direction: Direction; success: boolean; payload?: any; error?: string }> = [];

      for (const item of toSubmit) {
        try {
          const file = await dataUrlToFile(item.dataUrl!, `${item.dir}.jpg`);
          const fd = new FormData();
          fd.append('file', file);

          const res = await fetch('http://localhost:8000/infer', {
            method: 'POST',
            body: fd
          });

          if (!res.ok) {
            const txt = await res.text();
            responses.push({ direction: item.dir, success: false, error: `HTTP ${res.status}: ${txt}` });
            continue;
          }

          const json = await res.json();
          responses.push({ direction: item.dir, success: true, payload: json });
        } catch (err: any) {
          responses.push({ direction: item.dir, success: false, error: String(err?.message || err) });
        }
      }

      const evals = responses.map((r) => {
        if (!r.success) {
          return { signal_num: DIR_TO_NUM[r.direction], green_time: 0, label: GREEN_TO_LABEL[0], score: 0, raw: { error: r.error } };
        }
        const payload = r.payload ?? {};
        let gt = 0;
        if (payload.green_time != null) gt = Number(payload.green_time);
        else if (payload.traffic_label != null) {
          const label = String(payload.traffic_label);
          const map: Record<string, number> = { 'emergency': 120, 'high-traffic': 80, 'medium-traffic': 60, 'low-traffic': 30, 'no-traffic': 0 };
          gt = map[label] ?? 0;
        } else {
          gt = 0;
        }
        const label = GREEN_TO_LABEL[gt] ?? 'no-traffic';
        const score = extractScoreFromPayload(payload);
        return { signal_num: DIR_TO_NUM[r.direction], green_time: gt, label, score, raw: payload };
      });

      evals.sort((a, b) => {
        const pa = CLASS_PRIORITY[a.label] ?? 0;
        const pb = CLASS_PRIORITY[b.label] ?? 0;
        if (pa !== pb) return pb - pa;
        return b.score - a.score;
      });

      const ordered = evals.map((e) => ({ signal_num: e.signal_num, green_time: e.green_time, score: e.score, label: e.label }));

      setOrderedQueue(ordered);
      setLastResult({ responses, evals, ordered });

      if (onApplyAISuggestions && junction) {
        onApplyAISuggestions(junction.id, ordered);
      }
    } catch (err: any) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
    }
  };

  if (!junction) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center text-muted-foreground">
          <Circle className="h-16 w-16 mx-auto mb-4 opacity-20" />
          <p>Select a junction to configure AI control</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">AI Intersection Visualization</CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="relative bg-slate-50 dark:bg-slate-900 rounded-lg p-4 border-2 border-slate-200 dark:border-slate-700">
            <IntersectionVisualization
              junction={junction}
              uploadedImages={uploadedImages}
              onImageUpload={onImageUpload}
              showUploadControls={true}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-xl">AI Suggestions</CardTitle>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          {/* Show actual current green signal computed from junction.signals (keeps UI intact) */}
{junction && (
  (() => {
    // compute which direction currently has green (if any)
    const dirOrder: Array<{ dir: Direction; num: number }> = [
      { dir: 'north', num: 1 },
      { dir: 'east',  num: 2 },
      { dir: 'west',  num: 3 },
      { dir: 'south', num: 4 }
    ];
    let currentSignalNum: number | null = null;
    for (const d of dirOrder) {
      if (junction.signals[d.dir] && junction.signals[d.dir].color === 'green') {
        currentSignalNum = d.num;
        break;
      }
    }

    if (currentSignalNum == null) {
      // if none green, optionally show lastApplied value from aiQueues if available
      return (
        <div className="flex items-center gap-3 p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
          <Clock className="h-5 w-5 text-yellow-600" />
          <div>
            <p className="text-sm font-medium">Last Green Signal</p>
            <p className="text-lg font-bold">—</p>
          </div>
        </div>
      );
    }

    return (
      <div className="flex items-center gap-3 p-3 bg-green-50 dark:bg-green-950 rounded-lg">
        <Clock className="h-5 w-5 text-green-600" />
        <div>
          <p className="text-sm font-medium">Last Green Signal</p>
          <p className="text-lg font-bold">#{currentSignalNum}</p>
        </div>
      </div>
    );
  })()
)}


          <div>
            <h3 className="font-semibold mb-3">AI Suggested Queue</h3>

            <div className="space-y-2">
              {orderedQueue.length === 0 ? (
                <div className="text-sm text-muted-foreground">No AI result yet. Upload images and click Submit.</div>
              ) : (
                orderedQueue.map((item, idx) => (
                  <div
                    key={item.signal_num}
                    className="flex items-center gap-3 p-3 bg-card border rounded-lg hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 font-bold">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Signal #{item.signal_num}</p>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <span className="capitalize">{item.label}</span>
                        <span>· score: {item.score?.toFixed(3)}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-muted-foreground">Green time</p>
                      <p className="text-xl font-bold text-blue-600 dark:text-blue-400">{item.green_time}s</p>
                    </div>
                    <ArrowRight className="h-5 w-5 text-muted-foreground" />
                  </div>
                ))
              )}
            </div>

            <div className="mt-4">
              <Button onClick={handleSubmitToAI} disabled={busy} className="bg-indigo-600 hover:bg-indigo-700">
                {busy ? 'Submitting...' : 'Submit Uploaded Images to AI (decide order)'}
              </Button>
            </div>

            {error && <div className="text-sm text-red-600 mt-3">{error}</div>}

            {lastResult && (
              <div className="mt-3 text-xs text-muted-foreground whitespace-pre-wrap">
                <details>
                  <summary className="cursor-pointer">Show last AI responses / evaluation</summary>
                  <pre className="text-xs mt-2 p-2 bg-gray-50 rounded border">{JSON.stringify(lastResult, null, 2)}</pre>
                </details>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
