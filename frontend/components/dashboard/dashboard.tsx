// dashboard.tsx
import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Sidebar } from './sidebar';
import { JunctionMonitoringTab } from './junction-monitoring-tab';
import { AIControlTab } from './ai-control-tab';
import { ManualControlTab } from './manual-control-tab';
import { MOCK_JUNCTIONS, MOCK_ALERTS } from '@/data/mock-junctions';
import type { Junction, Direction, SignalColor, Alert } from '@/types/junction';
import { LogOut, Activity } from 'lucide-react';
import type { User } from '@/types/auth';
import { generateAISuggestions } from '@/utils/ai-mock';

interface MediaSet {
  image?: string;
  audio?: string;
  video?: string;
}

interface DashboardProps {
  user: User;
  onLogout: () => void;
}

type QueueItem = { signal_num: number; green_time: number; score?: number; label?: string };

const NUM_TO_DIR: Record<number, Direction> = {
  1: 'north',
  2: 'east',
  3: 'west',
  4: 'south'
};

export function Dashboard({ user, onLogout }: DashboardProps) {
  const [junctions, setJunctions] = useState<Junction[]>(MOCK_JUNCTIONS);
  const [selectedJunctions, setSelectedJunctions] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'monitoring' | 'ai' | 'manual'>('monitoring');
  const [selectedJunctionId, setSelectedJunctionId] = useState<string | null>(null);

  // uploadedMedia: junctionId -> direction -> { image?, audio?, video? (data URLs) }
  const [uploadedMedia, setUploadedMedia] = useState<Record<string, Record<Direction, MediaSet>>>({});

  const [aiQueues, setAiQueues] = useState<Record<string, { ordered: QueueItem[]; idx: number }>>({});

  useEffect(() => {
    if (user.primaryJunctionId) {
      setSelectedJunctions([user.primaryJunctionId]);
    }
  }, [user.primaryJunctionId]);

  useEffect(() => {
    const interval = setInterval(() => {
      setJunctions((prev) =>
        prev.map((junction) => {
          const updatedSignals = { ...junction.signals };
          Object.keys(updatedSignals).forEach((dir) => {
            const direction = dir as Direction;
            if (updatedSignals[direction].timer > 0) {
              updatedSignals[direction] = {
                ...updatedSignals[direction],
                timer: Math.max(0, updatedSignals[direction].timer - 1)
              };
            }
          });
          return { ...junction, signals: updatedSignals };
        })
      );

      setJunctions((prev) =>
        prev.map((junction) => {
          if (junction.controlMode !== 'ai') return junction;

          const q = aiQueues[junction.id];
          if (!q || !q.ordered || q.ordered.length === 0) return junction;

          const updatedSignals = { ...junction.signals };
          const anyTimer = Object.values(updatedSignals).some((s) => s.timer > 0);

          if (!anyTimer) {
            const nextIdx = (q.idx + 1) % q.ordered.length;
            const nextItem = q.ordered[nextIdx];

            Object.keys(updatedSignals).forEach((dkey) => {
              const dir = dkey as Direction;
              const sigNumForDir = (dir === 'north') ? 1 : (dir === 'east') ? 2 : (dir === 'west') ? 3 : 4;
              if (sigNumForDir === nextItem.signal_num) {
                updatedSignals[dir] = { color: 'green', timer: nextItem.green_time };
              } else {
                updatedSignals[dir] = { color: 'red', timer: 0 };
              }
            });

            setAiQueues((prevQ) => ({ ...(prevQ || {}), [junction.id]: { ordered: q.ordered, idx: nextIdx } }));

            return {
              ...junction,
              signals: updatedSignals,
              lastUpdated: new Date()
            };
          }

          return junction;
        })
      );
    }, 1000);

    return () => clearInterval(interval);
  }, [aiQueues]);

  const handleToggleJunction = (junctionId: string) => {
    setSelectedJunctions((prev) =>
      prev.includes(junctionId)
        ? prev.filter((id) => id !== junctionId)
        : [...prev, junctionId]
    );
  };

  const handleSwitchControl = (junctionId: string, mode: 'ai' | 'manual') => {
    setJunctions((prev) =>
      prev.map((junction) =>
        junction.id === junctionId ? { ...junction, controlMode: mode } : junction
      )
    );
    setSelectedJunctionId(junctionId);
    setActiveTab(mode);
  };

  const handleApplyManualChanges = (
    junctionId: string,
    direction: Direction,
    color: SignalColor,
    timer: number
  ) => {
    setJunctions((prev) =>
      prev.map((junction) => {
        if (junction.id === junctionId) {
          const updatedSignals = { ...junction.signals };
          Object.keys(updatedSignals).forEach((dir) => {
            const d = dir as Direction;
            if (d === direction) {
              updatedSignals[d] = { color, timer: color === 'green' ? timer : 0 };
            } else {
              updatedSignals[d] = { color: 'red', timer: 0 };
            }
          });
          return {
            ...junction,
            signals: updatedSignals,
            lastUpdated: new Date(),
            controlMode: 'manual' as const
          };
        }
        return junction;
      })
    );
  };

  const handleApplyAISuggestionsFromUI = (
    junctionId: string,
    ordered: Array<{ signal_num: number; green_time: number; reason?: string; score?: number; label?: string }>
  ) => {
    setAiQueues((prev) => ({
      ...(prev || {}),
      [junctionId]: { ordered: ordered.map(o => ({ signal_num: o.signal_num, green_time: o.green_time, score: o.score, label: o.label })), idx: 0 }
    }));

    setJunctions((prev) =>
      prev.map((junction) => {
        if (junction.id !== junctionId) return junction;

        const newSignals = { ...junction.signals };
        const first = ordered?.[0];

        Object.keys(newSignals).forEach((dirKey) => {
          const dir = dirKey as Direction;
          const sigNum = (dir === 'north') ? 1 : (dir === 'east') ? 2 : (dir === 'west') ? 3 : 4;
          if (first && sigNum === first.signal_num) {
            newSignals[dir] = { color: 'green', timer: first.green_time };
          } else {
            newSignals[dir] = { color: 'red', timer: 0 };
          }
        });

        return {
          ...junction,
          signals: newSignals,
          lastUpdated: new Date(),
          controlMode: 'ai' as const
        };
      })
    );
  };

  const monitoredJunctions = junctions.filter((j) => selectedJunctions.includes(j.id));
  const selectedJunction = junctions.find((j) => j.id === selectedJunctionId) || monitoredJunctions[0] || null;
  const aiState = selectedJunction ? generateAISuggestions(selectedJunction.id) : null;

  // Handles any media upload (image/audio/video) and stores as Data URL
  const handleMediaUpload = (direction: Direction, file: File, mediaType: 'image' | 'audio' | 'video') => {
    if (!selectedJunction) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      setUploadedMedia((prev) => ({
        ...prev,
        [selectedJunction.id]: {
          ...(prev[selectedJunction.id] || { north: {}, east: {}, south: {}, west: {} }),
          [direction]: {
            ...(prev[selectedJunction.id]?.[direction] || {}),
            [mediaType]: reader.result as string
          }
        }
      }));
    };

    // Use readAsDataURL for all types so we can send as blob later
    reader.readAsDataURL(file);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex flex-col">
      <header className="bg-white dark:bg-slate-950 border-b shadow-sm sticky top-0 z-10">
        <div className="px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl font-bold text-blue-600">🚦</span>
              <div>
                <h1 className="text-2xl font-bold">TRAFFIX</h1>
                <p className="text-sm text-muted-foreground">Traffic Management System</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm font-medium">{user.name}</p>
                <p className="text-xs text-muted-foreground">{user.role}</p>
              </div>
              <Button type="button" variant="outline" onClick={onLogout}>
                <LogOut className="mr-2 h-4 w-4" />
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <div className="w-80 flex-shrink-0">
          <Sidebar
            junctions={junctions}
            selectedJunctions={selectedJunctions}
            onToggleJunction={handleToggleJunction}
          />
        </div>

        <main className="flex-1 overflow-auto p-6">
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="space-y-4">
            <TabsList className="grid w-full max-w-2xl grid-cols-3">
              <TabsTrigger value="monitoring">Junction Monitoring</TabsTrigger>
              <TabsTrigger value="ai">AI Control</TabsTrigger>
              <TabsTrigger value="manual">Manual Control</TabsTrigger>
            </TabsList>

            <TabsContent value="monitoring">
              <JunctionMonitoringTab
                junctions={monitoredJunctions}
                onSwitchControl={handleSwitchControl}
              />
            </TabsContent>

            <TabsContent value="ai">
              <AIControlTab 
                junction={selectedJunction}
                uploadedMedia={selectedJunction ? uploadedMedia[selectedJunction.id] || { north: {}, east: {}, south: {}, west: {} } : { north: {}, east: {}, south: {}, west: {} }}
                onMediaUpload={(direction, file, mediaType) => handleMediaUpload(direction, file, mediaType)}
                onApplyAISuggestions={handleApplyAISuggestionsFromUI}
              />
            </TabsContent>

            <TabsContent value="manual">
              <ManualControlTab
                junction={selectedJunction}
                aiSuggestedTimer={aiState?.currentSuggestedTimer || 45}
                onApplyChanges={handleApplyManualChanges}
                uploadedImages={selectedJunction ? (uploadedMedia[selectedJunction.id] || {}) : {}}
              />
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  );
}
