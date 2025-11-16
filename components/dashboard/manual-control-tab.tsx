import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, Play } from 'lucide-react';
import type { Junction, Direction, SignalColor } from '@/types/junction';
import { IntersectionVisualization } from './intersection-visualization';

interface ManualControlTabProps {
  junction: Junction | null;
  aiSuggestedTimer: number;
  onApplyChanges: (junctionId: string, direction: Direction, color: SignalColor, timer: number) => void;
  uploadedImages: Record<Direction, string>;
}

const directions: { key: Direction; label: string }[] = [
  { key: 'north', label: '#1' },
  { key: 'south', label: '#2' },
  { key: 'east', label: '#3' },
  { key: 'west', label: '#4' }
];

export function ManualControlTab({ junction, aiSuggestedTimer, onApplyChanges, uploadedImages }: ManualControlTabProps) {
  const [selectedDirection, setSelectedDirection] = useState<Direction>('north');
  const [timer, setTimer] = useState('30');

  if (!junction) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center text-muted-foreground">
          <Play className="h-16 w-16 mx-auto mb-4 opacity-20" />
          <p>Select a junction to enable manual control</p>
        </div>
      </div>
    );
  }

  const handleApply = () => {
    if (selectedDirection && timer) {
      onApplyChanges(junction.id, selectedDirection, 'green', parseInt(timer));
      setTimer('30');
    }
  };

  return (
    <div className="w-full space-y-6">
      <Card className="w-full">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl">Manual Control Panel</CardTitle>
            <Badge variant="outline">{junction.name}</Badge>
          </div>
        </CardHeader>
        <CardContent className="p-6 space-y-6">
          {/* Visualization - Full Width */}
          <div className="space-y-4">
            <Label className="text-base font-semibold">Intersection Visualization</Label>
            <div className="relative bg-slate-50 dark:bg-slate-900 rounded-lg p-4 border-2 border-slate-200 dark:border-slate-700">
              <IntersectionVisualization
                junction={junction}
                uploadedImages={uploadedImages}
                showUploadControls={false}
              />
            </div>
          </div>

          {/* Manual Controls - Full Width, Stacked Below */}
          <div className="space-y-6 max-w-2xl">
            <div>
              <Label htmlFor="direction" className="text-base font-semibold mb-3 block">Select Signal</Label>
              <select
                id="direction"
                value={selectedDirection}
                onChange={(e) => setSelectedDirection(e.target.value as Direction)}
                className="w-full px-4 py-2 border border-input rounded-md bg-background text-foreground text-base"
              >
                <option value="north">#1</option>
                <option value="south">#2</option>
                <option value="east">#3</option>
                <option value="west">#4</option>
              </select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="manual-timer">Set Timer (sec)</Label>
              <Input
                id="manual-timer"
                type="number"
                min="1"
                max="300"
                value={timer}
                onChange={(e) => setTimer(e.target.value)}
                placeholder="Enter duration in seconds"
                className="text-base"
              />
              <p className="text-xs text-muted-foreground">
                AI Suggested: {aiSuggestedTimer}s
              </p>
            </div>

            <Button
              type="button"
              onClick={handleApply}
              className="w-full h-12 text-base bg-green-600 hover:bg-green-700 text-white"
              size="lg"
            >
              <CheckCircle className="mr-2 h-5 w-5" />
              Apply Manual Green
            </Button>

            <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-800">
              <p className="text-sm font-medium text-green-900 dark:text-green-100 mb-2">
                Manual Override Ready:
              </p>
              <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                <p><span className="font-semibold">Signal:</span> {directions.find(d => d.key === selectedDirection)?.label}</p>
                <p><span className="font-semibold">Color:</span> Green</p>
                <p><span className="font-semibold">Timer:</span> {timer}s</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
