import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, ArrowUp, ArrowRight, User, MapPin, Hash, Bot, Hand } from 'lucide-react';
import type { Junction, Direction } from '@/types/junction';

interface JunctionMonitoringTabProps {
  junctions: Junction[];
  onSwitchControl: (junctionId: string, mode: 'ai' | 'manual') => void;
}

const getSignalColorClass = (color: string) => {
  switch (color) {
    case 'red':
      return 'text-red-500 fill-red-500';
    case 'yellow':
      return 'text-yellow-500 fill-yellow-500';
    case 'green':
      return 'text-green-500 fill-green-500';
    default:
      return 'text-gray-400';
  }
};

const directions: { key: Direction; label: string }[] = [
  { key: 'north', label: 'N' },
  { key: 'east', label: 'E' },
  { key: 'south', label: 'S' },
  { key: 'west', label: 'W' }
];

const arrows = [
  { label: 'Left', icon: ArrowLeft },
  { label: 'Straight', icon: ArrowUp },
  { label: 'Right', icon: ArrowRight }
];

export function JunctionMonitoringTab({
  junctions,
  onSwitchControl
}: JunctionMonitoringTabProps) {
  return (
    <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-4">
      {junctions.map((junction) => {
        const activeSignal = Object.entries(junction.signals).find(
          ([, state]) => state.color === 'green' && state.timer > 0
        );

        return (
          <Card key={junction.id} className="hover:shadow-lg transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <CardTitle className="text-lg">{junction.name}</CardTitle>
                  <div className="flex items-center gap-2 mt-2 text-sm text-muted-foreground">
                    <MapPin className="h-3 w-3" />
                    <span>{junction.location}</span>
                  </div>
                </div>
                <Badge
                  variant={junction.controlMode === 'ai' ? 'secondary' : 'default'}
                  className="ml-2"
                >
                  {junction.controlMode === 'ai' ? (
                    <>
                      <Bot className="h-3 w-3 mr-1" />
                      AI
                    </>
                  ) : (
                    <>
                      <Hand className="h-3 w-3 mr-1" />
                      Manual
                    </>
                  )}
                </Badge>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="aspect-video bg-muted rounded-md overflow-hidden">
                <img
                  src={junction.imageUrl}
                  alt={junction.name}
                  className="w-full h-full object-cover"
                />
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="flex items-center gap-2">
                  <Hash className="h-3 w-3 text-muted-foreground" />
                  <span className="font-medium">ID:</span>
                  <span className="text-muted-foreground">{junction.id}</span>
                </div>
                <div className="flex items-center gap-2">
                  <User className="h-3 w-3 text-muted-foreground" />
                  <span className="font-medium">Officer:</span>
                </div>
                <div className="col-span-2 text-muted-foreground pl-5">
                  {junction.responsibleOfficer}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2">
                {arrows.map(({ label, icon: Icon }) => {
                  // Cycle through the four directions to get a representative signal status
                  const signalStates = [junction.signals.north, junction.signals.east, junction.signals.south];
                  const index = arrows.indexOf(arrows.find(a => a.label === label) || arrows[0]);
                  const signal = signalStates[index] || junction.signals.north;
                  
                  return (
                    <div
                      key={label}
                      className="flex flex-col items-center justify-center p-3 bg-muted rounded"
                    >
                      <span className="text-xs font-medium text-muted-foreground mb-2">
                        {label}
                      </span>
                      <Icon className={`h-6 w-6 ${getSignalColorClass(signal.color)}`} />
                    </div>
                  );
                })}
              </div>

              {activeSignal && (
                <div className="flex items-center justify-between bg-green-50 dark:bg-green-950 p-3 rounded">
                  <span className="text-sm font-medium">Active Timer:</span>
                  <span className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {activeSignal[1].timer}s
                  </span>
                </div>
              )}

              <div className="grid grid-cols-2 gap-2 pt-2">
                <Button
                  type="button"
                  variant={junction.controlMode === 'ai' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => onSwitchControl(junction.id, 'ai')}
                  className="w-full"
                >
                  <Bot className="h-4 w-4 mr-1" />
                  AI Control
                </Button>
                <Button
                  type="button"
                  variant={junction.controlMode === 'manual' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => onSwitchControl(junction.id, 'manual')}
                  className="w-full"
                >
                  <Hand className="h-4 w-4 mr-1" />
                  Manual
                </Button>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
