import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Search, MapPin } from 'lucide-react';
import type { Junction } from '@/types/junction';

interface SidebarProps {
  junctions: Junction[];
  selectedJunctions: string[];
  onToggleJunction: (junctionId: string) => void;
}

export function Sidebar({ junctions, selectedJunctions, onToggleJunction }: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredJunctions = junctions.filter(
    (junction) =>
      junction.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      junction.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      junction.location.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="h-full flex flex-col bg-white dark:bg-slate-950 border-r">
      <div className="p-4 border-b">
        <div className="relative">
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search junctions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-2">
          {filteredJunctions.map((junction) => {
            const isSelected = selectedJunctions.includes(junction.id);
            return (
              <Card
                key={junction.id}
                className={`cursor-pointer transition-all hover:shadow-md ${
                  isSelected ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-950' : ''
                }`}
                onClick={() => onToggleJunction(junction.id)}
              >
                <CardHeader className="p-3 pb-2">
                  <div className="flex items-start justify-between gap-2">
                    <CardTitle className="text-sm font-semibold">{junction.name}</CardTitle>
                    {isSelected && (
                      <Badge variant="default" className="text-xs">
                        Active
                      </Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="p-3 pt-0">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <MapPin className="h-3 w-3" />
                    <span>{junction.location}</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <span className="text-xs font-medium">ID: {junction.id}</span>
                    <Badge
                      variant={junction.controlMode === 'ai' ? 'secondary' : 'outline'}
                      className="text-xs"
                    >
                      {junction.controlMode === 'ai' ? 'AI' : 'Manual'}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
