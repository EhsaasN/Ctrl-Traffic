import type { AISuggestion, Direction, AIControlState } from '@/types/junction';

export function generateAISuggestions(junctionId: string): AIControlState {
  const suggestions: AISuggestion[] = [
    {
      direction: 'north',
      currentColor: 'red',
      suggestedTimer: 45,
      priority: 1
    },
    {
      direction: 'east',
      currentColor: 'red',
      suggestedTimer: 38,
      priority: 2
    },
    {
      direction: 'south',
      currentColor: 'red',
      suggestedTimer: 42,
      priority: 3
    },
    {
      direction: 'west',
      currentColor: 'red',
      suggestedTimer: 35,
      priority: 4
    }
  ];

  return {
    junctionId,
    lastGreenSignal: 'north',
    suggestedQueue: suggestions,
    currentSuggestedTimer: 45
  };
}
