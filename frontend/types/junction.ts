export type SignalColor = 'red' | 'yellow' | 'green';
export type Direction = 'north' | 'south' | 'east' | 'west';
export type ControlMode = 'ai' | 'manual';

export interface SignalState {
  color: SignalColor;
  timer: number;
}

export interface Junction {
  id: string;
  name: string;
  location: string;
  imageUrl: string;
  signals: Record<Direction, SignalState>;
  lastUpdated: Date;
  responsibleOfficer: string;
  controlMode: ControlMode;
  directionImages?: Record<Direction, string>;
}

export interface Alert {
  id: string;
  message: string;
  type: 'info' | 'warning' | 'error';
  timestamp: Date;
}

export interface AISuggestion {
  direction: Direction;
  currentColor: SignalColor;
  suggestedTimer: number;
  priority: number;
}

export interface AIControlState {
  junctionId: string;
  lastGreenSignal: Direction | null;
  suggestedQueue: AISuggestion[];
  currentSuggestedTimer: number;
}
