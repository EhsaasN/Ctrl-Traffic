import React from 'react';
import { Upload, Timer } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import type { Junction, Direction, SignalColor } from '@/types/junction';

interface IntersectionVisualizationProps {
  junction: Junction;
  uploadedImages?: Record<Direction, string>;
  onImageUpload?: (direction: Direction, file: File) => void;
  showUploadControls?: boolean;
}

// Horizontal Traffic Light Component (Red/Yellow/Green in a row)
const HorizontalTrafficLight = ({ color }: { color: SignalColor }) => {
  const getActiveColor = (lightColor: SignalColor, currentColor: SignalColor) => {
    if (lightColor === currentColor) {
      switch (lightColor) {
        case 'red':
          return 'bg-red-500 shadow-[0_0_20px_rgba(239,68,68,0.8)]';
        case 'yellow':
          return 'bg-yellow-500 shadow-[0_0_20px_rgba(234,179,8,0.8)]';
        case 'green':
          return 'bg-green-500 shadow-[0_0_20px_rgba(34,197,94,0.8)]';
      }
    }
    return 'bg-gray-400';
  };

  return (
    <div className="bg-gray-800 rounded-lg p-2 shadow-lg">
      <div className="flex flex-row gap-2">
        <div className={`w-6 h-6 rounded-full ${getActiveColor('red', color)} transition-all`} />
        <div className={`w-6 h-6 rounded-full ${getActiveColor('yellow', color)} transition-all`} />
        <div className={`w-6 h-6 rounded-full ${getActiveColor('green', color)} transition-all`} />
      </div>
    </div>
  );
};

// Arrow SVG Component (from reference code)
const ArrowSVG = ({ 
  color = 'green', 
  size = '16px', 
  direction = 'up' 
}: { 
  color?: string; 
  size?: string; 
  direction?: 'up' | 'down' | 'left' | 'right';
}) => {
  let transform = '';
  switch (direction) {
    case 'right': transform = 'rotate(90 10 10)'; break;
    case 'down': transform = 'rotate(180 10 10)'; break;
    case 'left': transform = 'rotate(270 10 10)'; break;
    case 'up':
    default: transform = ''; break;
  }

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      aria-hidden="true"
      style={{ display: 'block' }}
    >
      <path
        d="M 10 0 L 20 10 L 14 10 L 14 20 L 6 20 L 6 10 L 0 10 Z"
        fill={color}
        transform={transform}
      />
    </svg>
  );
};

// TrafficLightBox Component (from reference code, adapted for React/TypeScript)
const TrafficLightBox = ({
  arrows = [],
  orientation = 'horizontal',
  boxBackgroundColor = '#FFC107',
  cavityColor = '#333',
  cavityDiameter = '50px',
  arrowSize = '24px'
}: {
  arrows: Array<{ color: string; direction: 'up' | 'down' | 'left' | 'right' }>;
  orientation?: 'vertical' | 'horizontal';
  boxBackgroundColor?: string;
  cavityColor?: string;
  cavityDiameter?: string;
  arrowSize?: string;
}) => {
  const boxStyles: React.CSSProperties = {
    backgroundColor: boxBackgroundColor,
    borderRadius: '10px',
    padding: '12px',
    boxShadow: '2px 2px 5px rgba(0,0,0,0.2)',
    display: 'flex',
    flexDirection: orientation === 'vertical' ? 'column' : 'row',
    gap: '8px',
    width: 'fit-content'
  };

  const cavityStyles: React.CSSProperties = {
    width: cavityDiameter,
    height: cavityDiameter,
    borderRadius: '50%',
    backgroundColor: cavityColor,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  };

  return (
    <div style={boxStyles}>
      {arrows.map((arrow, index) => (
        <div key={index} style={cavityStyles}>
          <ArrowSVG 
            color={arrow.color} 
            size={arrowSize} 
            direction={arrow.direction} 
          />
        </div>
      ))}
    </div>
  );
};

// Signal Unit Component - Horizontal cluster with Traffic Light, Timer, and Directional Arrow
const SignalUnit = ({
  direction,
  color,
  timer,
  isActive,
  scale = 1,
  rotation = 0
}: {
  direction: Direction;
  color: SignalColor;
  timer: number;
  isActive: boolean;
  scale?: number;
  rotation?: number;
}) => {
  // Determine which arrow should be displayed based on direction
  const getArrow = () => {
    // Always show the arrow, but color it based on active state
    const arrowColor = isActive && color === 'green' ? '#22c55e' : '#6b7280'; // Green if active, gray if not
    
    // >>> CHANGED: map to point toward CENTER (not travel direction)
    // north -> down, east -> left, south -> up, west -> right
    switch (direction) {
      case 'north':
        return { color: arrowColor, direction: 'down' as const };
      case 'south':
        return { color: arrowColor, direction: 'up' as const };
      case 'east':
        return { color: arrowColor, direction: 'up' as const };
      case 'west':
        return { color: arrowColor, direction: 'up' as const };
    }
  };

  const arrow = getArrow();
  const arrows = [arrow];

  return (
    <div className="flex flex-row items-center gap-3" style={{ transform: `scale(${scale}) rotate(${rotation}deg)`, transformOrigin: 'center' }}>
      {/* 1. Traffic Light Symbol (horizontal red/yellow/green) */}
      <HorizontalTrafficLight color={color} />
      
      {/* 2. Stopwatch Icon with Timer */}
      <div className={`flex items-center gap-1 px-2 py-1 rounded shadow-lg ${
        color === 'green' 
          ? 'bg-green-100 border-2 border-green-400' 
          : color === 'yellow'
          ? 'bg-yellow-100 border-2 border-yellow-400'
          : 'bg-red-100 border-2 border-red-400'
      }`}>
        <Timer className={`h-4 w-4 ${
          color === 'green' 
            ? 'text-green-600' 
            : color === 'yellow'
            ? 'text-yellow-600'
            : 'text-red-600'
        }`} />
        <span className={`text-sm font-bold ${
          color === 'green' 
            ? 'text-green-600' 
            : color === 'yellow'
            ? 'text-yellow-600'
            : 'text-red-600'
        }`}>
          {timer > 0 ? `${timer}s` : '0s'}
        </span>
      </div>

      {/* 3. Directional Traffic Signal (Arrow Signal Light) - MANDATORY */}
      <TrafficLightBox
        arrows={arrows}
        orientation="horizontal"
        boxBackgroundColor="#FFC107"
        cavityDiameter="40px"
        arrowSize="20px"
      />
    </div>
  );
};

export function IntersectionVisualization({
  junction,
  uploadedImages = {} as Record<Direction, string>,
  onImageUpload,
  showUploadControls = false
}: IntersectionVisualizationProps) {
  const handleImageUpload = (direction: Direction, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onImageUpload) {
      onImageUpload(direction, file);
    }
  };

  // PERMANENT BUILT-IN BACKGROUND IMAGE - Hard-coded, always visible, no upload option
  // This image is permanently embedded in the visualization and cannot be changed by users
  const backgroundImage = '/image_4d4f39.jpg';

  return (
    <div className="relative w-full aspect-square max-w-4xl mx-auto bg-slate-100">
      {/* Permanent Background Image - Intersection Map (image_4d4f39.jpg) */}
      <img
        src={backgroundImage}
        alt="Intersection Map"
        className="absolute inset-0 w-full h-full object-contain"
        style={{ zIndex: 0 }}
        onError={(e) => {
          console.error('Failed to load background image:', backgroundImage);
          (e.target as HTMLImageElement).style.display = 'none';
        }}
      />

      {/* Signal Units positioned on sidewalks/shoulders, off the road */}
      {/* All four roads MUST have signal units */}
      
      {/* North Road Signal Unit - Past zebra crossing, moved slightly further into intersection (north) */}
      <div className="absolute top-[26%] left-1/2 -translate-x-1/2 z-20">
        <SignalUnit
          direction="north"
          color={junction.signals.north.color}
          timer={junction.signals.north.timer}
          isActive={junction.signals.north.color === 'green'}
          scale={0.72}
        />
      </div>

      {/* South Road Signal Unit - Past zebra crossing, moved slightly further into intersection (south) */}
      <div className="absolute bottom-[26%] left-1/2 -translate-x-1/2 z-20">
        <SignalUnit
          direction="south"
          color={junction.signals.south.color}
          timer={junction.signals.south.timer}
          isActive={junction.signals.south.color === 'green'}
          scale={0.72}
        />
      </div>

      {/* East Road Signal Unit - Past zebra crossing, nudged slightly toward intersection (east) */}
      <div className="absolute top-1/2 right-[15%] -translate-y-1/2 z-20">
        <SignalUnit
          direction="east"
          color={junction.signals.east.color}
          timer={junction.signals.east.timer}
          isActive={junction.signals.east.color === 'green'}
          scale={0.72}
          rotation={-90}
        />
      </div>

      {/* West Road Signal Unit - Past zebra crossing, nudged slightly toward intersection (west) */}
      <div className="absolute top-1/2 left-[15%] -translate-y-1/2 z-20">
        <SignalUnit
          direction="west"
          color={junction.signals.west.color}
          timer={junction.signals.west.timer}
          isActive={junction.signals.west.color === 'green'}
          scale={0.72}
          rotation={90}
        />
      </div>

      {/* Traffic Images positioned on road lanes, clearly behind zebra crossings */}
      
      {/* North Road Image - On the road lane, approaching from top, well back from intersection */}
      {uploadedImages.north ? (
        <div className="absolute top-[2%] left-1/2 -translate-x-1/2 z-5" style={{ width: '24%', maxWidth: '320px' }}>
          <div className="relative">
            <img
              src={uploadedImages.north}
              alt="North traffic view"
              className="w-full h-auto object-cover rounded border-2 border-blue-500 shadow-lg"
              style={{ maxHeight: '200px', minHeight: '160px' }}
            />
            {showUploadControls && (
              <Button
                type="button"
                variant="secondary"
                size="sm"
                className="absolute top-1 right-1 opacity-90 hover:opacity-100 h-6 w-6 p-0 bg-white/90"
                onClick={() => document.getElementById('upload-north')?.click()}
              >
                <Upload className="h-3 w-3" />
              </Button>
            )}
            <Input
              type="file"
              accept="image/*"
              onChange={(e) => handleImageUpload('north', e)}
              className="hidden"
              id="upload-north"
            />
          </div>
        </div>
      ) : showUploadControls ? (
        <div className="absolute top-[8%] left-1/2 -translate-x-1/2 z-15">
          <Button
            type="button"
            variant="default"
            size="sm"
            onClick={() => document.getElementById('upload-north')?.click()}
            className="bg-blue-500 hover:bg-blue-600 text-white"
          >
            <Upload className="h-3 w-3 mr-1" />
            #1
          </Button>
          <Input
            type="file"
            accept="image/*"
            onChange={(e) => handleImageUpload('north', e)}
            className="hidden"
            id="upload-north"
          />
        </div>
      ) : null}

      {/* South Road Image - On the road lane, approaching from bottom, well back from intersection */}
      {uploadedImages.south ? (
        <div className="absolute bottom-[2%] left-1/2 -translate-x-1/2 z-5" style={{ width: '24%', maxWidth: '320px' }}>
          <div className="relative">
            <img
              src={uploadedImages.south}
              alt="South traffic view"
              className="w-full h-auto object-cover rounded border-2 border-blue-500 shadow-lg"
              style={{ maxHeight: '200px', minHeight: '160px' }}
            />
            {showUploadControls && (
              <Button
                type="button"
                variant="secondary"
                size="sm"
                className="absolute top-1 right-1 opacity-90 hover:opacity-100 h-6 w-6 p-0 bg-white/90"
                onClick={() => document.getElementById('upload-south')?.click()}
              >
                <Upload className="h-3 w-3" />
              </Button>
            )}
            <Input
              type="file"
              accept="image/*"
              onChange={(e) => handleImageUpload('south', e)}
              className="hidden"
              id="upload-south"
            />
          </div>
        </div>
      ) : showUploadControls ? (
        <div className="absolute bottom-[8%] left-1/2 -translate-x-1/2 z-15">
          <Button
            type="button"
            variant="default"
            size="sm"
            onClick={() => document.getElementById('upload-south')?.click()}
            className="bg-blue-500 hover:bg-blue-600 text-white"
          >
            <Upload className="h-3 w-3 mr-1" />
            #4
          </Button>
          <Input
            type="file"
            accept="image/*"
            onChange={(e) => handleImageUpload('south', e)}
            className="hidden"
            id="upload-south"
          />
        </div>
      ) : null}

      {/* East Road Image - On the road lane, approaching from right, well back from intersection */}
      {uploadedImages.east ? (
        <div className="absolute top-1/2 -translate-y-1/2 right-[-4%] z-5" style={{ width: '24%', maxWidth: '320px' }}>
          <div className="relative">
            <img
              src={uploadedImages.east}
              alt="East traffic view"
              className="w-full h-auto object-cover rounded border-2 border-blue-500 shadow-lg"
              style={{ maxHeight: '200px', minHeight: '160px' }}
            />
            {showUploadControls && (
              <Button
                type="button"
                variant="secondary"
                size="sm"
                className="absolute top-1 right-1 opacity-90 hover:opacity-100 h-6 w-6 p-0 bg-white/90"
                onClick={() => document.getElementById('upload-east')?.click()}
              >
                <Upload className="h-3 w-3" />
              </Button>
            )}
            <Input
              type="file"
              accept="image/*"
              onChange={(e) => handleImageUpload('east', e)}
              className="hidden"
              id="upload-east"
            />
          </div>
        </div>
      ) : showUploadControls ? (
        <div className="absolute top-1/2 right-[5%] -translate-y-1/2 z-15">
          <Button
            type="button"
            variant="default"
            size="sm"
            onClick={() => document.getElementById('upload-east')?.click()}
            className="bg-blue-500 hover:bg-blue-600 text-white"
          >
            <Upload className="h-3 w-3 mr-1" />
            #2
          </Button>
          <Input
            type="file"
            accept="image/*"
            onChange={(e) => handleImageUpload('east', e)}
            className="hidden"
            id="upload-east"
          />
        </div>
      ) : null}

      {/* West Road Image - On the road lane, approaching from left, well back from intersection */}
      {uploadedImages.west ? (
        <div className="absolute top-1/2 -translate-y-1/2 left-[-4%] z-5" style={{ width: '24%', maxWidth: '320px' }}>
          <div className="relative">
            <img
              src={uploadedImages.west}
              alt="West traffic view"
              className="w-full h-auto object-cover rounded border-2 border-blue-500 shadow-lg"
              style={{ maxHeight: '200px', minHeight: '160px' }}
            />
            {showUploadControls && (
              <Button
                type="button"
                variant="secondary"
                size="sm"
                className="absolute top-1 right-1 opacity-90 hover:opacity-100 h-6 w-6 p-0 bg-white/90"
                onClick={() => document.getElementById('upload-west')?.click()}
              >
                <Upload className="h-3 w-3" />
              </Button>
            )}
            <Input
              type="file"
              accept="image/*"
              onChange={(e) => handleImageUpload('west', e)}
              className="hidden"
              id="upload-west"
            />
          </div>
        </div>
      ) : showUploadControls ? (
        <div className="absolute top-1/2 left-[5%] -translate-y-1/2 z-15">
          <Button
            type="button"
            variant="default"
            size="sm"
            onClick={() => document.getElementById('upload-west')?.click()}
            className="bg-blue-500 hover:bg-blue-600 text-white"
          >
            <Upload className="h-3 w-3 mr-1" />
            #3
          </Button>
          <Input
            type="file"
            accept="image/*"
            onChange={(e) => handleImageUpload('west', e)}
            className="hidden"
            id="upload-west"
          />
        </div>
      ) : null}
    </div>
  );
}
