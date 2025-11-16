// intersection-visualization.tsx
import React, { useEffect, useRef, useState } from 'react';
import { Upload, Timer } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { Junction, Direction, SignalColor } from '@/types/junction';

interface MediaSet {
  image?: string;
  audio?: string;
  video?: string;
}

interface IntersectionVisualizationProps {
  junction: Junction;
  uploadedMedia?: Record<Direction, MediaSet>;
  uploadedImages?: Record<Direction, string>;
  onMediaUpload?: (direction: Direction, file: File, mediaType: 'image' | 'audio' | 'video') => void;
  onImageUpload?: (direction: Direction, file: File) => void;
  showUploadControls?: boolean;
}

const DIRECTION_NUMBER: Record<Direction, number> = {
  north: 1,
  east: 2,
  west: 3,
  south: 4
};

const offscreenStyle: React.CSSProperties = {
  position: 'absolute',
  left: '-9999px',
  width: '1px',
  height: '1px',
  overflow: 'hidden'
};

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

const ArrowSVG = ({ color = 'green', size = '16px', direction = 'up' }: { color?: string; size?: string; direction?: 'up' | 'down' | 'left' | 'right'; }) => {
  let transform = '';
  switch (direction) {
    case 'right': transform = 'rotate(90 10 10)'; break;
    case 'down': transform = 'rotate(180 10 10)'; break;
    case 'left': transform = 'rotate(270 10 10)'; break;
    case 'up':
    default: transform = ''; break;
  }
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden="true" style={{ display: 'block' }}>
      <path d="M 10 0 L 20 10 L 14 10 L 14 20 L 6 20 L 6 10 L 0 10 Z" fill={color} transform={transform} />
    </svg>
  );
};

const TrafficLightBox = ({ arrows = [], orientation = 'horizontal', boxBackgroundColor = '#FFC107', cavityColor = '#333', cavityDiameter = '50px', arrowSize = '24px' }: { arrows: Array<{ color: string; direction: 'up' | 'down' | 'left' | 'right' }>; orientation?: 'vertical' | 'horizontal'; boxBackgroundColor?: string; cavityColor?: string; cavityDiameter?: string; arrowSize?: string; }) => {
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
          <ArrowSVG color={arrow.color} size={arrowSize} direction={arrow.direction} />
        </div>
      ))}
    </div>
  );
};

const SignalUnit = ({ direction, color, timer, isActive, scale = 1, rotation = 0 }: { direction: Direction; color: SignalColor; timer: number; isActive: boolean; scale?: number; rotation?: number; }) => {
  const getArrow = () => {
    const arrowColor = isActive && color === 'green' ? '#22c55e' : '#6b7280';
    switch (direction) {
      case 'north': return { color: arrowColor, direction: 'down' as const };
      case 'south': return { color: arrowColor, direction: 'up' as const };
      case 'east': return { color: arrowColor, direction: 'up' as const };
      case 'west': return { color: arrowColor, direction: 'up' as const };
    }
  };
  const arrow = getArrow();
  const arrows = [arrow];
  return (
    <div className="flex flex-row items-center gap-3" style={{ transform: `scale(${scale}) rotate(${rotation}deg)`, transformOrigin: 'center' }}>
      <HorizontalTrafficLight color={color} />
      <div className={`flex items-center gap-1 px-2 py-1 rounded shadow-lg ${ color === 'green' ? 'bg-green-100 border-2 border-green-400' : color === 'yellow' ? 'bg-yellow-100 border-2 border-yellow-400' : 'bg-red-100 border-2 border-red-400' }`}>
        <Timer className={`h-4 w-4 ${ color === 'green' ? 'text-green-600' : color === 'yellow' ? 'text-yellow-600' : 'text-red-600' }`} />
        <span className={`text-sm font-bold ${ color === 'green' ? 'text-green-600' : color === 'yellow' ? 'text-yellow-600' : 'text-red-600' }`}>{timer > 0 ? `${timer}s` : '0s'}</span>
      </div>
      <TrafficLightBox arrows={arrows} orientation="horizontal" boxBackgroundColor="#FFC107" cavityDiameter="40px" arrowSize="20px" />
    </div>
  );
};

export function IntersectionVisualization({ junction, uploadedMedia, uploadedImages, onMediaUpload, onImageUpload, showUploadControls = false }: IntersectionVisualizationProps) {
  const [localPreviews, setLocalPreviews] = useState<Record<Direction, MediaSet>>({
    north: {}, east: {}, south: {}, west: {}
  });

  const objectUrlsRef = useRef<string[]>([]);
  useEffect(() => {
    return () => {
      objectUrlsRef.current.forEach((u) => {
        try { URL.revokeObjectURL(u); } catch { /* ignore */ }
      });
      objectUrlsRef.current = [];
    };
  }, []);

  const backgroundImage = '/image_4d4f39.jpg';
  const acceptStr = '.jpg,.jpeg,.png,.wav,.mp4,image/*,audio/wav,video/mp4';

  const pickMediaTypeFromFile = (file: File): 'image' | 'audio' | 'video' => {
    if (file.type.startsWith('image/')) return 'image';
    if (file.type.startsWith('audio/')) return 'audio';
    if (file.type.startsWith('video/')) return 'video';
    const n = file.name.toLowerCase();
    if (n.endsWith('.wav')) return 'audio';
    if (n.endsWith('.mp4')) return 'video';
    return 'image';
  };

  const handleUnifiedFileChange = (direction: Direction, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const mediaType = pickMediaTypeFromFile(file);

    const url = URL.createObjectURL(file);
    objectUrlsRef.current.push(url);

    setLocalPreviews((prev) => {
      const copy = { ...prev, [direction]: { ...(prev[direction] || {}) } };
      if (mediaType === 'image') copy[direction].image = url;
      if (mediaType === 'audio') copy[direction].audio = url;
      if (mediaType === 'video') copy[direction].video = url;
      return copy;
    });

    if (onMediaUpload) onMediaUpload(direction, file, mediaType);
    if (mediaType === 'image' && onImageUpload) onImageUpload(direction, file);

    e.currentTarget.value = '';
  };

  const getPreviewFor = (direction: Direction, type: 'image' | 'audio' | 'video'): string | undefined => {
    if (uploadedMedia && uploadedMedia[direction]) {
      const val = uploadedMedia[direction][type];
      if (val) return val;
    }
    if (type === 'image' && uploadedImages && uploadedImages[direction]) {
      return uploadedImages[direction];
    }
    const local = localPreviews[direction] && localPreviews[direction][type];
    return local;
  };

  const hasAnyPreview = (direction: Direction) => {
    return !!(getPreviewFor(direction, 'image') || getPreviewFor(direction, 'audio') || getPreviewFor(direction, 'video'));
  };

  // ensure previews are smaller and behind signals (z-index)
  const renderPreviewsFor = (direction: Direction) => {
    const img = getPreviewFor(direction, 'image');
    const aud = getPreviewFor(direction, 'audio');
    const vid = getPreviewFor(direction, 'video');

    return (
      <div className="mt-1 space-y-2" style={{ zIndex: 10 }}>
        {img && (
          <img src={img} alt={`${direction} preview`} className="w-full h-auto object-cover rounded border-2 border-blue-500 shadow-lg" style={{ maxHeight: 140, minHeight: 100 }} />
        )}
        {aud && (
          <div className="w-full">
            <audio controls src={aud} className="w-full" />
          </div>
        )}
        {vid && (
          <div className="w-full">
            <video controls playsInline src={vid} className="w-full rounded border-2 border-blue-500 shadow-lg" style={{ maxHeight: 180 }} />
          </div>
        )}
      </div>
    );
  };

  // render upload button (used either inside preview or in the original standalone area)
  const renderUploadButtonContent = (dir: Direction, uploadId: string) => (
    <>
      <Button type="button" variant="default" size="sm" onClick={() => document.getElementById(uploadId)?.click()} className="bg-blue-500 hover:bg-blue-600 text-white flex items-center gap-2">
        <Upload className="h-3 w-3" />
        <span className="font-semibold">#{DIRECTION_NUMBER[dir]}</span>
      </Button>
      <input id={uploadId} type="file" accept={acceptStr} onChange={(e) => handleUnifiedFileChange(dir, e)} style={offscreenStyle} />
    </>
  );

  return (
    <div className="relative w-full aspect-square max-w-4xl mx-auto bg-slate-100">
      <img src={backgroundImage} alt="Intersection Map" className="absolute inset-0 w-full h-full object-contain" style={{ zIndex: 0 }} onError={(e) => { console.error('Failed to load background image:', backgroundImage); (e.target as HTMLImageElement).style.display = 'none'; }} />

      {/* Signals: kept on top */}
      <div className="absolute top-[26%] left-1/2 -translate-x-1/2 z-30">
        <SignalUnit direction="north" color={junction.signals.north.color} timer={junction.signals.north.timer} isActive={junction.signals.north.color === 'green'} scale={0.72} />
      </div>
      <div className="absolute bottom-[26%] left-1/2 -translate-x-1/2 z-30">
        <SignalUnit direction="south" color={junction.signals.south.color} timer={junction.signals.south.timer} isActive={junction.signals.south.color === 'green'} scale={0.72} />
      </div>
      <div className="absolute top-1/2 right-[15%] -translate-y-1/2 z-30">
        <SignalUnit direction="east" color={junction.signals.east.color} timer={junction.signals.east.timer} isActive={junction.signals.east.color === 'green'} scale={0.72} rotation={-90} />
      </div>
      <div className="absolute top-1/2 left-[15%] -translate-y-1/2 z-30">
        <SignalUnit direction="west" color={junction.signals.west.color} timer={junction.signals.west.timer} isActive={junction.signals.west.color === 'green'} scale={0.72} rotation={90} />
      </div>

      {/* NORTH: if preview exists, render preview block with internal button; else render standalone button like original */}
      {hasAnyPreview('north') ? (
        <div className="absolute top-[2%] left-1/2 -translate-x-1/2 z-10" style={{ width: '24%', maxWidth: 320 }}>
          <div className="relative">
            {renderPreviewsFor('north')}
            {showUploadControls && <div className="absolute top-1 right-1">{renderUploadButtonContent('north', 'upload-north')}</div>}
          </div>
        </div>
      ) : showUploadControls ? (
        <div className="absolute top-[8%] left-1/2 -translate-x-1/2 z-15">
          {renderUploadButtonContent('north', 'upload-north')}
        </div>
      ) : null}

      {/* SOUTH */}
      {hasAnyPreview('south') ? (
        <div className="absolute bottom-[2%] left-1/2 -translate-x-1/2 z-10" style={{ width: '24%', maxWidth: 320 }}>
          <div className="relative">
            {renderPreviewsFor('south')}
            {showUploadControls && <div className="absolute top-1 right-1">{renderUploadButtonContent('south', 'upload-south')}</div>}
          </div>
        </div>
      ) : showUploadControls ? (
        <div className="absolute bottom-[8%] left-1/2 -translate-x-1/2 z-15">
          {renderUploadButtonContent('south', 'upload-south')}
        </div>
      ) : null}

      {/* EAST */}
      {hasAnyPreview('east') ? (
        <div className="absolute top-1/2 -translate-y-1/2 right-[-4%] z-10" style={{ width: '24%', maxWidth: 320 }}>
          <div className="relative">
            {renderPreviewsFor('east')}
            {showUploadControls && <div className="absolute top-1 right-1">{renderUploadButtonContent('east', 'upload-east')}</div>}
          </div>
        </div>
      ) : showUploadControls ? (
        <div className="absolute top-1/2 right-[5%] -translate-y-1/2 z-15">
          {renderUploadButtonContent('east', 'upload-east')}
        </div>
      ) : null}

      {/* WEST */}
      {hasAnyPreview('west') ? (
        <div className="absolute top-1/2 -translate-y-1/2 left-[-4%] z-10" style={{ width: '24%', maxWidth: 320 }}>
          <div className="relative">
            {renderPreviewsFor('west')}
            {showUploadControls && <div className="absolute top-1 right-1">{renderUploadButtonContent('west', 'upload-west')}</div>}
          </div>
        </div>
      ) : showUploadControls ? (
        <div className="absolute top-1/2 left-[5%] -translate-y-1/2 z-15">
          {renderUploadButtonContent('west', 'upload-west')}
        </div>
      ) : null}
    </div>
  );
}
