import type { Junction, Alert } from '@/types/junction';

export const MOCK_JUNCTIONS: Junction[] = [
  {
    id: 'J001',
    name: 'Junction-01: City Center',
    location: 'Main St & 1st Ave',
    imageUrl: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdxCbQyBgYWcPB5kmnqMUdx3Z1CINYBq6g1w&s',
    signals: {
      north: { color: 'green', timer: 45 },
      south: { color: 'red', timer: 0 },
      east: { color: 'red', timer: 0 },
      west: { color: 'red', timer: 0 }
    },
    lastUpdated: new Date(),
    responsibleOfficer: 'Traffic Police',
    controlMode: 'ai'
  },
  {
    id: 'J002',
    name: 'Junction-02: Market Square',
    location: 'Market Rd & Commerce St',
    imageUrl: 'https://tse4.mm.bing.net/th/id/OIP.--6KE9JdEiVKem0Vi1FZBQHaEh?pid=Api&P=0&h=180',
    signals: {
      north: { color: 'red', timer: 0 },
      south: { color: 'red', timer: 0 },
      east: { color: 'green', timer: 32 },
      west: { color: 'red', timer: 0 }
    },
    lastUpdated: new Date(),
    responsibleOfficer: 'Traffic Controller',
    controlMode: 'ai'
  },
  {
    id: 'J003',
    name: 'Junction-03: Airport Road',
    location: 'Airport Rd & Terminal Way',
    imageUrl: 'https://www.shutterstock.com/image-illustration/3d-illustration-4-way-junction-260nw-2350135343.jpg',
    signals: {
      north: { color: 'yellow', timer: 3 },
      south: { color: 'red', timer: 0 },
      east: { color: 'red', timer: 0 },
      west: { color: 'red', timer: 0 }
    },
    lastUpdated: new Date(),
    responsibleOfficer: 'Officer Smith',
    controlMode: 'manual'
  },
  {
    id: 'J004',
    name: 'Junction-04: Hospital Junction',
    location: 'Hospital Rd & Medical Center Dr',
    imageUrl: 'https://c8.alamy.com/comp/AC4ABH/aerial-view-above-four-way-urban-intersection-san-francisco-california-AC4ABH.jpg',
    signals: {
      north: { color: 'red', timer: 0 },
      south: { color: 'green', timer: 28 },
      east: { color: 'red', timer: 0 },
      west: { color: 'red', timer: 0 }
    },
    lastUpdated: new Date(),
    responsibleOfficer: 'Officer Johnson',
    controlMode: 'ai'
  },
  {
    id: 'J005',
    name: 'Junction-05: University Gate',
    location: 'College Rd & University Blvd',
    imageUrl: '/assets/images/default.svg',
    signals: {
      north: { color: 'red', timer: 0 },
      south: { color: 'red', timer: 0 },
      east: { color: 'red', timer: 0 },
      west: { color: 'green', timer: 51 }
    },
    lastUpdated: new Date(),
    responsibleOfficer: 'Officer Davis',
    controlMode: 'ai'
  },
  {
    id: 'J006',
    name: 'Junction-06: Industrial Area',
    location: 'Factory Rd & Industrial Park',
    imageUrl: '/assets/images/default.svg',
    signals: {
      north: { color: 'green', timer: 19 },
      south: { color: 'red', timer: 0 },
      east: { color: 'red', timer: 0 },
      west: { color: 'red', timer: 0 }
    },
    lastUpdated: new Date(),
    responsibleOfficer: 'Officer Martinez',
    controlMode: 'ai'
  }
];

export const MOCK_ALERTS: Alert[] = [
  {
    id: 'A001',
    message: 'Junction-03 manual override active',
    type: 'info',
    timestamp: new Date(Date.now() - 5 * 60000)
  },
  {
    id: 'A002',
    message: 'Heavy traffic detected at Junction-01',
    type: 'warning',
    timestamp: new Date(Date.now() - 12 * 60000)
  },
  {
    id: 'A003',
    message: 'System sync completed successfully',
    type: 'info',
    timestamp: new Date(Date.now() - 25 * 60000)
  }
];
