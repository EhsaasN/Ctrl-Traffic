export interface User {
  id: string;
  authorizedId: string;
  name: string;
  role: string;
  primaryJunctionId?: string;
}

export const MOCK_USERS: Record<string, { password: string; user: User }> = {
  'ADMIN001': {
    password: 'traffic2024',
    user: {
      id: '1',
      authorizedId: 'ADMIN001',
      name: 'Chief Traffic Officer',
      role: 'Admin',
      primaryJunctionId: 'J001'
    }
  },
  'POLICE001': {
    password: 'police2024',
    user: {
      id: '2',
      authorizedId: 'POLICE001',
      name: 'Traffic Controller',
      role: 'Operator',
      primaryJunctionId: 'J002'
    }
  }
};
