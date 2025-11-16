import { useState } from 'react';
import { LoginPage } from '@/components/login-page';
import { Dashboard } from '@/components/dashboard/dashboard';
import type { User } from '@/types/auth';

function App() {
  const [user, setUser] = useState<User | null>(null);

  if (!user) {
    return <LoginPage onLogin={setUser} />;
  }

  return <Dashboard user={user} onLogout={() => setUser(null)} />;
}

export default App;