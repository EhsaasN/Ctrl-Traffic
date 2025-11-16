import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { ShieldCheck, Lock, User as UserIcon } from 'lucide-react';
import { MOCK_USERS, type User } from '@/types/auth';

interface LoginPageProps {
  onLogin: (user: User) => void;
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const [authorizedId, setAuthorizedId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    const userRecord = MOCK_USERS[authorizedId];
    if (userRecord && userRecord.password === password) {
      onLogin(userRecord.user);
    } else {
      setError('Invalid credentials. Access denied.');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-white/10 backdrop-blur-sm rounded-full p-4 border-2 border-white/20">
              <ShieldCheck className="h-12 w-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">Traffix Admin Portal</h1>
          <p className="text-blue-200 text-lg">Traffic Control Center</p>
        </div>

        <Card className="shadow-2xl border-blue-300/20">
          <CardHeader>
            <CardTitle className="text-2xl">Secure Login</CardTitle>
            <CardDescription>Authorized Access Only</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="authorizedId">Authorized ID</Label>
                <div className="relative">
                  <UserIcon className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="authorizedId"
                    type="text"
                    placeholder="Enter your authorized ID"
                    value={authorizedId}
                    onChange={(e) => setAuthorizedId(e.target.value)}
                    className="pl-9"
                    required
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="password">Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-9"
                    required
                  />
                </div>
              </div>

              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded text-sm">
                  {error}
                </div>
              )}

              <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700">
                Login
              </Button>

              <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground pt-2">
                <ShieldCheck className="h-3 w-3" />
                <span>Secure connection established</span>
              </div>
            </form>

            <div className="mt-6 pt-4 border-t text-xs text-muted-foreground">
              <p className="font-semibold mb-1">Demo Credentials:</p>
              <p>ID: ADMIN001 / Password: traffic2024</p>
              <p>ID: POLICE001 / Password: police2024</p>
            </div>
          </CardContent>
        </Card>

        <div className="text-center mt-6 text-sm text-blue-200">
          <p>&copy; 2024 Traffix Traffic Control System</p>
        </div>
      </div>
    </div>
  );
}
