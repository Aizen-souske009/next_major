'use client';

import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { Button } from '@/components/ui/button';

export default function UserDebug() {
  const { data: session } = useSession();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/users');
      const data = await response.json();
      setUsers(data.users || []);
    } catch (error) {
      console.error('Error fetching users:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (session) {
      fetchUsers();
    }
  }, [session]);

  if (!session) return null;

  return (
    <div className="bg-card rounded-lg shadow-md p-6 border border-border mt-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-medium">User Database Debug</h3>
        <Button onClick={fetchUsers} disabled={loading}>
          {loading ? 'Loading...' : 'Refresh'}
        </Button>
      </div>
      
      <div className="space-y-4">
        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
          <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">Current Session:</h4>
          <pre className="text-sm text-green-700 dark:text-green-300 overflow-x-auto">
            {JSON.stringify({
              id: session.user?.id,
              name: session.user?.name,
              email: session.user?.email,
              role: session.user?.role,
              lastLogin: session.user?.lastLogin
            }, null, 2)}
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
            Database Users ({users.length}):
          </h4>
          <div className="max-h-64 overflow-y-auto">
            {users.map((user) => (
              <div key={user.id} className="mb-3 p-3 bg-white dark:bg-gray-800 rounded border">
                <div className="text-sm space-y-1">
                  <p><strong>Name:</strong> {user.name}</p>
                  <p><strong>Email:</strong> {user.email}</p>
                  <p><strong>Role:</strong> {user.role}</p>
                  <p><strong>Last Login:</strong> {user.lastLogin ? new Date(user.lastLogin).toLocaleString() : 'Never'}</p>
                  <p><strong>Created:</strong> {new Date(user.createdAt).toLocaleString()}</p>
                  <p><strong>Updated:</strong> {new Date(user.updatedAt).toLocaleString()}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}