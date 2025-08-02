'use client';

import { useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { redirect } from 'next/navigation';
import SidebarLayout from '@/components/sidebar-layout';
import DeepfakeDetector from '@/components/deepfake-detector';

export default function AnalyzePage() {
    const { data: session, status } = useSession();

    // Redirect if user not authenticated
    useEffect(() => {
        if (status === 'unauthenticated') {
            redirect('/login');
        }
    }, [status]);

    if (status === 'loading') {
        return (
            <main className="flex min-h-screen bg-background text-foreground items-center justify-center p-4">
                <p className="text-muted-foreground">Loading...</p>
            </main>
        );
    }

    return (
        <SidebarLayout>
            <DeepfakeDetector />
        </SidebarLayout>
    );
}