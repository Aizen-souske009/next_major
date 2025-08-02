'use client';

import { useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { redirect } from 'next/navigation';
import SidebarLayout from '@/components/sidebar-layout';
import AnalysisHistory from '@/components/analysis-history';
import { FileText } from 'lucide-react';

export default function ReportsPage() {
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
            <div className="p-6 space-y-6">
                {/* Header */}
                <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl p-6 shadow-lg text-white">
                    <div className="flex items-center gap-3">
                        <FileText className="h-8 w-8" />
                        <div>
                            <h1 className="text-2xl font-bold">Analysis Reports</h1>
                            <p className="text-purple-100 mt-1">
                                View and manage your image analysis history
                            </p>
                        </div>
                    </div>
                </div>

                {/* Analysis History Component */}
                <AnalysisHistory />
            </div>
        </SidebarLayout>
    );
}