'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { redirect } from 'next/navigation';
import Image from 'next/image';
import SidebarLayout from '@/components/sidebar-layout';
import { Upload, Activity, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Logo from '@/components/Logo'; // <-- Import the new logo
import RecentAnalyses from '@/components/recent-analyses';

export default function Dashboard() {
    const { data: session, status } = useSession();

    // State for uploaded images and reports
    const [uploadedImages, setUploadedImages] = useState([]);
    const [scanReports, setScanReports] = useState([]);

    // Redirect if user not authenticated
    useEffect(() => {
        if (status === 'unauthenticated') {
            redirect('/login');
        }
    }, [status]);

    // Simulate fetching user's uploaded images and scan reports
    useEffect(() => {
        const fetchData = async () => {
            const userUploads = []; // Replace with actual fetched data
            const userReports = []; // Replace with actual fetched data
            setUploadedImages(userUploads);
            setScanReports(userReports);
        };
        fetchData();
    }, []);

    if (status === 'loading') {
        return (
            <main className="flex min-h-screen bg-background text-foreground items-center justify-center p-4">
                <p className="text-muted-foreground">Loading...</p>
            </main>
        );
    }

    return (
        <SidebarLayout>
            <div className="p-6 space-y-8">
                {/* Header */}
                <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl p-6 shadow-lg text-white">
                    <div className="flex items-center gap-4">
                        {/* Logo */}
                        <div className="flex items-center gap-2">
                            <Logo /> {/* New animated logo */}
                            <span className="text-xl font-semibold tracking-wide">
                                DeepTrace AI
                            </span>
                        </div>

                        {session?.user?.image && (
                            <div className="relative h-14 w-14 rounded-full border-2 border-white overflow-hidden group ml-auto">
                                <Image
                                    src={session.user.image}
                                    alt="Profile"
                                    fill
                                    sizes="64px"
                                    className="object-cover"
                                />
                                <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                    <p className="text-xs">{session?.user?.name}</p>
                                </div>
                            </div>
                        )}
                    </div>
                    <p className="text-sm opacity-80 mt-2">Your AI image scan dashboard</p>
                </div>

                {/* Upload Section */}
                <div className="bg-card rounded-lg shadow-md p-6 border border-border">
                    <h3 className="text-xl font-medium mb-4 flex items-center gap-2">
                        <Upload className="h-5 w-5 text-primary" />
                        Upload Images
                    </h3>
                    <div className="text-center space-y-4">
                        <p className="text-gray-600 dark:text-gray-400">
                            Upload an image to detect if it's AI-generated or real using our trained model
                        </p>
                        <Button 
                            onClick={() => window.location.href = '/dashboard/analyze'}
                            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                        >
                            Start AI Detection
                        </Button>
                    </div>

                    {/* Recent Uploads */}
                    {uploadedImages.length > 0 ? (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                            {uploadedImages.map((img, idx) => (
                                <div key={idx} className="border rounded-lg overflow-hidden shadow-sm">
                                    <img src={img.url} alt="Upload" className="w-full h-32 object-cover" />
                                    <p className="text-xs text-center text-muted-foreground mt-1">{img.result}</p>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-muted-foreground mt-4 text-center">
                            You haven’t uploaded any images yet.
                        </p>
                    )}
                </div>

                {/* Recent Analyses */}
                <div className="bg-card rounded-lg shadow-md p-6 border border-border">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-xl font-medium flex items-center gap-2">
                            <Activity className="h-5 w-5 text-primary" />
                            Recent Analyses
                        </h3>
                        <Button 
                            variant="outline"
                            onClick={() => window.location.href = '/dashboard/reports'}
                        >
                            View All
                        </Button>
                    </div>
                    
                    <RecentAnalyses />
                </div>

            </div>
        </SidebarLayout>
    );
}
