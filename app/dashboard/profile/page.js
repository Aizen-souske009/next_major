'use client';

import { useEffect, useRef } from 'react';
import { useSession } from 'next-auth/react';
import { redirect } from 'next/navigation';
import Image from 'next/image';
import SidebarLayout from '@/components/sidebar-layout';
import { User, Mail, Calendar, Cpu, Edit } from 'lucide-react';

// Card Components (Shadcn Style)
const Card = ({ className, ...props }) => (
    <div className={`rounded-lg border bg-card text-card-foreground shadow-sm ${className}`} {...props} />
);
const CardHeader = ({ className, ...props }) => (
    <div className={`flex flex-col space-y-1.5 p-6 ${className}`} {...props} />
);
const CardTitle = ({ className, ...props }) => (
    <h3 className={`text-2xl font-semibold leading-none tracking-tight ${className}`} {...props} />
);
const CardContent = ({ className, ...props }) => <div className={`p-6 pt-0 ${className}`} {...props} />;

export default function Profile() {
    const { data: session, status } = useSession();
    const cardRef = useRef(null);

    // Redirect if not authenticated
    useEffect(() => {
        if (status === 'unauthenticated') redirect('/login');
    }, [status]);

    // Subtle 3D hover effect
    useEffect(() => {
        const card = cardRef.current;
        if (!card) return;

        const handleMouseMove = (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const rotateX = (y - rect.height / 2) / 30;
            const rotateY = (rect.width / 2 - x) / 30;
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        };

        const handleMouseLeave = () => (card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0)');
        card.addEventListener('mousemove', handleMouseMove);
        card.addEventListener('mouseleave', handleMouseLeave);

        return () => {
            card.removeEventListener('mousemove', handleMouseMove);
            card.removeEventListener('mouseleave', handleMouseLeave);
        };
    }, []);

    if (status === 'loading') {
        return (
            <main className="flex min-h-screen bg-background text-foreground items-center justify-center p-4">
                <p className="text-muted-foreground">Loading profile...</p>
            </main>
        );
    }

    const formatDate = (dateString) => {
        if (!dateString) return 'Not available';
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
        });
    };

    return (
        <SidebarLayout>
            <div className="p-6 pt-16 md:pt-6">
                <h1 className="text-3xl font-bold mb-6">Profile</h1>

                <div className="max-w-3xl mx-auto">
                    <Card
                        ref={cardRef}
                        className="overflow-hidden transition-all duration-300 ease-out border-primary/10 hover:border-primary/30 hover:shadow-md bg-gradient-to-b from-background to-background/70"
                    >
                        {/* Top accent line */}
                        <div className="h-1 w-full bg-gradient-to-r from-fuchsia-500 to-cyan-500 opacity-80" />

                        <CardHeader className="relative pb-2">
                            <div className="flex flex-col md:flex-row items-center gap-6">
                                {/* Profile image */}
                                <div className="relative h-24 w-24 rounded-full overflow-hidden border-2 border-primary/10 shadow-md">
                                    {session?.user?.image ? (
                                        <Image src={session.user.image} alt="Profile" fill sizes="96px" className="object-cover" />
                                    ) : (
                                        <div className="h-full w-full flex items-center justify-center">
                                            <User className="h-12 w-12 text-primary/40" />
                                        </div>
                                    )}
                                </div>

                                <div className="text-center md:text-left">
                                    <CardTitle>{session?.user?.name || 'User'}</CardTitle>
                                    <p className="text-muted-foreground">{session?.user?.email}</p>

                                    {/* Edit Button */}
                                    <button className="mt-3 inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium border border-primary/20 bg-primary/10 text-primary hover:bg-primary/20 transition">
                                        <Edit className="w-4 h-4 mr-1" /> Edit Profile
                                    </button>
                                </div>
                            </div>
                        </CardHeader>

                        <CardContent>
                            <div className="space-y-4">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {/* Name */}
                                    <div className="flex items-center gap-3 p-4 rounded-lg border bg-card shadow-sm hover:border-primary/30">
                                        <div className="h-10 w-10 rounded-full bg-fuchsia-50 dark:bg-fuchsia-900/20 flex items-center justify-center">
                                            <User className="h-5 w-5 text-fuchsia-600 dark:text-fuchsia-400" />
                                        </div>
                                        <div>
                                            <p className="text-sm text-muted-foreground">Full Name</p>
                                            <p className="font-medium">{session?.user?.name || 'Not provided'}</p>
                                        </div>
                                    </div>

                                    {/* Email */}
                                    <div className="flex items-center gap-3 p-4 rounded-lg border bg-card shadow-sm hover:border-primary/30">
                                        <div className="h-10 w-10 rounded-full bg-cyan-50 dark:bg-cyan-900/20 flex items-center justify-center">
                                            <Mail className="h-5 w-5 text-cyan-600 dark:text-cyan-400" />
                                        </div>
                                        <div>
                                            <p className="text-sm text-muted-foreground">Email</p>
                                            <p className="font-medium">{session?.user?.email || 'Not provided'}</p>
                                        </div>
                                    </div>
                                </div>

                                {/* Account Created */}
                                <div className="flex items-center gap-3 p-4 rounded-lg border bg-card shadow-sm hover:border-primary/30">
                                    <div className="h-10 w-10 rounded-full bg-indigo-50 dark:bg-indigo-900/20 flex items-center justify-center">
                                        <Calendar className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
                                    </div>
                                    <div>
                                        <p className="text-sm text-muted-foreground">Account Created</p>
                                        <p className="font-medium">{formatDate(session?.user?.createdAt)}</p>
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </SidebarLayout>
    );
}
