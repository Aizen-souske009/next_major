'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useSession, signOut } from 'next-auth/react';
import { Activity, Home, User, FileText, LogOut, Menu, X, ChevronRight } from 'lucide-react';
import { ModeToggle } from './dark-mode';
import { Button } from './ui/button';
import UserAvatar from './user-avatar'; // <-- New component

const SidebarLayout = ({ children }) => {
    const pathname = usePathname();
    const { data: session } = useSession();
    const [collapsed, setCollapsed] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);

    const navItems = [
        { label: 'Dashboard', href: '/dashboard', icon: <Home className="w-5 h-5" /> },
        { label: 'Profile', href: '/dashboard/profile', icon: <User className="w-5 h-5" /> },
        { label: 'Analyze', href: '/dashboard/analyze', icon: <Activity className="w-5 h-5" /> }, // Changed
        { label: 'Reports', href: '/dashboard/reports', icon: <FileText className="w-5 h-5" /> },
    ];

    const handleSignOut = () => {
        signOut({ callbackUrl: '/' });
    };

    return (
        <div className="flex h-screen bg-background">
            {/* Mobile toggle button */}
            <div className="md:hidden fixed top-4 left-4 z-50">
                <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setMobileOpen(!mobileOpen)}
                    className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border border-gray-200 dark:border-gray-800 shadow-md"
                >
                    {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
                </Button>
            </div>

            {/* Mobile sidebar */}
            <div
                className={`
                    fixed inset-y-0 left-0 z-40 w-64 bg-white/90 dark:bg-gray-900/90 backdrop-blur-md border-r border-gray-200 dark:border-gray-800 shadow-lg
                    transform transition-transform duration-300 ease-in-out md:hidden
                    ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}
                `}
            >
                <SidebarContent
                    navItems={navItems}
                    pathname={pathname}
                    session={session}
                    handleSignOut={handleSignOut}
                    collapsed={false}
                    setCollapsed={() => { }}
                    isMobile={true}
                    closeMobile={() => setMobileOpen(false)}
                />
            </div>

            {/* Desktop sidebar */}
            <div
                className={`
                    hidden md:block relative z-30 bg-white/90 dark:bg-gray-900/90 backdrop-blur-md border-r border-gray-200 dark:border-gray-800 shadow-lg
                    transition-all duration-300 ease-in-out
                    ${collapsed ? 'w-20' : 'w-64'}
                `}
            >
                <SidebarContent
                    navItems={navItems}
                    pathname={pathname}
                    session={session}
                    handleSignOut={handleSignOut}
                    collapsed={collapsed}
                    setCollapsed={setCollapsed}
                    isMobile={false}
                />
            </div>

            {/* Main content */}
            <div className="flex-1 overflow-auto">
                <main className="min-h-screen">{children}</main>
            </div>
        </div>
    );
};

const SidebarContent = ({
    navItems,
    pathname,
    session,
    handleSignOut,
    collapsed,
    setCollapsed,
    isMobile,
    closeMobile,
}) => {
    const profileImage = session?.user?.image || '/images/default-avatar.png';

    return (
        <div className="flex flex-col h-full">
            {/* Logo + collapse */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
                <Link href="/" className="flex items-center">
                    {!collapsed && (
                        <span className="ml-1 text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-fuchsia-400 via-cyan-400 to-fuchsia-300">
                            DeepTrace AI
                        </span>
                    )}
                </Link>
                {!isMobile && (
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setCollapsed(!collapsed)}
                        className="text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                    >
                        <ChevronRight
                            className={`h-5 w-5 transition-transform ${collapsed ? 'rotate-180' : ''}`}
                        />
                    </Button>
                )}
            </div>

            {/* User profile */}
            <div className={`p-4 border-b border-gray-200 dark:border-gray-800 ${collapsed ? 'text-center' : ''}`}>
                <div className={`flex ${collapsed ? 'flex-col items-center' : 'items-center'}`}>
                    <UserAvatar
                        src={profileImage}
                        size={40}
                        alt={session?.user?.name || 'User'}
                        className="border border-gray-300 dark:border-gray-700"
                    />
                    {!collapsed && (
                        <div className="ml-3 overflow-hidden">
                            <p className="text-sm font-medium truncate">{session?.user?.name || 'User'}</p>
                            <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                                {session?.user?.email || 'user@example.com'}
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 overflow-y-auto py-4">
                <ul className="space-y-1 px-2">
                    {navItems.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <li key={item.href}>
                                <Link
                                    href={item.href}
                                    onClick={isMobile ? closeMobile : undefined}
                                    className={`
                                        flex items-center px-3 py-2 rounded-md text-sm font-medium transition-all duration-200
                                        ${isActive
                                            ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border-l-4 border-gray-900 dark:border-gray-100'
                                            : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white'}
                                        ${collapsed ? 'justify-center' : ''}
                                    `}
                                >
                                    <span>{item.icon}</span>
                                    {!collapsed && <span className="ml-3">{item.label}</span>}
                                </Link>
                            </li>
                        );
                    })}
                </ul>
            </nav>

            {/* Footer */}
            <div className="p-4 border-t border-gray-200 dark:border-gray-800">
                <div className={`flex ${collapsed ? 'flex-col items-center space-y-4' : 'items-center justify-between'}`}>
                    <ModeToggle />
                    {!collapsed ? (
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={handleSignOut}
                            className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800"
                        >
                            <LogOut className="h-4 w-4 mr-2" />
                            Sign Out
                        </Button>
                    ) : (
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={handleSignOut}
                            className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 mt-2"
                        >
                            <LogOut className="h-5 w-5" />
                        </Button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SidebarLayout;
