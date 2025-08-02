"use client";

import React from "react";
import Link from "next/link";
import { Menu, Activity } from "lucide-react";
import { useSession, signOut } from "next-auth/react";

import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

import { Button } from "@/components/ui/button";
import { ModeToggle } from "@/components/dark-mode";
import Logo from "@/components/Logo"; // <-- New Logo Component

const BRAND = "DeepTrace AI";

const navItems = [{ label: "Home", href: "/" }];

export default function Navbar() {
  const { data: session, status } = useSession();

  const handleSignOut = () => {
    signOut({ callbackUrl: "/" });
  };

  return (
    <nav className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-purple-200/40 dark:border-purple-900/40 fixed w-full top-0 left-0 right-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0 flex items-center space-x-2">
            <Logo /> {/* Custom animated logo */}
            <Link
              href="/"
              className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-fuchsia-400 via-cyan-400 to-fuchsia-300"
            >
              {BRAND}
            </Link>
          </div>

          {/* Desktop Nav */}
          <div className="hidden md:flex space-x-4 items-center">
            {navItems.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className="text-gray-600 hover:text-fuchsia-600 dark:text-gray-300 dark:hover:text-fuchsia-400 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 hover:bg-fuchsia-50 dark:hover:bg-fuchsia-900/30"
              >
                {item.label}
              </Link>
            ))}

            {status === "authenticated" ? (
              <>
                <Button variant="outline" asChild>
                  <Link href="/dashboard">Dashboard</Link>
                </Button>
                <Button
                  className="bg-gradient-to-r from-fuchsia-600 to-cyan-600 text-white hover:from-fuchsia-700 hover:to-cyan-700"
                  onClick={handleSignOut}
                >
                  Sign Out
                </Button>
              </>
            ) : (
              <Button
                className="bg-gradient-to-r from-fuchsia-600 to-cyan-600 text-white hover:from-fuchsia-700 hover:to-cyan-700"
                asChild
              >
                <Link href="/login">Sign In</Link>
              </Button>
            )}

            <ModeToggle />
          </div>

          {/* Mobile Nav */}
          <div className="md:hidden flex items-center space-x-3">
            <ModeToggle />
            <Sheet>
              <SheetTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Open menu"
                  className="text-fuchsia-600 dark:text-fuchsia-400 hover:bg-fuchsia-50 dark:hover:bg-fuchsia-900/30"
                >
                  <Menu className="h-6 w-6" />
                </Button>
              </SheetTrigger>
              <SheetContent
                side="right"
                className="border-l border-fuchsia-200/40 dark:border-fuchsia-900/40"
              >
                <SheetHeader>
                  <SheetTitle className="text-fuchsia-600 dark:text-fuchsia-400 flex items-center">
                    <Activity className="mr-2 h-5 w-5" />
                    Menu
                  </SheetTitle>
                </SheetHeader>
                <div className="flex flex-col space-y-4 mt-6">
                  {navItems.map((item) => (
                    <Link
                      key={item.label}
                      href={item.href}
                      className="text-gray-600 hover:text-fuchsia-600 dark:text-gray-300 dark:hover:text-fuchsia-400 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 hover:bg-fuchsia-50 dark:hover:bg-fuchsia-900/30"
                    >
                      {item.label}
                    </Link>
                  ))}

                  {status === "authenticated" ? (
                    <div className="pt-4 space-y-3">
                      <Button variant="outline" className="w-full" asChild>
                        <Link href="/dashboard">Dashboard</Link>
                      </Button>
                      <Button
                        className="w-full bg-gradient-to-r from-fuchsia-600 to-cyan-600 text-white hover:from-fuchsia-700 hover:to-cyan-700"
                        onClick={handleSignOut}
                      >
                        Sign Out
                      </Button>
                    </div>
                  ) : (
                    <div className="pt-4">
                      <Button
                        className="w-full bg-gradient-to-r from-fuchsia-600 to-cyan-600 text-white hover:from-fuchsia-700 hover:to-cyan-700"
                        asChild
                      >
                        <Link href="/login">Sign In</Link>
                      </Button>
                    </div>
                  )}
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>
    </nav>
  );
}
