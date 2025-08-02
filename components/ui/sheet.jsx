"use client";

import * as React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

export function Sheet({ children, ...props }) {
    return <Dialog.Root {...props}>{children}</Dialog.Root>;
}

export function SheetTrigger({ children, ...props }) {
    return (
        <Dialog.Trigger asChild {...props}>
            {children}
        </Dialog.Trigger>
    );
}

export function SheetContent({ className, children, ...props }) {
    return (
        <Dialog.Portal>
            <Dialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
            <Dialog.Content
                {...props}
                className={cn(
                    "fixed top-0 right-0 h-full w-80 bg-white dark:bg-gray-900 shadow-lg z-50 p-6 animate-in slide-in-from-right duration-200",
                    className
                )}
            >
                <button
                    className="absolute top-4 right-4 rounded-full p-2 hover:bg-gray-200 dark:hover:bg-gray-700"
                    onClick={() => Dialog.Root.close?.()}
                >
                    <X className="h-5 w-5 text-gray-800 dark:text-gray-200" />
                </button>
                {children}
            </Dialog.Content>
        </Dialog.Portal>
    );
}

export function SheetHeader({ className, ...props }) {
    return <div className={cn("mb-4", className)} {...props} />;
}

export function SheetTitle({ className, ...props }) {
    return (
        <Dialog.Title
            className={cn(
                "text-lg font-semibold text-gray-900 dark:text-gray-100",
                className
            )}
            {...props}
        />
    );
}
