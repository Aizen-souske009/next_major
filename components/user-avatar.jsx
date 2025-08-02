'use client';

import Image from 'next/image';
import { useMemo } from 'react';

// domains you allow
const ALLOWED = ['avatars.githubusercontent.com', 'lh3.googleusercontent.com'];

function isAllowedSrc(src) {
    if (!src) return false;
    try {
        const url = new URL(src);
        return ALLOWED.includes(url.hostname);
    } catch {
        return false;
    }
}

export default function UserAvatar({ src, size = 40, alt = 'User Avatar', className = '' }) {
    const safe = useMemo(() => isAllowedSrc(src), [src]);

    const fallback = '/images/default-avatar.png'; // put this in /public/images/

    if (!safe) {
        return (
            <Image
                src={fallback}
                alt={alt}
                width={size}
                height={size}
                className={`rounded-full ${className}`}
            />
        );
    }

    return (
        <Image
            src={src}
            alt={alt}
            width={size}
            height={size}
            className={`rounded-full ${className}`}
        />
    );
}
