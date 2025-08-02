// components/icons/custom-icons.js
export const NeuralGridIcon = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <circle cx="6" cy="6" r="2" />
        <circle cx="18" cy="6" r="2" />
        <circle cx="6" cy="18" r="2" />
        <circle cx="18" cy="18" r="2" />
        <line x1="6" y1="8" x2="6" y2="16" strokeWidth="1.5" />
        <line x1="18" y1="8" x2="18" y2="16" strokeWidth="1.5" />
        <line x1="8" y1="6" x2="16" y2="6" strokeWidth="1.5" />
        <line x1="8" y1="18" x2="16" y2="18" strokeWidth="1.5" />
    </svg>
);

export const BrainScanIcon = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path d="M12 2c4 0 6 3 6 7s-2 7-6 7-6-3-6-7 2-7 6-7z" strokeWidth="1.5" />
        <path d="M8 10h8M12 7v6" strokeWidth="1.5" />
    </svg>
);

export const AnalyticsIcon = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path d="M3 3v18h18" strokeWidth="1.5" />
        <rect x="6" y="12" width="3" height="6" />
        <rect x="11" y="9" width="3" height="9" />
        <rect x="16" y="6" width="3" height="12" />
    </svg>
);

export const UserShieldIcon = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <circle cx="12" cy="8" r="4" strokeWidth="1.5" />
        <path d="M4 20c0-4 4-7 8-7s8 3 8 7" strokeWidth="1.5" />
        <path d="M18 8v4l3 2 3-2V8l-3-2z" strokeWidth="1.5" />
    </svg>
);
