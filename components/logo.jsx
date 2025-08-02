"use client";

import { motion } from "framer-motion";

export default function Logo() {
    return (
        <motion.div
            className="p-2 rounded-full bg-white/20 backdrop-blur-md shadow-md" // <--- Background for visibility
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
        >
            <motion.svg
                className="h-10 w-10"
                viewBox="0 0 100 100"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
            >
                {/* Outer Glow Circle */}
                <motion.circle
                    cx="50"
                    cy="50"
                    r="45"
                    stroke="url(#gradient)"
                    strokeWidth="2.5"
                    initial={{ strokeDasharray: 280, strokeDashoffset: 280 }}
                    animate={{ strokeDashoffset: 0 }}
                    transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
                />

                {/* Inner Grid Dots */}
                <motion.circle cx="50" cy="30" r="2" fill="#f0f" />
                <motion.circle cx="70" cy="50" r="2" fill="#0ff" />
                <motion.circle cx="50" cy="70" r="2" fill="#f0f" />
                <motion.circle cx="30" cy="50" r="2" fill="#0ff" />

                {/* Gradient */}
                <defs>
                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#f0f" />
                        <stop offset="100%" stopColor="#0ff" />
                    </linearGradient>
                </defs>
            </motion.svg>
        </motion.div>
    );
}
