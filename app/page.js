"use client";

import { useRef } from "react";
import Link from "next/link";
import Navbar from "@/components/navbar";
import { Button } from "@/components/ui/button";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(ScrollTrigger);

export default function Home() {
  const heroRef = useRef(null);
  const featuresRef = useRef(null);
  const glitchLines = useRef([]);
  const scanGrid = useRef(null);
  const circle1 = useRef(null);
  const circle2 = useRef(null);
  const floatingDots = useRef([]);

  useGSAP(() => {
    // Hero fade-in
    gsap.fromTo(
      "[data-gt='hero']",
      { y: 40, opacity: 0 },
      { y: 0, opacity: 1, stagger: 0.15, duration: 1, ease: "power3.out" }
    );

    // Features reveal
    gsap.fromTo(
      "[data-gt='feature']",
      { y: 20, opacity: 0 },
      {
        y: 0,
        opacity: 1,
        stagger: 0.08,
        duration: 0.6,
        ease: "power2.out",
        scrollTrigger: { trigger: featuresRef.current, start: "top 80%" },
      }
    );

    // Glitch lines animation
    glitchLines.current.forEach((line, i) => {
      gsap.to(line, {
        x: "100%",
        duration: 3 + i * 2,
        repeat: -1,
        ease: "linear",
      });
    });

    // Scan grid pulse
    gsap.to(scanGrid.current, {
      opacity: 0.25,
      scale: 1.05,
      duration: 3,
      repeat: -1,
      yoyo: true,
      ease: "sine.inOut",
    });

    // Inner Circles Floating Motion
    gsap.to(circle1.current, {
      rotation: 360,
      scale: 1.1,
      duration: 8,
      repeat: -1,
      yoyo: true,
      ease: "sine.inOut",
    });

    gsap.to(circle2.current, {
      rotation: -360,
      scale: 1.05,
      duration: 12,
      repeat: -1,
      yoyo: true,
      ease: "sine.inOut",
    });

    // Floating Dots Animation
    floatingDots.current.forEach((dot, i) => {
      gsap.to(dot, {
        x: gsap.utils.random(-100, 100),
        y: gsap.utils.random(-100, 100),
        duration: gsap.utils.random(4, 8),
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
        delay: i * 0.2,
      });
    });

    // Heading shimmer
    gsap.to(".shimmer", {
      backgroundPosition: "200% center",
      duration: 3,
      repeat: -1,
      ease: "linear",
    });
  });

  const features = [
    { title: "AI Manipulation Detection", desc: "Classify images as real, AI‑generated, or morphed using deep models." },
    { title: "Pixel‑Level Forensics", desc: "Heatmaps show where edits, inpaints, or splices likely occurred." },
    { title: "Scan History & Reports", desc: "Every upload is stored; review past scans with confidence scores." },
  ];

  return (
    <main className="relative flex flex-col min-h-screen bg-black text-white overflow-hidden">
      <Navbar />

      {/* Background Animations */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Glitch Lines */}
        {[...Array(4)].map((_, i) => (
          <div
            key={i}
            ref={(el) => (glitchLines.current[i] = el)}
            className={`absolute top-[${i * 25}%] w-full h-[2px] bg-gradient-to-r from-fuchsia-600/40 to-cyan-600/40`}
          />
        ))}

        {/* Pixel Grid */}
        <div
          ref={scanGrid}
          className="absolute inset-0 bg-[radial-gradient(circle,rgba(255,255,255,0.05)_1px,transparent_1px)] [background-size:20px_20px]"
        />

        {/* Inner Circles */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
          <div
            ref={circle1}
            className="absolute w-[220px] h-[220px] border border-fuchsia-400/40 rounded-full"
          ></div>
          <div
            ref={circle2}
            className="absolute w-[160px] h-[160px] border border-cyan-400/40 rounded-full"
          ></div>
        </div>

        {/* Floating Dots */}
        {[...Array(25)].map((_, i) => (
          <div
            key={i}
            ref={(el) => (floatingDots.current[i] = el)}
            className="absolute w-2 h-2 bg-cyan-400 rounded-full opacity-60 blur-[1px]"
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
            }}
          />
        ))}
      </div>

      {/* HERO */}
      <section
        ref={heroRef}
        className="relative flex flex-col items-center justify-center text-center px-6 pt-32 pb-28 flex-1"
      >
        <h1
          data-gt="hero"
          className="relative z-10 max-w-4xl text-5xl md:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-fuchsia-400 via-cyan-400 to-fuchsia-300 shimmer bg-[length:200%_100%]"
        >
          DeepTrace AI
        </h1>
        <p data-gt="hero" className="relative z-10 mt-6 max-w-2xl text-lg md:text-xl text-gray-300">
          Tracing the truth in every pixel.
        </p>
        <p data-gt="hero" className="relative z-10 mt-4 max-w-xl text-base md:text-lg text-gray-400">
          Upload an image and let our models detect AI generation, manipulation, or authenticity in seconds.
        </p>
        <div data-gt="hero" className="relative z-10 mt-10">
          <Button asChild className="px-8 py-4 text-lg bg-gradient-to-r from-fuchsia-600 to-cyan-600 hover:from-fuchsia-700 hover:to-cyan-700">
            <Link href="/login">Get Started</Link>
          </Button>
        </div>
      </section>

      {/* FEATURES */}
      <section ref={featuresRef} className="py-24 px-6 bg-gradient-to-b from-transparent to-fuchsia-50/5">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-10 text-center">
          {features.map((f, i) => (
            <div
              key={i}
              data-gt="feature"
              className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 hover:scale-105 transition-transform shadow-lg"
            >
              <h3 className="text-2xl font-semibold text-fuchsia-300 mb-3">{f.title}</h3>
              <p className="text-gray-300">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 text-center bg-gradient-to-r from-fuchsia-600 to-cyan-600">
        <h2 className="text-3xl font-bold mb-4">Ready to analyze your images?</h2>
        <Button asChild variant="outline" className="bg-white text-fuchsia-700 hover:bg-gray-100">
          <Link href="/login">Sign In Now</Link>
        </Button>
      </section>
    </main>
  );
}
