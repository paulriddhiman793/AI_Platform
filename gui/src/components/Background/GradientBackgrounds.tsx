"use client";

import { motion } from "framer-motion";

interface GradientOrbProps {
  size?: number;
  color?: string;
  blur?: number;
  x?: number;
  y?: number;
  delay?: number;
  className?: string;
}

export function GradientOrb({ 
  size = 400, 
  color = "rgba(99, 102, 241, 0.4)",
  blur = 80,
  x = 50,
  y = 50,
  delay = 0,
  className = ""
}: GradientOrbProps) {
  return (
    <motion.div
      className={`fixed rounded-full pointer-events-none ${className}`}
      style={{
        width: size,
        height: size,
        left: `${x}%`,
        top: `${y}%`,
        transform: "translate(-50%, -50%)",
        background: `radial-gradient(circle at center, ${color} 0%, transparent 70%)`,
        filter: `blur(${blur}px)`,
        pointerEvents: "none",
        zIndex: -1,
      }}
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ 
        opacity: 1, 
        scale: 1,
        x: [0, 20, -20, 0],
        y: [0, -20, 20, 0],
      }}
      transition={{
        duration: 20,
        repeat: Infinity,
        ease: "easeInOut",
        delay,
      }}
    />
  );
}

interface AuroraProps {
  className?: string;
  colors?: string[];
}

export function Aurora({ className = "", colors = ["#6366f1", "#10b981", "#f59e0b"] }: AuroraProps) {
  return (
    <div className={`fixed inset-0 -z-10 overflow-hidden ${className}`}>
      {colors.map((color, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full opacity-20"
          style={{
            width: "600px",
            height: "600px",
            background: `radial-gradient(circle at center, ${color} 0%, transparent 70%)`,
            filter: "blur(100px)",
            top: `${20 + i * 25}%`,
            left: `${10 + i * 30}%`,
            transform: "translate(-50%, -50%)",
          }}
          animate={{
            x: [0, 100, -100, 0],
            y: [0, -100, 100, 0],
            scale: [1, 1.2, 0.8, 1],
            opacity: [0.15, 0.25, 0.15, 0.15],
          }}
          transition={{
            duration: 15 + i * 3,
            repeat: Infinity,
            ease: "easeInOut",
            delay: i * 2,
          }}
        />
      ))}
    </div>
  );
}

interface MeshGradientProps {
  className?: string;
  animate?: boolean;
}

export function MeshGradient({ className = "", animate = true }: MeshGradientProps) {
  return (
    <motion.div
      className={`fixed inset-0 -z-10 ${className}`}
      style={{
        background: `
          radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
          radial-gradient(ellipse at 80% 80%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
          radial-gradient(ellipse at 80% 20%, rgba(245, 158, 11, 0.1) 0%, transparent 50%),
          radial-gradient(ellipse at 20% 80%, rgba(239, 68, 68, 0.08) 0%, transparent 50%)
        `
      }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
    >
      {animate && (
        <>
          <motion.div
            className="absolute inset-0"
            style={{
              background: `
                radial-gradient(ellipse at 30% 30%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 70%, rgba(16, 185, 129, 0.06) 0%, transparent 60%)
              `
            }}
            animate={{
              scale: [1, 1.1, 1],
              rotate: [0, 2, -2, 0],
            }}
            transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div
            className="absolute inset-0"
            style={{
              background: `
                radial-gradient(ellipse at 70% 30%, rgba(245, 158, 11, 0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 30% 70%, rgba(239, 68, 68, 0.05) 0%, transparent 60%)
              `
            }}
            animate={{
              scale: [1, 0.95, 1],
              rotate: [0, -1, 1, 0],
            }}
            transition={{ duration: 25, repeat: Infinity, ease: "easeInOut", delay: 2 }}
          />
        </>
      )}
    </motion.div>
  );
}

export function NoiseOverlay({ className = "", opacity = 0.03 }: { className?: string; opacity?: number }) {
  return (
    <div
      className={`fixed inset-0 -z-10 ${className}`}
      style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        opacity,
        pointerEvents: "none",
      }}
    />
  );
}

export function GridPattern({ 
  className = "", 
  size = 60, 
  color = "rgba(99, 102, 241, 0.03)",
  animate = false 
}: { className?: string; size?: number; color?: string; animate?: boolean }) {
  return (
    <motion.div
      className={`fixed inset-0 -z-10 ${className}`}
      style={{
        backgroundImage: `
          linear-gradient(${color} 1px, transparent 1px),
          linear-gradient(90deg, ${color} 1px, transparent 1px)
        `,
        backgroundSize: `${size}px ${size}px`,
      }}
      animate={animate ? { 
        backgroundPosition: ["0 0", `${size}px ${size}px`, "0 0"] 
      } : {}}
      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
    />
  );
}

export function Vignette({ className = "", intensity = 0.5 }: { className?: string; intensity?: number }) {
  return (
    <div
      className={`fixed inset-0 -z-10 pointer-events-none ${className}`}
      style={{
        boxShadow: `inset 0 0 ${800 * intensity}px ${400 * intensity}px rgba(0, 0, 0, ${intensity})`,
        pointerEvents: "none",
      }}
    />
  );
}