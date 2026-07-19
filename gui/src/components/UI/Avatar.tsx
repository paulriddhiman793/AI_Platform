"use client";

import { motion } from "framer-motion";
import { clsx } from "clsx";
import { forwardRef, ReactNode } from "react";

interface AvatarProps {
  src?: string;
  alt?: string;
  name?: string;
  size?: "xs" | "sm" | "md" | "lg" | "xl" | "2xl";
  shape?: "circle" | "rounded" | "square";
  status?: "online" | "busy" | "away" | "offline";
  border?: boolean;
  glow?: boolean;
  fallback?: ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

const sizeMap: Record<string, number> = {
  xs: 24,
  sm: 32,
  md: 40,
  lg: 48,
  xl: 56,
  "2xl": 72,
};

const getInitials = (name: string) => {
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);
};

const shapeClasses: Record<string, string> = {
  circle: "rounded-full",
  rounded: "rounded-xl",
  square: "rounded-lg",
};

const statusColors: Record<string, string> = {
  online: "#10b981",
  busy: "#f59e0b",
  away: "#f59e0b",
  offline: "#6b7280",
};

export const Avatar = forwardRef<HTMLDivElement, AvatarProps>(
  (
    {
      src,
      alt,
      name,
      size = "md",
      shape = "circle",
      status,
      border = true,
      glow = false,
      fallback,
      className,
      style,
      ...props
    },
    ref
  ) => {
    const sizePx = sizeMap[size] ?? sizeMap.md;
    const fontSize = sizePx * 0.35;

    const shapeClass = shapeClasses[shape] ?? shapeClasses.circle;

    const gradientBorder = (
      <motion.div
        className="absolute -inset-0.5"
        style={{
          borderRadius: shape === "circle" ? "50%" : shape === "rounded" ? "1rem" : "0.5rem",
          background: "linear-gradient(135deg, #6366f1, #8b5cf6, #10b981, #f59e0b)",
          zIndex: -1,
          opacity: 0.6,
        }}
        animate={{ rotate: [0, 360] }}
        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
      />
    );

    return (
      <motion.div
        ref={ref}
        className={clsx(
          "relative inline-flex items-center justify-center overflow-hidden",
          shapeClass,
          className
        )}
        style={{
          width: sizePx,
          height: sizePx,
          fontSize,
          ...style,
        }}
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        {...props}
      >
        {glow && gradientBorder}
        {src ? (
          <motion.img
            src={src}
            alt={alt ?? name ?? "Avatar"}
            className="w-full h-full object-cover"
            style={{ borderRadius: "inherit" }}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          />
        ) : name ? (
<motion.div
            className="w-full h-full flex items-center justify-center font-semibold select-none"
            style={{
              background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #10b981 100%)",
              backgroundSize: "200% 200%",
              color: "white",
              fontWeight: 600,
            }}
            initial={{ opacity: 0, rotate: -90 }}
            animate={{ 
              opacity: 1, 
              rotate: 0, 
              backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] 
            }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          >
            {getInitials(name)}
          </motion.div>
        ) : fallback ? (
          <motion.div
            className="w-full h-full flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {fallback}
          </motion.div>
        ) : (
          <motion.div
            className="w-full h-full flex items-center justify-center bg-gradient-to-br from-zinc-700 to-zinc-800"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.svg
              className="w-1/2 h-1/2 text-zinc-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
              />
            </motion.svg>
          </motion.div>
        )}
        
        {border && (
          <motion.div
            className="absolute inset-0"
            style={{
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "inherit",
            }}
          />
        )}

        {status && (
          <motion.div
            className="absolute bottom-0 right-0 flex items-center justify-center"
            style={{
              width: sizePx * 0.22,
              height: sizePx * 0.22,
              background: statusColors[status] ?? statusColors.offline,
              border: `2px solid #09090b`,
              borderRadius: "50%",
              boxShadow: `0 0 0 2px #09090b, 0 0 12px ${statusColors[status] ?? statusColors.offline}`,
            }}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            {(status === "online" || status === "busy") && (
              <motion.div
                className="absolute inset-0 rounded-full"
                style={{ background: statusColors[status] ?? statusColors.offline, opacity: 0.4 }}
                animate={{ scale: [1, 2], opacity: [0.6, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
              />
            )}
          </motion.div>
        )}
      </motion.div>
    );
  }
);

interface AvatarGroupProps {
  avatars: Array<{ src?: string; name?: string; alt?: string; status?: "online" | "busy" | "away" | "offline" }>;
  size?: "xs" | "sm" | "md" | "lg" | "xl";
  maxVisible?: number;
  overlap?: number;
  className?: string;
}

export function AvatarGroup({ 
  avatars, 
  size = "md", 
  maxVisible = 5, 
  overlap = 8,
  className 
}: AvatarGroupProps) {
  const visibleAvatars = avatars.slice(0, maxVisible);
  const remaining = avatars.length - maxVisible;

  return (
    <div className={clsx("flex items-center", className)}>
      {visibleAvatars.map((avatar, index) => (
        <motion.div
          key={avatar.name ?? avatar.src ?? index}
          style={{ 
            zIndex: maxVisible - index,
            marginLeft: index > 0 ? -overlap : 0,
          }}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.05, type: "spring", stiffness: 300, damping: 20 }}
        >
          <Avatar
            src={avatar.src}
            name={avatar.name}
            alt={avatar.alt}
            status={avatar.status}
            size={size}
            border
            glow={index === 0}
          />
        </motion.div>
      ))}
      {remaining > 0 && (
        <motion.div
          style={{ 
            zIndex: 0, 
            marginLeft: -overlap,
            background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
          }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: maxVisible * 0.05, type: "spring" }}
        >
          <Avatar
            name={`+${remaining}`}
            size={size}
            border
            className="font-semibold"
          />
        </motion.div>
      )}
    </div>
  );
}