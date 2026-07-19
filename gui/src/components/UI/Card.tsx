"use client";

import { motion, HTMLMotionProps } from "framer-motion";
import { clsx } from "clsx";
import { ReactNode, forwardRef } from "react";

interface CardProps extends Omit<HTMLMotionProps<"div">, "children"> {
  children: ReactNode;
  variant?: "default" | "elevated" | "outlined" | "glass" | "gradient";
  padding?: "none" | "sm" | "md" | "lg" | "xl";
  hover?: boolean;
  interactive?: boolean;
  glow?: boolean;
  border?: boolean;
  className?: string;
}

const paddingStyles: Record<string, string> = {
  none: "p-0",
  sm: "p-4",
  md: "p-6",
  lg: "p-8",
  xl: "p-10",
};

const variantStyles: Record<string, string> = {
  default: "bg-zinc-950/50 backdrop-blur-sm border border-zinc-800/50",
  elevated: "bg-zinc-950/80 backdrop-blur-md shadow-2xl shadow-black/50 border border-zinc-800/30",
  outlined: "bg-transparent border-2 border-zinc-700/50",
  glass: "bg-white/5 backdrop-blur-xl border border-white/10",
  gradient: "bg-gradient-to-br from-zinc-900/80 via-zinc-950 to-zinc-900/80 border border-zinc-800/30",
};

export const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      children,
      variant = "default",
      padding = "md",
      hover = false,
      interactive = false,
      glow = false,
      border = false,
      className,
      ...props
    },
    ref
  ) => {
    const { style } = props;
    return (
      <motion.div
        ref={ref}
        className={clsx(
          "relative rounded-2xl overflow-hidden",
          variantStyles[variant] ?? variantStyles.default,
          paddingStyles[padding] ?? paddingStyles.md,
          border && "border border-zinc-700/50",
          hover && "cursor-pointer",
          className
        )}
        style={{
          ...style,
        }}
        initial={{ opacity: 0, y: 20, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        whileHover={hover ? { y: -4, scale: 1.01, boxShadow: "0 20px 40px -12px rgba(0, 0, 0, 0.4)" } : undefined}
        whileTap={interactive ? { scale: 0.99 } : undefined}
        transition={{ duration: 0.3, ease: "easeOut" }}
        {...props}
      >
        {glow && (
          <motion.div
            className="absolute inset-0 -z-10 rounded-[inherit] opacity-0 pointer-events-none"
            animate={{ opacity: [0, 0.15, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            style={{
              background: "linear-gradient(135deg, #6366f1, #8b5cf6, #10b981, #f59e0b)",
              filter: "blur(60px)",
            }}
          />
        )}

        <div className="relative z-10">
          {children}
        </div>

        {interactive && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-primary-500/10 to-emerald-500/10 opacity-0 pointer-events-none"
            initial={{ scaleX: 0, opacity: 0 }}
            whileHover={{ scaleX: 1, opacity: 1 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          />
        )}
      </motion.div>
    );
  }
);

Card.displayName = "Card";

interface CardHeaderProps extends Omit<HTMLMotionProps<"div">, "children"> {
  children: ReactNode;
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  avatar?: ReactNode;
  className?: string;
}

export function CardHeader({ 
  children, 
  title, 
  subtitle, 
  action, 
  avatar, 
  className,
  ...props 
}: CardHeaderProps) {
  return (
    <motion.div
      className={clsx("flex items-start justify-between gap-4 mb-4", className)}
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      {...props}
    >
      <div className="flex items-start gap-3 flex-1 min-w-0">
        {avatar && (
          <motion.div
            className="flex-shrink-0"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
          >
            {avatar}
          </motion.div>
        )}
        <div className="flex-1 min-w-0">
          {title && (
            <motion.h3
              className="text-lg font-bold text-white truncate"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1, duration: 0.3 }}
            >
              {title}
            </motion.h3>
          )}
          {subtitle && (
            <motion.p
              className="text-sm text-zinc-400 mt-0.5 truncate"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2, duration: 0.3 }}
            >
              {subtitle}
            </motion.p>
          )}
          {children && (
            <motion.div
              className="mt-2"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              {children}
            </motion.div>
          )}
        </div>
      </div>
      {action && (
        <motion.div
          className="flex-shrink-0 ml-4"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          {action}
        </motion.div>
      )}
    </motion.div>
  );
}

interface CardContentProps extends Omit<HTMLMotionProps<"div">, "children"> {
  children: ReactNode;
  className?: string;
}

export function CardContent({ children, className, ...props }: CardContentProps) {
  return (
    <motion.div
      className={clsx("flex-1", className)}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      {...props}
    >
      {children}
    </motion.div>
  );
}

interface CardFooterProps extends Omit<HTMLMotionProps<"div">, "children"> {
  children: ReactNode;
  className?: string;
  divided?: boolean;
}

export function CardFooter({ children, className, divided = true, ...props }: CardFooterProps) {
  return (
    <motion.div
      className={clsx(
        "flex items-center justify-end gap-3 mt-6 pt-4",
        divided && "border-t border-zinc-800/50",
        className
      )}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2, ease: "easeOut" }}
      {...props}
    >
      {children}
    </motion.div>
  );
}

interface CardGridProps {
  children: ReactNode;
  columns?: 1 | 2 | 3 | 4;
  gap?: number;
  className?: string;
}

export function CardGrid({ children, columns = 3, gap = 6, className }: CardGridProps) {
  const columnClasses = {
    1: "grid-cols-1",
    2: "grid-cols-1 md:grid-cols-2",
    3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
  };

  return (
    <motion.div
      className={clsx(
        "grid",
        className,
        columnClasses[columns],
        `gap-${gap}`
      )}
      initial="initial"
      animate="animate"
      variants={{
        initial: { opacity: 0 },
        animate: {
          opacity: 1,
          transition: {
            staggerChildren: 0.08,
          },
        },
      }}
    >
      {typeof children === "function" ? (children as (props: {}) => React.ReactNode)({}) : children}
    </motion.div>
  );
}

interface StatsCardProps {
  label: string;
  value: string | number;
  change?: number;
  trend?: "up" | "down" | "neutral";
  icon?: ReactNode;
  color?: "primary" | "success" | "warning" | "danger" | "info";
  loading?: boolean;
  className?: string;
}

export function StatsCard({ 
  label, 
  value, 
  change, 
  trend = "neutral", 
  icon, 
  color = "primary",
  loading = false,
  className 
}: StatsCardProps) {
  const colorStyles = {
    primary: "text-primary-400 bg-primary-500/10 border-primary-500/20",
    success: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    warning: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    danger: "text-red-400 bg-red-500/10 border-red-500/20",
    info: "text-cyan-400 bg-cyan-500/10 border-cyan-500/20",
  };

  const trendColors = {
    up: "text-emerald-400",
    down: "text-red-400",
    neutral: "text-zinc-500",
  };

  return (
    <Card variant="glass" padding="md" hover className={className}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <motion.p
            className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1"
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {label}
          </motion.p>
          
          {loading ? (
            <motion.div
              className="h-10 w-3/4 bg-zinc-800/50 rounded animate-pulse"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            />
          ) : (
            <motion.div
              className="flex items-baseline gap-2 flex-wrap"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.4 }}
            >
              <span className="text-3xl font-bold text-white tabular-nums">{value}</span>
              {change !== undefined && change !== 0 && (
                <motion.span
                  className={clsx("flex items-center gap-1 text-sm font-semibold", trendColors[trend])}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2, type: "spring", stiffness: 300, damping: 20 }}
                >
                  {trend === "up" ? (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" /></svg>
                  ) : trend === "down" ? (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V5" /></svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" /></svg>
                  )}
                  <span className="font-semibold">{Math.abs(Number(change))}%</span>
                </motion.span>
              )}
            </motion.div>
          )}
        </div>
        
        {icon && (
          <motion.div
            className={clsx("flex-shrink-0 p-3 rounded-xl", colorStyles[color])}
            initial={{ opacity: 0, scale: 0.8, x: 20 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 300, damping: 20 }}
            whileHover={{ scale: 1.1, rotate: 5 }}
          >
            {icon}
          </motion.div>
        )}
      </div>
      
      {change !== undefined && (
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-1 overflow-hidden"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.5, duration: 0.8, ease: "easeOut" }}
        >
          <motion.div
            className="h-full"
            style={{
              background: trend === "up" 
                ? "linear-gradient(90deg, #10b981, #34d399)"
                : trend === "down"
                ? "linear-gradient(90deg, #ef4444, #f87171)"
                : "linear-gradient(90deg, #6b7280, #9ca3af)",
              transformOrigin: "left"
            }}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: Math.min(Math.abs(Number(change)) / 100, 1) }}
            transition={{ delay: 0.5, duration: 1, ease: "easeOut" }}
          />
        </motion.div>
      )}
    </Card>
  );
}