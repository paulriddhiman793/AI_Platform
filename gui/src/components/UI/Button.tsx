"use client";

import { motion } from "framer-motion";
import { clsx } from "clsx";
import { forwardRef, ReactNode } from "react";

interface ButtonProps {
  children: ReactNode;
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger" | "success";
  size?: "xs" | "sm" | "md" | "lg" | "xl";
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  fullWidth?: boolean;
  glow?: boolean;
  ripple?: boolean;
  disabled?: boolean;
  className?: string;
  style?: React.CSSProperties;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
}

const variantStyles: Record<string, string> = {
  primary: "bg-gradient-to-r from-primary-600 to-primary-500 text-white hover:from-primary-500 hover:to-primary-400 shadow-lg shadow-primary-500/25",
  secondary: "bg-gradient-to-r from-zinc-800 to-zinc-900 text-white hover:from-zinc-700 hover:to-zinc-800 border border-zinc-700/50",
  outline: "bg-transparent text-primary-400 border-2 border-primary-500/50 hover:bg-primary-500/10 hover:border-primary-400 hover:text-white",
  ghost: "bg-transparent text-zinc-300 hover:bg-zinc-800/50 hover:text-white",
  danger: "bg-gradient-to-r from-red-600 to-red-500 text-white hover:from-red-500 hover:to-red-400 shadow-lg shadow-red-500/25",
  success: "bg-gradient-to-r from-emerald-600 to-emerald-500 text-white hover:from-emerald-500 hover:to-emerald-400 shadow-lg shadow-emerald-500/25",
};

const sizeStyles: Record<string, string> = {
  xs: "px-3 py-1.5 text-xs gap-1.5",
  sm: "px-4 py-2 text-sm gap-2",
  md: "px-6 py-3 text-base gap-2.5",
  lg: "px-8 py-4 text-lg gap-3",
  xl: "px-10 py-5 text-xl gap-3",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = "primary",
      size = "md",
      loading = false,
      leftIcon,
      rightIcon,
      fullWidth = false,
      glow = false,
      ripple = true,
      disabled,
      className,
      style,
      onClick,
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || loading;

    return (
      <motion.button
        ref={ref}
        className={clsx(
          "relative inline-flex items-center justify-center font-semibold rounded-xl",
          "transition-all duration-200 ease-out",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/50 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-950",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          variantStyles[variant],
          sizeStyles[size],
          fullWidth && "w-full",
          className
        )}
        style={{
          position: "relative",
          overflow: "hidden",
          ...style,
        }}
        disabled={isDisabled}
        onClick={onClick}
        {...props}
      >
        <span className="relative flex items-center justify-center gap-2.5">
          {!loading && leftIcon && (
            <motion.span
              className="flex-shrink-0"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
            >
              {leftIcon}
            </motion.span>
          )}

          <motion.span
            className="relative z-10"
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: 0.1 }}
          >
            {children}
          </motion.span>

          {!loading && rightIcon && (
            <motion.span
              className="flex-shrink-0"
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
            >
              {rightIcon}
            </motion.span>
          )}

          {loading && (
            <motion.div
              className="flex items-center justify-center"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
            >
              <motion.svg
                className="w-5 h-5 text-current"
                fill="none"
                viewBox="0 0 24 24"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <motion.circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <motion.path
                  className="opacity-75"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="4"
                  strokeLinecap="round"
                  d="M12 2a10 10 0 0 1 10 10"
                  animate={{ 
                    strokeDashoffset: [0, 50, 0],
                    strokeDasharray: ["31.4 31.4", "15.7 31.4", "31.4 31.4"],
                  }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                />
              </motion.svg>
              <motion.span
                className="sr-only"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                Loading...
              </motion.span>
            </motion.div>
          )}
        </span>
      </motion.button>
    );
  }
);

Button.displayName = "Button";

interface IconButtonProps extends Omit<ButtonProps, "children" | "size"> {
  icon: ReactNode;
  "aria-label": string;
  size?: "xs" | "sm" | "md" | "lg";
}

export function IconButton({ 
  icon, 
  "aria-label": ariaLabel, 
  size = "md",
  variant = "ghost",
  className,
  ...props
}: IconButtonProps) {
  const sizeMap = {
    xs: "p-1.5",
    sm: "p-2",
    md: "p-2.5",
    lg: "p-3.5",
  };

  return (
    <Button
      variant={variant}
      size="md"
      className={clsx(sizeMap[size], "aspect-square", className)}
      aria-label={ariaLabel}
      {...props}
    >
      <span className="flex items-center justify-center">
        {icon}
      </span>
    </Button>
  );
}

interface ToggleButtonProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  onIcon?: ReactNode;
  offIcon?: ReactNode;
  onLabel?: string;
  offLabel?: string;
  variant?: "outline" | "ghost";
  size?: "xs" | "sm" | "md" | "lg";
  className?: string;
  disabled?: boolean;
}

export function ToggleButton({ 
  checked, 
  onChange, 
  onIcon, 
  offIcon, 
  onLabel = "On", 
  offLabel = "Off",
  variant = "outline",
  size = "md",
  className,
  disabled,
}: ToggleButtonProps) {
  return (
    <Button
      variant={checked ? "primary" : variant}
      size={size}
      onClick={() => onChange(!checked)}
      aria-pressed={checked}
      disabled={disabled}
      className={className}
    >
      <motion.span
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        key={checked ? "on" : "off"}
      >
        {checked ? (onIcon || onLabel) : (offIcon || offLabel)}
      </motion.span>
    </Button>
  );
}