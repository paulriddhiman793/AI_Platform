"use client";

import { motion } from "framer-motion";
import { clsx } from "clsx";
import { forwardRef, ReactNode } from "react";

interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size" | "type"> {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  leftElement?: ReactNode;
  rightElement?: ReactNode;
  fullWidth?: boolean;
  variant?: "default" | "filled" | "outlined" | "ghost";
  size?: "xs" | "sm" | "md" | "lg";
  type?: string;
}

const sizeStyles: Record<string, string> = {
  xs: "px-3 py-1.5 text-xs gap-1.5",
  sm: "px-4 py-2 text-sm gap-2",
  md: "px-4 py-3 text-base gap-2",
  lg: "px-6 py-4 text-lg gap-2.5",
};

const variantStyles: Record<string, string> = {
  default: "bg-zinc-900/50 border-zinc-700 focus:border-primary-500 focus:ring-primary-500/20",
  filled: "bg-zinc-800/50 border-transparent focus:bg-zinc-800 focus:ring-primary-500/20",
  outlined: "bg-transparent border-2 border-zinc-700 focus:border-primary-500 focus:ring-primary-500/20",
  ghost: "bg-transparent border-transparent focus:bg-zinc-800/30 focus:ring-primary-500/20",
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      fullWidth = true,
      variant = "default",
      size = "md",
      className,
      style,
      id,
      ...props
    },
    ref
  ) => {
    const inputId = id || `input-${Math.random().toString(36).slice(2)}`;

    const disabled = props.disabled ?? false;
    const required = props.required ?? false;
    const error = props.error;
    const helperText = props.helperText;
    const showHelper = helperText || error;

    // Extract custom props to avoid spreading them to the input element
    const { leftIcon, rightIcon, leftElement, rightElement, ...inputProps } = props;

    return (
      <motion.div
        className={clsx("relative flex flex-col gap-1.5", fullWidth && "w-full")}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        {label && (
          <motion.label
            htmlFor={id}
            className={clsx(
              "flex items-center gap-1.5 text-sm font-medium text-zinc-300",
              "transition-colors duration-200",
              required && "after:content-['*'] after:text-red-500 after:ml-0.5"
            )}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 }}
          >
            {label}
          </motion.label>
        )}

        <motion.div
          className={clsx(
            "relative flex items-center",
            sizeStyles[size],
            variantStyles[variant],
            "rounded-xl transition-all duration-200",
            "focus-within:ring-2 focus-within:ring-primary-500/20",
            "hover:border-zinc-600",
            disabled && "opacity-50 cursor-not-allowed",
            error && "border-red-500/50 focus-within:ring-red-500/20",
            className
          )}
          style={{
            ...style,
          }}
        >
          {(leftIcon || leftElement) && (
            <motion.div
              className="absolute left-3 top-1/2 -translate-y-1/2 flex items-center justify-center text-zinc-500 pointer-events-none"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              {leftElement || leftIcon}
            </motion.div>
          )}

          <motion.input
            ref={ref}
            disabled={disabled}
            required={required}
            aria-invalid={error ? "true" : "false"}
            aria-describedby={clsx(error && `${inputId}-error`, helperText && `${inputId}-helper`)}
            className={clsx(
              "flex-1 bg-transparent outline-none placeholder:text-zinc-500/50",
              "text-white disabled:text-zinc-500",
              leftIcon && "pl-10",
              rightIcon && "pr-10",
              leftElement && "pl-10",
              rightElement && "pr-10",
              size === "xs" && "text-xs",
              size === "sm" && "text-sm",
              size === "md" && "text-base",
              size === "lg" && "text-lg"
            )}
            style={{ background: "transparent" }}
            {...(inputProps as any)}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            whileFocus={{ scale: 1.005 }}
          />

          {(rightIcon || rightElement) && (
            <motion.div
              className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center justify-center text-zinc-500 pointer-events-none"
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              {rightElement || rightIcon}
            </motion.div>
          )}

          {error && (
            <motion.div
              className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center justify-center text-red-500"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 0.5, repeat: Infinity }}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </motion.div>
          )}
        </motion.div>

        {showHelper && (
          <motion.p
            id={error ? `${inputId}-error` : `${inputId}-helper`}
            className={clsx(
              "text-sm transition-colors duration-200",
              error ? "text-red-500" : "text-zinc-500"
            )}
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: 0.3 }}
            role={error ? "alert" : undefined}
          >
            {error || helperText}
          </motion.p>
        )}
      </motion.div>
    );
  }
);

Input.displayName = "Input";

interface TextareaProps extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "size"> {
  label?: string;
  error?: string;
  helperText?: string;
  fullWidth?: boolean;
  variant?: "default" | "filled" | "outlined" | "ghost";
  size?: "xs" | "sm" | "md" | "lg";
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      label,
      error,
      helperText,
      fullWidth = true,
      variant = "default",
      size = "md",
      rows = 4,
      disabled = false,
      required = false,
      className,
      style,
      id,
      ...props
    },
    ref
  ) => {
    const inputId = id || `textarea-${Math.random().toString(36).slice(2)}`;

    const isError = Boolean(error);
    const showHelper = helperText || error;

    return (
      <motion.div
        className={clsx("flex flex-col gap-1.5", fullWidth && "w-full")}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        {label && (
          <motion.label
            htmlFor={inputId}
            className={clsx(
              "flex items-center gap-1.5 text-sm font-medium text-zinc-300",
              "transition-colors duration-200",
              required && "after:content-['*'] after:text-red-500 after:ml-0.5"
            )}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 }}
          >
            {label}
          </motion.label>
        )}

        <motion.div
          className={clsx(
            "relative",
            variantStyles[variant],
            "rounded-xl transition-all duration-200",
            "focus-within:ring-2 focus-within:ring-primary-500/20",
            "hover:border-zinc-600",
            disabled && "opacity-50 cursor-not-allowed",
            isError && "border-red-500/50 focus-within:ring-red-500/20",
            className
          )}
          style={{
            ...style,
          }}
        >
          <motion.textarea
            ref={ref}
            disabled={disabled}
            required={required}
            aria-invalid={isError}
            aria-describedby={clsx(isError && `${id}-error`, helperText && `${id}-helper`)}
            className={clsx(
              "w-full bg-transparent resize-y outline-none placeholder:text-zinc-500/50",
              "text-white disabled:text-zinc-500",
              "px-4 py-3",
              size === "xs" && "text-xs",
              size === "sm" && "text-sm",
              size === "md" && "text-base",
              size === "lg" && "text-lg"
            )}
            {...(props as any)}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            whileFocus={{ scale: 1.002 }}
          />

          {error && (
            <motion.div
              className="absolute right-3 bottom-3 flex items-center justify-center text-red-500"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 0.5, repeat: Infinity }}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </motion.div>
          )}
        </motion.div>

        {showHelper && (
          <motion.p
            id={error ? `${id}-error` : `${id}-helper`}
            className={clsx(
              "text-sm transition-colors duration-200",
              error ? "text-red-500" : "text-zinc-500"
            )}
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: 0.3 }}
            role={error ? "alert" : undefined}
          >
            {error || helperText}
          </motion.p>
        )}
      </motion.div>
    );
  }
);

Textarea.displayName = "Textarea";

interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "size"> {
  label?: string;
  error?: string;
  helperText?: string;
  options: Array<{ value: string; label: string; disabled?: boolean }>;
  placeholder?: string;
  fullWidth?: boolean;
  variant?: "default" | "filled" | "outlined" | "ghost";
  size?: "xs" | "sm" | "md" | "lg";
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      label,
      error,
      helperText,
      options,
      placeholder,
      fullWidth = true,
      variant = "default",
      size = "md",
      disabled = false,
      required = false,
      className,
      style,
      id,
      ...props
    },
    ref
  ) => {
    const inputId = id || `select-${Math.random().toString(36).slice(2)}`;
    const errorId = `${inputId}-error`;
    const helperId = `${inputId}-helper`;

    const isError = Boolean(error);
    const showHelper = helperText || error;

    return (
      <motion.div
        className={clsx("flex flex-col gap-1.5", fullWidth && "w-full")}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        {label && (
          <motion.label
            htmlFor={inputId}
            className={clsx(
              "flex items-center gap-1.5 text-sm font-medium text-zinc-300",
              "transition-colors duration-200",
              required && "after:content-['*'] after:text-red-500 after:ml-0.5"
            )}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 }}
          >
            {label}
          </motion.label>
        )}

        <motion.div
          className={clsx(
            "relative",
            sizeStyles[size],
            variantStyles[variant],
            "rounded-xl transition-all duration-200",
            "focus-within:ring-2 focus-within:ring-primary-500/20",
            "hover:border-zinc-600",
            disabled && "opacity-50 cursor-not-allowed",
            isError && "border-red-500/50 focus-within:ring-red-500/20",
            className
          )}
          style={{
            ...style,
          }}
        >
          <motion.select
            ref={ref}
            id={inputId}
            disabled={disabled}
            required={required}
            aria-invalid={isError}
            aria-describedby={clsx(isError && errorId, helperText && helperId)}
            className={clsx(
              "w-full bg-transparent outline-none appearance-none",
              "text-white disabled:text-zinc-500",
              "pr-10",
              size === "xs" && "text-xs",
              size === "sm" && "text-sm",
              size === "md" && "text-base",
              size === "lg" && "text-lg"
            )}
            {...(props as any)}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            whileFocus={{ scale: 1.002 }}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {options.map((option) => (
              <option key={option.value} value={option.value} disabled={option.disabled}>
                {option.label}
              </option>
            ))}
          </motion.select>

          <motion.div
            className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
            initial={{ opacity: 0, rotate: -90 }}
            animate={{ opacity: 1, rotate: 0 }}
          >
            <svg className="w-5 h-5 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </motion.div>

          {isError && (
            <motion.div
              className="absolute right-3 bottom-3 text-red-500"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 0.5, repeat: Infinity }}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </motion.div>
          )}
        </motion.div>

        {showHelper && (
          <motion.p
            id={error ? errorId : helperId}
            className={clsx(
              "text-sm transition-colors duration-200",
              error ? "text-red-500" : "text-zinc-500"
            )}
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: 0.3 }}
            role={error ? "alert" : undefined}
          >
            {error || helperText}
          </motion.p>
        )}
      </motion.div>
    );
  }
);

Select.displayName = "Select";