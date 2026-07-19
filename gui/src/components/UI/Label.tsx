"use client";

import { clsx } from "clsx";
import { LabelHTMLAttributes, forwardRef } from "react";

export const Label = forwardRef<HTMLLabelElement, LabelHTMLAttributes<HTMLLabelElement>>(
  ({ children, className, ...props }, ref) => {
    return (
      <label
        ref={ref}
        className={clsx(
          "flex items-center gap-1.5 text-sm font-medium text-zinc-300",
          "transition-colors duration-200",
          className
        )}
        {...props}
      >
        {children}
      </label>
    );
  }
);

Label.displayName = "Label";