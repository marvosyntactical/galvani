import { useEffect, useRef, useState, type ReactNode } from "react";

interface Props {
  title: string;
  children: ReactNode;
  /** label inside the trigger button, defaults to "?" */
  label?: string;
}

/**
 * A small `?`-style button that opens a popover with help text.
 * Used next to controls that need inline explanation (upload format, etc.).
 */
export function HelpPopover({ title, children, label = "?" }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="help-popover" ref={ref}>
      <button
        type="button"
        className="help-trigger"
        aria-label={`Help: ${title}`}
        onClick={() => setOpen((o) => !o)}
      >
        {label}
      </button>
      {open && (
        <div className="help-panel" role="dialog" aria-label={title}>
          <div className="help-panel-header">
            <strong>{title}</strong>
            <button
              type="button"
              className="help-close"
              onClick={() => setOpen(false)}
              aria-label="Close"
            >
              ×
            </button>
          </div>
          <div className="help-panel-body">{children}</div>
        </div>
      )}
    </div>
  );
}
