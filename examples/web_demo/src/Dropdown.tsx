import { useEffect, useRef, useState } from "react";

/**
 * A custom dropdown that does NOT use the native <select> element.
 *
 * Why this exists: Firefox renders native <select> popups as XUL popups
 * that are extremely sensitive to overlapping WebGL canvases repainting,
 * pointer events on global listeners (like @react-three/drei's
 * OrbitControls), or any other reflow on the document. The result is
 * dropdowns that flicker open for a single frame and immediately dismiss
 * on Firefox -- even though they work fine in Chromium.
 *
 * This component renders the dropdown menu as a regular absolutely-
 * positioned div inside our React tree. No native browser popup machinery.
 * Works identically on Firefox, Chromium, Safari.
 *
 * API matches the subset of <select> we actually used elsewhere.
 */
export interface DropdownOption<T extends string | number = string> {
  value: T;
  label: string;
}

interface Props<T extends string | number> {
  id?: string;
  value: T;
  options: DropdownOption<T>[];
  onChange: (next: T) => void;
  disabled?: boolean;
}

export function Dropdown<T extends string | number>({
  id,
  value,
  options,
  onChange,
  disabled = false,
}: Props<T>) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  // Click outside / Escape -> close.
  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const current = options.find((o) => o.value === value);

  return (
    <div className="custom-dropdown" ref={rootRef} id={id}>
      <button
        type="button"
        className="cd-toggle"
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="cd-value">{current?.label ?? String(value)}</span>
        <span className="cd-caret">{open ? "▴" : "▾"}</span>
      </button>
      {open && (
        <ul className="cd-menu" role="listbox">
          {options.map((opt) => (
            <li
              key={String(opt.value)}
              role="option"
              aria-selected={opt.value === value}
              className={opt.value === value ? "selected" : ""}
              onClick={() => {
                onChange(opt.value);
                setOpen(false);
              }}
            >
              {opt.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
