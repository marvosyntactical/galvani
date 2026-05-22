import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

interface Props {
  title: string;
  children: ReactNode;
  label?: string;
}

/**
 * `?` button that opens a help popover. The panel is rendered into a
 * portal at document.body and pinned to the viewport — that way no
 * ancestor stacking context (canvas-host, sidebar, drei's Stats panel)
 * can ever clip it.
 */
export function HelpPopover({ title, children, label = "?" }: Props) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);
  const trigRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    // Pin the panel under the trigger, clamped to viewport.
    const trig = trigRef.current;
    if (trig) {
      const r = trig.getBoundingClientRect();
      const width = 320;
      const left = Math.max(
        8,
        Math.min(window.innerWidth - width - 8, r.right - width),
      );
      const top = Math.min(window.innerHeight - 200, r.bottom + 6);
      setCoords({ top, left });
    }
    function onDoc(e: MouseEvent) {
      if (
        !trigRef.current?.contains(e.target as Node) &&
        !panelRef.current?.contains(e.target as Node)
      ) {
        setOpen(false);
      }
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
    <>
      <button
        ref={trigRef}
        type="button"
        className="help-trigger"
        aria-label={`Help: ${title}`}
        onClick={() => setOpen((o) => !o)}
      >
        {label}
      </button>
      {open && coords &&
        createPortal(
          <div
            ref={panelRef}
            className="help-panel help-panel-portal"
            role="dialog"
            aria-label={title}
            style={{ top: coords.top, left: coords.left }}
          >
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
          </div>,
          document.body,
        )}
    </>
  );
}
