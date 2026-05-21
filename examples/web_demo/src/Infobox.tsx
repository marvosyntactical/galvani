import { useState, type ReactNode } from "react";

interface Props {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}

export function Infobox({ title, children, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`infobox ${open ? "open" : ""}`}>
      <button
        className="infobox-header"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="caret">{open ? "▾" : "▸"}</span>
        <span className="title">{title}</span>
      </button>
      {open && <div className="infobox-body">{children}</div>}
    </div>
  );
}
