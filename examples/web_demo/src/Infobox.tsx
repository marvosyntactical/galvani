import type { ReactNode } from "react";

interface Props {
  id: string;
  title: string;
  children: ReactNode;
  /** Controlled accordion: parent decides which card is open. */
  openId: string | null;
  onToggle: (id: string | null) => void;
}

export function Infobox({ id, title, children, openId, onToggle }: Props) {
  const open = openId === id;
  return (
    <section className={`infobox ${open ? "open" : ""}`}>
      <button
        className="infobox-header"
        onClick={() => onToggle(open ? null : id)}
        aria-expanded={open}
      >
        <span className="infobox-marker" aria-hidden />
        <span className="infobox-title">{title}</span>
        <span className="infobox-chevron" aria-hidden>
          {open ? "−" : "+"}
        </span>
      </button>
      <div className="infobox-collapse" aria-hidden={!open}>
        <div className="infobox-body">{children}</div>
      </div>
    </section>
  );
}
