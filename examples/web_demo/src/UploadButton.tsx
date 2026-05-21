import { useRef } from "react";
import type { Payload } from "./payload";

interface Props {
  onLoad: (p: Payload, name: string) => void;
  onError: (msg: string) => void;
}

/**
 * Lets the user drop in their own payload JSON (in our schema v2 format)
 * for visualization. Useful for: trying scenarios baked locally without
 * a redeploy, sharing custom-baked payloads via download links.
 *
 * The file is parsed in-memory, schema-checked, and handed back to App.
 */
export function UploadButton({ onLoad, onError }: Props) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <>
      <button className="upload-btn" onClick={() => ref.current?.click()}>
        ⬆ Load custom JSON…
      </button>
      <input
        ref={ref}
        type="file"
        accept="application/json,.json"
        style={{ display: "none" }}
        onChange={async (e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          try {
            const text = await f.text();
            const p = JSON.parse(text) as Payload;
            if (p.metadata?.schema_version !== 2) {
              onError(
                `Schema version ${p.metadata?.schema_version} not supported (need 2).`,
              );
              return;
            }
            if (!Array.isArray(p.neurons) || !Array.isArray(p.rates)) {
              onError("Missing 'neurons' or 'rates' array.");
              return;
            }
            onLoad(p, f.name);
          } catch (err) {
            onError(`Failed to parse ${f.name}: ${err}`);
          }
          // Allow re-uploading the same file later.
          if (ref.current) ref.current.value = "";
        }}
      />
    </>
  );
}
