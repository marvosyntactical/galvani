/**
 * Per-cell-type baseline colors and short descriptions.
 *
 * Colors are picked to be perceptually distinct and biologically suggestive:
 *  - EPG (ring core)         -> warm orange
 *  - PEN_a (velocity loop)   -> teal
 *  - PEN_b (velocity loop)   -> light blue
 *  - Delta7 (inhibition)     -> magenta (matches inhibitory convention)
 *  - KC*  (mushroom body)    -> pale yellow/green family
 *  - APL  (mushroom body inhibitor) -> magenta
 */

import * as THREE from "three";

export interface CellTypeInfo {
  baseColor: THREE.Color;
  shortLabel: string;
  blurb: string;
}

function c(hex: string): THREE.Color {
  return new THREE.Color(hex);
}

export const CELL_TYPE_INFO: Record<string, CellTypeInfo> = {
  // HD ring types
  EPG: {
    baseColor: c("#f78c3c"),
    shortLabel: "EPG",
    blurb:
      "Cholinergic. Form a ring attractor in the ellipsoid body. Each EPG codes one azimuthal heading.",
  },
  "PEN_a(PEN1)": {
    baseColor: c("#3ad9c9"),
    shortLabel: "PEN_a",
    blurb:
      "Cholinergic. Asymmetric loop with EPG cells. Injects angular-velocity drive that rotates the bump.",
  },
  "PEN_b(PEN2)": {
    baseColor: c("#62a3ff"),
    shortLabel: "PEN_b",
    blurb:
      "Cholinergic. Second velocity-integration channel, complements PEN_a with offset PB->EB shift.",
  },
  Delta7: {
    baseColor: c("#d65bff"),
    shortLabel: "Delta7",
    blurb:
      "Glutamatergic (inhibitory in fly). Broad PB-wide inhibition that stabilises the bump shape.",
  },

  // Mushroom body
  "KCg-m": { baseColor: c("#cae74f"), shortLabel: "KCg-m", blurb: "Kenyon cell, gamma main." },
  "KCg-d": { baseColor: c("#aedf45"), shortLabel: "KCg-d", blurb: "Kenyon cell, gamma dorsal." },
  "KCg-t": { baseColor: c("#88c93b"), shortLabel: "KCg-t", blurb: "Kenyon cell, gamma terminal." },
  "KCab-m": { baseColor: c("#f0c43a"), shortLabel: "KCab-m", blurb: "Kenyon cell, alpha/beta main." },
  "KCab-c": { baseColor: c("#e9aa2d"), shortLabel: "KCab-c", blurb: "Kenyon cell, alpha/beta core." },
  "KCab-s": { baseColor: c("#d99020"), shortLabel: "KCab-s", blurb: "Kenyon cell, alpha/beta surface." },
  "KCab-p": { baseColor: c("#bc7615"), shortLabel: "KCab-p", blurb: "Kenyon cell, alpha/beta posterior." },
  "KCa'b'-m": { baseColor: c("#7be4b4"), shortLabel: "KCa'b'-m", blurb: "Kenyon cell, alpha'/beta' main." },
  "KCa'b'-ap1": { baseColor: c("#5ac38f"), shortLabel: "KCa'b'-ap1", blurb: "Kenyon cell, alpha'/beta' ap1." },
  "KCa'b'-ap2": { baseColor: c("#3da76d"), shortLabel: "KCa'b'-ap2", blurb: "Kenyon cell, alpha'/beta' ap2." },
  APL: {
    baseColor: c("#d65bff"),
    shortLabel: "APL",
    blurb:
      "GABAergic. Single neuron that pools KC activity and inhibits the whole KC population: the source of the mushroom body's sparse code.",
  },
};

const FALLBACK = c("#aaaaaa");

export function baseColorFor(cellType: string): THREE.Color {
  const info = CELL_TYPE_INFO[cellType];
  return info ? info.baseColor : FALLBACK;
}

export function labelFor(cellType: string): string {
  return CELL_TYPE_INFO[cellType]?.shortLabel ?? cellType;
}
