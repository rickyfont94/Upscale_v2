import React, {
  createContext, useContext, useReducer, useEffect, useRef, useCallback, useState, ReactNode, DragEvent, ClipboardEvent, useId, useMemo, ChangeEvent
} from "react";
import { createRoot } from "react-dom/client";
import { GoogleGenAI, Modality } from "@google/genai";
import clsx from "clsx";
import { twMerge } from "tailwind-merge";
import { Upload, Wand2, Undo2, Redo2, ClipboardCopy, Download, RotateCcw, ChevronDown, PenSquare, Trash2, Eye, X, ChevronsLeftRight, GalleryVerticalEnd } from 'lucide-react';

// React 19 runtime check (non-crashing)
;(window as any).React = React;
const v = (window as any).React?.version;
if (!v?.startsWith("19.")) {
  console.error(`React version mismatch. Expected 19.x, loaded: ${v ?? "unknown"}`);
}

/* ---------- constants ---------- */
const GEMINI_MODEL_NAME: ModelName = "gemini-2.5-flash-image-preview";
const MAX_IMAGE_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024; // 15MB
const BATCH_CONCURRENCY = 2;
const ENHANCEMENT_LOADING_MESSAGES = [
  "Analyzing portrait details...",
  "Applying advanced color correction...",
  "Retouching skin tones with care...",
  "Enhancing facial features subtly...",
  "Perfecting the background...",
  "This can take a moment, great results are coming!",
];

/* ---------- utils ---------- */
const apiKey =
  (globalThis as any).AI_STUDIO_API_KEY ??
  (import.meta as any).env?.VITE_API_KEY ??
  ((globalThis as any).process?.env?.API_KEY);

const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

function cn(...i: any[]) { return twMerge(clsx(i)); }

function downloadFile(content: string, fileName: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(link.href);
}

async function convertImage(base64: string, mimeType: 'image/jpeg' | 'image/webp', quality: number): Promise<string> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            if (!ctx) return reject(new Error("Could not get canvas context"));
            ctx.drawImage(img, 0, 0);
            resolve(canvas.toDataURL(mimeType, quality / 100));
        };
        img.onerror = (err) => reject(err);
        img.src = base64;
    });
}

/* ---------- types ---------- */
type ModelName = "gemini-2.5-flash-image-preview";
type ToolStates = {
  superResolution: { enabled: boolean; upscale: "2x" | "4x" | "8x" };
  colorTone: {
    enabled: boolean; whiteBalance: "auto" | "custom";
    temp: number; tint: number; exposure: number; contrast: number;
    highlights: number; shadows: number; vibrance: number; saturation: number;
    skinTone: "neutral" | "slightly warm";
  };
  facialRetouch: { enabled: boolean; intensity: "low" | "medium"; eyeEnhance: boolean; teethWhiten: boolean };
  preserveDetails: { enabled: boolean; strength: 'medium' | 'high' };
  hairStyling: { enabled: boolean; mode: "keep" | "color" | "custom"; color: string; customInstruction: string };
  background: { enabled: boolean; mode: "keep" | "solid" | "blur" | "custom"; solidColor: string; customInstruction: string };
  crop: { enabled: boolean; aspectRatio: "original" | "1:1" | "4:5" | "16:9" };
  distractions: { enabled: boolean; list: string };
  noiseAndOptics: { enabled: boolean; lumaNoise: number; chromaNoise: number; caFix: number; vignette: number; distortion: number; };
};
type ImageFile = { name: string; type: string; data: string };
type Preset = { id: string; name: string; tools: ToolStates; isBuiltIn?: boolean };
type BatchItem = { id: string; file: ImageFile; status: 'pending' | 'processing' | 'done' | 'error' | 'cancelled'; result?: string; prompt?: string; error?: string, preset?: ToolStates };

type AppState = {
  history: { tools: ToolStates }[]; historyIndex: number;
  currentImage: ImageFile | null; enhancedImage: string | null;
  isLoading: boolean;
  toast: { id: number; message: string; type: "success" | "error" } | null;
  activeModel: ModelName; tools: ToolStates; currentPrompt: string;
  userPresets: Preset[]; showBuiltInPresets: boolean; batchQueue: BatchItem[];
  enhancedLibrary: BatchItem[];
};
type Action =
  | { type: "UPDATE_TOOL"; payload: { tool: keyof ToolStates; settings: Partial<ToolStates[keyof ToolStates]> } }
  | { type: "UNDO" } | { type: "REDO" }
  | { type: "SET_IMAGE"; payload: ImageFile | null }
  | { type: "SELECT_BATCH_ITEM"; payload: BatchItem }
  | { type: "ENHANCE_START" } | { type: "ENHANCE_SUCCESS"; payload: { image: string, prompt: string } } | { type: "ENHANCE_FAILURE"; payload: string }
  | { type: "SHOW_TOAST"; payload: { message: string, type: 'success' | 'error' } }
  | { type: "HIDE_TOAST" }
  | { type: "RESET_STATE" }
  | { type: "TOGGLE_BUILTIN_PRESETS" }
  | { type: "SAVE_PRESET"; payload: { preset: Preset } }
  | { type: "LOAD_PRESET"; payload: { tools: ToolStates } }
  | { type: "DELETE_PRESET"; payload: { id: string } }
  | { type: "RENAME_PRESET"; payload: { id: string, name: string } }
  | { type: "IMPORT_PRESETS"; payload: { presets: Preset[] } }
  | { type: "ADD_TO_BATCH"; payload: { files: ImageFile[] } }
  | { type: "UPDATE_BATCH_ITEM"; payload: Partial<BatchItem> & { id: string } }
  | { type: "CLEAR_BATCH" }
  | { type: "APPLY_PRESET_TO_BATCH"; payload: { tools: ToolStates } }
  | { type: "SET_ENHANCED_IMAGE"; payload: string | null }
  | { type: "ADD_TO_LIBRARY"; payload: { item: BatchItem } }
  | { type: "REMOVE_FROM_LIBRARY"; payload: { id: string } }
  | { type: "CLEAR_LIBRARY" }
  | { type: "LOAD_LIBRARY"; payload: { library: BatchItem[] } };


/* ---------- state ---------- */
const initialToolStates: ToolStates = {
  superResolution: { enabled: true, upscale: "2x" },
  colorTone: {
    enabled: true, whiteBalance: "auto",
    temp: 0, tint: 0, exposure: 0, contrast: 0, highlights: 0, shadows: 0, vibrance: 0, saturation: 0,
    skinTone: "neutral"
  },
  facialRetouch: { enabled: true, intensity: "low", eyeEnhance: true, teethWhiten: true },
  preserveDetails: { enabled: false, strength: 'medium' },
  hairStyling: { enabled: true, mode: "color", color: "#5C3D2E", customInstruction: "" },
  background: { enabled: false, mode: "keep", solidColor: "#ffffff", customInstruction: "" },
  crop: { enabled: false, aspectRatio: "original" },
  distractions: { enabled: false, list: "" },
  noiseAndOptics: { enabled: false, lumaNoise: 0, chromaNoise: 0, caFix: 0, vignette: 0, distortion: 0 },
};

const initialState: AppState = {
  history: [{ tools: initialToolStates }], historyIndex: 0,
  currentImage: null, enhancedImage: null, isLoading: false, toast: null,
  activeModel: GEMINI_MODEL_NAME, tools: initialToolStates, currentPrompt: "",
  userPresets: [], showBuiltInPresets: false, batchQueue: [], enhancedLibrary: [],
};

const appReducer = (state: AppState, action: Action): AppState => {
  switch (action.type) {
    case "UPDATE_TOOL": {
      const newTools = { ...state.tools, [action.payload.tool]: { ...state.tools[action.payload.tool], ...action.payload.settings } } as ToolStates;
      const newHistory = [...state.history.slice(0, state.historyIndex + 1), { tools: newTools }];
      return { ...state, tools: newTools, history: newHistory, historyIndex: newHistory.length - 1 };
    }
    case "UNDO":
      if (state.historyIndex > 0) {
        const i = state.historyIndex - 1;
        return { ...state, historyIndex: i, tools: state.history[i].tools };
      }
      return state;
    case "REDO":
      if (state.historyIndex < state.history.length - 1) {
        const i = state.historyIndex + 1;
        return { ...state, historyIndex: i, tools: state.history[i].tools };
      }
      return state;
    case "SET_IMAGE": 
        return { ...state, currentImage: action.payload, enhancedImage: null, isLoading: false, history: [{ tools: state.tools }], historyIndex: 0 };
    case "SELECT_BATCH_ITEM":
        return { ...state, currentImage: action.payload.file, enhancedImage: action.payload.result || null, tools: action.payload.preset || state.tools, history: [{ tools: action.payload.preset || state.tools }], historyIndex: 0 };
    case "ENHANCE_START": return { ...state, isLoading: true, enhancedImage: null };
    case "ENHANCE_SUCCESS": return { ...state, isLoading: false, enhancedImage: action.payload.image, currentPrompt: action.payload.prompt };
    case "ENHANCE_FAILURE": return { ...state, isLoading: false };
    case "SHOW_TOAST": return { ...state, toast: { id: Date.now(), ...action.payload } };
    case "HIDE_TOAST": return { ...state, toast: null };
    case "RESET_STATE": return { ...initialState, currentImage: state.currentImage, userPresets: state.userPresets, showBuiltInPresets: state.showBuiltInPresets, batchQueue: state.batchQueue, enhancedLibrary: state.enhancedLibrary };
    case "TOGGLE_BUILTIN_PRESETS": return { ...state, showBuiltInPresets: !state.showBuiltInPresets };
    case "SAVE_PRESET": return { ...state, userPresets: [...state.userPresets, action.payload.preset] };
    case "LOAD_PRESET": {
      const newHistory = [...state.history.slice(0, state.historyIndex + 1), { tools: action.payload.tools }];
      return { ...state, tools: action.payload.tools, history: newHistory, historyIndex: newHistory.length - 1 };
    }
    case "DELETE_PRESET": return { ...state, userPresets: state.userPresets.filter(p => p.id !== action.payload.id) };
    case "RENAME_PRESET": return { ...state, userPresets: state.userPresets.map(p => p.id === action.payload.id ? { ...p, name: action.payload.name } : p) };
    case "IMPORT_PRESETS": {
      const incomingPresets = action.payload.presets.filter(p => p.id && p.name && p.tools && !p.isBuiltIn);
      const updatedPresets = [...state.userPresets];
      incomingPresets.forEach(p => { if (!updatedPresets.some(up => up.id === p.id)) { updatedPresets.push(p); } });
      return { ...state, userPresets: updatedPresets };
    }
    case "ADD_TO_BATCH": {
        const newItems: BatchItem[] = action.payload.files.map(file => ({
            id: `${file.name}-${Date.now()}-${Math.random()}`, file, status: 'pending', preset: state.tools
        }));
        return { ...state, batchQueue: [...state.batchQueue, ...newItems] };
    }
    case "UPDATE_BATCH_ITEM": return { ...state, batchQueue: state.batchQueue.map(item => item.id === action.payload.id ? { ...item, ...action.payload } : item) };
    case "CLEAR_BATCH": return { ...state, batchQueue: [] };
    case "APPLY_PRESET_TO_BATCH": return { ...state, batchQueue: state.batchQueue.map(item => item.status === 'pending' ? { ...item, preset: action.payload.tools } : item) };
    case "SET_ENHANCED_IMAGE": return { ...state, enhancedImage: action.payload };
    case "ADD_TO_LIBRARY": return { ...state, enhancedLibrary: [action.payload.item, ...state.enhancedLibrary.filter(item => item.id !== action.payload.item.id)] };
    case "REMOVE_FROM_LIBRARY": return { ...state, enhancedLibrary: state.enhancedLibrary.filter(item => item.id !== action.payload.id) };
    case "CLEAR_LIBRARY": return { ...state, enhancedLibrary: [] };
    case "LOAD_LIBRARY": return { ...state, enhancedLibrary: action.payload.library };
    default: return state;
  }
};

const AppContext = createContext<{ state: AppState; dispatch: React.Dispatch<Action> } | undefined>(undefined);
const AppProvider = ({ children }: { children: React.ReactNode }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);
  return <AppContext.Provider value={{ state, dispatch }}>{children}</AppContext.Provider>;
};
const useAppContext = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within an AppProvider");
  return ctx;
};

/* ---------- presets ---------- */
const BUILT_IN_PRESETS: Readonly<Preset[]> = [
    {
        id: 'builtin_linkedin', name: 'LinkedIn Clean', isBuiltIn: true,
        tools: { ...initialToolStates,
            superResolution: { enabled: true, upscale: '2x' },
            colorTone: { ...initialToolStates.colorTone, enabled: true, exposure: 1, contrast: 1 },
            facialRetouch: { ...initialToolStates.facialRetouch, enabled: true, intensity: 'low', eyeEnhance: true, teethWhiten: true },
            distractions: { enabled: true, list: 'minor lint on collar, stray hairs' },
        }
    },
    {
        id: 'builtin_studio', name: 'Studio Neutral', isBuiltIn: true,
        tools: { ...initialToolStates,
            superResolution: { enabled: true, upscale: '2x' },
            colorTone: { ...initialToolStates.colorTone, enabled: true },
            facialRetouch: { ...initialToolStates.facialRetouch, enabled: true, intensity: 'low' },
            background: { ...initialToolStates.background, enabled: true, mode: 'solid', solidColor: '#e0e0e0' }
        }
    },
    {
        id: 'builtin_warm', name: 'Soft Warm', isBuiltIn: true,
        tools: { ...initialToolStates,
            superResolution: { enabled: true, upscale: '2x' },
            colorTone: { ...initialToolStates.colorTone, enabled: true, temp: 2, vibrance: 2, skinTone: 'slightly warm' },
            facialRetouch: { ...initialToolStates.facialRetouch, enabled: true, intensity: 'low' },
        }
    }
];

/* ---------- prompt + AI ---------- */
const buildPrompt = (t: ToolStates): string => {
  const ops: string[] = [];
  if (t.superResolution.enabled) ops.push(`SUPER RESOLUTION: Perform a high-quality ${t.superResolution.upscale} super-resolution upscale, effectively doubling the image's pixel dimensions. The primary goal is to reconstruct and synthesize photorealistic fine details. Enhance textures like skin pores, individual hair strands, and fabric weaves with exceptional clarity. The final image must be sharp and artifact-free. Do not simply enlarge and smooth the image; new, believable detail must be generated.`);
  if (t.colorTone.enabled) {
    let wb = `WB ${t.colorTone.whiteBalance}`;
    if (t.colorTone.whiteBalance === "custom") wb += ` (temp:${t.colorTone.temp}, tint:${t.colorTone.tint})`;
    ops.push(`COLOR & EXPOSURE: ${wb}; exposure ${t.colorTone.exposure}; contrast ${t.colorTone.contrast}; highlights/shadows recover; vibrance ${t.colorTone.vibrance}; saturation ${t.colorTone.saturation}; skin tone ${t.colorTone.skinTone}.`);
  }
  if (t.facialRetouch.enabled) ops.push(`FACIAL RETOUCH (SUBTLE): intensity ${t.facialRetouch.intensity}; eyes ${t.facialRetouch.eyeEnhance ? "on" : "off"}; teeth ${t.facialRetouch.teethWhiten ? "on" : "off"}; keep pores/texture; reduce shine; soften fine wrinkles/under-eye; reduce glasses glare; DO NOT change identity/age/facial structure/expression.`);
  if (t.preserveDetails.enabled) ops.push(`PRESERVE DETAILS (${t.preserveDetails.strength.toUpperCase()}): Meticulously preserve unique, defining facial features like moles, scars, freckles, and birthmarks. Do not remove, soften, or alter them. Ensure original skin and fabric textures are fully maintained.`);
  if (t.hairStyling.enabled && t.hairStyling.mode !== 'keep') {
    let hair = `mode: ${t.hairStyling.mode}`;
    if (t.hairStyling.mode === 'color') hair += `, color: ${t.hairStyling.color}`;
    if (t.hairStyling.mode === 'custom') hair += `, instruction: "${t.hairStyling.customInstruction}"`;
    ops.push(`HAIR STYLING: ${hair}; maintain realistic texture/shine; precise masking; no color bleed.`);
  }
  if (t.background.enabled) {
    let bg = `mode: ${t.background.mode}`;
    if (t.background.mode === "solid") bg += `, color: ${t.background.solidColor}`;
    if (t.background.mode === "custom") bg += `, instruction: "${t.background.customInstruction}"`;
    ops.push(`BACKGROUND: ${bg}; hair-safe matting; realistic shadows; no halos/spill.`);
  }
  if (t.crop.enabled) ops.push(`CROP/ROTATE: aspect ${t.crop.aspectRatio}; straightened; safe headroom; keep ears/hair.`);
  if (t.distractions.enabled && t.distractions.list.trim()) ops.push(`OBJECT REMOVAL: Carefully and seamlessly remove the following distractions: "${t.distractions.list.trim()}". Fill the area with realistic, context-aware content. The edit must be undetectable.`);
  if (t.noiseAndOptics.enabled) {
    const corrections = [];
    if (t.noiseAndOptics.lumaNoise > 0) corrections.push(`luma noise reduction (${t.noiseAndOptics.lumaNoise}/10)`);
    if (t.noiseAndOptics.chromaNoise > 0) corrections.push(`chroma noise reduction (${t.noiseAndOptics.chromaNoise}/10)`);
    if (t.noiseAndOptics.caFix > 0) corrections.push(`chromatic aberration correction (${t.noiseAndOptics.caFix}/10)`);
    if (t.noiseAndOptics.vignette !== 0) corrections.push(`vignette correction (${t.noiseAndOptics.vignette}/10)`);
    if (t.noiseAndOptics.distortion !== 0) corrections.push(`lens distortion correction (${t.noiseAndOptics.distortion}/10)`);
    if (corrections.length > 0) ops.push(`NOISE & OPTICS CORRECTION: ${corrections.join(', ')}.`);
  }
  
  const numberedOps = ops.map((op, i) => `${i + 1}) ${op}`).join("\n");

  return `Enhance this PORTRAIT while preserving identity and realism.

ORDER OF OPERATIONS:
${numberedOps}

CONSISTENCY: same intensity across faces; consistent skin tone.
LIMITS: no style transfer; no makeup/clothing/body reshaping/new objects.
OUTPUT: single high-res PNG (sRGB).`;
};

const enhanceImage = async (image: ImageFile, tools: ToolStates, model: ModelName) => {
  if (!ai) throw new Error("Missing API key. Set AI_STUDIO_API_KEY, VITE_API_KEY, or process.env.API_KEY.");
  const prompt = buildPrompt(tools);
  const base64Data = image.data.split(",")[1];
  const imagePart = { inlineData: { data: base64Data, mimeType: image.type } };
  const textPart = { text: prompt };

  const response = await ai.models.generateContent({
    model, contents: { parts: [imagePart, textPart] },
    config: { responseModalities: [Modality.IMAGE, Modality.TEXT] }
  });

  const cand = response.candidates?.[0];
  for (const part of cand?.content?.parts ?? []) {
    if ((part as any).inlineData) {
      const p: any = part;
      return { image: `data:${p.inlineData.mimeType};base64,${p.inlineData.data}`, prompt };
    }
  }
  throw new Error(response.text || "The AI did not return an image. It may have refused the request.");
};

/* ---------- UI primitives ---------- */
const Button = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost' | 'destructive' }>(({ className, variant = 'secondary', ...props }, ref) => {
    const variants = {
        primary: "bg-primary text-primary-foreground hover:bg-primary/90",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80 border border-border",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
    };
    return <button ref={ref} className={cn("inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 gap-2", variants[variant], className)} {...props} />;
});
const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) =>
  <div ref={ref} className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)} {...props} />
);
const Label = React.forwardRef<HTMLLabelElement, React.LabelHTMLAttributes<HTMLLabelElement>>(({ className, ...props }, ref) =>
  <label ref={ref} className={cn("text-sm font-medium leading-none", className)} {...props} />
);
const Select = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(({ className, children, ...props }, ref) =>
  <select ref={ref} className={cn("h-10 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2", className)} {...props}>{children}</select>
);
const Switch = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { checked: boolean }>(({ className, checked, id, ...props }, ref) =>
  <button type="button" role="switch" aria-checked={checked} ref={ref} id={id}
    className={cn("peer inline-flex h-[24px] w-[44px] shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2", checked ? "bg-green-600" : "bg-slate-300", className)}
    {...props}><span className={cn("pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform", checked ? "translate-x-5" : "translate-x-0")} /></button>
);
const Slider = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...props }, ref) =>
  <input ref={ref} type="range" className={cn("w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary", className)} {...props} />
);
const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...props }, ref) =>
  <textarea ref={ref} className={cn("flex min-h-[80px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2", className)} {...props} />
);
const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...props }, ref) =>
  <input ref={ref} className={cn("flex h-10 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm", className)} {...props} />
);
const Modal = ({ open, onClose, title, children }: { open: boolean, onClose: () => void, title: string, children: ReactNode }) => {
    if (!open) return null;
    return (
        <div className="fixed inset-0 bg-black/50 z-[9999] flex items-center justify-center p-4 animate-fade-in" onClick={onClose}>
            <div className="bg-card rounded-lg shadow-xl w-full max-w-md" onClick={e => e.stopPropagation()}>
                <div className="p-4 border-b border-border flex justify-between items-center">
                    <h3 className="text-lg font-semibold text-foreground">{title}</h3>
                    <Button variant="ghost" className="h-8 px-2" onClick={onClose} aria-label="Close modal"><Icon name="close" /></Button>
                </div>
                <div className="p-4">{children}</div>
            </div>
        </div>
    );
};

const iconMap = {
    upload: Upload, enhance: Wand2, undo: Undo2, redo: Redo2, copy: ClipboardCopy, download: Download, reset: RotateCcw,
    chevronDown: ChevronDown, edit: PenSquare, delete: Trash2, view: Eye, close: X, chevronsLeftRight: ChevronsLeftRight,
    library: GalleryVerticalEnd,
};
type IconName = keyof typeof iconMap;
const Icon = ({ name, ...props }: { name: IconName } & React.SVGProps<SVGSVGElement> & { size?: number | string }) => {
    const LucideIcon = iconMap[name];
    if (!LucideIcon) return null;
    const defaultProps = { strokeWidth: 1.75, size: 20, 'aria-hidden': true };
    return <LucideIcon {...defaultProps} {...props} />;
};

/* ---------- app components ---------- */
interface AccordionItemProps { title: string; children: ReactNode; enabled: boolean; onToggle: (enabled: boolean) => void }
const AccordionItem = ({ title, children, enabled, onToggle }: AccordionItemProps) => {
  const [open, setOpen] = useState(true);
  const id = useId();
  return (
    <Card className="mb-2 overflow-hidden bg-white">
      <div className="flex items-center justify-between p-3 cursor-pointer" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={id}>
        <div className="flex items-center gap-3">
          <Switch checked={enabled} onClick={(e) => { e.stopPropagation(); onToggle(!enabled); }} aria-label={`Toggle ${title}`} />
          <span className="font-semibold select-none text-foreground">{title}</span>
        </div>
        <Icon name="chevronDown" className={cn("transition-transform text-muted-foreground", open ? "rotate-180" : "")} />
      </div>
      {open && <div id={id} className={cn("p-4 border-t border-border", !enabled && "opacity-50 pointer-events-none")}>{children}</div>}
    </Card>
  );
};

const ToolControl = ({ label, children, value, controlId }: { label: ReactNode; children: ReactNode; value?: string | number, controlId?: string }) => (
  <div className="grid grid-cols-3 items-center gap-4 mb-3">
    <Label htmlFor={controlId} className="text-muted-foreground col-span-1">{label}</Label>
    <div className="flex items-center col-span-2">
      <div className="flex-grow">{children}</div>
      {value !== undefined && <span className="ml-3 w-10 text-right text-sm font-mono">{value}</span>}
    </div>
  </div>
);

const PresetsManager = () => {
    const { state, dispatch } = useAppContext();
    const { tools, userPresets, showBuiltInPresets } = state;
    const [isManageModalOpen, setManageModalOpen] = useState(false);
    const [presetToRename, setPresetToRename] = useState<Preset | null>(null);
    const [newName, setNewName] = useState("");
    const importRef = useRef<HTMLInputElement>(null);

    const availablePresets = useMemo(() => {
        const all = [...userPresets];
        if (showBuiltInPresets) all.push(...BUILT_IN_PRESETS);
        return all;
    }, [userPresets, showBuiltInPresets]);

    const handleSave = () => {
        const name = prompt("Enter a name for your preset:");
        if (name) {
            const newPreset: Preset = { id: `user_${Date.now()}`, name, tools };
            dispatch({ type: "SAVE_PRESET", payload: { preset: newPreset } });
            dispatch({ type: "SHOW_TOAST", payload: { message: `Preset "${name}" saved!`, type: "success" } });
        }
    };

    const handleLoad = (e: ChangeEvent<HTMLSelectElement>) => {
        const id = e.target.value;
        if (!id) return;
        const preset = availablePresets.find(p => p.id === id);
        if (preset) {
            dispatch({ type: "LOAD_PRESET", payload: { tools: preset.tools } });
            dispatch({ type: "SHOW_TOAST", payload: { message: `Preset "${preset.name}" loaded!`, type: "success" } });
        }
        e.target.value = "";
    };

    const handleExport = () => {
        downloadFile(JSON.stringify(userPresets, null, 2), "ai-portrait-presets.json", "application/json");
    };

    const handleImport = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const presets = JSON.parse(event.target?.result as string);
                if (Array.isArray(presets)) {
                    dispatch({ type: "IMPORT_PRESETS", payload: { presets } });
                    dispatch({ type: "SHOW_TOAST", payload: { message: "Presets imported successfully!", type: "success" } });
                } else throw new Error("Invalid format");
            } catch (error) {
                dispatch({ type: "SHOW_TOAST", payload: { message: "Failed to import presets. Invalid file.", type: "error" } });
            }
        };
        reader.readAsText(file);
    };

    return (
        <div className="mb-6">
            <h2 className="text-2xl font-bold mb-4 text-foreground">Presets</h2>
            <Card className="p-4 bg-white">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <Select onChange={handleLoad} value="">
                        <option value="">Apply a Preset...</option>
                        {showBuiltInPresets && <optgroup label="Built-in Presets">{BUILT_IN_PRESETS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</optgroup>}
                        {userPresets.length > 0 && <optgroup label="My Presets">{userPresets.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</optgroup>}
                    </Select>
                    <div className="flex items-center gap-2">
                        <Button onClick={handleSave} className="flex-grow">Save Current</Button>
                        <Button onClick={() => setManageModalOpen(true)}>Manage</Button>
                    </div>
                </div>
                <div className="flex items-center justify-start gap-2 mt-3">
                    <Switch id="toggle-builtins" checked={showBuiltInPresets} onClick={() => dispatch({ type: "TOGGLE_BUILTIN_PRESETS" })} />
                    <Label htmlFor="toggle-builtins" className="text-sm select-none cursor-pointer">Show Built-in Presets</Label>
                </div>
            </Card>

            <Modal open={isManageModalOpen} onClose={() => setManageModalOpen(false)} title="Manage Presets">
                <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-2">
                    {userPresets.map(p => (
                        <div key={p.id} className="flex items-center justify-between gap-2 p-2 rounded-md hover:bg-secondary">
                            {presetToRename?.id === p.id ? (
                                <Input value={newName} onChange={e => setNewName(e.target.value)} onKeyDown={e => {
                                    if(e.key === 'Enter' && newName.trim()) {
                                        dispatch({ type: 'RENAME_PRESET', payload: { id: p.id, name: newName.trim() }});
                                        setPresetToRename(null);
                                    } else if(e.key === 'Escape') setPresetToRename(null);
                                }} autoFocus />
                            ) : (
                                <span className="text-sm">{p.name}</span>
                            )}
                            <div className="flex gap-1">
                                <Button variant="ghost" className="h-8 px-2" aria-label={`Rename ${p.name}`} onClick={() => { setPresetToRename(p); setNewName(p.name); }}>
                                  <Icon name="edit" />
                                </Button>
                                <Button variant="ghost" className="h-8 px-2" aria-label={`Delete ${p.name}`} onClick={() => { if(confirm(`Delete "${p.name}"?`)) dispatch({ type: 'DELETE_PRESET', payload: {id: p.id} })}}>
                                  <Icon name="delete" />
                                </Button>
                            </div>
                        </div>
                    ))}
                    {userPresets.length === 0 && <p className="text-muted-foreground text-sm text-center py-4">You have no saved presets.</p>}
                </div>
                <div className="flex gap-2 mt-4 pt-4 border-t border-border">
                    <Button onClick={() => importRef.current?.click()} className="flex-grow">Import</Button>
                    <input type="file" ref={importRef} accept=".json" className="hidden" onChange={handleImport} />
                    <Button onClick={handleExport} disabled={userPresets.length === 0} className="flex-grow">Export</Button>
                </div>
            </Modal>
        </div>
    );
};

const ToolsPanel = () => {
  const { state, dispatch } = useAppContext();
  const { tools } = state;
  const updateTool = (tool: keyof ToolStates, settings: Partial<ToolStates[keyof ToolStates]>) =>
    dispatch({ type: "UPDATE_TOOL", payload: { tool, settings } });
  
  const hairColorPresets = ["#000000", "#3B2A24", "#5C3D2E", "#A47C5A", "#B89364", "#DDC084", "#8B4513", "#C0C0C0"];

  const ids = {
      upscale: useId(), wb: useId(), temp: useId(), tint: useId(), exposure: useId(), contrast: useId(), highlights: useId(), 
      shadows: useId(), vibrance: useId(), saturation: useId(), skinTone: useId(), intensity: useId(), eyeEnhance: useId(), 
      teethWhiten: useId(), hairMode: useId(), hairColor: useId(), hairCustom: useId(), bgMode: useId(), bgColor: useId(), bgCustom: useId(), aspect: useId(), distractions: useId(),
      preserveStrength: useId(), luma: useId(), chroma: useId(), ca: useId(), vignette: useId(), distortion: useId()
  };

  return (
    <div className="lg:col-span-1 h-fit lg:sticky top-6">
      <PresetsManager />
      <h2 className="text-2xl font-bold mb-4 text-foreground">Enhancement Tools</h2>
      <AccordionItem title="Super Resolution" enabled={tools.superResolution.enabled} onToggle={(e) => updateTool("superResolution", { enabled: e })}>
        <ToolControl label="Upscale" controlId={ids.upscale}><Select id={ids.upscale} value={tools.superResolution.upscale} onChange={(e) => updateTool("superResolution", { upscale: e.target.value as "2x" | "4x" | "8x" })}><option value="2x">2x</option><option value="4x">4x</option><option value="8x">8x (Max)</option></Select></ToolControl>
      </AccordionItem>
      <AccordionItem title="Color & Tone" enabled={tools.colorTone.enabled} onToggle={(e) => updateTool("colorTone", { enabled: e })}>
          <ToolControl label="White Balance" controlId={ids.wb}><Select id={ids.wb} value={tools.colorTone.whiteBalance} onChange={(e) => updateTool("colorTone", { whiteBalance: e.target.value as "auto" | "custom" })}><option value="auto">Auto</option><option value="custom">Custom</option></Select></ToolControl>
          {tools.colorTone.whiteBalance === 'custom' && (<><ToolControl label="Temperature" value={tools.colorTone.temp} controlId={ids.temp}><Slider id={ids.temp} min="-10" max="10" value={tools.colorTone.temp} onChange={(e) => updateTool("colorTone", { temp: +e.target.value })} onDoubleClick={() => updateTool("colorTone", { temp: 0 })} /></ToolControl><ToolControl label="Tint" value={tools.colorTone.tint} controlId={ids.tint}><Slider id={ids.tint} min="-10" max="10" value={tools.colorTone.tint} onChange={(e) => updateTool("colorTone", { tint: +e.target.value })} onDoubleClick={() => updateTool("colorTone", { tint: 0 })} /></ToolControl></>)}
          <ToolControl label="Exposure" value={tools.colorTone.exposure} controlId={ids.exposure}><Slider id={ids.exposure} min="-10" max="10" value={tools.colorTone.exposure} onChange={(e) => updateTool("colorTone", { exposure: +e.target.value })} onDoubleClick={() => updateTool("colorTone", { exposure: 0 })} /></ToolControl>
          <ToolControl label="Contrast" value={tools.colorTone.contrast} controlId={ids.contrast}><Slider id={ids.contrast} min="-10" max="10" value={tools.colorTone.contrast} onChange={(e) => updateTool("colorTone", { contrast: +e.target.value })} onDoubleClick={() => updateTool("colorTone", { contrast: 0 })} /></ToolControl>
          <ToolControl label="Highlights" value={tools.colorTone.highlights} controlId={ids.highlights}><Slider id={ids.highlights} min="-10" max="10" value={tools.colorTone.highlights} onChange={(e) => updateTool("colorTone", { highlights: +e.target.value })} onDoubleClick={() => updateTool("colorTone", { highlights: 0 })} /></ToolControl>
          <ToolControl label="Shadows" value={tools.colorTone.shadows} controlId={ids.shadows}><Slider id={ids.shadows} min="-10" max="10" value={tools.colorTone.shadows} onChange={(e) => updateTool("colorTone", { shadows: +e.target.value })} onDoubleClick={() => updateTool("colorTone", { shadows: 0 })} /></ToolControl>
          <ToolControl label="Vibrance" value={tools.colorTone.vibrance} controlId={ids.vibrance}><Slider id={ids.vibrance} min="-10" max="10" value={tools.colorTone.vibrance} onChange={(e) => updateTool("colorTone", { vibrance: +e.target.value })} onDoubleClick={() => updateTool("colorTone", { vibrance: 0 })} /></ToolControl>
          <ToolControl label="Saturation" value={tools.colorTone.saturation} controlId={ids.saturation}><Slider id={ids.saturation} min="-10" max="10" value={tools.colorTone.saturation} onChange={(e) => updateTool("colorTone", { saturation: +e.target.value })} onDoubleClick={() => updateTool("colorTone", { saturation: 0 })} /></ToolControl>
          <ToolControl label="Skin Tone" controlId={ids.skinTone}><Select id={ids.skinTone} value={tools.colorTone.skinTone} onChange={(e) => updateTool("colorTone", { skinTone: e.target.value as "neutral" | "slightly warm" })}><option value="neutral">Neutral</option><option value="slightly warm">Slightly Warm</option></Select></ToolControl>
      </AccordionItem>
      <AccordionItem title="Facial Retouch" enabled={tools.facialRetouch.enabled} onToggle={(e) => updateTool("facialRetouch", { enabled: e })}>
        <ToolControl label="Intensity" controlId={ids.intensity}><Select id={ids.intensity} value={tools.facialRetouch.intensity} onChange={(e) => updateTool("facialRetouch", { intensity: e.target.value as "low" | "medium" })}><option value="low">Low</option><option value="medium">Medium</option></Select></ToolControl>
        <ToolControl label="Enhance Eyes" controlId={ids.eyeEnhance}><div className="w-full flex justify-end"><Switch id={ids.eyeEnhance} checked={tools.facialRetouch.eyeEnhance} onClick={() => updateTool("facialRetouch", { eyeEnhance: !tools.facialRetouch.eyeEnhance })} /></div></ToolControl>
        <ToolControl label="Whiten Teeth" controlId={ids.teethWhiten}><div className="w-full flex justify-end"><Switch id={ids.teethWhiten} checked={tools.facialRetouch.teethWhiten} onClick={() => updateTool("facialRetouch", { teethWhiten: !tools.facialRetouch.teethWhiten })} /></div></ToolControl>
      </AccordionItem>
      <AccordionItem title="Preserve Details" enabled={tools.preserveDetails.enabled} onToggle={(e) => updateTool("preserveDetails", { enabled: e })}><ToolControl label="Strength" controlId={ids.preserveStrength}><Select id={ids.preserveStrength} value={tools.preserveDetails.strength} onChange={(e) => updateTool("preserveDetails", { strength: e.target.value as 'medium' | 'high' })}><option value="medium">Medium</option><option value="high">High</option></Select></ToolControl></AccordionItem>
      <AccordionItem title="Hair Styling" enabled={tools.hairStyling.enabled} onToggle={(e) => updateTool("hairStyling", { enabled: e })}>
          <Select id={ids.hairMode} value={tools.hairStyling.mode} onChange={(e) => updateTool("hairStyling", { mode: e.target.value as any })}>
              <option value="keep">Keep Original</option>
              <option value="color">Change Color</option>
              <option value="custom">Custom Style</option>
          </Select>
          {tools.hairStyling.mode === "color" && (<><div className="flex items-center gap-2 mt-2"><Input id={ids.hairColor} type="color" value={tools.hairStyling.color} onChange={(e) => updateTool("hairStyling", { color: e.target.value })} className="p-1 h-10 w-14" /><Input aria-label="Hex color" type="text" value={tools.hairStyling.color} onChange={(e) => updateTool("hairStyling", { color: e.target.value })} className="flex-grow" /></div><div className="flex flex-wrap gap-2 mt-3 justify-center">{hairColorPresets.map(c => (<button key={c} type="button" aria-label={`Set hair color to ${c}`} onClick={() => updateTool("hairStyling", { color: c })} className={cn("w-6 h-6 rounded-full border-2 transition-transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2", tools.hairStyling.color.toLowerCase() === c.toLowerCase() ? 'border-primary' : 'border-border')} style={{ backgroundColor: c }} />))}</div></>)}
          {tools.hairStyling.mode === "custom" && <Textarea id={ids.hairCustom} value={tools.hairStyling.customInstruction} onChange={(e) => updateTool("hairStyling", { customInstruction: e.target.value })} className="mt-2" placeholder="e.g. rainbow streaks, curly and voluminous" />}
      </AccordionItem>
      <AccordionItem title="Background" enabled={tools.background.enabled} onToggle={(e) => updateTool("background", { enabled: e })}><Select id={ids.bgMode} value={tools.background.mode} onChange={(e) => updateTool("background", { mode: e.target.value as ToolStates["background"]["mode"] })}><option value="keep">Keep Original</option><option value="solid">Solid Color</option><option value="blur">Blur/Bokeh</option><option value="custom">Custom</option></Select>{tools.background.mode === "solid" && (<div className="flex items-center gap-2 mt-2"><Input id={ids.bgColor} type="color" value={tools.background.solidColor} onChange={(e) => updateTool("background", { solidColor: e.target.value })} className="p-1 h-10 w-14" /><Input aria-label="Hex color" type="text" value={tools.background.solidColor} onChange={(e) => updateTool("background", { solidColor: e.target.value })} className="flex-grow" /></div>)}{tools.background.mode === "custom" && <Textarea id={ids.bgCustom} value={tools.background.customInstruction} onChange={(e) => updateTool("background", { customInstruction: e.target.value })} className="mt-2" placeholder="e.g. a serene beach at sunset" />}</AccordionItem>
      <AccordionItem title="Crop & Straighten" enabled={tools.crop.enabled} onToggle={(e) => updateTool("crop", { enabled: e })}><ToolControl label="Aspect Ratio" controlId={ids.aspect}><Select id={ids.aspect} value={tools.crop.aspectRatio} onChange={(e) => updateTool("crop", { aspectRatio: e.target.value as any })}><option value="original">Original</option><option value="1:1">1:1 (Square)</option><option value="4:5">4:5 (Portrait)</option><option value="16:9">16:9 (Widescreen)</option></Select></ToolControl></AccordionItem>
      <AccordionItem title="Remove Distractions" enabled={tools.distractions.enabled} onToggle={(e) => updateTool("distractions", { enabled: e })}>
        <Textarea 
          id={ids.distractions} 
          value={tools.distractions.list} 
          onChange={(e) => updateTool("distractions", { list: e.target.value })} 
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              const value = e.currentTarget.value.trim();
              if (value && !value.endsWith(',')) {
                updateTool("distractions", { list: value + ', ' });
              }
            }
          }}
          placeholder="e.g. blemish on cheek, stray hair, lint on collar" />
        <p className="text-xs text-muted-foreground mt-1">Separate items with a comma or press Enter.</p>
      </AccordionItem>
      <AccordionItem title="Noise & Optics" enabled={tools.noiseAndOptics.enabled} onToggle={(e) => updateTool("noiseAndOptics", { enabled: e })}>
        <ToolControl label="Luma Noise" value={tools.noiseAndOptics.lumaNoise} controlId={ids.luma}><Slider id={ids.luma} min="0" max="10" value={tools.noiseAndOptics.lumaNoise} onChange={(e) => updateTool("noiseAndOptics", { lumaNoise: +e.target.value })} onDoubleClick={() => updateTool("noiseAndOptics", { lumaNoise: 0 })} /></ToolControl>
        <ToolControl label="Chroma Noise" value={tools.noiseAndOptics.chromaNoise} controlId={ids.chroma}><Slider id={ids.chroma} min="0" max="10" value={tools.noiseAndOptics.chromaNoise} onChange={(e) => updateTool("noiseAndOptics", { chromaNoise: +e.target.value })} onDoubleClick={() => updateTool("noiseAndOptics", { chromaNoise: 0 })} /></ToolControl>
        <ToolControl label="Fix CA" value={tools.noiseAndOptics.caFix} controlId={ids.ca}><Slider id={ids.ca} min="0" max="10" value={tools.noiseAndOptics.caFix} onChange={(e) => updateTool("noiseAndOptics", { caFix: +e.target.value })} onDoubleClick={() => updateTool("noiseAndOptics", { caFix: 0 })} /></ToolControl>
        <ToolControl label="Vignette" value={tools.noiseAndOptics.vignette} controlId={ids.vignette}><Slider id={ids.vignette} min="-10" max="10" value={tools.noiseAndOptics.vignette} onChange={(e) => updateTool("noiseAndOptics", { vignette: +e.target.value })} onDoubleClick={() => updateTool("noiseAndOptics", { vignette: 0 })} /></ToolControl>
        <ToolControl label="Distortion" value={tools.noiseAndOptics.distortion} controlId={ids.distortion}><Slider id={ids.distortion} min="-10" max="10" value={tools.noiseAndOptics.distortion} onChange={(e) => updateTool("noiseAndOptics", { distortion: +e.target.value })} onDoubleClick={() => updateTool("noiseAndOptics", { distortion: 0 })} /></ToolControl>
      </AccordionItem>
    </div>
  );
};

const ImageViewer = () => {
  const { state } = useAppContext();
  const { currentImage, enhancedImage, isLoading } = state;
  const [sliderPos, setSliderPos] = useState(50);
  const sliderRef = useRef<HTMLDivElement>(null);
  const [loadingMessage, setLoadingMessage] = useState(ENHANCEMENT_LOADING_MESSAGES[0]);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [isSliding, setIsSliding] = useState(false);
  const panStartRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (isLoading) {
      let i = 0; const interval = setInterval(() => { i = (i + 1) % ENHANCEMENT_LOADING_MESSAGES.length; setLoadingMessage(ENHANCEMENT_LOADING_MESSAGES[i]); }, 2500);
      return () => clearInterval(interval);
    }
  }, [isLoading]);

  const handleSliderMove = (clientX: number) => {
    if (!sliderRef.current) return;
    const rect = sliderRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };
  
  const onMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
        setIsPanning(true);
        panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    }
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (isSliding) {
        handleSliderMove(e.clientX);
    } else if (isPanning && zoom > 1) {
        setPan({ x: e.clientX - panStartRef.current.x, y: e.clientY - panStartRef.current.y });
    }
  };

  const onMouseUp = () => { setIsPanning(false); setIsSliding(false); };
  const handleZoom = (e: React.ChangeEvent<HTMLInputElement>) => { const newZoom = parseFloat(e.target.value); setZoom(newZoom); if (newZoom <= 1) setPan({ x: 0, y: 0 }); };
  const resetZoomAndPan = () => { setZoom(1); setPan({ x: 0, y: 0 }); };
  
  const Placeholder = () => <div className="w-full aspect-[4/5] bg-secondary rounded-lg flex flex-col items-center justify-center border-2 border-dashed border-border text-center p-4"><Upload className="text-muted-foreground mb-4 h-12 w-12" strokeWidth={1.5} /><p className="text-foreground font-semibold">Upload, drop, or paste an image</p><p className="text-muted-foreground text-sm mt-1">to begin your AI-powered enhancement.</p></div>;
  if (!currentImage) return <div className="flex justify-center"><div className="w-full max-w-2xl"><Placeholder /></div></div>;

  const imageStyle = { transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`, transformOrigin: 'center center', willChange: 'transform' } as const;
  const canPan = zoom > 1;

  return (
    <div className="flex flex-col items-center">
      <div className="w-full max-w-2xl">
        <div ref={sliderRef} className={cn("relative select-none w-full aspect-[4/5] rounded-lg overflow-hidden border border-border shadow-lg", canPan ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-auto')}
            onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseUp}>
          <img src={currentImage.data} alt="Original" style={imageStyle} className="absolute inset-0 w-full h-full object-contain" draggable={false} />
          {enhancedImage && !isLoading && (
            <>
              <div className="absolute inset-0 w-full h-full" style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}>
                <img src={enhancedImage} alt="Enhanced" style={imageStyle} className="absolute inset-0 w-full h-full object-contain" draggable={false} />
              </div>
              <div
                className="absolute top-0 bottom-0 z-10 flex items-center -translate-x-1/2 cursor-ew-resize group"
                style={{ left: `${sliderPos}%` }}
                onMouseDown={(e) => { e.stopPropagation(); setIsSliding(true); }}
              >
                <div className="w-1 h-full bg-white/50 backdrop-blur-sm shadow-md group-hover:bg-white transition-colors" />
                <div className="absolute grid w-12 h-12 text-white transition-transform rounded-full shadow-lg bg-primary place-items-center group-hover:scale-110">
                  <Icon name="chevronsLeftRight" size={24} />
                </div>
              </div>
            </>
          )}
          {isLoading && <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex flex-col justify-center items-center text-center p-4"><div className="loader"></div><p className="mt-4 text-foreground font-semibold">{loadingMessage}</p></div>}
        </div>
        {enhancedImage && !isLoading && (
          <div className="w-full bg-card border border-border rounded-lg p-3 mt-4 flex items-center justify-between gap-4 shadow-sm">
            <Label htmlFor="zoom-slider" className="text-sm font-medium">Zoom</Label>
            <Slider id="zoom-slider" min="1" max="4" step="0.05" value={zoom} onChange={handleZoom} className="flex-grow" />
            <span className="text-sm font-mono w-12 text-center">{zoom.toFixed(2)}x</span>
            <Button variant="secondary" onClick={resetZoomAndPan} className="h-8 px-3">Reset</Button>
          </div>
        )}
      </div>
    </div>
  );
};

interface HeaderProps { onEnhance: () => void; onUpload: (files: FileList) => void; onLibraryOpen: () => void; }
const Header = ({ onEnhance, onUpload, onLibraryOpen }: HeaderProps) => {
  const { state, dispatch } = useAppContext();
  const { isLoading, currentImage, enhancedImage, historyIndex, history, tools, currentPrompt, batchQueue } = state;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState<'png' | 'jpeg' | 'webp'>('png');
  const [exportQuality, setExportQuality] = useState(92);

  const hasPendingBatch = useMemo(() => batchQueue.some(item => item.status === 'pending'), [batchQueue]);
  const isProcessing = useMemo(() => isLoading || batchQueue.some(item => item.status === 'processing'), [isLoading, batchQueue]);
  const canEnhance = useMemo(() => (currentImage || hasPendingBatch) && !isProcessing && apiKey, [currentImage, hasPendingBatch, isProcessing, apiKey]);


  const download = async (type: "png" | "jpeg" | "webp" | "txt" | "json") => {
    if (!enhancedImage || !currentImage) return;
    const base = currentImage.name.replace(/\.[^.]+$/, "");
    if (type === "txt") { downloadFile(currentPrompt, `${base}-prompt.txt`, "text/plain"); } 
    else if (type === "json") { downloadFile(JSON.stringify(tools, null, 2), `${base}-recipe.json`, "application/json"); }
    else {
        let imageData = enhancedImage;
        if (type !== 'png') imageData = await convertImage(enhancedImage, `image/${type}`, exportQuality);
        const a = document.createElement("a"); a.href = imageData; a.download = `${base}-enhanced.${type}`; document.body.appendChild(a); a.click(); document.body.removeChild(a); 
    }
    setOpen(false);
  };
  
  const handleCopyPrompt = () => { if(!currentPrompt) return; navigator.clipboard.writeText(currentPrompt); dispatch({ type: "SHOW_TOAST", payload: { message: "Prompt copied to clipboard!", type: 'success' } }); };

  return (
    <header className="text-center mb-8 flex flex-col items-center">
      <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-foreground">AI Portrait Studio</h1>
      <p className="text-lg text-muted-foreground mt-2 max-w-2xl">Professional-quality portrait retouching powered by Gemini. Upload a photo to get started.</p>
      <div className="flex flex-wrap items-center justify-center gap-2 mt-6">
        <Button variant="secondary" onClick={() => fileInputRef.current?.click()} title="Upload New Image(s)"><Icon name="upload" />{currentImage ? "Add/Change" : "Upload"}</Button>
        <input ref={fileInputRef} type="file" className="hidden" accept="image/*" multiple onChange={(e) => e.target.files && onUpload(e.target.files)} />
        <Button variant="primary" onClick={onEnhance} disabled={!canEnhance} title="Enhance Image"><Icon name="enhance" />{isProcessing ? "Processing..." : (hasPendingBatch ? "Start Batch" : "Enhance")}</Button>
        <Button variant="secondary" onClick={() => dispatch({ type: "UNDO" })} disabled={historyIndex === 0} title="Undo (Ctrl+Z)" aria-label="Undo"><Icon name="undo" /></Button>
        <Button variant="secondary" onClick={() => dispatch({ type: "REDO" })} disabled={historyIndex === history.length - 1} title="Redo (Ctrl+Shift+Z)" aria-label="Redo"><Icon name="redo" /></Button>
        <Button variant="secondary" onClick={handleCopyPrompt} disabled={!enhancedImage} title="Copy Prompt" aria-label="Copy enhancement prompt"><Icon name="copy" /></Button>
        <div className="relative">
          <Button variant="secondary" onClick={() => setOpen(!open)} disabled={!enhancedImage} title="Download" aria-label="Download enhanced image"><Icon name="download" /></Button>
          {open && (<div className="absolute top-full mt-2 w-64 bg-card rounded-md shadow-lg z-10 text-left border border-border p-3">
            <div className="grid grid-cols-2 gap-2 mb-3">
              <Button variant={exportFormat === 'png' ? 'primary': 'secondary'} onClick={() => setExportFormat('png')} className="h-9">PNG</Button>
              <Button variant={exportFormat === 'jpeg' ? 'primary': 'secondary'} onClick={() => setExportFormat('jpeg')} className="h-9">JPEG</Button>
              <Button variant={exportFormat === 'webp' ? 'primary': 'secondary'} onClick={() => setExportFormat('webp')} className="h-9">WebP</Button>
            </div>
            {exportFormat !== 'png' && <div className="mb-3"><Label htmlFor="quality">Quality: {exportQuality}</Label><Slider id="quality" min="1" max="100" value={exportQuality} onChange={e => setExportQuality(+e.target.value)}/></div>}
            <Button variant="primary" className="w-full mb-2" onClick={() => download(exportFormat)}>Download Image</Button>
            <div className="flex gap-2"><Button onClick={() => download("txt")} className="flex-grow">Prompt (.txt)</Button><Button onClick={() => download("json")} className="flex-grow">Recipe (.json)</Button></div>
          </div>)}
        </div>
        <Button variant="secondary" onClick={() => dispatch({ type: "RESET_STATE" })} disabled={!currentImage} title="Reset All Tools" aria-label="Reset all tools"><Icon name="reset" /></Button>
        <Button variant="secondary" onClick={onLibraryOpen} title="Open Enhanced Library"><Icon name="library" />Library</Button>
      </div>
    </header>
  );
};

const BatchQueue = () => {
    const { state, dispatch } = useAppContext();
    const { batchQueue, userPresets, showBuiltInPresets, currentImage } = state;
    const availablePresets = useMemo(() => [
        { id: 'current', name: 'Current Tool Settings', tools: state.tools },
        ...userPresets,
        ...(showBuiltInPresets ? BUILT_IN_PRESETS : [])
    ], [userPresets, showBuiltInPresets, state.tools]);

    if(batchQueue.length === 0) return null;

    const handleApplyPreset = (e: ChangeEvent<HTMLSelectElement>) => {
        const id = e.target.value;
        const preset = availablePresets.find(p => p.id === id);
        if(preset) dispatch({ type: 'APPLY_PRESET_TO_BATCH', payload: { tools: preset.tools }});
    };
    
    const StatusBadge = ({status}: {status: BatchItem['status']}) => {
        const colors = { pending: 'bg-gray-200 text-gray-800', processing: 'bg-blue-200 text-blue-800 animate-pulse', done: 'bg-green-200 text-green-800', error: 'bg-red-200 text-red-800', cancelled: 'bg-yellow-200 text-yellow-800'};
        return <span className={`px-2 py-1 text-xs font-medium rounded-full ${colors[status]}`}>{status}</span>
    }

    return (
        <div className="fixed bottom-0 left-0 right-0 z-40">
            <div className="container mx-auto px-4 sm:px-6 lg:px-8">
                <Card className="bg-white/80 backdrop-blur-sm border-t-2 shadow-2xl max-h-[40vh] flex flex-col">
                    <header className="p-3 border-b flex flex-wrap items-center justify-between gap-3">
                        <h3 className="font-semibold text-lg">Batch Queue ({batchQueue.length})</h3>
                        <div className="flex items-center gap-2">
                           <Select onChange={handleApplyPreset} className="h-9 text-xs w-48"><option>Apply Preset to All...</option>{availablePresets.map(p=><option key={p.id} value={p.id}>{p.name}</option>)}</Select>
                           <Button onClick={()=>dispatch({type: 'CLEAR_BATCH'})} variant="destructive" className="h-9 text-xs">Clear</Button>
                        </div>
                    </header>
                    <div className="overflow-y-auto p-2 space-y-2">
                        {batchQueue.map(item => (
                            <div key={item.id} className={cn("flex items-center gap-3 p-2 bg-secondary/50 rounded-md transition-all cursor-pointer hover:bg-secondary", currentImage?.data === item.file.data && "ring-2 ring-primary")}
                                 onClick={() => dispatch({type: 'SELECT_BATCH_ITEM', payload: item })}>
                                <img src={item.result || item.file.data} className="w-12 h-12 object-cover rounded-md" />
                                <div className="flex-grow overflow-hidden"><p className="text-sm font-medium truncate">{item.file.name}</p><p className="text-xs text-muted-foreground truncate">{item.error || ''}</p></div>
                                <div className="flex items-center gap-2 flex-shrink-0">
                                  <StatusBadge status={item.status} />
                                  {item.status === 'done' && <Button variant="ghost" className="h-8 px-2" aria-label="View result" onClick={(e) => { e.stopPropagation(); dispatch({type: 'SET_ENHANCED_IMAGE', payload: item.result!})}}><Icon name="view" /></Button>}
                                  {(item.status === 'pending' || item.status === 'processing') && <Button variant="ghost" className="h-8 px-2" aria-label="Cancel processing" onClick={(e) => { e.stopPropagation(); dispatch({type: 'UPDATE_BATCH_ITEM', payload: {id: item.id, status: 'cancelled'}})}}><Icon name="close" /></Button>}
                                </div>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>
        </div>
    );
};

const LibraryModal = ({ open, onClose }: { open: boolean, onClose: () => void }) => {
    const { state, dispatch } = useAppContext();
    const { enhancedLibrary } = state;

    const handleSelect = (item: BatchItem) => {
        dispatch({ type: 'SELECT_BATCH_ITEM', payload: item });
        onClose();
    };

    const handleDelete = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        dispatch({ type: 'REMOVE_FROM_LIBRARY', payload: { id } });
    };

    const handleClear = () => {
        if (confirm('Are you sure you want to clear the entire library? This cannot be undone.')) {
            dispatch({ type: 'CLEAR_LIBRARY' });
        }
    };

    return (
        <Modal open={open} onClose={onClose} title={`Enhanced Library (${enhancedLibrary.length})`}>
            <div className="flex flex-col h-full">
                {enhancedLibrary.length > 0 && (
                    <div className="flex justify-end mb-4 border-b pb-4">
                        <Button variant="destructive" onClick={handleClear}>Clear All</Button>
                    </div>
                )}
                {enhancedLibrary.length === 0 ? (
                    <p className="text-center text-muted-foreground py-8">Your enhanced images will appear here.</p>
                ) : (
                    <div className="grid grid-cols-3 sm:grid-cols-4 gap-3 max-h-[60vh] overflow-y-auto pr-2 -mr-2">
                        {enhancedLibrary.map(item => item.result ? (
                            <div key={item.id} className="relative group cursor-pointer aspect-square" onClick={() => handleSelect(item)}>
                                <img src={item.result} alt={item.file.name} className="w-full h-full object-cover rounded-md bg-secondary" />
                                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center p-1">
                                    <p className="text-white text-xs text-center font-medium line-clamp-2">{item.file.name}</p>
                                </div>
                                <Button variant="destructive" className="absolute top-1 right-1 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity z-10" onClick={(e) => handleDelete(e, item.id)}>
                                    <Icon name="delete" size={14}/>
                                </Button>
                            </div>
                        ): null)}
                    </div>
                )}
            </div>
        </Modal>
    );
};


const Toast = () => {
  const { state, dispatch } = useAppContext();
  const { toast } = state;
  useEffect(() => {
    if (toast) { const t = setTimeout(() => dispatch({ type: "HIDE_TOAST" }), 4000); return () => clearTimeout(t); }
  }, [toast, dispatch]);
  if (!toast) return null;
  const isSuccess = toast.type === "success";
  return <div className={cn("fixed top-5 right-5 z-50 text-white font-bold rounded-lg shadow-lg p-4 animate-fade-in-down", isSuccess ? "bg-green-600" : "bg-destructive")}>{toast.message}</div>;
};

const App = () => {
  const { state, dispatch } = useAppContext();
  const { currentImage, tools, activeModel, batchQueue, enhancedLibrary } = state;
  const [isDragging, setIsDragging] = useState(false);
  const [isLibraryOpen, setLibraryOpen] = useState(false);

  useEffect(() => {
    if (!(window as any).React?.version?.startsWith("19.")) {
      console.error(
        `React version mismatch. Expected: 19.x, Loaded: ${(window as any).React?.version ?? 'unknown'}.`
      );
    }
  }, []);

  useEffect(() => {
      try {
        const storedPresets = localStorage.getItem('ai-portrait-presets');
        if (storedPresets) dispatch({ type: 'IMPORT_PRESETS', payload: { presets: JSON.parse(storedPresets) } });
        const storedLibrary = localStorage.getItem('ai-portrait-library');
        if (storedLibrary) dispatch({ type: 'LOAD_LIBRARY', payload: { library: JSON.parse(storedLibrary) } });
      } catch (e) { console.error("Could not load from localStorage", e)}
  }, []);

  useEffect(() => {
      localStorage.setItem('ai-portrait-presets', JSON.stringify(state.userPresets));
  }, [state.userPresets]);
  
  useEffect(() => {
      localStorage.setItem('ai-portrait-library', JSON.stringify(enhancedLibrary));
  }, [enhancedLibrary]);

  const processFiles = (files: FileList | File[]) => {
    const imageFiles: File[] = Array.from(files).filter(f => f.type.startsWith("image/"));
    if (imageFiles.length === 0) { dispatch({ type: "SHOW_TOAST", payload: { message: "No valid image files selected.", type: "error" } }); return; }
    
    const oversized = imageFiles.find(f => f.size > MAX_IMAGE_UPLOAD_SIZE_BYTES);
    if (oversized) { dispatch({ type: "SHOW_TOAST", payload: { message: `File ${oversized.name} is too large (max 15MB).`, type: "error" } }); return; }
    
    const fileReaders: Promise<ImageFile>[] = imageFiles.map(file => new Promise((resolve) => {
        const r = new FileReader();
        r.onloadend = () => resolve({ name: file.name, type: file.type, data: r.result as string });
        r.readAsDataURL(file);
    }));
    Promise.all(fileReaders).then(results => {
        if(!currentImage || imageFiles.length > 1) dispatch({ type: "SET_IMAGE", payload: results[0] });
        dispatch({ type: 'ADD_TO_BATCH', payload: {files: results }});
        dispatch({ type: "SHOW_TOAST", payload: { message: `${results.length} image(s) added to queue.`, type: "success" } });
    });
  };

  const handleEnhance = useCallback(async () => {
    if (batchQueue.some(item => item.status === 'pending')) {
        const pending = batchQueue.filter(item => item.status === 'pending');
        if (pending.length > 0) {
            dispatch({ type: "SHOW_TOAST", payload: { message: `Starting batch enhancement for ${pending.length} images.`, type: "success" } });
        }
        return; // The useEffect hook will pick up the pending items and process them.
    }
    
    if (!currentImage) return dispatch({ type: "SHOW_TOAST", payload: { message: "Please upload an image first.", type: "error" } });
    
    dispatch({ type: "ENHANCE_START" });
    try {
      const result = await enhanceImage(currentImage, tools, activeModel);
      dispatch({ type: "ENHANCE_SUCCESS", payload: result });
      const finishedItem: BatchItem = {
          id: `${currentImage.name}-${Date.now()}`, file: currentImage, status: 'done',
          result: result.image, prompt: result.prompt, preset: tools
      };
      dispatch({ type: "ADD_TO_LIBRARY", payload: { item: finishedItem } });
      dispatch({ type: "SHOW_TOAST", payload: { message: "Image enhanced and saved to library!", type: "success" } });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "An unknown error occurred.";
      dispatch({ type: "ENHANCE_FAILURE", payload: msg });
      dispatch({ type: "SHOW_TOAST", payload: { message: msg, type: "error" } });
    }
  }, [currentImage, tools, activeModel, dispatch, batchQueue]);
  
  useEffect(() => {
      const processingItems = batchQueue.filter(i => i.status === 'processing').length;
      const pendingItems = batchQueue.filter(i => i.status === 'pending');
      if (processingItems >= BATCH_CONCURRENCY || pendingItems.length === 0) return;

      const itemsToProcess = pendingItems.slice(0, BATCH_CONCURRENCY - processingItems);
      itemsToProcess.forEach(async item => {
          dispatch({ type: 'UPDATE_BATCH_ITEM', payload: { id: item.id, status: 'processing' }});
          try {
              const result = await enhanceImage(item.file, item.preset || tools, activeModel);
              const finishedItem: BatchItem = { ...item, status: 'done', result: result.image, prompt: result.prompt };
              dispatch({ type: 'UPDATE_BATCH_ITEM', payload: finishedItem });
              dispatch({ type: 'ADD_TO_LIBRARY', payload: { item: finishedItem } });
          } catch(err) {
              const msg = err instanceof Error ? err.message : "An unknown error occurred.";
              dispatch({ type: 'UPDATE_BATCH_ITEM', payload: { id: item.id, status: 'error', error: msg }});
          }
      });
  }, [batchQueue, tools, activeModel, dispatch]);

  const onDragOver = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const onDragLeave = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const onDrop = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); if (e.dataTransfer.files?.length) processFiles(e.dataTransfer.files); };
  const onPaste = (e: ClipboardEvent<HTMLDivElement>) => { if (e.clipboardData.files?.length) processFiles(e.clipboardData.files); };

  const hotkeyHandlers = useMemo(() => ({
    'z': () => dispatch({ type: 'UNDO' }),
    'Z': () => dispatch({ type: 'REDO' }),
    'u': () => dispatch({ type: 'UPDATE_TOOL', payload: { tool: 'superResolution', settings: { enabled: !state.tools.superResolution.enabled }}}),
    'r': () => dispatch({ type: 'UPDATE_TOOL', payload: { tool: 'facialRetouch', settings: { enabled: !state.tools.facialRetouch.enabled }}}),
    'c': () => dispatch({ type: 'UPDATE_TOOL', payload: { tool: 'crop', settings: { enabled: !state.tools.crop.enabled }}}),
    'b': () => dispatch({ type: 'UPDATE_TOOL', payload: { tool: 'background', settings: { enabled: !state.tools.background.enabled }}}),
  }), [dispatch, state.tools]);
  useHotkeys(hotkeyHandlers);

  return (
    <div className="min-h-screen p-4 sm:p-6 lg:p-8" onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave} onPaste={onPaste}>
      {!apiKey && <div className="bg-destructive text-destructive-foreground p-3 rounded-md text-center mb-4 fixed top-0 left-1/2 -translate-x-1/2 mt-4 z-50 shadow-lg animate-fade-in-down"><strong>Warning:</strong> API Key is not configured. The application will not function.</div>}
      <Header onEnhance={handleEnhance} onUpload={(files) => processFiles(files)} onLibraryOpen={() => setLibraryOpen(true)} />
      <main className="grid grid-cols-1 lg:grid-cols-3 gap-8 pb-48">
        <ToolsPanel />
        <div className="lg:col-span-2"><ImageViewer /></div>
      </main>
      <BatchQueue />
      <LibraryModal open={isLibraryOpen} onClose={() => setLibraryOpen(false)} />
      <Toast />
      {isDragging && (<div className="fixed inset-0 bg-primary/20 backdrop-blur-sm z-50 flex items-center justify-center pointer-events-none animate-fade-in"><div className="text-2xl font-bold text-primary-foreground p-8 bg-primary rounded-lg shadow-2xl">Drop your image(s) here</div></div>)}
    </div>
  );
};

function useHotkeys(h: Record<string, () => void>) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return;
      if ((e.metaKey || e.ctrlKey) && !e.altKey) {
        const key = e.shiftKey ? e.key.toUpperCase() : e.key;
        if (h[key]) { e.preventDefault(); h[key](); }
      } else if (h[e.key]) {
        e.preventDefault(); h[e.key]();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [h]);
}

const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(<React.StrictMode><AppProvider><App /></AppProvider></React.StrictMode>);
}