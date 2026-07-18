import type { ModelPreset } from "./models.js";
import { formatMB } from "./models.js";

type WebGpuLimitMap = Record<string, number | string>;

export type HardwareSnapshot = {
  webgpuSupported: boolean;
  webgpuReason: string;
  adapterInfo?: Record<string, unknown>;
  features: string[];
  limits: WebGpuLimitMap;
  deviceMemoryGB?: number;
  logicalProcessors?: number;
  storageQuotaBytes?: number;
  storageUsageBytes?: number;
  secureContext: boolean;
  crossOriginIsolated: boolean;
  userAgent: string;
  platform?: string;
};

/**
 * Whether this is a Chromium-based browser (Chrome, Edge, Opera, Brave, …), per the UA-Client-Hints
 * brand list. onnxruntime-web's WebGPU backend is developed and tested against Chromium; Firefox's
 * and Safari's newer WebGPU implementations still fail on some of its generated WGSL and its
 * quantized-weight session transforms (errors that only surface at load/inference time), so
 * wasm-capable runtimes prefer wasm off-Chromium even when the WebGPU probe passes.
 * navigator.userAgentData is itself Chromium-only, and every Chromium new enough for WebGPU (113+)
 * ships it, so the brand list is a reliable signal.
 */
export function isChromiumBased(): boolean {
  if (typeof navigator === "undefined") return false;
  const nav = navigator as Navigator & { userAgentData?: { brands?: Array<{ brand: string }> } };
  return nav.userAgentData?.brands?.some((entry) => entry.brand === "Chromium") ?? false;
}

const LIMIT_NAMES = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupsPerDimension",
  "maxBindGroups",
  "maxBindingsPerBindGroup",
  "maxStorageBuffersPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxTextureDimension2D",
];

export async function collectHardwareSnapshot(): Promise<HardwareSnapshot> {
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    userAgentData?: { platform?: string };
  };

  // Read off globalThis (not window) so the probe also works in a worker global scope.
  const scope = globalThis as typeof globalThis & { isSecureContext?: boolean; crossOriginIsolated?: boolean };

  const base: HardwareSnapshot = {
    webgpuSupported: false,
    webgpuReason: "navigator.gpu is not available.",
    features: [],
    limits: {},
    deviceMemoryGB: nav.deviceMemory,
    logicalProcessors: nav.hardwareConcurrency,
    secureContext: scope.isSecureContext === true,
    crossOriginIsolated: scope.crossOriginIsolated === true,
    userAgent: navigator.userAgent,
    platform: nav.userAgentData?.platform ?? navigator.platform,
  };

  try {
    const estimate = await navigator.storage?.estimate?.();
    base.storageQuotaBytes = estimate?.quota;
    base.storageUsageBytes = estimate?.usage;
  } catch {
    // Storage quota is a best-effort browser hint; ignore unsupported/blocked cases.
  }

  if (!("gpu" in navigator)) {
    return base;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { ...base, webgpuReason: "WebGPU exists, but no adapter was returned." };
    }

    const limits = adapter.limits as unknown as Record<string, number>;
    const pickedLimits: WebGpuLimitMap = {};
    for (const name of LIMIT_NAMES) {
      if (typeof limits[name] === "number") {
        pickedLimits[name] = limits[name];
      }
    }

    let canCreateDevice = false;
    try {
      const device = await adapter.requestDevice();
      canCreateDevice = true;
      device.destroy();
    } catch {
      canCreateDevice = false;
    }

    const info = adapter.info as unknown as Record<string, unknown> | undefined;
    return {
      ...base,
      webgpuSupported: canCreateDevice,
      webgpuReason: canCreateDevice ? "WebGPU adapter and device are available." : "Adapter exists, but requestDevice() failed.",
      adapterInfo: info,
      features: Array.from(adapter.features).sort(),
      limits: pickedLimits,
    };
  } catch (error) {
    return {
      ...base,
      webgpuReason: error instanceof Error ? error.message : String(error),
    };
  }
}

export function modelSafetyHint(snapshot: HardwareSnapshot, model: ModelPreset): string {
  const vram = formatMB(model.vramRequiredMB);
  const maxStorage = numberLimit(snapshot, "maxStorageBufferBindingSize");
  const deviceMemory = snapshot.deviceMemoryGB;

  const sourceText =
    model.vramSource === "estimated"
      ? `This custom model is configured with an estimated GPU memory requirement of about ${vram}.`
      : `WebLLM metadata says this model requires about ${vram} of GPU memory.`;

  const parts = [
    sourceText,
    "Browsers do not expose total or free VRAM, so this is a heuristic; the actual load attempt is the definitive test.",
  ];

  if (typeof maxStorage === "number") {
    parts.push(`Adapter maxStorageBufferBindingSize is ${formatBytes(maxStorage)}.`);
  }

  if (typeof deviceMemory === "number") {
    parts.push(`navigator.deviceMemory reports about ${deviceMemory} GB of system memory, but this is not VRAM.`);
  }

  return parts.join(" ");
}

export function formatHardwareSnapshot(snapshot: HardwareSnapshot, model?: ModelPreset): string {
  const lines = [
    `WebGPU: ${snapshot.webgpuSupported ? "available" : "unavailable"}`,
    `Reason: ${snapshot.webgpuReason}`,
    `Secure context: ${snapshot.secureContext ? "yes" : "no"}`,
    `crossOriginIsolated: ${snapshot.crossOriginIsolated ? "yes" : "no"}`,
    `Logical processors: ${snapshot.logicalProcessors ?? "unknown"}`,
    `Approx system memory: ${snapshot.deviceMemoryGB ? `${snapshot.deviceMemoryGB} GB` : "unavailable"}`,
    `Browser storage: ${formatStorage(snapshot)}`,
    `Platform: ${snapshot.platform ?? "unknown"}`,
  ];

  if (snapshot.adapterInfo && Object.keys(snapshot.adapterInfo).length > 0) {
    lines.push("", "GPU adapter info:", JSON.stringify(snapshot.adapterInfo, null, 2));
  }

  if (snapshot.features.length > 0) {
    lines.push("", `GPU features (${snapshot.features.length}):`, snapshot.features.join(", "));
  }

  if (Object.keys(snapshot.limits).length > 0) {
    lines.push("", "Selected GPU limits:");
    for (const [key, value] of Object.entries(snapshot.limits)) {
      lines.push(`  ${key}: ${typeof value === "number" && key.toLowerCase().includes("size") ? formatBytes(value) : value}`);
    }
  }

  if (model) {
    lines.push("", "Selected model safety hint:", modelSafetyHint(snapshot, model));
  }

  return lines.join("\n");
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return "unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatStorage(snapshot: HardwareSnapshot): string {
  if (typeof snapshot.storageQuotaBytes !== "number") {
    return "unavailable";
  }
  const usage = typeof snapshot.storageUsageBytes === "number" ? formatBytes(snapshot.storageUsageBytes) : "unknown";
  return `${usage} used / ${formatBytes(snapshot.storageQuotaBytes)} quota`;
}

function numberLimit(snapshot: HardwareSnapshot, key: string): number | undefined {
  const value = snapshot.limits[key];
  return typeof value === "number" ? value : undefined;
}
