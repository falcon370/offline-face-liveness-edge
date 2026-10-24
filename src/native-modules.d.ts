/**
 * Loose ambient declarations for the native peer dependencies.
 *
 * These let the integration layer (hook + component) type-check in this library
 * package without the heavy native modules installed. The host app (Datalake)
 * installs the real packages, which ship their own precise types that take
 * precedence at app build time.
 */

declare module 'react-native-mediapipe' {
  import type { ComponentType } from 'react';

  export enum RunningMode {
    IMAGE = 0,
    VIDEO = 1,
    LIVE_STREAM = 2,
  }

  export interface Category {
    categoryName: string;
    score: number;
    index?: number;
    displayName?: string;
  }

  export interface NormalizedLandmark {
    x: number;
    y: number;
    z?: number;
  }

  export interface FaceLandmarkDetectionResultBundle {
    results: Array<{
      faceLandmarks: NormalizedLandmark[][];
      faceBlendshapes?: Array<{ categories: Category[] }>;
      facialTransformationMatrixes?: Array<{ data: number[] }>;
    }>;
    timestampMs?: number;
  }

  export interface UseFaceLandmarkOptions {
    onResults?: (bundle: FaceLandmarkDetectionResultBundle) => void;
    onError?: (e: unknown) => void;
    runningMode?: RunningMode;
    numFaces?: number;
    minFaceDetectionConfidence?: number;
    minFacePresenceConfidence?: number;
    minTrackingConfidence?: number;
  }

  export function useFaceLandmarkDetection(
    onResults: (bundle: FaceLandmarkDetectionResultBundle) => void,
    onError: (e: unknown) => void,
    runningMode: RunningMode,
    modelPath: string,
    options?: Partial<UseFaceLandmarkOptions>
  ): { cameraViewProps: Record<string, unknown> };

  export const MediapipeCamera: ComponentType<Record<string, unknown>>;
}

declare module 'react-native-fast-tflite' {
  export interface TensorflowModel {
    runSync(
      inputs: (Float32Array | Uint8Array | Int32Array)[]
    ): (Float32Array | Uint8Array | Int32Array)[];
    run(
      inputs: (Float32Array | Uint8Array | Int32Array)[]
    ): Promise<(Float32Array | Uint8Array | Int32Array)[]>;
  }
  export interface TensorflowModelState {
    state: 'loading' | 'loaded' | 'error';
    model?: TensorflowModel;
    error?: Error;
  }
  export function useTensorflowModel(
    source: number | { url: string },
    delegate?: 'default' | 'core-ml' | 'android-gpu' | 'nnapi'
  ): TensorflowModelState;
  export function loadTensorflowModel(
    source: number | { url: string },
    delegate?: string
  ): Promise<TensorflowModel>;
}

declare module 'react-native-mmkv' {
  export class MMKV {
    constructor(config?: { id?: string; encryptionKey?: string; path?: string });
    set(key: string, value: string | number | boolean): void;
    getString(key: string): string | undefined;
    getNumber(key: string): number | undefined;
    getBoolean(key: string): boolean | undefined;
    delete(key: string): void;
    getAllKeys(): string[];
    clearAll(): void;
  }
}

/**
 * Minimal 'react-native' surface used by the bundled UI component. The host app
 * has the full react-native types installed which override this at build time.
 */
declare module 'react-native' {
  import type { ComponentType, ReactNode } from 'react';
  export type ViewStyle = Record<string, unknown>;
  export type TextStyle = Record<string, unknown>;
  export interface ViewProps {
    style?: ViewStyle | ViewStyle[] | unknown;
    pointerEvents?: 'auto' | 'none' | 'box-none' | 'box-only';
    children?: ReactNode;
  }
  export interface TextProps {
    style?: TextStyle | TextStyle[] | unknown;
    children?: ReactNode;
  }
  export const View: ComponentType<ViewProps>;
  export const Text: ComponentType<TextProps>;
  export const StyleSheet: {
    create<T extends Record<string, unknown>>(styles: T): T;
    absoluteFill: ViewStyle;
  };
}

declare module '@react-native-community/netinfo' {
  export interface NetInfoState {
    isConnected: boolean | null;
    isInternetReachable: boolean | null;
    type: string;
  }
  export function fetch(): Promise<NetInfoState>;
  export function addEventListener(
    listener: (state: NetInfoState) => void
  ): () => void;
}
