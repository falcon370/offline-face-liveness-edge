/**
 * useFaceAuth — wires the native peers into a ready-to-use FaceAuthSession.
 *
 * Responsibilities:
 *  - construct the encrypted EnrollmentStore (MMKV) and SyncQueue
 *  - load the TFLite models (MobileFaceNet + MiniFASNet) via fast-tflite
 *  - expose enroll()/verify() runners and a syncNow() that flushes + purges
 *
 * The single native seam that must be completed per-host is pixel extraction
 * for the TFLite models (turning the current camera frame into an aligned 112x112
 * / 80x80 tensor). We accept `embedNow`/`passiveScoreNow` callbacks so the host
 * can plug in a vision-camera frame processor or a takePhoto+resize path; the
 * example app shows the resize-plugin approach.
 */

import { useEffect, useMemo, useRef } from 'react';
import { MMKV } from 'react-native-mmkv';
import { useTensorflowModel } from 'react-native-fast-tflite';
import NetInfo from '@react-native-community/netinfo';

import { FaceAuthSession } from '../session';
import { EnrollmentStore } from '../storage/enrollmentStore';
import { SyncQueue, type RemoteUploader } from '../sync/syncQueue';
import type { EmbedNow, PassiveScoreNow } from '../session';
import type { ChallengeEngineConfig } from '../liveness/challengeEngine';

export interface UseFaceAuthOptions {
  /** Encryption key for the local biometric store (provision securely). */
  encryptionKey: string;
  /** Uploader for attendance sync (MockServer.uploader() in the demo). */
  uploader: RemoteUploader;
  /** Pixel→embedding seam (see module doc). */
  embedNow: EmbedNow;
  /** Pixel→passive-score seam. */
  passiveScoreNow: PassiveScoreNow;
  matchThreshold?: number;
  challengeConfig?: Partial<ChallengeEngineConfig>;
  /** Auto-flush the queue when connectivity returns. Default true. */
  autoSyncOnConnect?: boolean;
}

export interface FaceAuthApi {
  session: FaceAuthSession;
  store: EnrollmentStore;
  syncQueue: SyncQueue;
  /** Manually flush pending attendance records (uploads then purges). */
  syncNow: () => Promise<{ uploaded: number; remaining: number }>;
}

export function useFaceAuth(opts: UseFaceAuthOptions): FaceAuthApi {
  const recognition = useTensorflowModel(
    require('../../models/mobilefacenet.tflite'),
    'default'
  );
  const antiSpoof = useTensorflowModel(
    require('../../models/minifasnet_v2.tflite'),
    'default'
  );

  // Keep latest loaded models available to the injected seams if the host wants
  // to delegate inference to the library (optional convenience).
  const modelsRef = useRef({ recognition, antiSpoof });
  modelsRef.current = { recognition, antiSpoof };

  const api = useMemo<FaceAuthApi>(() => {
    const kv = new MMKV({ id: 'faceauth', encryptionKey: opts.encryptionKey });
    const store = new EnrollmentStore(kv);
    const syncQueue = new SyncQueue(kv, opts.uploader);
    const session = new FaceAuthSession({
      store,
      syncQueue,
      embedNow: opts.embedNow,
      passiveScoreNow: opts.passiveScoreNow,
      matchThreshold: opts.matchThreshold,
      challengeConfig: opts.challengeConfig,
    });
    return {
      session,
      store,
      syncQueue,
      syncNow: () => syncQueue.flush(),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opts.encryptionKey]);

  // Opportunistic sync: flush + purge whenever the device regains connectivity.
  useEffect(() => {
    if (opts.autoSyncOnConnect === false) return;
    const unsub = NetInfo.addEventListener((state) => {
      if (state.isConnected) {
        api.syncNow().catch(() => {
          /* keep queued; retried on next connect */
        });
      }
    });
    return unsub;
  }, [api, opts.autoSyncOnConnect]);

  return api;
}
