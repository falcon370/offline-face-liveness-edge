/**
 * Demo wiring for react-native-face-auth.
 *
 * This is a local replacement for the library's `useFaceAuth` hook, so this demo
 * can load both TFLite models directly and hand them to `FaceAuthCamera` alongside
 * the pure session/store/sync objects.
 *
 * The pure pipeline logic (session, store, sync, matcher) is imported straight
 * from the sibling library source at ../../src — those modules have no native
 * imports; they take the native objects (MMKV, the TFLite model) via injection.
 */

import { useEffect, useMemo, useRef } from 'react';
import { MMKV } from 'react-native-mmkv';
import { useTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite';
import NetInfo from '@react-native-community/netinfo';

import { FaceAuthSession } from '../../src/session';
import { EnrollmentStore } from '../../src/storage/enrollmentStore';
import { SyncQueue } from '../../src/sync/syncQueue';
import { MockServer } from '../../src/sync/mockEndpoint';
import type { Embedding } from '../../src/types';

const ENCRYPTION_KEY = 'demo-key-rotate-in-production';

export interface FaceAuthDemoApi {
  session: FaceAuthSession;
  store: EnrollmentStore;
  syncQueue: SyncQueue;
  mockServer: MockServer;
  /** The loaded MobileFaceNet model (undefined until ready); passed to the camera worklet. */
  model: TensorflowModel | undefined;
  modelState: string;
  /** The loaded MiniFASNet v2 model (undefined until ready); passed to the camera worklet. */
  fasModel: TensorflowModel | undefined;
  /** The camera worklet writes the latest embedding here; embedNow() reads it. */
  embeddingRef: React.MutableRefObject<Embedding | null>;
  /** Written each frame with MiniFASNet P(live); read by session.passiveScoreNow(). */
  passiveScoreRef: React.MutableRefObject<number>;
  syncNow: () => Promise<{ uploaded: number; remaining: number }>;
}

export function useFaceAuthDemo(): FaceAuthDemoApi {
  const recognition = useTensorflowModel(
    require('../assets/models/mobilefacenet.tflite'),
    'default'
  );
  const model = recognition.state === 'loaded' ? recognition.model : undefined;

  const antiSpoof = useTensorflowModel(
    require('../assets/models/minifasnet_v2.tflite'),
    'default'
  );
  const fasModel = antiSpoof.state === 'loaded' ? antiSpoof.model : undefined;

  // The seam: the frame processor stashes the most recent embedding here, and
  // the session's embedNow() reads it once active liveness passes.
  const embeddingRef = useRef<Embedding | null>(null);
  // Passive liveness score: starts at 1 (passes the gate) until the first real
  // MiniFASNet score arrives. The camera worklet overwrites this every frame
  // once fasModel is loaded.
  const passiveScoreRef = useRef<number>(1);

  const api = useMemo<Omit<FaceAuthDemoApi, 'model' | 'modelState' | 'fasModel' | 'passiveScoreRef'>>(() => {
    const kv = new MMKV({ id: 'faceauth', encryptionKey: ENCRYPTION_KEY });
    const store = new EnrollmentStore(kv);
    const mockServer = new MockServer();
    const syncQueue = new SyncQueue(kv, mockServer.uploader({ latencyMs: 200 }));
    const session = new FaceAuthSession({
      store,
      syncQueue,
      embedNow: () => embeddingRef.current,
      passiveScoreNow: () => passiveScoreRef.current,
      matchThreshold: 0.4,
      // Blink + smile only — unambiguous and avoids front-camera yaw-sign issues.
      // Thresholds tuned for MLKit's smiling / eye-open probabilities.
      challengeConfig: {
        challenges: ['blink', 'smile'],
        randomize: false, // blink first, then smile — predictable for the demo
        smileThreshold: 0.4,
        // Tuned against live 12fps MLKit data on this front camera:
        //  - a firm blink peaks at closed ~0.45-0.59, so 0.40 catches it; smile-
        //    induced eye-narrowing only reaches ~0.30, so it won't false-trigger.
        //  - during active use the eyes-open floor sits ~0.22-0.28 (not ~0.05), so
        //    the re-open gate must be 0.28, not lower, or it never registers "open".
        blinkClosedThreshold: 0.4,
        blinkOpenThreshold: 0.28,
        // Generous per-challenge window. At 15s a user reading the prompt could
        // time out, which silently fails liveness — and in enroll that left the
        // screen stuck on "Please smile". FaceAuthCamera also auto-restarts on a
        // timeout, but a long window means it rarely triggers in the first place.
        perChallengeTimeoutMs: 60000,
        // Passive anti-spoof gate DISABLED (0 = never fails on passive score).
        //
        // Measured on-device (RMX5003, front camera): MiniFASNet emits a constant
        // saturated [0.000, 0.004, 0.995] for a live, moving face — byte-identical
        // across every frame sampled. The output does not respond to its input, so
        // the gate rejected 100% of genuine faces while detecting no spoofs.
        // ChallengeEngine.setPassiveScore() latches `spoof` permanently on the first
        // sub-threshold frame, so verification died ~2 frames in, every time.
        //
        // Do NOT "fix" this by pointing LIVE_CLASS_INDEX at class 2 — a constant
        // output means a print/replay attack scores identically to a live face, so
        // that would only disguise the gate as working. Re-enable (0.6) once the
        // model is validated against a labeled print/replay set. Prime suspects:
        // the reference implementation feeds BGR at a 2.7x context crop, while
        // FaceAuthCamera feeds RGB at 1.15x. See FaceAuthDemo/README.md's known caveat.
        minPassiveScore: 0,
      },
    });
    return { session, store, syncQueue, mockServer, embeddingRef, syncNow: () => syncQueue.flush() };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Opportunistic sync + purge whenever connectivity returns.
  useEffect(() => {
    const unsub = NetInfo.addEventListener((state) => {
      if (state.isConnected) api.syncNow().catch(() => {});
    });
    return unsub;
  }, [api]);

  return { ...api, model, modelState: recognition.state, fasModel, passiveScoreRef };
}
