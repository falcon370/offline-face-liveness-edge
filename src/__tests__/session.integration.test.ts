import { FaceAuthSession } from '../session';
import { EnrollmentStore, type KeyValueStore } from '../storage/enrollmentStore';
import { SyncQueue } from '../sync/syncQueue';
import { MockServer } from '../sync/mockEndpoint';
import { l2Normalize } from '../recognition/matcher';
import type { Blendshapes, ChallengeType, Embedding, FaceFrame, HeadPose } from '../types';

class MemKV implements KeyValueStore {
  private m = new Map<string, string>();
  set(k: string, v: string) { this.m.set(k, v); }
  getString(k: string) { return this.m.get(k); }
  delete(k: string) { this.m.delete(k); }
  getAllKeys() { return [...this.m.keys()]; }
}

function f(t: number, bs: Partial<Blendshapes> = {}, pose: Partial<HeadPose> = {}): FaceFrame {
  return {
    timestampMs: t,
    box: { x: 0.3, y: 0.3, width: 0.4, height: 0.4 },
    landmarks: [],
    blendshapes: { eyeBlinkLeft: 0, eyeBlinkRight: 0, mouthSmileLeft: 0, mouthSmileRight: 0, jawOpen: 0, ...bs },
    headPose: { yaw: 0, pitch: 0, roll: 0, ...pose },
    facePresent: true,
  };
}

/** Drive the engine through blink→smile (fixed order) to reach 'live'. */
function passLiveness(step: (frame: FaceFrame) => unknown) {
  let t = 0;
  step(f(t, { eyeBlinkLeft: 0.8, eyeBlinkRight: 0.8 }));
  step(f((t += 100), { eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 }));
  step(f((t += 100), { mouthSmileLeft: 0.8, mouthSmileRight: 0.8 }));
  step(f((t += 100), { mouthSmileLeft: 0.8, mouthSmileRight: 0.8 }));
  step(f((t += 100), { mouthSmileLeft: 0.8, mouthSmileRight: 0.8 }));
}

/** Drive an enrollment: calibrate the neutral pose, then move to each direction. */
function sweepEnroll(enroll: { step: (frame: FaceFrame) => unknown }) {
  let t = 0;
  // Look straight to auto-calibrate the neutral / capture center.
  for (let i = 0; i < 8; i++) enroll.step(f((t += 50), {}, { yaw: 0, pitch: 0 }));
  // Move the head toward each target in order: up, down, then both yaw sides
  // (opposite signs — the runner is mirror-agnostic about which is which).
  const moves: Partial<HeadPose>[] = [
    { yaw: 0, pitch: 20 }, // up
    { yaw: 0, pitch: -20 }, // down
    { yaw: 20, pitch: 0 }, // one side
    { yaw: -20, pitch: 0 }, // the other side
  ];
  for (const m of moves) {
    for (let k = 0; k < 2; k++) enroll.step(f((t += 50), {}, m));
  }
}

describe('FaceAuthSession (enroll → verify → sync)', () => {
  const aliceVec: Embedding = l2Normalize(Float32Array.from([1, 0, 0.1, 0]));
  const challengeConfig = { challenges: ['blink', 'smile'] as ChallengeType[], randomize: false, minPassiveScore: 0.5 };

  function makeSession() {
    const kv = new MemKV();
    const store = new EnrollmentStore(kv);
    const server = new MockServer();
    const syncQueue = new SyncQueue(kv, server.uploader({ latencyMs: 0 }));
    const session = new FaceAuthSession({
      store,
      syncQueue,
      embedNow: () => aliceVec,
      passiveScoreNow: () => 0.95,
      challengeConfig,
    });
    return { session, store, syncQueue, server };
  }

  it('enrolls a live face and then verifies the same identity', async () => {
    const { session, store, syncQueue, server } = makeSession();

    const enroll = session.startEnrollment('alice');
    // Sweep through the five head-pose regions to capture per-pose templates.
    sweepEnroll(enroll);
    expect(enroll.isDone()).toBe(true);
    const saved = store.get('alice');
    expect(saved).not.toBeNull();
    expect(saved!.templates).toHaveLength(5); // center + up/down/left/right

    const verify = session.startVerification();
    passLiveness((fr) => verify.step(fr));
    expect(verify.isDone()).toBe(true);

    const out = verify.complete({ deviceId: 'field-01' });
    expect(out).not.toBeNull();
    expect(out!.verification.matched).toBe(true);
    expect(out!.verification.personId).toBe('alice');
    expect(out!.liveness.outcome).toBe('live');

    // Attendance record was queued; flushing uploads then purges it.
    expect(syncQueue.pendingCount()).toBe(1);
    const r = await syncQueue.flush();
    expect(r.uploaded).toBe(1);
    expect(syncQueue.pendingCount()).toBe(0);
    expect(server.received.size).toBe(1);
  });

  it('rejects re-enrolling a face that is already in the gallery', () => {
    const { session, store } = makeSession();

    // First enrollment of this face succeeds.
    const first = session.startEnrollment('alice');
    sweepEnroll(first);
    expect(first.result()).not.toBeNull();
    expect(store.list()).toHaveLength(1);

    // Same face (embedNow still returns aliceVec), different name → duplicate.
    const second = session.startEnrollment('alice-again');
    sweepEnroll(second);

    expect(second.isDone()).toBe(true);
    expect(second.result()).toBeNull();
    const rej = second.rejection();
    expect(rej).not.toBeNull();
    expect(rej!.reason).toBe('duplicate');
    expect(rej!.personId).toBe('alice');
    // No second identity was written.
    expect(store.list()).toHaveLength(1);
    expect(store.get('alice-again')).toBeNull();
  });

  it('does not match an unknown face but still records the attempt', () => {
    const kv = new MemKV();
    const store = new EnrollmentStore(kv);
    store.save({ personId: 'alice', template: aliceVec, enrolledAt: '', sampleCount: 1 });

    // Probe a different identity at verify time.
    const other = l2Normalize(Float32Array.from([0, 1, 0, 0]));
    const session = new FaceAuthSession({
      store,
      syncQueue: new SyncQueue(kv, async (b) => b.map((x) => x.id)),
      embedNow: () => other,
      passiveScoreNow: () => 0.95,
      challengeConfig,
    });
    const verify = session.startVerification();
    passLiveness((fr) => verify.step(fr));
    const out = verify.complete();
    expect(out!.verification.matched).toBe(false);
    expect(out!.record.verified).toBe(false);
  });
});
