/**
 * High-level orchestration of the enroll and verify flows.
 *
 * FaceAuthSession is framework-agnostic: it consumes FaceFrame samples (from the
 * MediaPipe adapter) plus injected model runners (passive anti-spoof + embedder)
 * and drives the challenge engine, matching, and the attendance/sync record.
 * The React layer (LivenessCamera / useFaceAuth) is a thin wrapper over this.
 */

import { ChallengeEngine, type ChallengeEngineConfig } from './liveness/challengeEngine';
import { EnrollmentStore } from './storage/enrollmentStore';
import { SyncQueue } from './sync/syncQueue';
import { averageTemplate, identify, DEFAULT_MATCH_THRESHOLD } from './recognition/matcher';
import type {
  AttendanceRecord,
  ChallengeState,
  Embedding,
  EnrolledIdentity,
  FaceFrame,
  LivenessResult,
  VerificationResult,
} from './types';

/** Produces an embedding for the currently-aligned face crop (native-backed). */
export type EmbedNow = () => Embedding | null;
/** Produces the latest passive anti-spoof live-probability (0..1). */
export type PassiveScoreNow = () => number;

export interface FaceAuthDeps {
  store: EnrollmentStore;
  syncQueue: SyncQueue;
  embedNow: EmbedNow;
  passiveScoreNow: PassiveScoreNow;
  /** Cosine threshold for a positive match. */
  matchThreshold?: number;
  /**
   * Cosine threshold for rejecting a duplicate enrollment (the same face already
   * in the gallery). Defaults to `matchThreshold`. Set higher to be stricter
   * about what counts as "the same person".
   */
  duplicateThreshold?: number;
  challengeConfig?: Partial<ChallengeEngineConfig>;
  /** Id generator for attendance records (default: time+random). */
  newId?: () => string;
}

function defaultId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export class FaceAuthSession {
  constructor(private readonly deps: FaceAuthDeps) {}

  /** Begin a liveness-gated verification. Drive with `step()` per frame. */
  startVerification(): VerifyRunner {
    return new VerifyRunner(this.deps);
  }

  /**
   * Begin a multi-angle enrollment sweep for `personId`. The user is guided to
   * present several head poses (center/left/right/up/down); one template is
   * captured per pose. Drive with `step()` per frame.
   */
  startEnrollment(personId: string, config?: EnrollSweepConfig): EnrollRunner {
    return new EnrollRunner(this.deps, personId, config);
  }
}

/** Drives a single liveness challenge + recognition pass. */
export class VerifyRunner {
  private readonly engine: ChallengeEngine;
  private result: { liveness: LivenessResult; verification: VerificationResult } | null = null;

  constructor(private readonly deps: FaceAuthDeps) {
    this.engine = new ChallengeEngine(deps.challengeConfig);
  }

  /** Feed one frame. Returns the active challenge, or null once complete. */
  step(frame: FaceFrame): ChallengeState | null {
    this.engine.setPassiveScore(this.deps.passiveScoreNow());
    const state = this.engine.update(frame);
    if (this.engine.isDone() && this.result === null) this.finish();
    return state;
  }

  private finish(): void {
    const liveness = this.engine.result();
    if (liveness.outcome !== 'live') {
      this.result = {
        liveness,
        verification: {
          matched: false,
          personId: null,
          similarity: -1,
          threshold: this.deps.matchThreshold ?? DEFAULT_MATCH_THRESHOLD,
        },
      };
      return;
    }
    // Liveness passed -> capture embedding and match against the gallery.
    const probe = this.deps.embedNow();
    const verification: VerificationResult = probe
      ? identify(probe, this.deps.store.list(), this.deps.matchThreshold)
      : {
          matched: false,
          personId: null,
          similarity: -1,
          threshold: this.deps.matchThreshold ?? DEFAULT_MATCH_THRESHOLD,
        };
    this.result = { liveness, verification };
  }

  isDone(): boolean {
    return this.result !== null;
  }

  /**
   * Final result; also enqueues an attendance record for later sync/purge.
   * Returns null until the flow is complete.
   */
  complete(meta?: AttendanceRecord['meta']): {
    liveness: LivenessResult;
    verification: VerificationResult;
    record: AttendanceRecord;
  } | null {
    if (!this.result) return null;
    const record: AttendanceRecord = {
      id: (this.deps.newId ?? defaultId)(),
      personId: this.result.verification.personId ?? 'unknown',
      timestamp: new Date().toISOString(),
      verified: this.result.verification.matched,
      similarity: this.result.verification.similarity,
      liveness: this.result.liveness,
      meta,
    };
    this.deps.syncQueue.enqueue(record);
    return { ...this.result, record };
  }
}

/** Why an enrollment did not persist a new identity. */
export interface EnrollRejection {
  reason: 'duplicate';
  /** The already-enrolled identity this face matched. */
  personId: string;
  /** Cosine similarity to that identity. */
  similarity: number;
}

/** A head-movement direction captured during enrollment. */
export type EnrollDirection = 'up' | 'down' | 'left' | 'right';

export interface EnrollSweepConfig {
  /** Directions to capture, in prompt order. Default: up, down, left, right. */
  directions?: EnrollDirection[];
  /** Offset (deg) from the neutral pose required to count as a directional move. */
  moveDeg?: number;
  /** Max |yaw| (deg) tolerated while auto-calibrating the neutral pose. */
  calibYawDeg?: number;
  /** Frames of roughly-straight pose to average into the neutral reference. */
  calibrationFrames?: number;
  /** Consecutive frames a direction must be held before its sample is captured. */
  holdFrames?: number;
}

export const DEFAULT_ENROLL_SWEEP_CONFIG: Required<EnrollSweepConfig> = {
  directions: ['up', 'down', 'left', 'right'],
  // Measured relative to the auto-calibrated neutral, so up/down are symmetric
  // regardless of the camera's pitch bias. 12° is an easy, deliberate head move.
  moveDeg: 12,
  calibYawDeg: 12,
  calibrationFrames: 6,
  holdFrames: 1,
};

export type EnrollPhase = 'calibrating' | 'sweeping';

/** Progress snapshot returned from `EnrollRunner.step()`. */
export interface EnrollProgress {
  /** 'calibrating' = establishing the neutral/center; 'sweeping' = directions. */
  phase: EnrollPhase;
  /** 0..1 toward completion (captured directions / total). */
  progress: number;
  /** Directions captured so far. */
  captured: EnrollDirection[];
  /** The direction the head is currently moved toward this frame, or null. */
  current: EnrollDirection | null;
  /** The next direction to prompt for (first uncaptured), or null when done. */
  target: EnrollDirection | null;
}

/**
 * Drives an iPhone-Face-ID-style enrollment: the user looks straight to establish
 * a neutral reference (auto-calibrated, so a camera's pose bias doesn't matter),
 * then moves their head to each prompted direction (up/down/left/right). One
 * embedding is captured per direction. The on-demand head movement also serves as
 * liveness (a static photo can't move on cue). The captured per-direction
 * embeddings are stored as `templates`, so later matching can use the best pose —
 * fixing the "chin lifted = unrecognized" problem.
 */
export class EnrollRunner {
  private readonly cfg: Required<EnrollSweepConfig>;
  private readonly directions: EnrollDirection[];

  // Auto-calibrated neutral pose (set once enough straight frames are seen).
  private neutral: { yaw: number; pitch: number } | null = null;
  private calibSum = { yaw: 0, pitch: 0 };
  private calibCount = 0;

  private centerSample: Embedding | null = null;
  private readonly dirSamples = new Map<EnrollDirection, Embedding>();
  private holdDir: EnrollDirection | null = null;
  private holdCount = 0;
  // Yaw sign of the first side-turn captured; the second must be the opposite.
  private firstYawSign: number | null = null;

  private saved: EnrolledIdentity | null = null;
  private rejected: EnrollRejection | null = null;

  constructor(
    private readonly deps: FaceAuthDeps,
    private readonly personId: string,
    config: EnrollSweepConfig = {}
  ) {
    this.cfg = { ...DEFAULT_ENROLL_SWEEP_CONFIG, ...config };
    this.directions = this.cfg.directions;
  }

  /** Feed one frame; returns calibration/direction progress for the UI. */
  step(frame: FaceFrame): EnrollProgress {
    if (this.saved !== null || this.rejected !== null) return this.snapshot(null);

    if (!frame.facePresent) {
      this.holdDir = null;
      this.holdCount = 0;
      return this.snapshot(null);
    }

    const { yaw, pitch } = frame.headPose;

    // Phase 1 — calibrate the neutral pose from a few roughly-straight frames,
    // then capture the center template.
    if (this.neutral === null) {
      if (Math.abs(yaw) <= this.cfg.calibYawDeg) {
        this.calibSum.yaw += yaw;
        this.calibSum.pitch += pitch;
        this.calibCount++;
        if (this.calibCount >= this.cfg.calibrationFrames) {
          const e = this.deps.embedNow();
          if (e) {
            this.neutral = {
              yaw: this.calibSum.yaw / this.calibCount,
              pitch: this.calibSum.pitch / this.calibCount,
            };
            this.centerSample = e;
          }
        }
      } else {
        // Not straight enough — restart the calibration average.
        this.calibSum = { yaw: 0, pitch: 0 };
        this.calibCount = 0;
      }
      return this.snapshot(null);
    }

    // Phase 2 — target-driven capture. We only check whether the head has moved
    // far enough along the CURRENT target's axis, ignoring incidental drift on the
    // other axis (so a side-turn with a little pitch still counts as a side-turn).
    // The two side turns are mirror-agnostic: turn one way, then the other —
    // whichever physical side is which, they just need opposite yaw signs.
    const dy = yaw - this.neutral.yaw;
    const dp = pitch - this.neutral.pitch;
    const m = this.cfg.moveDeg;
    const target = this.directions.find((d) => !this.dirSamples.has(d)) ?? null;

    let moving = false;
    if (target === 'up') moving = dp >= m;
    else if (target === 'down') moving = dp <= -m;
    else if (target === 'left' || target === 'right') {
      if (Math.abs(dy) >= m) {
        const sign = dy >= 0 ? 1 : -1;
        // Require the opposite side from the first yaw turn already captured.
        moving = this.firstYawSign === null || sign !== this.firstYawSign;
      }
    }
    const current = moving ? target : null;

    if (current !== null) {
      if (current === this.holdDir) this.holdCount++;
      else {
        this.holdDir = current;
        this.holdCount = 1;
      }
      if (this.holdCount >= this.cfg.holdFrames) {
        const e = this.deps.embedNow();
        if (e) {
          this.dirSamples.set(current, e);
          if ((current === 'left' || current === 'right') && this.firstYawSign === null) {
            this.firstYawSign = dy >= 0 ? 1 : -1;
          }
          this.holdDir = null;
          this.holdCount = 0;
          if (this.captured().length >= this.directions.length) this.finalize();
        }
      }
    } else {
      this.holdDir = null;
      this.holdCount = 0;
    }

    return this.snapshot(current);
  }

  private captured(): EnrollDirection[] {
    return this.directions.filter((d) => this.dirSamples.has(d));
  }

  private snapshot(current: EnrollDirection | null): EnrollProgress {
    const captured = this.captured();
    const target = this.directions.find((d) => !this.dirSamples.has(d)) ?? null;
    return {
      phase: this.neutral === null ? 'calibrating' : 'sweeping',
      progress: captured.length / this.directions.length,
      captured,
      current,
      target,
    };
  }

  private finalize(): void {
    // center template first, then the captured directions in prompt order.
    const all: Embedding[] = [];
    if (this.centerSample) all.push(this.centerSample);
    for (const d of this.directions) {
      const s = this.dirSamples.get(d);
      if (s) all.push(s);
    }
    const template = averageTemplate(all);

    // Reject re-enrollment of a face already in the gallery: if this person's
    // average template matches an existing identity (best pose), it's the same
    // person enrolling again. We do not add a second record for them.
    const threshold = this.deps.duplicateThreshold ?? this.deps.matchThreshold;
    const existing = identify(template, this.deps.store.list(), threshold);
    if (existing.matched && existing.personId) {
      this.rejected = {
        reason: 'duplicate',
        personId: existing.personId,
        similarity: existing.similarity,
      };
      return;
    }

    const identity: EnrolledIdentity = {
      personId: this.personId,
      template,
      templates: all,
      enrolledAt: new Date().toISOString(),
      sampleCount: all.length,
    };
    this.deps.store.save(identity);
    this.saved = identity;
  }

  isDone(): boolean {
    return this.saved !== null || this.rejected !== null;
  }

  /** The enrolled identity, or null if nothing was saved (e.g. duplicate). */
  result(): EnrolledIdentity | null {
    return this.saved;
  }

  /** Set when enrollment was refused; null if it succeeded or is still running. */
  rejection(): EnrollRejection | null {
    return this.rejected;
  }
}
