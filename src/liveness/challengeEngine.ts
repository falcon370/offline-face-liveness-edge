/**
 * Active-liveness challenge engine.
 *
 * Issues a randomized sequence of human-verifiable challenges (blink, smile,
 * turn head) and confirms each from per-frame MediaPipe blendshapes + head pose.
 * This defeats static-photo attacks (a photo cannot blink on demand) and the
 * randomized order + per-challenge time window defeats pre-recorded replays.
 *
 * Pure logic: feed it FaceFrame samples via `update()`; it returns the current
 * state. Combine with the passive MiniFASNet score (see passive.ts) for the
 * final live/spoof decision — both must pass.
 */

import type {
  ChallengeState,
  ChallengeType,
  FaceFrame,
  LivenessResult,
} from '../types';

export interface ChallengeEngineConfig {
  /** Ordered/!randomized set of challenges to require. */
  challenges: ChallengeType[];
  /** Randomize challenge order on construction. Default true. */
  randomize: boolean;
  /** Max ms allowed per challenge before the whole flow fails. */
  perChallengeTimeoutMs: number;
  /** Blendshape value above which an eye is considered closed. */
  blinkClosedThreshold: number;
  /** Blendshape value below which an eye is considered open (hysteresis). */
  blinkOpenThreshold: number;
  /** Mean smile blendshape required to count as a smile. */
  smileThreshold: number;
  /** Absolute yaw (deg) required to count as a head turn. */
  turnYawDeg: number;
  /** Passive anti-spoof score required throughout (0..1). Set 0 to disable. */
  minPassiveScore: number;
  /**
   * Consecutive sub-threshold frames required before declaring a spoof.
   *
   * A single frame is not enough evidence: the anti-spoof model sees one 80x80
   * crop, and a momentary bad crop (mid-blink, motion blur, a hand crossing the
   * face) can dip one frame's score without the subject being a spoof. Latching
   * on the first dip made a whole session unrecoverable, since the outcome never
   * returns to `inProgress`. Requiring a short run keeps genuine attacks caught
   * (they score low on every frame, not one) while surviving transient noise.
   */
  passiveSpoofFrames: number;
}

export const DEFAULT_CHALLENGE_CONFIG: ChallengeEngineConfig = {
  challenges: ['blink', 'smile', 'turnLeft', 'turnRight'],
  randomize: true,
  perChallengeTimeoutMs: 6000,
  blinkClosedThreshold: 0.5,
  blinkOpenThreshold: 0.2,
  smileThreshold: 0.5,
  turnYawDeg: 20,
  minPassiveScore: 0.6,
  passiveSpoofFrames: 3,
};

function shuffle<T>(arr: T[], rng: () => number): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export class ChallengeEngine {
  private readonly cfg: ChallengeEngineConfig;
  private readonly order: ChallengeType[];
  private index = 0;
  private currentStartMs: number | null = null;
  private completed: ChallengeType[] = [];
  private outcome: LivenessResult['outcome'] = 'inProgress';
  private lastPassiveScore = 1;
  private lowPassiveFrames = 0;

  // Per-challenge transient detector state.
  private blinkPhase: 'waitClose' | 'waitOpen' = 'waitClose';
  private smileHoldFrames = 0;

  constructor(config: Partial<ChallengeEngineConfig> = {}, rng: () => number = Math.random) {
    this.cfg = { ...DEFAULT_CHALLENGE_CONFIG, ...config };
    this.order = this.cfg.randomize
      ? shuffle(this.cfg.challenges, rng)
      : [...this.cfg.challenges];
  }

  /**
   * Set the latest passive anti-spoof score; a sustained low score fails the
   * whole flow. `minPassiveScore: 0` disables the gate entirely.
   */
  setPassiveScore(score: number): void {
    this.lastPassiveScore = score;
    if (this.outcome !== 'inProgress' || this.cfg.minPassiveScore <= 0) return;
    if (score < this.cfg.minPassiveScore) {
      this.lowPassiveFrames++;
      if (this.lowPassiveFrames >= this.cfg.passiveSpoofFrames) this.outcome = 'spoof';
    } else {
      // One good frame clears the run — a spoof cannot produce them.
      this.lowPassiveFrames = 0;
    }
  }

  /**
   * Advance the state machine with one frame. Returns the current challenge
   * state (or null when the flow is complete).
   */
  update(frame: FaceFrame): ChallengeState | null {
    if (this.outcome !== 'inProgress') return this.currentState();
    if (this.index >= this.order.length) {
      this.outcome = 'live';
      return null;
    }

    if (this.currentStartMs === null) this.currentStartMs = frame.timestampMs;

    // Timeout guard.
    if (frame.timestampMs - this.currentStartMs > this.cfg.perChallengeTimeoutMs) {
      this.outcome = 'timeout';
      return this.currentState();
    }

    // Require a single present face throughout; absence resets detector phase.
    if (!frame.facePresent) {
      this.resetDetectors();
      return this.currentState();
    }

    const challenge = this.order[this.index];
    const done = this.evaluate(challenge, frame);
    if (done) this.advance(challenge);
    return this.currentState();
  }

  private evaluate(type: ChallengeType, frame: FaceFrame): boolean {
    switch (type) {
      case 'blink':
        return this.evaluateBlink(frame);
      case 'smile':
        return this.evaluateSmile(frame);
      case 'turnLeft':
        return frame.headPose.yaw <= -this.cfg.turnYawDeg;
      case 'turnRight':
        return frame.headPose.yaw >= this.cfg.turnYawDeg;
    }
  }

  /** A valid blink = both eyes close past threshold, then re-open (full cycle). */
  private evaluateBlink(frame: FaceFrame): boolean {
    const closed =
      (frame.blendshapes.eyeBlinkLeft + frame.blendshapes.eyeBlinkRight) / 2;
    if (this.blinkPhase === 'waitClose') {
      if (closed >= this.cfg.blinkClosedThreshold) this.blinkPhase = 'waitOpen';
      return false;
    }
    // waitOpen
    if (closed <= this.cfg.blinkOpenThreshold) return true;
    return false;
  }

  /** A valid smile = mean smile blendshape held above threshold for a few frames. */
  private evaluateSmile(frame: FaceFrame): boolean {
    const smile =
      (frame.blendshapes.mouthSmileLeft + frame.blendshapes.mouthSmileRight) / 2;
    if (smile >= this.cfg.smileThreshold) {
      this.smileHoldFrames++;
    } else {
      this.smileHoldFrames = 0;
    }
    return this.smileHoldFrames >= 3;
  }

  private advance(challenge: ChallengeType): void {
    this.completed.push(challenge);
    this.index++;
    this.currentStartMs = null;
    this.resetDetectors();
    if (this.index >= this.order.length) this.outcome = 'live';
  }

  private resetDetectors(): void {
    this.blinkPhase = 'waitClose';
    this.smileHoldFrames = 0;
  }

  private currentState(): ChallengeState | null {
    if (this.index >= this.order.length) return null;
    const type = this.order[this.index];
    let status: ChallengeState['status'] = 'active';
    if (this.outcome === 'timeout' || this.outcome === 'spoof') status = 'failed';
    return { type, status, progress: this.progressFor(type) };
  }

  private progressFor(type: ChallengeType): number {
    if (type === 'smile') return Math.min(1, this.smileHoldFrames / 3);
    if (type === 'blink') return this.blinkPhase === 'waitOpen' ? 0.5 : 0;
    return 0;
  }

  /** Final result snapshot. */
  result(): LivenessResult {
    return {
      outcome: this.outcome,
      passiveScore: this.lastPassiveScore,
      completed: [...this.completed],
    };
  }

  isDone(): boolean {
    return this.outcome !== 'inProgress';
  }
}
