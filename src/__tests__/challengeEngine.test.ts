import { ChallengeEngine } from '../liveness/challengeEngine';
import type { Blendshapes, FaceFrame, HeadPose } from '../types';

function frame(
  t: number,
  bs: Partial<Blendshapes> = {},
  pose: Partial<HeadPose> = {},
  facePresent = true
): FaceFrame {
  return {
    timestampMs: t,
    box: { x: 0.3, y: 0.3, width: 0.4, height: 0.4 },
    landmarks: [],
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0,
      mouthSmileRight: 0,
      jawOpen: 0,
      ...bs,
    },
    headPose: { yaw: 0, pitch: 0, roll: 0, ...pose },
    facePresent,
  };
}

describe('ChallengeEngine', () => {
  it('passes a fixed-order blink→smile→turnLeft sequence', () => {
    const engine = new ChallengeEngine({
      challenges: ['blink', 'smile', 'turnLeft'],
      randomize: false,
      minPassiveScore: 0.5,
    });
    engine.setPassiveScore(0.9);

    let t = 0;
    // blink: close then open
    expect(engine.update(frame(t, { eyeBlinkLeft: 0.8, eyeBlinkRight: 0.8 }))?.type).toBe('blink');
    engine.update(frame((t += 100), { eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 }));
    // now smile (hold 3 frames)
    expect(engine.update(frame((t += 100), { mouthSmileLeft: 0.8, mouthSmileRight: 0.8 }))?.type).toBe('smile');
    engine.update(frame((t += 100), { mouthSmileLeft: 0.8, mouthSmileRight: 0.8 }));
    engine.update(frame((t += 100), { mouthSmileLeft: 0.8, mouthSmileRight: 0.8 }));
    // turnLeft
    const afterSmile = engine.update(frame((t += 100), {}, { yaw: -30 }));
    // last challenge satisfied -> done
    expect(engine.isDone()).toBe(true);
    expect(engine.result().outcome).toBe('live');
    expect(engine.result().completed).toEqual(['blink', 'smile', 'turnLeft']);
    // state is null once complete
    expect(afterSmile).toBeNull();
  });

  it('fails as spoof when the passive score stays low', () => {
    const engine = new ChallengeEngine({
      challenges: ['blink'],
      randomize: false,
      minPassiveScore: 0.6,
      passiveSpoofFrames: 3,
    });
    for (let i = 0; i < 3; i++) {
      engine.setPassiveScore(0.2);
      engine.update(frame(i * 100, { eyeBlinkLeft: 0.8, eyeBlinkRight: 0.8 }));
    }
    expect(engine.result().outcome).toBe('spoof');
  });

  it('survives a single low passive frame (transient bad crop)', () => {
    const engine = new ChallengeEngine({
      challenges: ['blink'],
      randomize: false,
      minPassiveScore: 0.6,
      passiveSpoofFrames: 3,
    });
    engine.setPassiveScore(0.9);
    engine.update(frame(0));
    engine.setPassiveScore(0.2); // one-frame dip
    engine.update(frame(100));
    engine.setPassiveScore(0.9);
    engine.update(frame(200));
    expect(engine.result().outcome).not.toBe('spoof');
  });

  it('does not gate on passive score when minPassiveScore is 0', () => {
    const engine = new ChallengeEngine({
      challenges: ['blink'],
      randomize: false,
      minPassiveScore: 0,
      passiveSpoofFrames: 1,
    });
    for (let i = 0; i < 5; i++) {
      engine.setPassiveScore(0);
      engine.update(frame(i * 100));
    }
    expect(engine.result().outcome).not.toBe('spoof');
  });

  it('times out if a challenge is never completed', () => {
    const engine = new ChallengeEngine({
      challenges: ['blink'],
      randomize: false,
      perChallengeTimeoutMs: 1000,
      minPassiveScore: 0,
    });
    engine.update(frame(0));
    engine.update(frame(2000)); // past timeout, no blink
    expect(engine.result().outcome).toBe('timeout');
  });

  it('does not count a half-blink (close without re-open)', () => {
    const engine = new ChallengeEngine({ challenges: ['blink'], randomize: false, minPassiveScore: 0 });
    engine.update(frame(0, { eyeBlinkLeft: 0.9, eyeBlinkRight: 0.9 }));
    engine.update(frame(100, { eyeBlinkLeft: 0.9, eyeBlinkRight: 0.9 }));
    expect(engine.isDone()).toBe(false);
  });

  it('turnRight requires positive yaw beyond threshold', () => {
    const engine = new ChallengeEngine({ challenges: ['turnRight'], randomize: false, turnYawDeg: 20, minPassiveScore: 0 });
    expect(engine.update(frame(0, {}, { yaw: 10 }))?.type).toBe('turnRight'); // not enough
    expect(engine.isDone()).toBe(false);
    engine.update(frame(100, {}, { yaw: 25 }));
    expect(engine.result().outcome).toBe('live');
  });
});
