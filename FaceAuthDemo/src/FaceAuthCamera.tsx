/**
 * FaceAuthCamera — the native seam, implemented.
 *
 * A single react-native-vision-camera feeds one frame processor that:
 *   1. runs MLKit face detection (react-native-vision-camera-face-detector) to
 *      get per-frame eye-open / smile / head-angle signals for active liveness;
 *   2. crops+resizes the face to 112x112 and runs MobileFaceNet (fast-tflite) to
 *      produce the recognition embedding;
 * then hops back to JS (useRunOnJS) to (a) stash the latest embedding for
 * embedNow() and (b) step the liveness/verify FSM with a FaceFrame.
 *
 * The MLKit signals are mapped onto the library's MediaPipe-shaped Blendshapes at
 * this seam, so the pure ChallengeEngine / matcher are reused unchanged.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, Linking, Dimensions, type TextStyle } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useFrameProcessor,
  runAtTargetFps,
  type CameraRuntimeError,
} from 'react-native-vision-camera';
import { useFaceDetector } from 'react-native-vision-camera-face-detector';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRunOnJS } from 'react-native-worklets-core';
import type { TensorflowModel } from 'react-native-fast-tflite';

import { l2Normalize } from '../../src/recognition/matcher';
import { isFaceUsable } from '../../src/frame/preprocess';
import { CornerBrackets, ScanLine, NEON } from './hud';
import type {
  VerifyRunner,
  EnrollRunner,
  FaceAuthSession,
  EnrollRejection,
  EnrollDirection,
} from '../../src/session';
import type { Embedding, FaceFrame, ChallengeType, EnrolledIdentity } from '../../src/types';
import type { VerifyResult } from '../../src/components/LivenessCamera';

const PROMPTS: Record<ChallengeType, string> = {
  blink: 'Please blink',
  smile: 'Please smile 😊',
  turnLeft: 'Turn your head left',
  turnRight: 'Turn your head right',
};

const DIR_PROMPT: Record<EnrollDirection, string> = {
  up: 'Tilt your head up',
  down: 'Tilt your head down',
  // Mirror-agnostic: the user just turns one way, then the other.
  left: 'Turn your head to one side',
  right: 'Now turn to the other side',
};

interface FaceData {
  box: { x: number; y: number; width: number; height: number };
  yaw: number;
  pitch: number;
  roll: number;
  leftEyeOpen: number;
  rightEyeOpen: number;
  smiling: number;
}

export interface FaceAuthCameraProps {
  session: FaceAuthSession;
  model: TensorflowModel | undefined;
  /** MiniFASNet v2 passive anti-spoof model (optional; skipped if undefined). */
  fasModel?: TensorflowModel;
  embeddingRef: React.MutableRefObject<Embedding | null>;
  /** Written each frame with MiniFASNet P(live); read by session.passiveScoreNow(). */
  passiveScoreRef?: React.MutableRefObject<number>;
  mode: 'verify' | 'enroll';
  personId?: string;
  onVerified?: (r: VerifyResult) => void;
  onEnrolled?: (id: EnrolledIdentity) => void;
  /** Called when enrollment is refused (e.g. this face is already enrolled). */
  onEnrollRejected?: (r: EnrollRejection) => void;
}

const clamp01 = (n: number) => (n < 0 ? 0 : n > 1 ? 1 : n);

// Frontal-pose gate. "Side-wise" is specifically a left/right head turn = yaw, so
// we gate ONLY on yaw. Pitch (look up/down) and roll (head tilt) are not profile
// turns — and critically, MLKit's pitch estimate jumps when the eyes close, so
// including pitch would flag a blinking, forward-facing user as "turned" right as
// they blink. Yaw is also the axis that actually degrades face recognition.
// Profiles start well before 90°; 28° blocks a real turn while leaving normal
// phone-held angles (and blinks) untouched.
const FRONTAL_MAX_YAW = 30; // deg — reject side-wise (profile) turns

function isFrontal(d: FaceData): boolean {
  return Math.abs(d.yaw) <= FRONTAL_MAX_YAW;
}

export function FaceAuthCamera(props: FaceAuthCameraProps): React.ReactElement {
  const { session, model, fasModel, embeddingRef, passiveScoreRef, mode } = props;
  const device = useCameraDevice('front');
  // Derive rotation direction from sensor orientation so it works across devices.
  // landscape-left (90° CW sensor) needs CCW correction → false; landscape-right needs CW → true.
  const ROT_CW = device?.sensorOrientation === 'landscape-right';
  const [hasPermission, setHasPermission] = useState(false);
  const [prompt, setPrompt] = useState('Position your face in the frame');
  // On-screen frame-processor readout (see logJs).
  const [diag, setDiag] = useState('frame processor: no frames yet');
  // Verify mode: show the face-guide oval to help the user realign when they drift
  // out of frame mid-challenge (mirrors the enrollment centering experience).
  const [faceMissing, setFaceMissing] = useState(false);
  // Enrollment ring state (iPhone-style): which angular segments are captured.
  const [ring, setRing] = useState<{
    phase: 'calibrating' | 'sweeping';
    captured: EnrollDirection[];
    current: EnrollDirection | null;
    target: EnrollDirection | null;
    progress: number;
  } | null>(null);

  const { detectFaces } = useFaceDetector({
    performanceMode: 'accurate', // more robust detection in low light / harsh backlight
    classificationMode: 'all', // needed for smiling / eyes-open probabilities
    landmarkMode: 'none',
    contourMode: 'none',
    minFaceSize: 0.1, // allow smaller / partially-lit faces
  });
  const { resize } = useResizePlugin();

  const runnerRef = useRef<VerifyRunner | EnrollRunner | null>(null);
  const doneRef = useRef(false);
  // Last frame with a usable face + when we saw it — used to bridge transient
  // MLKit detection drops so the liveness hold counters don't keep resetting.
  const lastGoodRef = useRef<{ data: FaceData; t: number } | null>(null);
  // Detection drops more often in low light; bridge slightly longer so the
  // liveness/capture FSM doesn't keep resetting on transient losses.
  const FACE_GRACE_MS = 750;
  // When the head first started reading as turned. MLKit's pose estimate jitters
  // when the eyes close (blink) or the mouth moves (smile), briefly flagging a
  // forward-facing user as turned — so we only nudge "Look straight" once the turn
  // has persisted, not on these momentary blips.
  const turnedSinceRef = useRef<number | null>(null);
  const SUSTAINED_TURN_MS = 800;
  // Smile "hold": MLKit's smile probability flickers (a real smile can register
  // for only 1-2 frames before a tiny head shift drops it), and the challenge
  // needs 3 consecutive frames over threshold. We hold the smile signal with a
  // fast attack / slow release so a strong smile survives brief dropouts. (Blink
  // is intentionally NOT smoothed — it needs the raw signal to fall back for the
  // "eyes re-opened" check.)
  const smileHeldRef = useRef(0);
  const SMILE_RELEASE = 0.8; // per-frame decay of the held smile value

  // vision-camera occasionally emits a recoverable error while (re)creating the
  // capture session on Android (it self-recovers). Handling onError ourselves logs
  // it as a warning instead of letting it surface as a red-box console error.
  const onCameraError = useCallback((error: CameraRuntimeError) => {
    // eslint-disable-next-line no-console
    console.warn(`[Camera] recoverable ${error.code}: ${error.message}`);
  }, []);

  const getRunner = useCallback((): VerifyRunner | EnrollRunner => {
    if (runnerRef.current) return runnerRef.current;
    runnerRef.current =
      mode === 'verify'
        ? session.startVerification()
        : session.startEnrollment(props.personId ?? 'unknown');
    return runnerRef.current;
  }, [mode, props.personId, session]);

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);


  // Runs on the JS thread once per processed frame: update the embedding seam and
  // advance the liveness/verify or enroll FSM.
  const onFrame = useRunOnJS(
    (data: FaceData | null, embeddingArr: number[] | null, passiveScore: number) => {
      if (doneRef.current) return;

      const now = Date.now();

      const faceDetected =
        !!data && isFaceUsable(data.box, { minWidth: 0.1, centerTolerance: 0.5 });
      const frontal = faceDetected && isFrontal(data as FaceData);
      const faceTurned = faceDetected && !frontal;

      // Recognition gate. For VERIFY we only ever embed a frontal frame, so a
      // side-wise view can never produce a recognition result (the "no profile
      // scan" guarantee). For ENROLL we deliberately want off-frontal poses
      // (left/right/up/down) so the multi-angle sweep can capture them, so we embed
      // any detected face and let EnrollRunner pick samples by head pose.
      const allowEmbed = mode === 'enroll' ? faceDetected : frontal;
      if (allowEmbed && embeddingArr && embeddingArr.length > 0) {
        embeddingRef.current = l2Normalize(Float32Array.from(embeddingArr));
      }
      if (passiveScoreRef) passiveScoreRef.current = passiveScore;

      // Liveness: use the detected face's REAL eye/smile signals whenever a face is
      // present, independent of exact head pose. (Tying "face present" to strict
      // frontality made blink fragile — as the eyes close, MLKit's pose estimate
      // jitters, which would drop that very frame and reset the blink FSM.) Bridge
      // brief detection drops as before.
      let effData: FaceData | null = null;
      let facePresent = false;
      if (faceDetected) {
        effData = data as FaceData;
        facePresent = true;
        lastGoodRef.current = { data: data as FaceData, t: now };
      } else if (lastGoodRef.current && now - lastGoodRef.current.t < FACE_GRACE_MS) {
        // Transient detection drop — keep the FSM alive with the last good reading.
        effData = lastGoodRef.current.data;
        facePresent = true;
      }

      const box = effData ? effData.box : { x: 0, y: 0, width: 0, height: 0 };
      const lo = effData && effData.leftEyeOpen >= 0 ? effData.leftEyeOpen : 1;
      const ro = effData && effData.rightEyeOpen >= 0 ? effData.rightEyeOpen : 1;
      const sm = effData && effData.smiling >= 0 ? effData.smiling : 0;
      // Fast-attack/slow-release hold on smile (see smileHeldRef). Reset when no
      // face so a held smile can't carry across a real absence.
      const smileHeld = facePresent
        ? Math.max(sm, smileHeldRef.current * SMILE_RELEASE)
        : 0;
      smileHeldRef.current = smileHeld;

      // Only nudge "Look straight" once a turn has persisted (not on blink/smile
      // pose jitter).
      if (faceTurned) {
        if (turnedSinceRef.current === null) turnedSinceRef.current = now;
      } else {
        turnedSinceRef.current = null;
      }
      const sustainedTurn =
        turnedSinceRef.current !== null && now - turnedSinceRef.current > SUSTAINED_TURN_MS;

      const frame: FaceFrame = {
        timestampMs: now,
        box,
        landmarks: [],
        blendshapes: {
          eyeBlinkLeft: clamp01(1 - lo),
          eyeBlinkRight: clamp01(1 - ro),
          mouthSmileLeft: clamp01(smileHeld),
          mouthSmileRight: clamp01(smileHeld),
          jawOpen: 0,
        },
        headPose: effData
          ? { yaw: effData.yaw, pitch: effData.pitch, roll: effData.roll }
          : { yaw: 0, pitch: 0, roll: 0 },
        facePresent,
      };

      if (mode === 'verify') {
        const vr = getRunner() as VerifyRunner;
        const challenge = vr.step(frame);
        setFaceMissing(!facePresent);
        setPrompt(
          !facePresent
            ? 'Center your face in the circle'
            : sustainedTurn
            ? 'Look straight at the camera — don’t turn sideways'
            : challenge
            ? PROMPTS[challenge.type]
            : 'Verifying…'
        );
        if (vr.isDone()) {
          doneRef.current = true;
          const result = vr.complete();
          if (result) props.onVerified?.(result);
        }
      } else {
        // Multi-angle enrollment sweep (iPhone Face ID style): guide the user
        // through the head-pose regions; EnrollRunner captures one template per
        // pose. The on-demand rotation also serves as liveness.
        const er = getRunner() as EnrollRunner;
        const prog = er.step(frame);
        setRing({
          phase: prog.phase,
          captured: prog.captured,
          current: prog.current,
          target: prog.target,
          progress: prog.progress,
        });
        setPrompt(
          !facePresent
            ? 'Center your face in the circle'
            : prog.phase === 'calibrating'
            ? 'Hold still…'
            : prog.target
            ? DIR_PROMPT[prog.target]
            : 'Done!'
        );
        if (er.isDone()) {
          doneRef.current = true;
          const id = er.result();
          if (id) {
            props.onEnrolled?.(id);
          } else {
            const rej = er.rejection();
            if (rej) props.onEnrollRejected?.(rej);
          }
        }
      }
    },
    [getRunner, mode, embeddingRef, passiveScoreRef]
  );

  // Reports frame-processor faults (model/detector errors) that would otherwise
  // be invisible, since a throw inside the worklet silently wedges the FSM.
  // Also surfaces the state ON SCREEN: release builds send console output
  // nowhere (RN 0.76 bridgeless only forwards to Metro), so an on-screen
  // readout is the only way to see inside a standalone APK.
  const logJs = useRunOnJS((msg: string) => {
    // eslint-disable-next-line no-console
    console.warn(`[FaceAuthCamera] ${msg}`);
    setDiag(msg);
  }, []);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      // 12fps (was 6): a natural blink's fully-closed frame lasts ~100ms, so at
      // 6fps it fell between samples and the closed signal only reached ~0.3.
      // Sampling faster catches the closed peak. runAtTargetFps self-throttles, so
      // a slower device simply runs below 12 — never faster, never blocking.
      runAtTargetFps(12, () => {
        'worklet';
        let faces;
        try {
          faces = detectFaces(frame);
        } catch (e) {
          logJs(`detectFaces threw: ${String(e)}`);
          onFrame(null, null, 1);
          return;
        }
        if (!faces || faces.length === 0) {
          logJs(`0 faces | frame ${frame.width}x${frame.height} | detector OK`);
          onFrame(null, null, 1);
          return;
        }
        const face = faces[0];
        const fw = frame.width;
        const fh = frame.height;
        const b = face.bounds;

        // Rotate MLKit's upright-space bounds into the frame-buffer space (90°).
        let rx: number;
        let ry: number;
        let rw: number;
        let rh: number;
        if (ROT_CW) {
          rx = b.y;
          ry = fh - (b.x + b.width);
          rw = b.height;
          rh = b.width;
        } else {
          rx = fw - (b.y + b.height);
          ry = b.x;
          rw = b.height;
          rh = b.width;
        }

        // Square crop (so resize to 112x112 doesn't distort) with a small margin.
        const ccx = rx + rw / 2;
        const ccy = ry + rh / 2;
        let side = Math.max(rw, rh) * 1.15;
        let cx = ccx - side / 2;
        let cy = ccy - side / 2;
        if (cx < 0) cx = 0;
        if (cy < 0) cy = 0;
        if (cx + side > fw) side = fw - cx;
        if (cy + side > fh) side = fh - cy;
        if (side < 1) side = 1;
        const cw = side;
        const ch = side;

        logJs(
          `${faces.length} face | frame ${frame.width}x${frame.height} | boxW ${(
            cw / fw
          ).toFixed(2)} | mdl ${model != null ? 'y' : 'N'} fas ${fasModel != null ? 'y' : 'N'}`
        );

        let embedding: number[] | null = null;
        if (model != null) {
          const resized = resize(frame, {
            crop: { x: cx, y: cy, width: cw, height: ch },
            scale: { width: 112, height: 112 },
            pixelFormat: 'rgb',
            dataType: 'float32', // values 0..1
          });
          // The crop comes out sideways (landscape buffer). MobileFaceNet is not
          // rotation-invariant, so rotate it 90° to upright before embedding.
          const N = 112;
          const up = new Float32Array(N * N * 3);
          for (let oy = 0; oy < N; oy++) {
            for (let ox = 0; ox < N; ox++) {
              let sx: number;
              let sy: number;
              if (ROT_CW) {
                sx = oy;
                sy = N - 1 - ox;
              } else {
                sx = N - 1 - oy;
                sy = ox;
              }
              const si = (sy * N + sx) * 3;
              const di = (oy * N + ox) * 3;
              up[di] = resized[si];
              up[di + 1] = resized[si + 1];
              up[di + 2] = resized[si + 2];
            }
          }

          const input = new Float32Array(N * N * 3);
          // Illumination normalization for low light. We combine two steps:
          //  1) a black-point lift + contrast stretch using the crop's own
          //     brightness range, so a dim, low-contrast face uses the full 0..1
          //     range instead of being squashed into the dark end;
          //  2) auto-gamma that maps the (stretched) mean to mid-grey. Gamma lifts
          //     shadows NON-linearly — far more effective on dark faces than the
          //     old linear gain, which clipped highlights and left them muddy.
          // In good light this is ~a no-op (range≈full, mean≈0.5 → gamma≈1).
          // Measure brightness/contrast from a CENTER PATCH (middle 50% of the
          // crop) — that region is almost all skin, so the background sitting in
          // the margin'd corners no longer drives the correction. A bright backlit
          // background won't over-darken the face, and a dark background won't
          // over-brighten it. The stretch+gamma is still APPLIED to the whole crop
          // below; only the measurement is restricted to the center.
          const m0 = N >> 2; // N/4  (start of center patch)
          const m1 = N - m0; // 3N/4 (end of center patch)
          let sum = 0;
          let lo = 1;
          let hi = 0;
          let count = 0;
          for (let py = m0; py < m1; py++) {
            for (let px = m0; px < m1; px++) {
              const pi = (py * N + px) * 3;
              for (let c = 0; c < 3; c++) {
                const v = up[pi + c];
                sum += v;
                if (v < lo) lo = v;
                if (v > hi) hi = v;
                count++;
              }
            }
          }
          // Contrast stretch: pull the patch's darkest pixels toward 0 and its
          // brightest to 1. Guard a flat/near-black patch (hi-lo tiny).
          let span = hi - lo;
          if (span < 0.05) span = 0.05;
          const stretchedMean = span > 1e-3 ? (sum / count - lo) / span : 0.5;
          // Auto-gamma toward mid-grey on the stretched signal.
          const target = 0.5;
          let gamma = 1;
          if (stretchedMean > 1e-3 && stretchedMean < 0.999) {
            gamma = Math.log(target) / Math.log(stretchedMean);
            if (gamma < 0.35) gamma = 0.35; // cap shadow lift so noise doesn't explode
            else if (gamma > 1.4) gamma = 1.4;
          }
          for (let i = 0; i < input.length; i++) {
            let v = (up[i] - lo) / span; // contrast-stretched, ~0..1
            if (v < 0) v = 0;
            else if (v > 1) v = 1;
            v = Math.pow(v, gamma); // shadow-lifting gamma
            input[i] = v * 2 - 1; // -> [-1,1]
          }
          try {
            const out = model.runSync([input]);
            const e = out[0];
            const arr = new Array<number>(e.length);
            for (let i = 0; i < e.length; i++) arr[i] = e[i] as number;
            embedding = arr;
          } catch (e) {
            logJs(`mobilefacenet runSync threw: ${String(e)}`);
          }
        }

        // ── MiniFASNet v2 — 80×80 face crop, normalized [0,1] ────────────────
        let passiveScore = 1;
        if (fasModel != null) {
          try {
            const rgb80 = resize(frame, {
              crop: { x: cx, y: cy, width: cw, height: ch },
              scale: { width: 80, height: 80 },
              pixelFormat: 'rgb',
              dataType: 'float32',
            });
            const fasOut = fasModel.runSync([rgb80])[0] as Float32Array;
            // Numerically-stable softmax → P(live) at class index 1.
            let fmax = -Infinity;
            for (let i = 0; i < fasOut.length; i++) if (fasOut[i] > fmax) fmax = fasOut[i];
            let fsum = 0;
            const fexps: number[] = [];
            for (let i = 0; i < fasOut.length; i++) {
              const fe = Math.exp(fasOut[i] - fmax);
              fexps.push(fe);
              fsum += fe;
            }
            passiveScore = (fexps[1] ?? 0) / (fsum || 1);
          } catch (e) {
            logJs(`minifasnet runSync threw: ${String(e)}`);
            passiveScore = 1;
          }
        }

        onFrame(
          {
            box: { x: cx / fw, y: cy / fh, width: cw / fw, height: ch / fh },
            yaw: face.yawAngle,
            pitch: face.pitchAngle,
            roll: face.rollAngle,
            leftEyeOpen: face.leftEyeOpenProbability,
            rightEyeOpen: face.rightEyeOpenProbability,
            smiling: face.smilingProbability,
          },
          embedding,
          passiveScore
        );
      });
    },
    [detectFaces, resize, model, fasModel, onFrame, ROT_CW, logJs]
  );

  if (!device || !hasPermission) {
    return (
      <View style={styles.center}>
        <CornerBrackets />
        <Text style={styles.eyebrow}>◤ CAMERA · OFFLINE ◢</Text>
        <Text style={styles.info}>
          {!device ? 'NO FRONT CAMERA FOUND' : 'CAMERA PERMISSION REQUIRED'}
        </Text>
        {!hasPermission && (
          <Text style={styles.link} onPress={() => Linking.openSettings()}>
            [ open settings ]
          </Text>
        )}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
        onError={onCameraError}
        lowLightBoost={device.supportsLowLightBoost ? true : undefined}
      />
      <ScanLine height={Dimensions.get('window').height} color={mode === 'enroll' ? NEON.cyan : NEON.violet} />
      <CornerBrackets color={mode === 'enroll' ? NEON.cyan : NEON.violet} />
      {mode === 'enroll' && ring && <EnrollRing {...ring} />}
      {mode === 'verify' && faceMissing && (
        <View style={styles.ringWrap} pointerEvents="none">
          <View style={[styles.faceGuide, styles.faceGuideCalib]} />
        </View>
      )}
      <View style={styles.modeTag} pointerEvents="none">
        <Text style={styles.modeTagText}>
          {mode === 'enroll' ? '◈ ENROLLMENT SCAN' : '◈ IDENTITY VERIFICATION'}
        </Text>
      </View>
      <View style={styles.overlay} pointerEvents="none">
        <Text style={styles.prompt}>{prompt}</Text>
        <Text style={styles.diag}>{diag}</Text>
      </View>
    </View>
  );
}

const ENROLL_ARROWS: { key: EnrollDirection; glyph: string; pos: TextStyle }[] = [
  { key: 'up', glyph: '▲', pos: { top: 2, left: 0, right: 0, textAlign: 'center' } },
  { key: 'down', glyph: '▼', pos: { bottom: 2, left: 0, right: 0, textAlign: 'center' } },
  { key: 'left', glyph: '◀', pos: { left: 4, top: '50%', marginTop: -20 } },
  { key: 'right', glyph: '▶', pos: { right: 4, top: '50%', marginTop: -20 } },
];

/**
 * iPhone-Face-ID-style enrollment guide: a face-guide oval with up/down/left/right
 * chevrons that light green as each direction is captured (the next one to do
 * pulses amber). The ring grows toward full as progress completes.
 */
function EnrollRing(props: {
  phase: 'calibrating' | 'sweeping';
  captured: EnrollDirection[];
  current: EnrollDirection | null;
  target: EnrollDirection | null;
  progress: number;
}): React.ReactElement {
  const { captured, target, phase, progress } = props;
  return (
    <View style={styles.ringWrap} pointerEvents="none">
      <View style={styles.ringBox}>
        {/* Progress ring that grows from a dot to a full circle as it completes. */}
        <View
          style={[
            styles.growCircle,
            {
              width: 120 + progress * 150,
              height: 120 + progress * 150,
              borderRadius: (120 + progress * 150) / 2,
              opacity: 0.25 + progress * 0.55,
            },
          ]}
        />
        <View style={[styles.faceGuide, phase === 'calibrating' && styles.faceGuideCalib]} />
        {ENROLL_ARROWS.map((a) => {
          const done = captured.includes(a.key);
          const isTarget = phase === 'sweeping' && target === a.key;
          return (
            <Text
              key={a.key}
              style={[
                styles.arrow,
                a.pos,
                done ? styles.arrowDone : isTarget ? styles.arrowTarget : styles.arrowPending,
              ]}>
              {a.glyph}
            </Text>
          );
        })}
        <Text style={styles.ringPct}>{Math.round(progress * 100)}%</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: NEON.bg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: NEON.bg },
  eyebrow: {
    color: NEON.green,
    fontSize: 11,
    fontFamily: 'monospace',
    letterSpacing: 3,
    marginBottom: 14,
  },
  info: {
    color: NEON.textMain,
    fontSize: 15,
    fontFamily: 'monospace',
    letterSpacing: 1,
    textAlign: 'center',
    paddingHorizontal: 24,
  },
  link: { color: NEON.cyan, fontSize: 14, fontFamily: 'monospace', marginTop: 14 },
  modeTag: { position: 'absolute', top: 56, left: 0, right: 0, alignItems: 'center' },
  modeTagText: {
    color: NEON.textMain,
    fontFamily: 'monospace',
    fontSize: 11,
    letterSpacing: 2,
    backgroundColor: 'rgba(3,5,13,0.55)',
    borderWidth: 1,
    borderColor: 'rgba(34,229,255,0.35)',
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 4,
    overflow: 'hidden',
  },
  overlay: { position: 'absolute', bottom: 48, left: 0, right: 0, alignItems: 'center', paddingHorizontal: 24 },
  prompt: {
    color: NEON.textMain,
    fontSize: 18,
    fontWeight: '700',
    fontFamily: 'monospace',
    letterSpacing: 0.5,
    textAlign: 'center',
    backgroundColor: 'rgba(3,5,13,0.65)',
    borderWidth: 1,
    borderColor: 'rgba(34,229,255,0.4)',
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 4,
    overflow: 'hidden',
  },
  diag: {
    color: '#7CFFB2',
    fontFamily: 'monospace',
    fontSize: 11,
    textAlign: 'center',
    marginTop: 10,
    backgroundColor: 'rgba(3,5,13,0.8)',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 3,
    overflow: 'hidden',
  },
  track: { marginTop: 16, width: '80%', height: 8, borderRadius: 4, backgroundColor: 'rgba(255,255,255,0.3)' },
  fill: { height: 8, borderRadius: 4, backgroundColor: NEON.green },
  ringWrap: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
  ringBox: { position: 'relative', width: 300, height: 300, alignItems: 'center', justifyContent: 'center' },
  growCircle: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: NEON.cyan,
  },
  faceGuide: {
    position: 'absolute',
    width: 176,
    height: 226,
    borderRadius: 110,
    borderWidth: 1.5,
    borderColor: 'rgba(34,229,255,0.45)',
    borderStyle: 'dashed',
  },
  faceGuideCalib: { borderColor: 'rgba(178,107,255,0.9)' },
  arrow: { position: 'absolute', fontSize: 30, fontWeight: '700' },
  arrowPending: { color: 'rgba(94,120,150,0.5)' },
  arrowTarget: { color: NEON.amber, fontSize: 38, textShadowColor: 'rgba(255,201,77,0.8)', textShadowRadius: 12, textShadowOffset: { width: 0, height: 0 } },
  arrowDone: { color: NEON.green },
  ringPct: {
    position: 'absolute',
    color: NEON.cyan,
    fontSize: 18,
    fontWeight: '800',
    fontFamily: 'monospace',
    textShadowColor: 'rgba(34,229,255,0.7)',
    textShadowRadius: 10,
    textShadowOffset: { width: 0, height: 0 },
  },
});
