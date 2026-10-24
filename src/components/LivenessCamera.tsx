/**
 * LivenessCamera — turnkey camera UI that runs the liveness challenge flow and
 * (optionally) recognition, then hands back a result.
 *
 * Uses MediaPipe's Face Landmarker live-stream for detection + blendshapes
 * (active liveness), feeding each frame through a VerifyRunner/EnrollRunner. The
 * recognition embedding + passive anti-spoof score come from the `embedNow` /
 * `passiveScoreNow` seams supplied via useFaceAuth (see that hook's docs).
 */

import React, { useCallback, useRef, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import {
  MediapipeCamera,
  RunningMode,
  useFaceLandmarkDetection,
  type FaceLandmarkDetectionResultBundle,
} from 'react-native-mediapipe';

import { toFaceFrame, type MpFaceResult } from '../frame/mediapipeAdapter';
import type { FaceAuthSession, VerifyRunner, EnrollRunner } from '../session';
import type {
  AttendanceRecord,
  ChallengeState,
  ChallengeType,
  EnrolledIdentity,
  LivenessResult,
  VerificationResult,
} from '../types';

const LANDMARKER_MODEL = 'face_landmarker'; // resolved from bundled assets

const PROMPTS: Record<ChallengeType, string> = {
  blink: 'Please blink',
  smile: 'Please smile',
  turnLeft: 'Turn your head left',
  turnRight: 'Turn your head right',
};

export interface VerifyResult {
  liveness: LivenessResult;
  verification: VerificationResult;
  record: AttendanceRecord;
}

export interface LivenessCameraProps {
  session: FaceAuthSession;
  mode: 'verify' | 'enroll';
  /** Required for enroll mode. */
  personId?: string;
  enrollSamples?: number;
  onVerified?: (r: VerifyResult) => void;
  onEnrolled?: (identity: EnrolledIdentity) => void;
  onProgress?: (challenge: ChallengeState | null, progress: number) => void;
}

function bundleToResult(b: FaceLandmarkDetectionResultBundle): MpFaceResult | null {
  const first = b.results?.[0];
  const lm = first?.faceLandmarks?.[0];
  if (!first || !lm || lm.length === 0) return null;
  return {
    landmarks: lm.map((p) => ({ x: p.x, y: p.y })),
    blendshapes: first.faceBlendshapes?.[0]?.categories ?? [],
    transformMatrix: first.facialTransformationMatrixes?.[0]?.data,
    timestampMs: b.timestampMs ?? Date.now(),
  };
}

export function LivenessCamera(props: LivenessCameraProps): React.ReactElement {
  const { session, mode } = props;
  const [prompt, setPrompt] = useState<string>('Position your face in the frame');
  const [progress, setProgress] = useState(0);

  const runnerRef = useRef<VerifyRunner | EnrollRunner | null>(null);
  const doneRef = useRef(false);

  const getRunner = useCallback((): VerifyRunner | EnrollRunner => {
    if (runnerRef.current) return runnerRef.current;
    runnerRef.current =
      mode === 'verify'
        ? session.startVerification()
        : session.startEnrollment(props.personId ?? 'unknown');
    return runnerRef.current;
  }, [mode, props.personId, session]);

  const onResults = useCallback(
    (bundle: FaceLandmarkDetectionResultBundle) => {
      if (doneRef.current) return;
      const frame = toFaceFrame(bundleToResult(bundle), Date.now());
      const runner = getRunner();

      if (mode === 'verify') {
        const vr = runner as VerifyRunner;
        const challenge = vr.step(frame);
        setPrompt(challenge ? PROMPTS[challenge.type] : 'Verifying…');
        props.onProgress?.(challenge, challenge?.progress ?? 1);
        if (vr.isDone()) {
          doneRef.current = true;
          const result = vr.complete();
          if (result) props.onVerified?.(result);
        }
      } else {
        const er = runner as EnrollRunner;
        const prog = er.step(frame);
        setProgress(prog.progress);
        const DIR_PROMPT = {
          up: 'Tilt your head up',
          down: 'Tilt your head down',
          left: 'Turn your head to one side',
          right: 'Now turn to the other side',
        } as const;
        setPrompt(
          prog.phase === 'calibrating'
            ? 'Look straight at the camera'
            : prog.target
            ? DIR_PROMPT[prog.target]
            : 'Capturing…'
        );
        props.onProgress?.(null, prog.progress);
        if (er.isDone()) {
          doneRef.current = true;
          const identity = er.result();
          if (identity) props.onEnrolled?.(identity);
        }
      }
    },
    [getRunner, mode, props]
  );

  const onError = useCallback((e: unknown) => {
    setPrompt(`Camera error: ${String(e)}`);
  }, []);

  const { cameraViewProps } = useFaceLandmarkDetection(
    onResults,
    onError,
    RunningMode.LIVE_STREAM,
    LANDMARKER_MODEL,
    { numFaces: 1 }
  );

  return (
    <View style={styles.container}>
      <MediapipeCamera {...cameraViewProps} style={StyleSheet.absoluteFill} />
      <View style={styles.overlay} pointerEvents="none">
        <Text style={styles.prompt}>{prompt}</Text>
        {mode === 'enroll' && (
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${Math.round(progress * 100)}%` }]} />
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  overlay: {
    position: 'absolute',
    bottom: 48,
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  prompt: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
    textAlign: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 12,
    overflow: 'hidden',
  },
  progressTrack: {
    marginTop: 16,
    width: '80%',
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.3)',
  },
  progressFill: { height: 8, borderRadius: 4, backgroundColor: '#4ade80' },
});
