/**
 * FaceAuthDemo — offline facial recognition + liveness on-device.
 *
 * Home → Enroll (liveness-gated capture of your face) → Verify (attendance:
 * liveness + 1:N recognition) → sync+purge to a mock endpoint. Recognition runs
 * MobileFaceNet via the real frame-processor seam in FaceAuthCamera.
 */

import React, { useState } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  Pressable,
  StyleSheet,
  StatusBar,
  TextInput,
  Keyboard,
} from 'react-native';

import { useFaceAuthDemo } from './src/useFaceAuthDemo';
import { FaceAuthCamera } from './src/FaceAuthCamera';
import { CornerBrackets, PulseDot, NEON } from './src/hud';
import type { EnrollRejection } from '../src/session';
import type { EnrolledIdentity } from '../src/types';
import type { VerifyResult } from '../src/components/LivenessCamera';

type Screen = 'home' | 'enroll' | 'verify';

export default function App(): React.ReactElement {
  const [screen, setScreen] = useState<Screen>('home');
  const [status, setStatus] = useState('');
  const [name, setName] = useState('');
  // The id used for the current enrollment session (set when entering enroll).
  const [enrollId, setEnrollId] = useState('');
  const fa = useFaceAuthDemo();

  const startEnroll = () => {
    Keyboard.dismiss();
    const id = name.trim() || `person-${Date.now()}`;
    setEnrollId(id);
    setScreen('enroll');
  };

  const clearEnrolled = () => {
    for (const p of fa.store.list()) fa.store.remove(p.personId);
    setStatus('Cleared all enrolled identities');
  };

  const onEnrolled = (id: EnrolledIdentity) => {
    setStatus(`✅ Enrolled ${id.personId} (${id.sampleCount} samples)`);
    setName('');
    setScreen('home');
  };

  const onEnrollRejected = (r: EnrollRejection) => {
    // The captured face already belongs to an enrolled person — don't add a
    // duplicate identity for them.
    setStatus(
      `⚠️ This face is already enrolled as "${r.personId}" (sim ${r.similarity.toFixed(2)}). Duplicate rejected.`
    );
    setName('');
    setScreen('home');
  };

  const onVerified = async (r: VerifyResult) => {
    const msg = r.verification.matched
      ? `✅ ${r.verification.personId} (sim ${r.verification.similarity.toFixed(2)})`
      : `❌ no match · liveness: ${r.liveness.outcome}`;
    const res = await fa.syncNow();
    setStatus(`${msg} · synced ${res.uploaded}, server has ${fa.mockServer.received.size}`);
    setScreen('home');
  };

  if (screen === 'enroll') {
    return (
      <FaceAuthCamera
        session={fa.session}
        model={fa.model}
        fasModel={fa.fasModel}
        embeddingRef={fa.embeddingRef}
        passiveScoreRef={fa.passiveScoreRef}
        mode="enroll"
        personId={enrollId}
        onEnrolled={onEnrolled}
        onEnrollRejected={onEnrollRejected}
      />
    );
  }
  if (screen === 'verify') {
    return (
      <FaceAuthCamera
        session={fa.session}
        model={fa.model}
        fasModel={fa.fasModel}
        embeddingRef={fa.embeddingRef}
        passiveScoreRef={fa.passiveScoreRef}
        mode="verify"
        onVerified={onVerified}
      />
    );
  }

  const modelReady = fa.modelState === 'loaded';
  const people = fa.store.list();
  const enrolled = people.length > 0;

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={NEON.bg} />
      <CornerBrackets />

      <View style={styles.titleWrap}>
        <Text style={styles.eyebrow}>◤ SYSTEM · ONLINE ◢</Text>
        <Text style={styles.title}>DATALAKE FACE AUTH</Text>
        <View style={styles.titleUnderline} />
        <Text style={styles.subtitle}>ON-DEVICE RECOGNITION · LIVENESS · OFFLINE</Text>
      </View>

      <View style={styles.panel}>
        <Text style={styles.label}>◈ IDENTITY NAME</Text>
        <TextInput
          style={styles.input}
          value={name}
          onChangeText={setName}
          placeholder="e.g. Ravi Kumar"
          placeholderTextColor="#3d5170"
          autoCapitalize="words"
          returnKeyType="done"
        />

        <View style={styles.buttons}>
          <Button label="Enroll a Face" accent={NEON.cyan} disabled={!modelReady} onPress={startEnroll} />
          <Button
            label="Verify (Attendance)"
            accent={NEON.violet}
            disabled={!modelReady || !enrolled}
            onPress={() => setScreen('verify')}
          />
        </View>
      </View>

      <View style={styles.consolePanel}>
        <View style={styles.consoleRow}>
          <PulseDot color={modelReady ? NEON.green : NEON.amber} pulsing={!modelReady} />
          <Text style={styles.consoleText}>
            MODEL: {modelReady ? 'MOBILEFACENET · LOADED' : `LOADING… (${fa.modelState})`}
          </Text>
        </View>
        <View style={styles.consoleDivider} />
        <View style={styles.consoleRow}>
          <PulseDot color={enrolled ? NEON.cyan : NEON.textDim} />
          <Text style={styles.consoleText} numberOfLines={2}>
            ENROLLED ({people.length}): {enrolled ? people.map((p) => p.personId).join(', ') : 'none yet'}
          </Text>
        </View>
        {enrolled && (
          <Text style={styles.clear} onPress={clearEnrolled}>
            [ clear enrolled identities ]
          </Text>
        )}
        {!!status && <Text style={styles.status}>{status}</Text>}
        <Text style={styles.pending}>PENDING SYNC · {fa.syncQueue.pendingCount()}</Text>
      </View>
    </SafeAreaView>
  );
}

function Button({
  label,
  onPress,
  disabled,
  accent,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  accent: string;
}) {
  const color = disabled ? NEON.textDim : accent;
  return (
    <Pressable
      style={[
        styles.button,
        { borderColor: color, backgroundColor: disabled ? 'transparent' : `${accent}1f` },
      ]}
      onPress={onPress}
      disabled={disabled}>
      <Text style={[styles.buttonText, { color }]}>{label.toUpperCase()}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, justifyContent: 'center', backgroundColor: NEON.bg },
  titleWrap: { alignItems: 'center', marginBottom: 32 },
  eyebrow: {
    color: NEON.green,
    fontSize: 11,
    fontFamily: 'monospace',
    letterSpacing: 3,
    marginBottom: 10,
  },
  title: {
    color: NEON.cyan,
    fontSize: 26,
    fontWeight: '800',
    fontFamily: 'monospace',
    letterSpacing: 2,
    textAlign: 'center',
    textShadowColor: 'rgba(34,229,255,0.7)',
    textShadowRadius: 16,
    textShadowOffset: { width: 0, height: 0 },
  },
  titleUnderline: {
    width: 90,
    height: 2,
    backgroundColor: NEON.cyan,
    marginTop: 10,
    opacity: 0.7,
  },
  subtitle: {
    color: NEON.textDim,
    fontSize: 11,
    fontFamily: 'monospace',
    letterSpacing: 1.5,
    textAlign: 'center',
    marginTop: 12,
  },
  panel: {
    backgroundColor: NEON.panel,
    borderWidth: 1,
    borderColor: NEON.panelBorder,
    borderRadius: 4,
    padding: 18,
    marginBottom: 18,
  },
  label: {
    color: NEON.cyan,
    fontSize: 11,
    fontFamily: 'monospace',
    letterSpacing: 1.5,
    marginBottom: 8,
    marginLeft: 2,
  },
  input: {
    backgroundColor: 'rgba(3,5,13,0.6)',
    color: NEON.textMain,
    fontSize: 16,
    fontFamily: 'monospace',
    borderRadius: 4,
    paddingHorizontal: 14,
    paddingVertical: 12,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(34,229,255,0.25)',
  },
  buttons: { gap: 14 },
  button: {
    paddingVertical: 15,
    borderRadius: 4,
    alignItems: 'center',
    borderWidth: 1.5,
  },
  buttonText: { fontSize: 15, fontWeight: '700', fontFamily: 'monospace', letterSpacing: 1.5 },
  consolePanel: {
    backgroundColor: 'rgba(3,5,13,0.5)',
    borderWidth: 1,
    borderColor: 'rgba(178,107,255,0.25)',
    borderRadius: 4,
    padding: 16,
  },
  consoleRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  consoleDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: 'rgba(148,163,184,0.25)',
    marginVertical: 10,
  },
  consoleText: {
    color: NEON.textMain,
    fontFamily: 'monospace',
    fontSize: 12,
    flexShrink: 1,
    letterSpacing: 0.5,
  },
  clear: {
    color: NEON.red,
    textAlign: 'center',
    marginTop: 14,
    fontSize: 12,
    fontFamily: 'monospace',
    letterSpacing: 1,
  },
  status: {
    color: NEON.textMain,
    textAlign: 'center',
    marginTop: 14,
    fontSize: 14,
    fontFamily: 'monospace',
  },
  pending: {
    color: NEON.textDim,
    textAlign: 'center',
    marginTop: 12,
    fontSize: 11,
    fontFamily: 'monospace',
    letterSpacing: 1,
  },
});
