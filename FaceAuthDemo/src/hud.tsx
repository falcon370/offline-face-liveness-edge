/**
 * hud.tsx — shared sci-fi HUD decorations for the demo app's UI.
 * JS-only (Animated + StyleSheet), no native deps, so it hot-reloads on device.
 */

import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, View } from 'react-native';

export const NEON = {
  bg: '#03050d',
  bgAlt: '#070c1a',
  panel: 'rgba(10,18,36,0.55)',
  panelBorder: 'rgba(34,229,255,0.35)',
  cyan: '#22e5ff',
  cyanSoft: 'rgba(34,229,255,0.14)',
  violet: '#b26bff',
  violetSoft: 'rgba(178,107,255,0.14)',
  green: '#3dffa0',
  amber: '#ffc94d',
  red: '#ff5577',
  textMain: '#e7f8ff',
  textDim: '#5c7089',
};

const BRACKET = 22;

/** Four HUD-style targeting corner brackets, absolutely positioned over the parent. */
export function CornerBrackets({
  color = NEON.cyan,
  inset = 14,
}: {
  color?: string;
  inset?: number;
}): React.ReactElement {
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <View style={[styles.bracket, { top: inset, left: inset, borderTopWidth: 2, borderLeftWidth: 2, borderColor: color }]} />
      <View style={[styles.bracket, { top: inset, right: inset, borderTopWidth: 2, borderRightWidth: 2, borderColor: color }]} />
      <View style={[styles.bracket, { bottom: inset, left: inset, borderBottomWidth: 2, borderLeftWidth: 2, borderColor: color }]} />
      <View style={[styles.bracket, { bottom: inset, right: inset, borderBottomWidth: 2, borderRightWidth: 2, borderColor: color }]} />
    </View>
  );
}

/** A status dot that pulses an outward glow ring while `pulsing`, solid otherwise. */
export function PulseDot({ color, pulsing }: { color: string; pulsing?: boolean }): React.ReactElement {
  const scale = useRef(new Animated.Value(1)).current;
  const opacity = useRef(new Animated.Value(0.9)).current;

  useEffect(() => {
    if (!pulsing) return;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.parallel([
          Animated.timing(scale, { toValue: 2.2, duration: 900, useNativeDriver: true }),
          Animated.timing(opacity, { toValue: 0, duration: 900, useNativeDriver: true }),
        ]),
        Animated.timing(scale, { toValue: 1, duration: 0, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 0.9, duration: 0, useNativeDriver: true }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [pulsing, scale, opacity]);

  return (
    <View style={styles.dotWrap}>
      {pulsing && (
        <Animated.View
          style={[styles.dotRing, { borderColor: color, transform: [{ scale }], opacity }]}
        />
      )}
      <View style={[styles.dotCore, { backgroundColor: color }]} />
    </View>
  );
}

/** A slow vertical scanning beam sweeping across a view of the given height. */
export function ScanLine({ height, color = NEON.cyan }: { height: number; color?: string }): React.ReactElement {
  const y = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(y, { toValue: 1, duration: 2800, useNativeDriver: true }),
        Animated.timing(y, { toValue: 0, duration: 0, useNativeDriver: true }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [y]);

  return (
    <View style={[StyleSheet.absoluteFill, styles.scanClip]} pointerEvents="none">
      <Animated.View
        style={[
          styles.scan,
          {
            backgroundColor: color,
            transform: [{ translateY: y.interpolate({ inputRange: [0, 1], outputRange: [0, height] }) }],
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  bracket: { position: 'absolute', width: BRACKET, height: BRACKET },
  dotWrap: { width: 18, height: 18, alignItems: 'center', justifyContent: 'center' },
  dotCore: { width: 9, height: 9, borderRadius: 5 },
  dotRing: { position: 'absolute', width: 9, height: 9, borderRadius: 5, borderWidth: 1.5 },
  scanClip: { overflow: 'hidden' },
  scan: { position: 'absolute', left: 0, right: 0, height: 2, opacity: 0.5 },
});
