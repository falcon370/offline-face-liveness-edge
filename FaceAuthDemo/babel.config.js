module.exports = {
  presets: ['module:@react-native/babel-preset'],
  // Required by react-native-vision-camera frame processors, the resize plugin,
  // and react-native-vision-camera-face-detector (all use worklets-core).
  plugins: [['react-native-worklets-core/plugin']],
};
