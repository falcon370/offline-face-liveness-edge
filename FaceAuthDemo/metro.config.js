const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');
const path = require('path');

// The pure pipeline logic lives in the sibling library at ../src (no native
// imports — it takes native objects via injection). We watch it so the demo can
// import those modules directly, and register the TFLite/Task model extensions
// as assets so require() returns a loadable asset for react-native-fast-tflite.
const repoSrc = path.resolve(__dirname, '..', 'src');
const localNodeModules = path.resolve(__dirname, 'node_modules');

/**
 * @type {import('metro-config').MetroConfig}
 */
const config = {
  watchFolders: [repoSrc],
  resolver: {
    // Files under ../src are outside the project root, so Babel's injected
    // helper requires (@babel/runtime/*) and any bare imports must resolve back
    // to *this* app's node_modules.
    nodeModulesPaths: [localNodeModules],
    extraNodeModules: new Proxy(
      {},
      {get: (_target, name) => path.join(localNodeModules, name)},
    ),
  },
};

const merged = mergeConfig(getDefaultConfig(__dirname), config);
// assetExts must be appended to the defaults, not replace them.
merged.resolver.assetExts = [
  ...new Set([...getDefaultConfig(__dirname).resolver.assetExts, 'tflite', 'task']),
];
module.exports = merged;
