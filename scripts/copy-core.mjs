// Copies Capacitor's JS runtime into www/. The native shell only injects a minimal bridge; registerPlugin()
// (which native.js uses to reach the device plugins) lives in @capacitor/core, so the page must load it.
// Without it the app silently runs in web mode on the phone: no purchases, notifications or durable storage.
import { copyFileSync } from 'node:fs';
copyFileSync('node_modules/@capacitor/core/dist/capacitor.js', 'www/capacitor.js');
console.log('capacitor.js copied to www');
