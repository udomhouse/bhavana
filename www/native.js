/* Bhāvanā native bridge — connects the web app to device features through Capacitor plugins.
   In a normal browser this file sets window.NATIVE = null and the app runs in web mode. */
(function () {
  const Cap = window.Capacitor;
  if (!Cap || !Cap.isNativePlatform || !Cap.isNativePlatform()) { window.NATIVE = null; return; }
  const plug = (n) => Cap.registerPlugin(n);
  const TTS = plug('TextToSpeech'), LN = plug('LocalNotifications'), Pref = plug('Preferences'),
    KA = plug('KeepAwake'), IAP = plug('NativePurchases'), SB = plug('StatusBar'), App = plug('App'), Hap = plug('Haptics'), Splash = plug('SplashScreen');
  const quiet = (p) => (p && p.catch ? p.catch(() => {}) : p);
  // Never let one slow plugin hold the app: give up on a call after `ms`
  const within = (p, ms = 1500, fallback = null) => Promise.race([Promise.resolve(p).catch(() => fallback), new Promise((r) => setTimeout(() => r(fallback), ms))]);

  const N = {
    platform: Cap.getPlatform(),   // 'ios' | 'android'
    ttsReady: null,                // true when the device has a Thai voice
    onRange: null,                 // set by app.js: called with the char index of the word being spoken

    async init() {
      quiet(Splash.hide({ fadeOutDuration: 250 }));   // first, so the logo never blocks the app
      const r = await within(TTS.isLanguageSupported({ lang: 'th-TH' }));
      N.ttsReady = r ? !!r.supported : null;
      within(TTS.addListener('onRangeStart', (i) => N.onRange && N.onRange(i.start)));
      within(App.addListener('backButton', () => { if (!(window.onNativeBack && window.onNativeBack())) App.exitApp(); }));
      within(App.addListener('appStateChange', (s) => { if (s.isActive && window.onNativeResume) window.onNativeResume(); }));
      quiet(SB.setOverlaysWebView({ overlay: true }));
      const mq = matchMedia('(prefers-color-scheme: dark)');
      const style = () => quiet(SB.setStyle({ style: mq.matches ? 'DARK' : 'LIGHT' }));
      style(); if (mq.addEventListener) mq.addEventListener('change', style);
    },

    /* speech */
    speak(text, o) {
      return TTS.speak({ text, lang: 'th-TH', rate: o.rate, pitch: o.pitch, volume: 1, category: 'playback', queueStrategy: 0 });
    },
    stop() { return quiet(TTS.stop()); },
    openInstall() { return quiet(TTS.openInstall()); },

    /* meditation helpers */
    keepAwake() { quiet(KA.keepAwake()); },
    allowSleep() { quiet(KA.allowSleep()); },
    haptic() { quiet(Hap.impact({ style: 'MEDIUM' })); },

    /* durable storage (WebView localStorage can be cleared by the OS) */
    async restore() { try { const r = await within(Pref.get({ key: 'state' })); return r && r.value ? JSON.parse(r.value) : null; } catch (e) { return null; } },
    persist(state) { quiet(Pref.set({ key: 'state', value: JSON.stringify(state) })); },

    /* Premium: one-time "Non-Consumable" in-app purchase, same product id in App Store Connect and Google Play */
    PRODUCT: 'premium_lifetime',
    async premiumStatus() {   // true / false, or null when the store can't be reached (keep the cached value)
      try {
        const { purchases } = await IAP.getPurchases({ productType: 'inapp', onlyCurrentEntitlements: true });
        return purchases.some((p) => p.productIdentifier === N.PRODUCT && (N.platform === 'ios' || p.purchaseState === undefined || String(p.purchaseState) === '1'));
      } catch (e) { return null; }
    },
    async premiumPrice() {    // localized price from the store, e.g. "A$1.99" or "฿49.00"
      try { return (await IAP.getProduct({ productIdentifier: N.PRODUCT, productType: 'inapp' })).product.priceString || null; } catch (e) { return null; }
    },
    async buyPremium() {
      const t = await IAP.purchaseProduct({ productIdentifier: N.PRODUCT, productType: 'inapp', quantity: 1 });
      if (t && t.productIdentifier === N.PRODUCT) return true;
      return !!(await N.premiumStatus());
    },
    /* 7-day free trial: a FREE non-consumable "trial_7day" (Apple guideline 3.1.1 asks trials of one-time
       purchases to be a free in-app item named "XX-day Trial"). Its purchase date starts the trial, so it
       follows the Apple ID / Google account and survives reinstalling the app. */
    TRIAL: 'trial_7day',
    async trialStart() {      // ms timestamp the trial began, or null (not started / store unreachable)
      try {
        const { purchases } = await IAP.getPurchases({ productType: 'inapp' });
        const t = purchases.find((p) => p.productIdentifier === N.TRIAL);
        if (!t) return 0;
        const d = Date.parse(t.originalPurchaseDate || t.purchaseDate);
        return isNaN(d) ? Date.now() : d;
      } catch (e) { return null; }
    },
    async startTrial() {      // returns the start timestamp; falls back to "now" when the store has no trial item
      try { await IAP.getProduct({ productIdentifier: N.TRIAL, productType: 'inapp' }); }
      catch (e) { return Date.now(); }
      const t = await IAP.purchaseProduct({ productIdentifier: N.TRIAL, productType: 'inapp', quantity: 1 });
      const d = t && Date.parse(t.originalPurchaseDate || t.purchaseDate);
      return d && !isNaN(d) ? d : Date.now();
    },
    async restorePremium() { try { await IAP.restorePurchases(); } catch (e) {} return !!(await N.premiumStatus()); },

    /* notifications */
    async notifState() { const r = await within(LN.checkPermissions()); return r ? r.display : null; },   // 'granted' | 'denied' | 'prompt' | null
    async requestNotif() { try { return (await LN.requestPermissions()).display === 'granted'; } catch (e) { return false; } },
    async testNotif() {
      await LN.schedule({ notifications: [{ id: 9999, title: 'Bhāvanā 🙏', body: 'การแจ้งเตือนทำงานแล้ว จะเตือนวันพระและสตรีคให้ตามเวลา', schedule: { at: new Date(Date.now() + 5000) } }] });
    },
    // o = { pra:[{date:Date,label}], praOn, streakOn, doneToday, ask }
    async schedule(o) {
      try {
        let perm = await LN.checkPermissions();
        if (perm.display !== 'granted') {
          if (!o.ask || perm.display === 'denied') return false;
          perm = await LN.requestPermissions();
          if (perm.display !== 'granted') return false;
        }
        const pend = await LN.getPending();
        if (pend.notifications.length) await LN.cancel({ notifications: pend.notifications.map((n) => ({ id: n.id })) });
        const now = Date.now(), list = [];
        let id = 1;
        if (o.praOn) for (const p of o.pra) {
          const d = p.date;
          const eve = new Date(d.getFullYear(), d.getMonth(), d.getDate() - 1, 19, 0);
          const morn = new Date(d.getFullYear(), d.getMonth(), d.getDate(), 6, 30);
          if (eve > now) list.push({ id: id++, title: 'พรุ่งนี้วันพระ 🌕', body: p.label + ' · เตรียมใจสวดมนต์ รักษาศีล', schedule: { at: eve } });
          if (morn > now) list.push({ id: id++, title: 'วันนี้วันพระ 🙏', body: p.label + ' · เริ่มวันด้วยบทสวดสั้น ๆ สักบทไหม', schedule: { at: morn } });
        }
        if (o.streakOn) for (let k = 0; k < 14; k++) {
          const t = new Date(); t.setDate(t.getDate() + k); t.setHours(19, 30, 0, 0);
          if (k === 0 && (o.doneToday || t <= now)) continue;
          list.push({ id: 1000 + k, title: 'สตรีคของคุณรออยู่ 🔥', body: 'นั่งสมาธิ 5 นาทีก่อนนอน รักษาไฟไว้ให้ลุกต่อ', schedule: { at: t } });
        }
        list.sort((a, b) => a.schedule.at - b.schedule.at);
        const cut = list.slice(0, 60); // iOS keeps at most 64 pending notifications
        if (cut.length) await LN.schedule({ notifications: cut });
        return true;
      } catch (e) { return false; }
    },
  };
  window.NATIVE = N;
})();
