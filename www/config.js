/* Bhāvanā app settings.
   สนทนาธรรม (community room): paste your Supabase project URL and anon (public) key here.
   Supabase › Project Settings › API. The anon key is meant to ship inside apps; never put the service_role key here.
   Leave empty to hide the room's server features. */
window.BHAVANA_CONFIG = {
  supabaseUrl: 'https://dexkdilmsgwjyroxqoud.supabase.co',
  supabaseAnonKey: 'sb_publishable_DK_-gxitQydKdE83V3AScg__4Ozeeiv',
  // Test mode: shows a switch in Settings that turns Premium on without buying. ⚠️ Set to false before
  // sending a build to the App Store / Google Play.
  testMode: true,
};
