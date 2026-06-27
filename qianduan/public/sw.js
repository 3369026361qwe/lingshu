// 灵枢 PWA Service Worker — 离线缓存 + 风险告警推送
const CACHE = 'lingshu-v3.2.0';
const PRELOAD = ['/', '/api/health'];

// ── 安装: 预缓存核心资源 ──
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(PRELOAD))
  );
  self.skipWaiting();
});

// ── 激活: 清理旧缓存 ──
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── 请求拦截: 缓存优先 → 网络回退 ──
self.addEventListener('fetch', (e) => {
  // 跳过 API 请求（始终走网络）
  if (e.request.url.includes('/api/') || e.request.url.includes('/ws/')) {
    return;
  }
  e.respondWith(
    caches.match(e.request).then((cached) => {
      const fetched = fetch(e.request).then((res) => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, clone));
        }
        return res;
      });
      return cached || fetched;
    })
  );
});

// ── 推送通知: 风控告警 ──
self.addEventListener('push', (e) => {
  const data = e.data?.json() || { title: '灵枢风控', body: '请检查系统' };
  e.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: '/favicon.svg',
      badge: '/favicon.svg',
      tag: 'risk-alert',
      vibrate: [200, 100, 200],
      requireInteraction: data.level === 'CRITICAL',
    })
  );
});
