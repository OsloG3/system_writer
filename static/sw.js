// Bridge Trainer service worker: shows Web Push turn alerts while the page
// is closed. A focused /play window already alerts in-page, so the popup is
// suppressed there to avoid double notifications.

self.addEventListener("push", (event) => {
    let data = {};
    try {
        data = event.data ? event.data.json() : {};
    } catch (e) {
        data = { body: event.data ? event.data.text() : "" };
    }
    event.waitUntil(
        clients
            .matchAll({ type: "window", includeUncontrolled: true })
            .then((list) => {
                if (list.some((c) => c.focused)) return;
                const title = data.title || "Bridge Trainer";
                const opts = {
                    body: data.body || "",
                    tag: data.tag || "turn",
                    renotify: true,
                    icon: "/static/icon-192.png",
                    badge: "/static/icon-192.png",
                    data: data.data || {},
                };
                return self.registration
                    .showNotification(title, opts)
                    .catch(() => {
                        // older engines reject renotify/tag combos
                        delete opts.renotify;
                        return self.registration.showNotification(title, opts);
                    });
            }),
    );
});

self.addEventListener("notificationclick", (event) => {
    event.notification.close();
    event.waitUntil(
        clients
            .matchAll({ type: "window", includeUncontrolled: true })
            .then((list) => {
                for (const c of list) {
                    if ("focus" in c) return c.focus();
                }
                return clients.openWindow("/play");
            }),
    );
});
