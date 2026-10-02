package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	webpush "github.com/SherClockHolmes/webpush-go"
)

// ---- Web Push: turn alerts that reach closed pages ----
//
// The in-page turn alerts only work while the /play tab is open. For real
// background delivery the browser hands us a PushSubscription (an endpoint
// on the OS push service plus P-256 keys), we store it per user and, when a
// team board flips to a member's turn, POST a VAPID-signed encrypted
// notification to that endpoint (RFC 8291/8292 via webpush-go). The service
// worker (static/sw.js) shows it, suppressing the popup when a focused /play
// window already alerts in-page.
//
// The VAPID identity lives in data/vapid.json and is generated on first
// start. Requirements on the client: a secure origin (HTTPS or localhost);
// iOS only delivers to sites installed on the home screen (iOS 16.4+).

const pushTTL = 15 * time.Minute // stale turn alerts are dropped, not queued forever

// pushSubscription is one browser endpoint bound to a username.
type pushSubscription struct {
	User     string `json:"user"`
	Endpoint string `json:"endpoint"`
	P256DH   string `json:"p256dh"`
	Auth     string `json:"auth"`
	Added    string `json:"added"`
}

var (
	pushMu   sync.Mutex
	pushSubs = map[string]pushSubscription{} // keyed by endpoint (globally unique)

	vapidPriv    string
	vapidPub     string
	vapidSubject = "mailto:bridge-trainer@localhost"
)

func pushSubsPath() string { return filepath.Join(dataDir, "push_subs.json") }
func vapidPath() string    { return filepath.Join(dataDir, "vapid.json") }

func initPush() {
	if s := strings.TrimSpace(os.Getenv("PUSH_VAPID_SUBJECT")); s != "" {
		vapidSubject = s
	}
	if err := loadVAPID(); err != nil {
		log.Printf("Web push disabled: %v", err)
		return
	}
	loadPushSubs()
	log.Printf("Web push ready: %d subscription(s), subject %s", len(pushSubs), vapidSubject)
}

// loadVAPID reads the persisted identity or generates and saves a new one.
func loadVAPID() error {
	data, err := os.ReadFile(vapidPath())
	if err == nil {
		var k struct {
			Private string `json:"private"`
			Public  string `json:"public"`
		}
		if err := json.Unmarshal(data, &k); err == nil && k.Private != "" && k.Public != "" {
			vapidPriv, vapidPub = k.Private, k.Public
			return nil
		}
	}
	priv, pub, err := webpush.GenerateVAPIDKeys()
	if err != nil {
		return err
	}
	vapidPriv, vapidPub = priv, pub
	return writeJSONFile(vapidPath(), map[string]string{"private": priv, "public": pub})
}

func loadPushSubs() {
	data, err := os.ReadFile(pushSubsPath())
	if err != nil {
		return
	}
	var list []pushSubscription
	if err := json.Unmarshal(data, &list); err != nil {
		log.Printf("Could not parse push_subs.json: %v", err)
		return
	}
	pushMu.Lock()
	defer pushMu.Unlock()
	for _, s := range list {
		if s.Endpoint != "" && s.User != "" {
			pushSubs[s.Endpoint] = s
		}
	}
}

// savePushSubsLocked persists the subscription store. Callers hold pushMu.
func savePushSubsLocked() error {
	list := make([]pushSubscription, 0, len(pushSubs))
	for _, s := range pushSubs {
		list = append(list, s)
	}
	return writeJSONFile(pushSubsPath(), list)
}

func pushOptions() *webpush.Options {
	return &webpush.Options{
		Subscriber:      vapidSubject,
		TTL:             int(pushTTL.Seconds()),
		Urgency:         webpush.UrgencyHigh,
		VAPIDPublicKey:  vapidPub,
		VAPIDPrivateKey: vapidPriv,
	}
}

// sendPush delivers a payload to every endpoint of a user, dropping the ones
// the push service reports as gone.
func sendPush(user string, payload map[string]any) {
	go func() {
		data, _ := json.Marshal(payload)
		deliverPush(user, data)
	}()
}

func deliverPush(user string, data []byte) {
	if vapidPriv == "" || user == "" {
		return
	}
	pushMu.Lock()
	subs := make([]pushSubscription, 0, 2)
	for _, s := range pushSubs {
		if s.User == user {
			subs = append(subs, s)
		}
	}
	pushMu.Unlock()
	if len(subs) == 0 {
		return
	}
	opts := pushOptions()
	var dead []string
	for _, s := range subs {
		resp, err := webpush.SendNotification(data, &webpush.Subscription{
			Endpoint: s.Endpoint,
			Keys:     webpush.Keys{P256dh: s.P256DH, Auth: s.Auth},
		}, opts)
		if err != nil {
			log.Printf("push to %s failed: %v", s.Endpoint[:min(len(s.Endpoint), 40)], err)
			continue
		}
		resp.Body.Close()
		// 404/410: the subscription expired or was dropped by the service
		if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusGone {
			dead = append(dead, s.Endpoint)
		}
	}
	if len(dead) > 0 {
		pushMu.Lock()
		for _, ep := range dead {
			delete(pushSubs, ep)
		}
		if err := savePushSubsLocked(); err != nil {
			log.Printf("Could not persist push subscriptions: %v", err)
		}
		pushMu.Unlock()
	}
}

// sendTurnPush alerts one member that boards await their bid.
func sendTurnPush(user, gameID string, boards []int) {
	nums := make([]string, len(boards))
	for i, b := range boards {
		nums[i] = strconv.Itoa(b)
	}
	body := fmt.Sprintf("Boards %s are awaiting your bid", strings.Join(nums, ", "))
	if len(boards) == 1 {
		body = fmt.Sprintf("Board %s is awaiting your bid", nums[0])
	}
	sendPush(user, map[string]any{
		"title": "Your turn · Bridge Trainer",
		"body":  body,
		"tag":   "turn-" + gameID,
		"data":  map[string]any{"table": gameID, "boards": boards},
	})
}

// ---- Handlers ----

// GET /api/play/push/key - the VAPID public key for PushManager.subscribe
func handlePushKey(w http.ResponseWriter, r *http.Request) {
	if requireLogin(w, r) == "" {
		return
	}
	if vapidPub == "" {
		jsonError(w, http.StatusServiceUnavailable, "web push is not configured on this server")
		return
	}
	writeJSON(w, map[string]string{"key": vapidPub})
}

// POST /api/play/push/subscribe - store the browser's PushSubscription for
// the logged-in user (the endpoint doubles as the dedupe key)
func handlePushSubscribe(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	if vapidPub == "" {
		jsonError(w, http.StatusServiceUnavailable, "web push is not configured on this server")
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Endpoint string `json:"endpoint"`
		Keys     struct {
			P256DH string `json:"p256dh"`
			Auth   string `json:"auth"`
		} `json:"keys"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if !strings.HasPrefix(body.Endpoint, "https://") || body.Keys.P256DH == "" || body.Keys.Auth == "" {
		jsonError(w, http.StatusBadRequest, "invalid push subscription")
		return
	}
	pushMu.Lock()
	pushSubs[body.Endpoint] = pushSubscription{
		User:     username,
		Endpoint: body.Endpoint,
		P256DH:   body.Keys.P256DH,
		Auth:     body.Keys.Auth,
		Added:    time.Now().UTC().Format(time.RFC3339),
	}
	err := savePushSubsLocked()
	pushMu.Unlock()
	if err != nil {
		log.Printf("Could not persist push subscriptions: %v", err)
		jsonError(w, http.StatusInternalServerError, "could not save subscription")
		return
	}
	writeJSON(w, map[string]bool{"ok": true})
}

// POST /api/play/push/unsubscribe - drop one of the user's endpoints
func handlePushUnsubscribe(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Endpoint string `json:"endpoint"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	pushMu.Lock()
	s, ok := pushSubs[body.Endpoint]
	if ok && s.User == username {
		delete(pushSubs, body.Endpoint)
	}
	err := savePushSubsLocked()
	pushMu.Unlock()
	if err != nil {
		log.Printf("Could not persist push subscriptions: %v", err)
	}
	writeJSON(w, map[string]bool{"ok": ok && s.User == username})
}
