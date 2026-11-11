package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// cardViewTestEnv swaps the user/session stores and data dir for the test,
// seeds one account with a live session and returns its cookie.
func cardViewTestEnv(t *testing.T) *http.Cookie {
	t.Helper()
	oldDir, oldUsers, oldSessions := dataDir, users, sessions
	dataDir = t.TempDir()
	users = map[string]User{"alice": {Username: "alice", Created: "2026-01-01T00:00:00Z"}}
	sessions = map[string]Session{
		"tok-alice": {Username: "alice", Expires: time.Now().Add(time.Hour)},
	}
	t.Cleanup(func() {
		dataDir, users, sessions = oldDir, oldUsers, oldSessions
	})
	return &http.Cookie{Name: sessionCookie, Value: "tok-alice"}
}

func cardViewReq(method, path, body string, c *http.Cookie) *http.Request {
	var r *http.Request
	if body != "" {
		r = httptest.NewRequest(method, path, strings.NewReader(body))
	} else {
		r = httptest.NewRequest(method, path, nil)
	}
	if c != nil {
		r.AddCookie(c)
	}
	return r
}

func TestCardViewRequiresLogin(t *testing.T) {
	cardViewTestEnv(t)
	rr := httptest.NewRecorder()
	handleUserCardView(rr, cardViewReq("POST", "/api/user/card-view", `{"cardView":"diagram"}`, nil))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("logged out: want 401, got %d", rr.Code)
	}
}

func TestCardViewRejectsBadValues(t *testing.T) {
	c := cardViewTestEnv(t)
	for _, body := range []string{`{"cardView":"tiny"}`, `{"cardView":""}`, `not json`} {
		rr := httptest.NewRecorder()
		handleUserCardView(rr, cardViewReq("POST", "/api/user/card-view", body, c))
		if rr.Code != http.StatusBadRequest {
			t.Fatalf("body %q: want 400, got %d", body, rr.Code)
		}
	}
	// the account keeps the default
	if got := cardViewOf("alice"); got != "cards" {
		t.Fatalf("cardViewOf after rejects: %q", got)
	}
}

func TestCardViewSetPersistAndAuth(t *testing.T) {
	c := cardViewTestEnv(t)

	// default before any choice
	rr := httptest.NewRecorder()
	handleAuthStatus(rr, cardViewReq("GET", "/api/auth", "", c))
	var auth map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &auth); err != nil {
		t.Fatalf("auth status: %v", err)
	}
	if auth["cardView"] != "cards" {
		t.Fatalf("default cardView: %v", auth["cardView"])
	}

	// switch to diagram
	rr = httptest.NewRecorder()
	handleUserCardView(rr, cardViewReq("POST", "/api/user/card-view", `{"cardView":"diagram"}`, c))
	if rr.Code != http.StatusOK {
		t.Fatalf("set diagram: want 200, got %d (%s)", rr.Code, rr.Body.String())
	}
	if got := cardViewOf("alice"); got != "diagram" {
		t.Fatalf("cardViewOf: %q", got)
	}

	// it reached the on-disk user store
	data, err := os.ReadFile(filepath.Join(dataDir, "users.json"))
	if err != nil {
		t.Fatalf("users.json: %v", err)
	}
	var stored []User
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatalf("users.json parse: %v (%s)", err, data)
	}
	if len(stored) != 1 || stored[0].Username != "alice" || stored[0].CardView != "diagram" {
		t.Fatalf("users.json missing cardView: %s", data)
	}

	// GET /api/auth now reports it
	rr = httptest.NewRecorder()
	handleAuthStatus(rr, cardViewReq("GET", "/api/auth", "", c))
	if err := json.Unmarshal(rr.Body.Bytes(), &auth); err != nil {
		t.Fatalf("auth status: %v", err)
	}
	if auth["cardView"] != "diagram" {
		t.Fatalf("auth cardView: %v", auth["cardView"])
	}

	// and back to cards
	rr = httptest.NewRecorder()
	handleUserCardView(rr, cardViewReq("POST", "/api/user/card-view", `{"cardView":"cards"}`, c))
	if rr.Code != http.StatusOK {
		t.Fatalf("set cards: want 200, got %d", rr.Code)
	}
	if got := cardViewOf("alice"); got != "cards" {
		t.Fatalf("cardViewOf after reset: %q", got)
	}
}

func TestCardViewOfUnknownUser(t *testing.T) {
	cardViewTestEnv(t)
	if got := cardViewOf(""); got != "cards" {
		t.Fatalf("logged out: %q", got)
	}
	if got := cardViewOf("nobody"); got != "cards" {
		t.Fatalf("unknown account: %q", got)
	}
}
