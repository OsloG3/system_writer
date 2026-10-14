package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func googleTestEnv(t *testing.T) {
	t.Helper()
	oldDir, oldUsers, oldClientID := dataDir, users, googleClientID
	dataDir = t.TempDir()
	users = make(map[string]User)
	t.Cleanup(func() {
		dataDir, users, googleClientID = oldDir, oldUsers, oldClientID
	})
}

func TestGoogleUsername(t *testing.T) {
	googleTestEnv(t)
	if got := googleUsername("Alice.Bob@Example.com"); got != "alice_bob" {
		t.Fatalf("googleUsername: %q", got)
	}
	if got := googleUsername("ab@x.com"); got != "ab0" {
		t.Fatalf("short local part: %q", got)
	}
	if got := googleUsername(strings.Repeat("x", 40) + "@y.com"); len(got) != 24 {
		t.Fatalf("long local part: %q", got)
	}
	users["alice_bob"] = User{Username: "alice_bob"}
	if got := googleUsername("alice.bob@z.com"); got != "alice_bob-1" {
		t.Fatalf("collision: %q", got)
	}
	users["alice_bob-1"] = User{Username: "alice_bob-1"}
	if got := googleUsername("alice.bob@z.com"); got != "alice_bob-2" {
		t.Fatalf("second collision: %q", got)
	}
}

func TestGoogleUser(t *testing.T) {
	googleTestEnv(t)
	claims := &googleClaims{Sub: "g-123", Email: "carol@x.com", EmailVerified: true}

	u, err := googleUser(claims)
	if err != nil {
		t.Fatalf("googleUser: %v", err)
	}
	if u.Username != "carol" || u.GoogleID != "g-123" || u.PassHash != "" {
		t.Fatalf("new account: %+v", u)
	}
	if _, ok := users["carol"]; !ok {
		t.Fatal("account not stored")
	}

	// a second login with the same subject finds the same account
	again, err := googleUser(claims)
	if err != nil {
		t.Fatalf("googleUser again: %v", err)
	}
	if again.Username != "carol" {
		t.Fatalf("duplicate account: %+v", again)
	}

	// a verified email links to a pre-existing password account
	users["dave"] = User{Username: "dave", Email: "dave@x.com", Salt: "s", PassHash: "h", Iters: 1}
	linked, err := googleUser(&googleClaims{Sub: "g-999", Email: "dave@x.com", EmailVerified: true})
	if err != nil {
		t.Fatalf("googleUser link: %v", err)
	}
	if linked.Username != "dave" || linked.GoogleID != "g-999" || linked.PassHash != "h" {
		t.Fatalf("linked account: %+v", linked)
	}
	if len(users) != 2 {
		t.Fatalf("link created an extra account: %+v", users)
	}
}

func TestGoogleAuthDisabled(t *testing.T) {
	googleTestEnv(t)
	googleClientID = ""
	rr := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/auth/google", strings.NewReader(`{"credential":"x"}`))
	handleGoogleAuth(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404 without a client ID, got %d", rr.Code)
	}
}

func TestGoogleAuthRejectsGarbage(t *testing.T) {
	googleTestEnv(t)
	googleClientID = "test-client"
	t.Cleanup(func() { attempts = make(map[string]ipAttempts) })

	rr := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/auth/google", strings.NewReader(`{"credential":"not-a-jwt"}`))
	req.RemoteAddr = "192.0.2.1:1234"
	handleGoogleAuth(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for an invalid token, got %d", rr.Code)
	}

	rr = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "/api/auth/google", strings.NewReader(`{}`))
	handleGoogleAuth(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 without a credential, got %d", rr.Code)
	}
}
