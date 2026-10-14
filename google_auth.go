package main

// Google sign-in. The browser obtains an ID token ("credential") from the
// Google Identity Services script and posts it to /api/auth/google; the
// server verifies it against Google's published keys and starts a normal
// session cookie. Accounts are keyed by the Google subject ID; a matching
// email links to an existing password account, otherwise a new account is
// created with a username derived from the email.

import (
	"context"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// googleClientID is the OAuth client ID from the Google Cloud console.
// When empty, Google sign-in is disabled and the button is not shown.
var googleClientID = strings.TrimSpace(os.Getenv("GOOGLE_CLIENT_ID"))

const googleCertsURL = "https://www.googleapis.com/oauth2/v3/certs"

var (
	googleKeysMu      sync.Mutex
	googleKeys        map[string]*rsa.PublicKey // keyed by kid
	googleKeysFetched time.Time
)

// fetchGoogleKeys downloads Google's JWKS and builds the RSA public keys.
// Callers must hold googleKeysMu.
func fetchGoogleKeys(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", googleCertsURL, nil)
	if err != nil {
		return err
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("google certs: status %d", res.StatusCode)
	}
	var doc struct {
		Keys []struct {
			Kid string `json:"kid"`
			Kty string `json:"kty"`
			N   string `json:"n"`
			E   string `json:"e"`
		} `json:"keys"`
	}
	if err := json.NewDecoder(res.Body).Decode(&doc); err != nil {
		return err
	}
	keys := make(map[string]*rsa.PublicKey, len(doc.Keys))
	for _, k := range doc.Keys {
		if k.Kty != "RSA" || k.Kid == "" {
			continue
		}
		nb, err := base64.RawURLEncoding.DecodeString(k.N)
		if err != nil {
			return err
		}
		ebx, err := base64.RawURLEncoding.DecodeString(k.E)
		if err != nil {
			return err
		}
		e := 0
		for _, b := range ebx {
			e = e<<8 | int(b)
		}
		keys[k.Kid] = &rsa.PublicKey{N: new(big.Int).SetBytes(nb), E: e}
	}
	if len(keys) == 0 {
		return fmt.Errorf("google certs: no RSA keys")
	}
	googleKeys, googleKeysFetched = keys, time.Now()
	return nil
}

// googleKey returns the signing key for a kid, refreshing the cached set on
// an unknown kid (key rotation) or when it is older than an hour
func googleKey(ctx context.Context, kid string) (*rsa.PublicKey, error) {
	googleKeysMu.Lock()
	defer googleKeysMu.Unlock()
	if key, ok := googleKeys[kid]; ok && time.Since(googleKeysFetched) < time.Hour {
		return key, nil
	}
	if err := fetchGoogleKeys(ctx); err != nil {
		if key, ok := googleKeys[kid]; ok {
			return key, nil // serve stale keys when Google is unreachable
		}
		return nil, err
	}
	if key, ok := googleKeys[kid]; ok {
		return key, nil
	}
	return nil, jwt.ErrTokenSignatureInvalid
}

type googleClaims struct {
	Sub           string `json:"sub"`
	Email         string `json:"email"`
	EmailVerified bool   `json:"email_verified"`
	jwt.RegisteredClaims
}

// verifyGoogleIDToken checks the signature, issuer, audience and expiry of
// a Google ID token and returns its claims
func verifyGoogleIDToken(ctx context.Context, credential string) (*googleClaims, error) {
	keyfunc := func(t *jwt.Token) (any, error) {
		kid, _ := t.Header["kid"].(string)
		if kid == "" {
			return nil, jwt.ErrTokenInvalidId
		}
		return googleKey(ctx, kid)
	}
	var claims googleClaims
	_, err := jwt.ParseWithClaims(credential, &claims, keyfunc,
		jwt.WithValidMethods([]string{"RS256"}),
		jwt.WithAudience(googleClientID),
		jwt.WithExpirationRequired(),
	)
	if err != nil {
		return nil, err
	}
	if claims.Issuer != "https://accounts.google.com" && claims.Issuer != "accounts.google.com" {
		return nil, jwt.ErrTokenInvalidIssuer
	}
	if claims.Sub == "" || claims.Email == "" || !claims.EmailVerified {
		return nil, jwt.ErrTokenInvalidClaims
	}
	return &claims, nil
}

// googleUsername derives a free username from the Google email address.
// Callers must hold usersMu.
func googleUsername(email string) string {
	local, _, _ := strings.Cut(email, "@")
	var b strings.Builder
	for _, c := range strings.ToLower(local) {
		switch {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '_', c == '-':
			b.WriteRune(c)
		default:
			b.WriteRune('_')
		}
	}
	base := b.String()
	if len(base) > 24 {
		base = base[:24]
	}
	for len(base) < 3 {
		base += "0"
	}
	name := base
	for i := 1; ; i++ {
		if _, taken := users[name]; !taken {
			return name
		}
		name = base + "-" + strconv.Itoa(i)
	}
}

// googleUser finds or creates the local account for a verified Google login.
// Callers must hold usersMu.
func googleUser(claims *googleClaims) (User, error) {
	email := strings.ToLower(claims.Email)
	for _, u := range users {
		if u.GoogleID != "" && u.GoogleID == claims.Sub {
			return u, nil
		}
	}
	for key, u := range users {
		if u.Email == email && u.GoogleID == "" {
			u.GoogleID = claims.Sub
			users[key] = u
			if err := saveUsers(); err != nil {
				return User{}, err
			}
			return u, nil
		}
	}
	u := User{
		Username: googleUsername(email),
		Email:    email,
		GoogleID: claims.Sub,
		Created:  time.Now().UTC().Format(time.RFC3339),
	}
	users[u.Username] = u
	if err := saveUsers(); err != nil {
		delete(users, u.Username)
		return User{}, err
	}
	return u, nil
}

// POST /api/auth/google - exchanges a Google ID token for a session cookie
func handleGoogleAuth(w http.ResponseWriter, r *http.Request) {
	if googleClientID == "" {
		jsonError(w, http.StatusNotFound, "google sign-in is not configured")
		return
	}
	if !checkThrottle(w, r) {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Credential string `json:"credential"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Credential == "" {
		jsonError(w, http.StatusBadRequest, "missing google credential")
		return
	}
	claims, err := verifyGoogleIDToken(r.Context(), body.Credential)
	if err != nil {
		recordFailure(r)
		jsonError(w, http.StatusUnauthorized, "google sign-in failed")
		return
	}
	usersMu.Lock()
	user, err := googleUser(claims)
	if err != nil {
		usersMu.Unlock()
		log.Printf("Could not persist google user: %v", err)
		jsonError(w, http.StatusInternalServerError, "could not save user")
		return
	}
	usersMu.Unlock()

	recordSuccess(r)
	newSession(w, r, user.Username)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"ok": true, "username": user.Username})
}
