package main

import (
	"crypto/pbkdf2"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Node struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Children    []*Node `json:"children"`
}

type TreeNote struct {
	ID      string   `json:"id"`
	Name    string   `json:"name"`
	Owner   string   `json:"owner"`
	Editors []string `json:"editors"`
	Roots   []*Node  `json:"roots"` // Must be plural to match your JSON
}

type User struct {
	Username string `json:"username"`
	Salt     string `json:"salt"`
	PassHash string `json:"passHash"`
	Iters    int    `json:"iters"`
	Created  string `json:"created"`
}

type Session struct {
	Username string    `json:"username"`
	Expires  time.Time `json:"expires"`
}

// dataDir holds all persisted state (accounts, sessions, trees). It defaults
// to ./data but can be pointed outside the repo with ATHENEUM_DATA_DIR so
// that git operations / redeploys never touch user accounts.
var dataDir = "./data"

const sessionCookie = "atheneum_session"
const sessionTTL = 7 * 24 * time.Hour

// PBKDF2 cost for new accounts; accounts created before the iters field
// existed are stored with legacyPbkdf2Iters
const pbkdf2Iters = 600_000
const legacyPbkdf2Iters = 210_000

// Request body limits
const maxBodyAuth = 64 << 10 // login, register, editors, create
const maxBodyTree = 8 << 20  // tree PUT

// Brute-force protection for login and register
const maxFailures = 5

var (
	usersMu sync.Mutex
	users   = make(map[string]User) // keyed by lowercase username

	sessionsMu sync.Mutex
	sessions   = make(map[string]Session) // keyed by token
)

// dummySalt burns the same PBKDF2 cost on login attempts for unknown
// usernames so timing does not reveal which accounts exist
var dummySalt = []byte("0123456789abcdef")

func main() {
	// Allow the data directory to live outside the repo (recommended on servers)
	if d := strings.TrimSpace(os.Getenv("ATHENEUM_DATA_DIR")); d != "" {
		dataDir = d
	}

	// Ensure data directory exists
	if err := os.MkdirAll(dataDir, os.ModePerm); err != nil {
		log.Fatalf("Could not create data directory: %v", err)
	}
	log.Printf("Data directory: %s", dataDir)

	loadUsers()
	loadSessions()

	// API Endpoints
	http.HandleFunc("POST /api/register", handleRegister)
	http.HandleFunc("POST /api/login", handleLogin)
	http.HandleFunc("POST /api/logout", handleLogout)
	http.HandleFunc("GET /api/auth", handleAuthStatus)
	http.HandleFunc("GET /api/users", handleListUsers)
	http.HandleFunc("POST /api/new", handleCreateTree)
	http.HandleFunc("GET /api/list", handleListTrees)
	http.HandleFunc("GET /api/tree/{id}", handleTreeGet)
	http.HandleFunc("PUT /api/tree/{id}", handleTreePut)
	http.HandleFunc("DELETE /api/tree/{id}", handleTreeDelete)
	http.HandleFunc("POST /api/tree/{id}/editors", handleTreeEditors)
	http.HandleFunc("POST /api/tree/{id}/copy", handleTreeCopy)

	// Static/HTML Pages
	http.HandleFunc("/edit/", serveHTML("static/edit.html"))
	http.HandleFunc("/view/", serveHTML("static/view.html"))
	http.HandleFunc("/practice/", serveHTML("static/practice.html"))
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("./static/"))))
	http.HandleFunc("/", serveHTML("static/index.html")) // Homepage to create new

	fmt.Println("Server running on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", withSecurityHeaders(http.DefaultServeMux)))
}

// withSecurityHeaders adds conservative hardening headers to every response
func withSecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Referrer-Policy", "same-origin")
		next.ServeHTTP(w, r)
	})
}

// Generates a random hex ID
func generateID() string {
	bytes := make([]byte, 8)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

func writeJSONFile(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ---- User store ----

func usersPath() string    { return filepath.Join(dataDir, "users.json") }
func sessionsPath() string { return filepath.Join(dataDir, "sessions.json") }

func loadUsers() {
	data, err := os.ReadFile(usersPath())
	if err != nil {
		return
	}
	var list []User
	if err := json.Unmarshal(data, &list); err != nil {
		log.Printf("Could not parse users.json: %v", err)
		return
	}
	usersMu.Lock()
	defer usersMu.Unlock()
	for _, u := range list {
		users[strings.ToLower(u.Username)] = u
	}
	log.Printf("Loaded %d user(s)", len(users))
}

// saveUsers writes the user store to disk. Callers must hold usersMu.
func saveUsers() error {
	list := make([]User, 0, len(users))
	for _, u := range users {
		list = append(list, u)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Username < list[j].Username })
	return writeJSONFile(usersPath(), list)
}

func validUsername(name string) bool {
	if len(name) < 3 || len(name) > 32 {
		return false
	}
	for _, c := range name {
		switch {
		case c >= 'a' && c <= 'z':
		case c >= '0' && c <= '9':
		case c == '-' || c == '_':
		default:
			return false
		}
	}
	return true
}

func hashPassword(password string, salt []byte, iters int) (string, error) {
	key, err := pbkdf2.Key(sha256.New, password, salt, iters, 32)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(key), nil
}

func userExists(username string) bool {
	usersMu.Lock()
	defer usersMu.Unlock()
	_, ok := users[username]
	return ok
}

// ---- Sessions ----

func loadSessions() {
	data, err := os.ReadFile(sessionsPath())
	if err != nil {
		return
	}
	var m map[string]Session
	if err := json.Unmarshal(data, &m); err != nil {
		log.Printf("Could not parse sessions.json: %v", err)
		return
	}
	now := time.Now()
	sessionsMu.Lock()
	defer sessionsMu.Unlock()
	for token, s := range m {
		if s.Expires.After(now) {
			sessions[token] = s
		}
	}
}

// saveSessions writes the session store to disk, dropping expired entries.
// Callers must hold sessionsMu.
func saveSessions() error {
	now := time.Now()
	live := make(map[string]Session, len(sessions))
	for token, s := range sessions {
		if s.Expires.After(now) {
			live[token] = s
		}
	}
	for token := range sessions {
		if _, ok := live[token]; !ok {
			delete(sessions, token)
		}
	}
	return writeJSONFile(sessionsPath(), live)
}

// isHTTPS reports whether the request arrived over TLS, including via a
// reverse proxy that terminates TLS
func isHTTPS(r *http.Request) bool {
	return r.TLS != nil || strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https")
}

func newSession(w http.ResponseWriter, r *http.Request, username string) {
	b := make([]byte, 32)
	rand.Read(b)
	token := hex.EncodeToString(b)
	sessionsMu.Lock()
	sessions[token] = Session{Username: username, Expires: time.Now().Add(sessionTTL)}
	if err := saveSessions(); err != nil {
		log.Printf("Could not persist sessions: %v", err)
	}
	sessionsMu.Unlock()
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookie,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		Secure:   isHTTPS(r),
		SameSite: http.SameSiteLaxMode,
		MaxAge:   int(sessionTTL.Seconds()),
	})
}

// currentUser returns the username of the request's session, or "" when logged out
func currentUser(r *http.Request) string {
	c, err := r.Cookie(sessionCookie)
	if err != nil || c.Value == "" {
		return ""
	}
	sessionsMu.Lock()
	s, ok := sessions[c.Value]
	sessionsMu.Unlock()
	if !ok {
		return ""
	}
	if time.Now().After(s.Expires) {
		sessionsMu.Lock()
		delete(sessions, c.Value)
		saveSessions()
		sessionsMu.Unlock()
		return ""
	}
	return s.Username
}

func endSession(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(sessionCookie); err == nil {
		sessionsMu.Lock()
		delete(sessions, c.Value)
		saveSessions()
		sessionsMu.Unlock()
	}
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookie,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		Secure:   isHTTPS(r),
		MaxAge:   -1,
	})
}

func jsonError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// limitBody caps the request body size to protect against oversized payloads
func limitBody(w http.ResponseWriter, r *http.Request, max int64) bool {
	r.Body = http.MaxBytesReader(w, r.Body, max)
	return true
}

// ---- Brute-force throttling (per client IP) ----

type ipAttempts struct {
	failures    int
	lockedUntil time.Time
}

var (
	attemptsMu sync.Mutex
	attempts   = make(map[string]ipAttempts)
)

// clientIP extracts the peer IP; X-Forwarded-For is deliberately not trusted
func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// checkThrottle writes a 429 response and returns false while the IP is locked out
func checkThrottle(w http.ResponseWriter, r *http.Request) bool {
	ip := clientIP(r)
	attemptsMu.Lock()
	a := attempts[ip]
	attemptsMu.Unlock()
	if d := time.Until(a.lockedUntil); d > 0 {
		w.Header().Set("Retry-After", strconv.Itoa(int(d.Seconds())+1))
		jsonError(w, http.StatusTooManyRequests, "too many failed attempts, try again later")
		return false
	}
	return true
}

func recordFailure(r *http.Request) {
	ip := clientIP(r)
	attemptsMu.Lock()
	defer attemptsMu.Unlock()
	if len(attempts) > 10_000 {
		now := time.Now()
		for k, v := range attempts {
			if v.lockedUntil.Before(now) && v.failures < maxFailures {
				delete(attempts, k)
			}
		}
		if len(attempts) > 50_000 {
			attempts = make(map[string]ipAttempts)
		}
	}
	a := attempts[ip]
	a.failures++
	if a.failures >= maxFailures {
		shift := min(a.failures-maxFailures, 11) // cap backoff at 12h
		a.lockedUntil = time.Now().Add(time.Duration(1<<shift) * 15 * time.Second)
	}
	attempts[ip] = a
}

func recordSuccess(r *http.Request) {
	attemptsMu.Lock()
	delete(attempts, clientIP(r))
	attemptsMu.Unlock()
}

// requireLogin writes a 401 response and returns "" when there is no valid session
func requireLogin(w http.ResponseWriter, r *http.Request) string {
	username := currentUser(r)
	if username != "" {
		return username
	}
	jsonError(w, http.StatusUnauthorized, "login required")
	return ""
}

// ---- Auth handlers ----

// POST /api/register - creates an account and logs the user in
func handleRegister(w http.ResponseWriter, r *http.Request) {
	if !checkThrottle(w, r) {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	username := strings.ToLower(strings.TrimSpace(body.Username))
	if !validUsername(username) {
		jsonError(w, http.StatusBadRequest, "username must be 3-32 characters using letters, numbers, - or _")
		return
	}
	if len(body.Password) < 8 {
		jsonError(w, http.StatusBadRequest, "password must be at least 8 characters")
		return
	}

	salt := make([]byte, 16)
	rand.Read(salt)
	passHash, err := hashPassword(body.Password, salt, pbkdf2Iters)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, "could not hash password")
		return
	}

	usersMu.Lock()
	if _, exists := users[username]; exists {
		usersMu.Unlock()
		jsonError(w, http.StatusConflict, "username already taken")
		return
	}
	users[username] = User{
		Username: username,
		Salt:     hex.EncodeToString(salt),
		PassHash: passHash,
		Iters:    pbkdf2Iters,
		Created:  time.Now().UTC().Format(time.RFC3339),
	}
	if err := saveUsers(); err != nil {
		delete(users, username)
		usersMu.Unlock()
		log.Printf("Could not persist users: %v", err)
		jsonError(w, http.StatusInternalServerError, "could not save user")
		return
	}
	usersMu.Unlock()

	newSession(w, r, username)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"ok": true, "username": username})
}

// POST /api/login - exchanges username and password for a session cookie
func handleLogin(w http.ResponseWriter, r *http.Request) {
	if !checkThrottle(w, r) {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	username := strings.ToLower(strings.TrimSpace(body.Username))

	usersMu.Lock()
	user, ok := users[username]
	usersMu.Unlock()

	failed := func() {
		recordFailure(r)
		time.Sleep(300 * time.Millisecond)
		jsonError(w, http.StatusUnauthorized, "incorrect username or password")
	}
	iters := user.Iters
	if iters == 0 {
		iters = legacyPbkdf2Iters // accounts created before iters were stored
	}
	if !ok {
		// Spend the same PBKDF2 cost as a real lookup so response timing
		// does not reveal whether the account exists
		user.Salt = hex.EncodeToString(dummySalt)
	}
	salt, err := hex.DecodeString(user.Salt)
	if err != nil {
		failed()
		return
	}
	passHash, err := hashPassword(body.Password, salt, iters)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, "could not verify password")
		return
	}
	if !ok || subtle.ConstantTimeCompare([]byte(passHash), []byte(user.PassHash)) != 1 {
		failed()
		return
	}

	recordSuccess(r)
	newSession(w, r, user.Username)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"ok": true, "username": user.Username})
}

// POST /api/logout - ends the current session
func handleLogout(w http.ResponseWriter, r *http.Request) {
	endSession(w, r)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

// GET /api/auth - reports the logged in user for this browser
func handleAuthStatus(w http.ResponseWriter, r *http.Request) {
	username := currentUser(r)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"loggedIn": username != "",
		"username": username,
	})
}

// GET /api/users - lists registered usernames (for the sharing dialog)
func handleListUsers(w http.ResponseWriter, r *http.Request) {
	if requireLogin(w, r) == "" {
		return
	}
	usersMu.Lock()
	list := make([]string, 0, len(users))
	for name := range users {
		list = append(list, name)
	}
	usersMu.Unlock()
	sort.Strings(list)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string][]string{"users": list})
}

// ---- Tree permissions ----

// isOwner reports whether username owns the tree.
// Legacy trees created before accounts existed have no owner and any
// logged-in user may manage them.
func isOwner(tree *TreeNote, username string) bool {
	if username == "" {
		return false
	}
	if tree.Owner == "" {
		return true
	}
	return tree.Owner == username
}

// canEdit reports whether username may edit the tree: the owner or an editor
func canEdit(tree *TreeNote, username string) bool {
	if username == "" {
		return false
	}
	if isOwner(tree, username) {
		return true
	}
	for _, e := range tree.Editors {
		if e == username {
			return true
		}
	}
	return false
}

func loadTree(id string) (*TreeNote, error) {
	data, err := os.ReadFile(filepath.Join(dataDir, id+".json"))
	if err != nil {
		return nil, err
	}
	var tree TreeNote
	if err := json.Unmarshal(data, &tree); err != nil {
		return nil, err
	}
	tree.ID = id
	return &tree, nil
}

// ---- Tree handlers ----

// POST /api/new - Creates a new tree owned by the current user and returns the ID
func handleCreateTree(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}

	limitBody(w, r, maxBodyAuth)
	var body struct {
		Name string `json:"name"`
	}
	json.NewDecoder(r.Body).Decode(&body) // body is optional
	name := strings.TrimSpace(body.Name)
	if name == "" {
		name = "New System"
	}

	id := generateID()
	tree := TreeNote{
		ID:      id,
		Name:    name,
		Owner:   username,
		Editors: []string{},
		Roots: []*Node{ // Initialize as a slice
			{
				ID:          "root-" + generateID(),
				Name:        "New Root Node",
				Description: "Enter description here...",
				Children:    []*Node{},
			},
		},
	}

	if err := saveTree(id, &tree); err != nil {
		jsonError(w, http.StatusInternalServerError, "could not create tree")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": id})
}

// validID reports whether id is safe to use as a file name under dataDir
func validID(id string) bool {
	if id == "" || len(id) > 64 {
		return false
	}
	for _, c := range id {
		switch {
		case c >= 'a' && c <= 'z':
		case c >= 'A' && c <= 'Z':
		case c >= '0' && c <= '9':
		case c == '-' || c == '_':
		default:
			return false
		}
	}
	return true
}

// GET /api/tree/{id} - public read access
func handleTreeGet(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid tree ID")
		return
	}
	file, err := os.Open(filepath.Join(dataDir, id+".json"))
	if err != nil {
		jsonError(w, http.StatusNotFound, "tree not found")
		return
	}
	defer file.Close()
	w.Header().Set("Content-Type", "application/json")
	io.Copy(w, file)
}

// PUT /api/tree/{id} - only the owner and editors may save
func handleTreePut(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid tree ID")
		return
	}
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	existing, err := loadTree(id)
	if err != nil {
		jsonError(w, http.StatusNotFound, "tree not found")
		return
	}
	if !canEdit(existing, username) {
		jsonError(w, http.StatusForbidden, "only the owner and editors can edit this system")
		return
	}

	limitBody(w, r, maxBodyTree)
	var tree TreeNote
	if err := json.NewDecoder(r.Body).Decode(&tree); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	// Ownership and the editor list are managed server-side only
	tree.ID = id
	tree.Owner = existing.Owner
	tree.Editors = existing.Editors
	if err := saveTree(id, &tree); err != nil {
		jsonError(w, http.StatusInternalServerError, "could not save tree")
		return
	}
	w.WriteHeader(http.StatusOK)
}

// DELETE /api/tree/{id} - only the owner may delete
func handleTreeDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid tree ID")
		return
	}
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	existing, err := loadTree(id)
	if err != nil {
		jsonError(w, http.StatusNotFound, "tree not found")
		return
	}
	if !isOwner(existing, username) {
		jsonError(w, http.StatusForbidden, "only the owner can delete this system")
		return
	}
	if err := os.Remove(filepath.Join(dataDir, id+".json")); err != nil {
		jsonError(w, http.StatusNotFound, "tree not found")
		return
	}
	w.WriteHeader(http.StatusOK)
}

// POST /api/tree/{id}/copy - duplicates any public tree as a new system
// owned by the current user
func handleTreeCopy(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid tree ID")
		return
	}
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	src, err := loadTree(id)
	if err != nil {
		jsonError(w, http.StatusNotFound, "tree not found")
		return
	}

	name := strings.TrimSpace(src.Name)
	if name == "" {
		name = "Untitled Tree"
	}
	newID := generateID()
	tree := TreeNote{
		ID:      newID,
		Name:    name + " (copy)",
		Owner:   username,
		Editors: []string{},
		Roots:   src.Roots, // node IDs are scoped to a single tree, safe to reuse
	}
	if err := saveTree(newID, &tree); err != nil {
		jsonError(w, http.StatusInternalServerError, "could not create copy")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": newID})
}

// POST /api/tree/{id}/editors - the owner adds or removes editors.
// Body: {"username": "...", "action": "add"|"remove"}
func handleTreeEditors(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid tree ID")
		return
	}
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	tree, err := loadTree(id)
	if err != nil {
		jsonError(w, http.StatusNotFound, "tree not found")
		return
	}
	if !isOwner(tree, username) {
		jsonError(w, http.StatusForbidden, "only the owner can manage editors")
		return
	}

	limitBody(w, r, maxBodyAuth)
	var body struct {
		Username string `json:"username"`
		Action   string `json:"action"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	target := strings.ToLower(strings.TrimSpace(body.Username))
	if !validUsername(target) {
		jsonError(w, http.StatusBadRequest, "invalid username")
		return
	}
	if target == tree.Owner {
		jsonError(w, http.StatusBadRequest, "the owner is always an editor")
		return
	}
	if !userExists(target) {
		jsonError(w, http.StatusNotFound, "no user with that username")
		return
	}

	has := false
	kept := make([]string, 0, len(tree.Editors))
	for _, e := range tree.Editors {
		if e == target {
			has = true
			continue
		}
		kept = append(kept, e)
	}
	switch body.Action {
	case "add":
		if has {
			jsonError(w, http.StatusConflict, "user is already an editor")
			return
		}
		tree.Editors = append(kept, target)
	case "remove":
		if !has {
			jsonError(w, http.StatusNotFound, "user is not an editor")
			return
		}
		tree.Editors = kept
	default:
		jsonError(w, http.StatusBadRequest, `action must be "add" or "remove"`)
		return
	}

	if err := saveTree(id, tree); err != nil {
		jsonError(w, http.StatusInternalServerError, "could not save tree")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"ok": true, "editors": tree.Editors})
}

func saveTree(id string, tree *TreeNote) error {
	data, err := json.MarshalIndent(tree, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dataDir, id+".json"), data, 0644)
}

// Helper to serve HTML files
func serveHTML(filename string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, filename)
	}
}

func handleListTrees(w http.ResponseWriter, r *http.Request) {
	files, err := os.ReadDir(dataDir)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, "could not read data")
		return
	}

	type TreeInfo struct {
		ID      string   `json:"id"`
		Name    string   `json:"name"`
		Owner   string   `json:"owner"`
		Editors []string `json:"editors"`
	}
	list := make([]TreeInfo, 0, len(files))

	for _, f := range files {
		if filepath.Ext(f.Name()) != ".json" || f.Name() == "users.json" || f.Name() == "sessions.json" {
			continue
		}
		fileID := strings.TrimSuffix(f.Name(), ".json")
		if !validID(fileID) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dataDir, f.Name()))
		if err != nil {
			continue
		}
		var tree TreeNote
		if err := json.Unmarshal(data, &tree); err != nil {
			continue
		}
		if tree.ID == "" {
			tree.ID = fileID
		}

		name := strings.TrimSpace(tree.Name)
		if name == "" && len(tree.Roots) > 0 && tree.Roots[0] != nil {
			name = tree.Roots[0].Name // legacy files without a name field
		}
		if name == "" {
			name = "Untitled Tree"
		}
		editors := tree.Editors
		if editors == nil {
			editors = []string{}
		}

		list = append(list, TreeInfo{
			ID:      tree.ID,
			Name:    name,
			Owner:   tree.Owner,
			Editors: editors,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}
