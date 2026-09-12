package main

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
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
	ID    string  `json:"id"`
	Name  string  `json:"name"`
	Roots []*Node `json:"roots"` // Must be plural to match your JSON
}

const dataDir = "./data"
const sessionCookie = "atheneum_session"

var (
	editPasscode string
	sessionsMu   sync.Mutex
	sessions     = make(map[string]bool)
)

func main() {
	// Ensure data directory exists
	if err := os.MkdirAll(dataDir, os.ModePerm); err != nil {
		log.Fatalf("Could not create data directory: %v", err)
	}

	configurePasscode()

	// API Endpoints
	http.HandleFunc("/api/new", handleCreateTree)
	http.HandleFunc("/api/tree/", handleTreeAPI)
	http.HandleFunc("/api/list", handleListTrees)
	http.HandleFunc("/api/login", handleLogin)
	http.HandleFunc("/api/logout", handleLogout)
	http.HandleFunc("/api/auth", handleAuthStatus)

	// Static/HTML Pages
	http.HandleFunc("/edit/", serveHTML("static/edit.html"))
	http.HandleFunc("/view/", serveHTML("static/view.html"))
	http.HandleFunc("/practice/", serveHTML("static/practice.html"))
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("./static/"))))
	http.HandleFunc("/", serveHTML("static/index.html")) // Homepage to create new

	fmt.Println("Server running on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

// Generates a random hex ID
func generateID() string {
	bytes := make([]byte, 8)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

// Loads the edit passcode from the environment or passcode.txt,
// generating a temporary one if neither is configured
func configurePasscode() {
	if p := strings.TrimSpace(os.Getenv("ATHENEUM_PASSCODE")); p != "" {
		editPasscode = p
		log.Println("Edit passcode loaded from ATHENEUM_PASSCODE")
		return
	}
	if data, err := os.ReadFile("passcode.txt"); err == nil {
		if p := strings.TrimSpace(string(data)); p != "" {
			editPasscode = p
			log.Println("Edit passcode loaded from passcode.txt")
			return
		}
	}
	editPasscode = randomPasscode(12)
	log.Printf("No passcode configured (set ATHENEUM_PASSCODE or create passcode.txt). Generated for this run: %s", editPasscode)
}

func randomPasscode(n int) string {
	const alphabet = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
	b := make([]byte, n)
	rand.Read(b)
	for i := range b {
		b[i] = alphabet[int(b[i])%len(alphabet)]
	}
	return string(b)
}

func newSession(w http.ResponseWriter) {
	b := make([]byte, 32)
	rand.Read(b)
	token := hex.EncodeToString(b)
	sessionsMu.Lock()
	sessions[token] = true
	sessionsMu.Unlock()
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookie,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   7 * 24 * 3600,
	})
}

func hasSession(r *http.Request) bool {
	c, err := r.Cookie(sessionCookie)
	if err != nil || c.Value == "" {
		return false
	}
	sessionsMu.Lock()
	defer sessionsMu.Unlock()
	return sessions[c.Value]
}

func endSession(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(sessionCookie); err == nil {
		sessionsMu.Lock()
		delete(sessions, c.Value)
		sessionsMu.Unlock()
	}
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookie,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		MaxAge:   -1,
	})
}

// requireAuth writes a 401 response and returns false when there is no valid session
func requireAuth(w http.ResponseWriter, r *http.Request) bool {
	if hasSession(r) {
		return true
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	json.NewEncoder(w).Encode(map[string]string{"error": "passcode required"})
	return false
}

// POST /api/login - exchanges the passcode for a session cookie
func handleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		Passcode string `json:"passcode"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	if subtle.ConstantTimeCompare([]byte(body.Passcode), []byte(editPasscode)) != 1 {
		time.Sleep(300 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]string{"error": "incorrect passcode"})
		return
	}
	newSession(w)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

// POST /api/logout - ends the current session
func handleLogout(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	endSession(w, r)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

// GET /api/auth - reports whether this browser may edit
func handleAuthStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"unlocked": hasSession(r)})
}

// POST /api/new - Creates a new tree and returns the ID
func handleCreateTree(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !requireAuth(w, r) {
		return
	}

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
		ID:   id,
		Name: name,
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
		http.Error(w, "Could not create tree", http.StatusInternalServerError)
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

// GET, PUT or DELETE /api/tree/{id}
func handleTreeAPI(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/tree/")
	if !validID(id) {
		http.Error(w, "Invalid tree ID", http.StatusBadRequest)
		return
	}
	filePath := filepath.Join(dataDir, id+".json")

	switch r.Method {
	case http.MethodGet:
		file, err := os.Open(filePath)
		if err != nil {
			http.Error(w, "Tree not found", http.StatusNotFound)
			return
		}
		defer file.Close()
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, file)

	case http.MethodPut:
		if !requireAuth(w, r) {
			return
		}
		var tree TreeNote
		if err := json.NewDecoder(r.Body).Decode(&tree); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}
		tree.ID = id
		if err := saveTree(id, &tree); err != nil {
			http.Error(w, "Could not save tree", http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)

	case http.MethodDelete:
		if !requireAuth(w, r) {
			return
		}
		if err := os.Remove(filePath); err != nil {
			http.Error(w, "Tree not found", http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
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
		http.Error(w, "Could not read data", http.StatusInternalServerError)
		return
	}

	type TreeInfo struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	list := make([]TreeInfo, 0, len(files))

	for _, f := range files {
		if filepath.Ext(f.Name()) != ".json" {
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

		list = append(list, TreeInfo{
			ID:   tree.ID,
			Name: name,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}
