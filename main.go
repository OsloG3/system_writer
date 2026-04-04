package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type Node struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Children    []*Node `json:"children"`
}

type TreeNote struct {
	ID    string  `json:"id"`
	Roots []*Node `json:"roots"` // Must be plural to match your JSON
}

const dataDir = "./data"

func main() {
	// Ensure data directory exists
	os.MkdirAll(dataDir, os.ModePerm)

	// API Endpoints
	http.HandleFunc("/api/new", handleCreateTree)
	http.HandleFunc("/api/tree/", handleTreeAPI)
	http.HandleFunc("/api/list", handleListTrees)

	// Static/HTML Pages
	http.HandleFunc("/edit/", serveHTML("static/edit.html"))
	http.HandleFunc("/view/", serveHTML("static/view.html"))
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

// POST /api/new - Creates a new tree and returns the ID
func handleCreateTree(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	id := generateID()
	tree := TreeNote{
		ID: id,
		Roots: []*Node{ // Initialize as a slice
			{
				ID:          "root-" + generateID(),
				Name:        "New Root Node",
				Description: "Enter description here...",
				Children:    []*Node{},
			},
		},
	}

	saveTree(id, &tree)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": id})
}

// GET /api/tree/{id} or PUT /api/tree/{id}
func handleTreeAPI(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/tree/")
	filePath := filepath.Join(dataDir, id+".json")

	if r.Method == http.MethodGet {
		file, err := os.Open(filePath)
		if err != nil {
			http.Error(w, "Tree not found", http.StatusNotFound)
			return
		}
		defer file.Close()
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, file)

	} else if r.Method == http.MethodPut {
		var tree TreeNote
		if err := json.NewDecoder(r.Body).Decode(&tree); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}
		saveTree(id, &tree)
		w.WriteHeader(http.StatusOK)
	}
}

func saveTree(id string, tree *TreeNote) {
	data, _ := json.MarshalIndent(tree, "", "  ")
	os.WriteFile(filepath.Join(dataDir, id+".json"), data, 0644)
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
	var list []TreeInfo

	for _, f := range files {
		if filepath.Ext(f.Name()) == ".json" {
			data, _ := os.ReadFile(filepath.Join(dataDir, f.Name()))
			var tree TreeNote
			json.Unmarshal(data, &tree)

			name := "Untitled Tree"
			if len(tree.Roots) > 0 && tree.Roots[0] != nil {
				name = tree.Roots[0].Name
			}

			list = append(list, TreeInfo{
				ID:   tree.ID,
				Name: name,
			})
		}

	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}
