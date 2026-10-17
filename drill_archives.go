package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
)

// ---- Drill archives ----
//
// An archive is one user's named collection of card problems, stored per
// archive in data/drill_archives/<id>.json. Membership is by problem ID, so
// a problem can sit in any number of archives (and in several users') while
// staying a normal library problem: archives only ever narrow the library
// view, they never hide or move a problem.

const drillArchiveNameMax = 80

var (
	drillArchMu   sync.Mutex
	drillArchives = map[string]*DrillArchive{} // keyed by archive ID
)

type DrillArchive struct {
	ID       string   `json:"id"`
	Owner    string   `json:"owner"`
	Name     string   `json:"name"`
	DrillIDs []string `json:"drillIds,omitempty"`
	Created  string   `json:"created"`
	Updated  string   `json:"updated"`
}

func drillArchivesDir() string          { return filepath.Join(dataDir, "drill_archives") }
func drillArchivePath(id string) string { return filepath.Join(drillArchivesDir(), id+".json") }

func initDrillArchives() {
	if err := os.MkdirAll(drillArchivesDir(), 0o755); err != nil {
		log.Printf("Could not create drill archives directory: %v", err)
		return
	}
	files, err := filepath.Glob(filepath.Join(drillArchivesDir(), "*.json"))
	if err != nil {
		return
	}
	n := 0
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		var a DrillArchive
		if err := json.Unmarshal(data, &a); err != nil {
			log.Printf("Could not parse %s: %v", f, err)
			continue
		}
		if a.ID == "" {
			a.ID = strings.TrimSuffix(filepath.Base(f), ".json")
		}
		if !validID(a.ID) {
			continue
		}
		drillArchives[a.ID] = &a
		n++
	}
	if n > 0 {
		log.Printf("Loaded %d drill archive(s)", n)
	}
}

// saveDrillArchiveLocked persists an archive. Callers must hold drillArchMu.
func saveDrillArchiveLocked(a *DrillArchive) error {
	return writeJSONFile(drillArchivePath(a.ID), a)
}

// removeDrillFromArchives drops a deleted problem from every archive that
// still references it.
func removeDrillFromArchives(drillID string) {
	drillArchMu.Lock()
	defer drillArchMu.Unlock()
	for _, a := range drillArchives {
		if !slices.Contains(a.DrillIDs, drillID) {
			continue
		}
		a.DrillIDs = slices.DeleteFunc(a.DrillIDs, func(id string) bool { return id == drillID })
		a.Updated = time.Now().UTC().Format(time.RFC3339)
		if err := saveDrillArchiveLocked(a); err != nil {
			log.Printf("Could not persist drill archive %s: %v", a.ID, err)
		}
	}
}

// cleanArchiveName trims an archive name; empty or overlong names are rejected.
func cleanArchiveName(raw string) (string, bool) {
	name := strings.TrimSpace(raw)
	if name == "" || len([]rune(name)) > drillArchiveNameMax {
		return "", false
	}
	return name, true
}

// ---- Handlers ----

// GET /api/drill-archives - the caller's archives with their member IDs.
func handleDrillArchivesList(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	drillArchMu.Lock()
	type info struct {
		ID       string   `json:"id"`
		Name     string   `json:"name"`
		DrillIDs []string `json:"drillIds"`
		Count    int      `json:"count"`
		Updated  string   `json:"updated"`
	}
	list := []info{}
	for _, a := range drillArchives {
		if a.Owner != username {
			continue
		}
		ids := append([]string(nil), a.DrillIDs...)
		slices.Sort(ids)
		list = append(list, info{ID: a.ID, Name: a.Name, DrillIDs: ids, Count: len(ids), Updated: a.Updated})
	}
	drillArchMu.Unlock()
	slices.SortFunc(list, func(x, y info) int { return strings.Compare(x.Name, y.Name) })
	writeJSON(w, map[string]any{"archives": list})
}

// POST /api/drill-archives - create an empty archive. Body: {name}.
func handleDrillArchiveCreate(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	name, ok := cleanArchiveName(body.Name)
	if !ok {
		jsonError(w, http.StatusBadRequest, "the archive needs a name (at most 80 characters)")
		return
	}
	now := time.Now().UTC().Format(time.RFC3339)
	a := &DrillArchive{ID: generateID(), Owner: username, Name: name, Created: now, Updated: now}
	drillArchMu.Lock()
	drillArchives[a.ID] = a
	err := saveDrillArchiveLocked(a)
	drillArchMu.Unlock()
	if err != nil {
		log.Printf("Could not persist drill archive %s: %v", a.ID, err)
		jsonError(w, http.StatusInternalServerError, "could not save the archive")
		return
	}
	writeJSON(w, map[string]any{"id": a.ID, "name": a.Name, "drillIds": []string{}, "count": 0})
}

// lookupOwnedArchive resolves {id} to an archive owned by the caller.
func lookupOwnedArchive(w http.ResponseWriter, r *http.Request, username string) *DrillArchive {
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid archive ID")
		return nil
	}
	drillArchMu.Lock()
	a := drillArchives[id]
	drillArchMu.Unlock()
	if a == nil {
		jsonError(w, http.StatusNotFound, "no such archive")
		return nil
	}
	if a.Owner != username {
		jsonError(w, http.StatusForbidden, "only the archive's owner can manage it")
		return nil
	}
	return a
}

// PUT /api/drill-archives/{id} - rename an archive. Body: {name}.
func handleDrillArchivePut(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	name, ok := cleanArchiveName(body.Name)
	if !ok {
		jsonError(w, http.StatusBadRequest, "the archive needs a name (at most 80 characters)")
		return
	}
	a := lookupOwnedArchive(w, r, username)
	if a == nil {
		return
	}
	drillArchMu.Lock()
	a.Name = name
	a.Updated = time.Now().UTC().Format(time.RFC3339)
	err := saveDrillArchiveLocked(a)
	drillArchMu.Unlock()
	if err != nil {
		log.Printf("Could not persist drill archive %s: %v", a.ID, err)
		jsonError(w, http.StatusInternalServerError, "could not save the archive")
		return
	}
	writeJSON(w, map[string]any{"ok": true, "id": a.ID, "name": name})
}

// DELETE /api/drill-archives/{id} - remove an archive (its problems stay in
// the library).
func handleDrillArchiveDelete(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	a := lookupOwnedArchive(w, r, username)
	if a == nil {
		return
	}
	drillArchMu.Lock()
	delete(drillArchives, a.ID)
	drillArchMu.Unlock()
	if err := os.Remove(drillArchivePath(a.ID)); err != nil && !os.IsNotExist(err) {
		log.Printf("Could not remove drill archive file: %v", err)
	}
	writeJSON(w, map[string]bool{"ok": true})
}

// POST /api/drill-archives/{id}/drills - add or remove a problem.
// Body: {"drillId": "...", "action": "add"|"remove"}
func handleDrillArchiveDrills(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	a := lookupOwnedArchive(w, r, username)
	if a == nil {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		DrillID string `json:"drillId"`
		Action  string `json:"action"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if !validID(body.DrillID) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	has := slices.Contains(a.DrillIDs, body.DrillID)
	switch body.Action {
	case "add":
		if has {
			jsonError(w, http.StatusConflict, "the problem is already in this archive")
			return
		}
		if lookupDrill(body.DrillID) == nil {
			jsonError(w, http.StatusNotFound, "no such problem")
			return
		}
	case "remove":
		if !has {
			jsonError(w, http.StatusNotFound, "the problem is not in this archive")
			return
		}
	default:
		jsonError(w, http.StatusBadRequest, `action must be "add" or "remove"`)
		return
	}
	drillArchMu.Lock()
	if body.Action == "add" {
		a.DrillIDs = append(a.DrillIDs, body.DrillID)
	} else {
		a.DrillIDs = slices.DeleteFunc(a.DrillIDs, func(id string) bool { return id == body.DrillID })
	}
	a.Updated = time.Now().UTC().Format(time.RFC3339)
	err := saveDrillArchiveLocked(a)
	ids := append([]string(nil), a.DrillIDs...)
	drillArchMu.Unlock()
	if err != nil {
		log.Printf("Could not persist drill archive %s: %v", a.ID, err)
		jsonError(w, http.StatusInternalServerError, "could not save the archive")
		return
	}
	writeJSON(w, map[string]any{"ok": true, "drillIds": ids, "count": len(ids)})
}
