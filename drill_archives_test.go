package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"
)

// drillTestEnv points dataDir at a temp dir and resets the drill stores.
func drillTestEnv(t *testing.T) {
	t.Helper()
	old := dataDir
	dataDir = t.TempDir()
	if err := os.MkdirAll(drillsDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(drillArchivesDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		dataDir = old
		drillsMu.Lock()
		drills = map[string]*DrillProblem{}
		drillsMu.Unlock()
		drillSessMu.Lock()
		drillSess = map[string]*DrillSession{}
		drillSessMu.Unlock()
		drillArchMu.Lock()
		drillArchives = map[string]*DrillArchive{}
		drillArchMu.Unlock()
	})
}

// createTestDrill creates a problem through the handler and returns its ID.
func createTestDrill(t *testing.T, token string) string {
	t.Helper()
	body, _ := json.Marshal(drillProblem())
	rr := httptest.NewRecorder()
	handleDrillCreate(rr, drillReq(t, "POST", "/api/drills", string(body), token))
	if rr.Code != http.StatusOK {
		t.Fatalf("create drill: %d %s", rr.Code, rr.Body.String())
	}
	var out struct {
		ID string `json:"id"`
	}
	json.Unmarshal(rr.Body.Bytes(), &out)
	if out.ID == "" {
		t.Fatal("create returned no id")
	}
	return out.ID
}

// playThrough finishes one practice run of `id` for the token's user by
// playing the hero's single scripted card (the fixture ends in a claim).
func playThrough(t *testing.T, id, token string) {
	t.Helper()
	rr := httptest.NewRecorder()
	req := drillReq(t, "POST", "/api/drills/"+id+"/session", "", token)
	req.SetPathValue("id", id)
	handleDrillSessionNew(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("new session: %d %s", rr.Code, rr.Body.String())
	}
	var st struct {
		ID   string `json:"id"`
		Done bool   `json:"done"`
	}
	json.Unmarshal(rr.Body.Bytes(), &st)
	if st.Done {
		return
	}
	rr = httptest.NewRecorder()
	req = drillReq(t, "POST", "/api/drill/session/"+st.ID+"/card", `{"card":"5S"}`, token)
	req.SetPathValue("sid", st.ID)
	handleDrillSessionCard(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("play card: %d %s", rr.Code, rr.Body.String())
	}
	var done struct {
		Phase string `json:"phase"`
	}
	json.Unmarshal(rr.Body.Bytes(), &done)
	if done.Phase != "done" {
		t.Fatalf("expected a finished hand, got phase %q", done.Phase)
	}
}

// listFlags reads GET /api/drills and maps problem ID -> played flag.
func listFlags(t *testing.T, token string) map[string]bool {
	t.Helper()
	rr := httptest.NewRecorder()
	handleDrillsList(rr, drillReq(t, "GET", "/api/drills", "", token))
	if rr.Code != http.StatusOK {
		t.Fatalf("list: %d", rr.Code)
	}
	var out struct {
		Problems []struct {
			ID     string `json:"id"`
			Played bool   `json:"played"`
		} `json:"problems"`
	}
	json.Unmarshal(rr.Body.Bytes(), &out)
	m := map[string]bool{}
	for _, p := range out.Problems {
		m[p.ID] = p.Played
	}
	return m
}

func TestDrillPlayedFlag(t *testing.T) {
	drillTestEnv(t)
	alice, bob := drillLogin(t, "alice"), drillLogin(t, "bob")
	id := createTestDrill(t, alice)

	if flags := listFlags(t, bob); flags[id] {
		t.Fatal("fresh problem must be unplayed")
	}
	playThrough(t, id, bob)

	if flags := listFlags(t, bob); !flags[id] {
		t.Fatal("finished problem must be flagged played for the player")
	}
	if flags := listFlags(t, alice); flags[id] {
		t.Fatal("played flag must be per user")
	}

	// the stamp survives a restart: it is persisted on the problem file
	data, err := os.ReadFile(drillPath(id))
	if err != nil {
		t.Fatal(err)
	}
	var p DrillProblem
	if err := json.Unmarshal(data, &p); err != nil {
		t.Fatal(err)
	}
	if _, ok := p.Played["bob"]; !ok {
		t.Fatalf("played stamp not persisted: %s", data)
	}

	// an author edit keeps the played stamps (server-side fields)
	body, _ := json.Marshal(drillProblem())
	rr := httptest.NewRecorder()
	req := drillReq(t, "PUT", "/api/drills/"+id, string(body), alice)
	req.SetPathValue("id", id)
	handleDrillPut(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("put: %d %s", rr.Code, rr.Body.String())
	}
	drillsMu.Lock()
	_, kept := drills[id].Played["bob"]
	drillsMu.Unlock()
	if !kept {
		t.Fatal("edit dropped the played stamps")
	}
}

// archiveReq builds a request with both path values used by archive routes.
func archiveReq(t *testing.T, method, path, body, token, id, drillID string) *http.Request {
	t.Helper()
	req := drillReq(t, method, path, body, token)
	if id != "" {
		req.SetPathValue("id", id)
	}
	if drillID != "" {
		req.SetPathValue("drillId", drillID)
	}
	return req
}

func TestDrillArchives(t *testing.T) {
	drillTestEnv(t)
	alice, bob := drillLogin(t, "alice"), drillLogin(t, "bob")
	d1 := createTestDrill(t, alice)
	d2 := createTestDrill(t, bob)

	// logged out
	rr := httptest.NewRecorder()
	handleDrillArchivesList(rr, drillReq(t, "GET", "/api/drill-archives", "", ""))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("list logged out: %d", rr.Code)
	}

	// create
	rr = httptest.NewRecorder()
	handleDrillArchiveCreate(rr, drillReq(t, "POST", "/api/drill-archives", `{"name":"  Endplays  "}`, alice))
	if rr.Code != http.StatusOK {
		t.Fatalf("create archive: %d %s", rr.Code, rr.Body.String())
	}
	var created struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	json.Unmarshal(rr.Body.Bytes(), &created)
	if created.ID == "" || created.Name != "Endplays" {
		t.Fatalf("bad create response: %s", rr.Body.String())
	}
	if _, err := os.Stat(drillArchivePath(created.ID)); err != nil {
		t.Fatalf("archive not persisted: %v", err)
	}
	aid := created.ID

	// empty name rejected
	rr = httptest.NewRecorder()
	handleDrillArchiveCreate(rr, drillReq(t, "POST", "/api/drill-archives", `{"name":"  "}`, alice))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("empty name: %d", rr.Code)
	}

	// add problems (a problem of another author is fine)
	for _, d := range []string{d1, d2} {
		rr = httptest.NewRecorder()
		handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+aid+"/drills",
			`{"drillId":"`+d+`","action":"add"}`, alice, aid, ""))
		if rr.Code != http.StatusOK {
			t.Fatalf("add %s: %d %s", d, rr.Code, rr.Body.String())
		}
	}
	// duplicate add
	rr = httptest.NewRecorder()
	handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+aid+"/drills",
		`{"drillId":"`+d1+`","action":"add"}`, alice, aid, ""))
	if rr.Code != http.StatusConflict {
		t.Fatalf("duplicate add: %d", rr.Code)
	}
	// unknown problem
	rr = httptest.NewRecorder()
	handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+aid+"/drills",
		`{"drillId":"deadbeefdeadbeef","action":"add"}`, alice, aid, ""))
	if rr.Code != http.StatusNotFound {
		t.Fatalf("unknown drill: %d", rr.Code)
	}
	// bad action
	rr = httptest.NewRecorder()
	handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+aid+"/drills",
		`{"drillId":"`+d1+`","action":"shuffle"}`, alice, aid, ""))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("bad action: %d", rr.Code)
	}

	// bob cannot manage alice's archive
	rr = httptest.NewRecorder()
	handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+aid+"/drills",
		`{"drillId":"`+d1+`","action":"remove"}`, bob, aid, ""))
	if rr.Code != http.StatusForbidden {
		t.Fatalf("foreign membership change: %d", rr.Code)
	}

	// the same problem in a second archive of another user
	rr = httptest.NewRecorder()
	handleDrillArchiveCreate(rr, drillReq(t, "POST", "/api/drill-archives", `{"name":"Bob's favs"}`, bob))
	var bobs struct {
		ID string `json:"id"`
	}
	json.Unmarshal(rr.Body.Bytes(), &bobs)
	rr = httptest.NewRecorder()
	handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+bobs.ID+"/drills",
		`{"drillId":"`+d1+`","action":"add"}`, bob, bobs.ID, ""))
	if rr.Code != http.StatusOK {
		t.Fatalf("multi-archive add: %d", rr.Code)
	}

	// listings are per owner and carry the member IDs
	rr = httptest.NewRecorder()
	handleDrillArchivesList(rr, drillReq(t, "GET", "/api/drill-archives", "", alice))
	var la struct {
		Archives []struct {
			ID       string   `json:"id"`
			Name     string   `json:"name"`
			DrillIDs []string `json:"drillIds"`
			Count    int      `json:"count"`
		} `json:"archives"`
	}
	json.Unmarshal(rr.Body.Bytes(), &la)
	if len(la.Archives) != 1 || la.Archives[0].Count != 2 ||
		!slices.Equal(la.Archives[0].DrillIDs, sortedIDs(d1, d2)) {
		t.Fatalf("bad alice listing: %s", rr.Body.String())
	}
	rr = httptest.NewRecorder()
	handleDrillArchivesList(rr, drillReq(t, "GET", "/api/drill-archives", "", bob))
	if !strings.Contains(rr.Body.String(), "Bob's favs") || strings.Contains(rr.Body.String(), "Endplays") {
		t.Fatalf("listings must be per owner: %s", rr.Body.String())
	}

	// rename
	rr = httptest.NewRecorder()
	handleDrillArchivePut(rr, archiveReq(t, "PUT", "/api/drill-archives/"+aid, `{"name":"Squeezes"}`, alice, aid, ""))
	if rr.Code != http.StatusOK || !strings.Contains(rr.Body.String(), "Squeezes") {
		t.Fatalf("rename: %d %s", rr.Code, rr.Body.String())
	}
	rr = httptest.NewRecorder()
	handleDrillArchivePut(rr, archiveReq(t, "PUT", "/api/drill-archives/"+aid, `{"name":"Hax"}`, bob, aid, ""))
	if rr.Code != http.StatusForbidden {
		t.Fatalf("foreign rename: %d", rr.Code)
	}

	// remove a member; removing a non-member is 404
	rr = httptest.NewRecorder()
	handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+aid+"/drills",
		`{"drillId":"`+d2+`","action":"remove"}`, alice, aid, ""))
	if rr.Code != http.StatusOK {
		t.Fatalf("remove: %d %s", rr.Code, rr.Body.String())
	}
	rr = httptest.NewRecorder()
	handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+aid+"/drills",
		`{"drillId":"`+d2+`","action":"remove"}`, alice, aid, ""))
	if rr.Code != http.StatusNotFound {
		t.Fatalf("remove non-member: %d", rr.Code)
	}

	// deleting a problem sweeps it out of every archive
	rr = httptest.NewRecorder()
	req := drillReq(t, "DELETE", "/api/drills/"+d1, "", alice)
	req.SetPathValue("id", d1)
	handleDrillDelete(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("delete drill: %d", rr.Code)
	}
	drillArchMu.Lock()
	inAlice := slices.Contains(drillArchives[aid].DrillIDs, d1)
	inBob := slices.Contains(drillArchives[bobs.ID].DrillIDs, d1)
	drillArchMu.Unlock()
	if inAlice || inBob {
		t.Fatal("deleted problem must be swept from all archives")
	}

	// delete an archive: gone from memory, disk and the listing
	rr = httptest.NewRecorder()
	handleDrillArchiveDelete(rr, archiveReq(t, "DELETE", "/api/drill-archives/"+aid, "", bob, aid, ""))
	if rr.Code != http.StatusForbidden {
		t.Fatalf("foreign delete: %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	handleDrillArchiveDelete(rr, archiveReq(t, "DELETE", "/api/drill-archives/"+aid, "", alice, aid, ""))
	if rr.Code != http.StatusOK {
		t.Fatalf("delete archive: %d", rr.Code)
	}
	if _, err := os.Stat(drillArchivePath(aid)); !os.IsNotExist(err) {
		t.Fatal("archive file must be removed")
	}
	rr = httptest.NewRecorder()
	handleDrillArchivesList(rr, drillReq(t, "GET", "/api/drill-archives", "", alice))
	if strings.Contains(rr.Body.String(), "Squeezes") {
		t.Fatalf("deleted archive still listed: %s", rr.Body.String())
	}
}

func TestDrillArchiveReload(t *testing.T) {
	drillTestEnv(t)
	alice := drillLogin(t, "alice")
	d1 := createTestDrill(t, alice)
	rr := httptest.NewRecorder()
	handleDrillArchiveCreate(rr, drillReq(t, "POST", "/api/drill-archives", `{"name":"Kept"}`, alice))
	var created struct {
		ID string `json:"id"`
	}
	json.Unmarshal(rr.Body.Bytes(), &created)
	rr = httptest.NewRecorder()
	handleDrillArchiveDrills(rr, archiveReq(t, "POST", "/api/drill-archives/"+created.ID+"/drills",
		`{"drillId":"`+d1+`","action":"add"}`, alice, created.ID, ""))
	if rr.Code != http.StatusOK {
		t.Fatalf("add: %d", rr.Code)
	}

	// simulate a restart: wipe memory and reload from disk
	drillArchMu.Lock()
	drillArchives = map[string]*DrillArchive{}
	drillArchMu.Unlock()
	initDrillArchives()
	drillArchMu.Lock()
	a := drillArchives[created.ID]
	drillArchMu.Unlock()
	if a == nil || a.Name != "Kept" || !slices.Equal(a.DrillIDs, []string{d1}) {
		t.Fatalf("archive did not survive the reload: %+v", a)
	}
}

func sortedIDs(ids ...string) []string {
	out := append([]string(nil), ids...)
	slices.Sort(out)
	return out
}
