package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// analyzeShareEnv swaps the data dir and the user/session stores for the
// test, seeds one account with a live session and returns its cookie.
func analyzeShareEnv(t *testing.T) *http.Cookie {
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

func analyzeReq(method, path, body string, c *http.Cookie) *http.Request {
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

// oneBoard is a minimal valid share board: four hands, a call sequence and
// two plays. The share endpoint checks shape, not deck legality.
const oneBoard = `{"title":"4S by South","hands":["AK863.2.AK72.943","9.QJT987.J43.AK6","QJT7.AK6.QT9.QJ5","542.543.865.T872"],` +
	`"dealer":0,"vuln":1,"calls":["4S","P","P","P"],` +
	`"plays":[{"seat":1,"card":"CT"},{"seat":0,"card":"SA"}],` +
	`"resultTricks":11,"names":["Ann","Bob","alice","Eve"],"idxLabel":3,"contractStr":"4S S"}`

const oneResult = `{"dd_tricks":11,"target":10,"plays":[{"seat":1,"card":"CT","mistake":false}]}`

func TestAnalyzeShareRequiresLogin(t *testing.T) {
	analyzeShareEnv(t)
	rr := httptest.NewRecorder()
	body := `{"kind":"single","boards":[` + oneBoard + `],"results":[` + oneResult + `]}`
	handleAnalyzeShareCreate(rr, analyzeReq("POST", "/api/analyze/share", body, nil))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("logged out: want 401, got %d", rr.Code)
	}
}

func TestAnalyzeShareValidation(t *testing.T) {
	c := analyzeShareEnv(t)
	cases := map[string]string{
		"bad kind":          `{"kind":"set","boards":[` + oneBoard + `],"results":[` + oneResult + `]}`,
		"no boards":         `{"kind":"bulk","boards":[],"results":[]}`,
		"misaligned":        `{"kind":"bulk","boards":[` + oneBoard + `],"results":[]}`,
		"single no result":  `{"kind":"single","boards":[` + oneBoard + `],"results":[null]}`,
		"single two":        `{"kind":"single","boards":[` + oneBoard + `,` + oneBoard + `],"results":[` + oneResult + `,` + oneResult + `]}`,
		"missing hand":      `{"kind":"bulk","boards":[{"hands":["","x","y","z"],"dealer":0,"vuln":0}],"results":[null]}`,
		"bad dealer":        `{"kind":"bulk","boards":[{"hands":["a","b","c","d"],"dealer":9,"vuln":0}],"results":[null]}`,
		"bad play":          `{"kind":"bulk","boards":[{"hands":["a","b","c","d"],"dealer":0,"vuln":0,"plays":[{"seat":1,"card":"ZZ"}]}],"results":[null]}`,
		"result not object": `{"kind":"bulk","boards":[` + oneBoard + `],"results":[[1,2]]}`,
		"not json":          `nonsense`,
	}
	for name, body := range cases {
		rr := httptest.NewRecorder()
		handleAnalyzeShareCreate(rr, analyzeReq("POST", "/api/analyze/share", body, c))
		if rr.Code != http.StatusBadRequest {
			t.Errorf("%s: want 400, got %d (%s)", name, rr.Code, rr.Body.String())
		}
	}
}

func TestAnalyzeShareRoundTrip(t *testing.T) {
	c := analyzeShareEnv(t)
	rr := httptest.NewRecorder()
	body := `{"kind":"single","boards":[` + oneBoard + `],"results":[` + oneResult + `]}`
	handleAnalyzeShareCreate(rr, analyzeReq("POST", "/api/analyze/share", body, c))
	if rr.Code != http.StatusOK {
		t.Fatalf("create: want 200, got %d (%s)", rr.Code, rr.Body.String())
	}
	var created struct {
		ID  string `json:"id"`
		URL string `json:"url"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &created); err != nil {
		t.Fatalf("create body: %v", err)
	}
	if !validID(created.ID) || created.URL != "/analyze?share="+created.ID {
		t.Fatalf("bad create response: %+v", created)
	}

	// the snapshot landed on disk under data/analysis/
	if _, err := os.Stat(analysisPath(created.ID)); err != nil {
		t.Fatalf("snapshot file: %v", err)
	}

	// public read: no cookie needed
	rr = httptest.NewRecorder()
	req := analyzeReq("GET", "/api/analyze/share/"+created.ID, "", nil)
	req.SetPathValue("id", created.ID)
	handleAnalyzeShareGet(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("get: want 200, got %d (%s)", rr.Code, rr.Body.String())
	}
	var got SharedAnalysis
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("get body: %v", err)
	}
	if got.ID != created.ID || got.Kind != "single" || got.Owner != "alice" {
		t.Fatalf("snapshot header wrong: %+v", got)
	}
	if len(got.Boards) != 1 || len(got.Results) != 1 {
		t.Fatalf("snapshot payload: %d boards, %d results", len(got.Boards), len(got.Results))
	}
	b := got.Boards[0]
	if b.Title != "4S by South" || b.Dealer != 0 || b.Vuln != 1 || b.ContractStr != "4S S" {
		t.Fatalf("board fields wrong: %+v", b)
	}
	if b.Names != [4]string{"Ann", "Bob", "alice", "Eve"} {
		t.Fatalf("names wrong: %+v", b.Names)
	}
	if len(b.Plays) != 2 || b.Plays[0].Card != "TC" || b.Plays[1].Seat != 0 {
		t.Fatalf("plays wrong: %+v", b.Plays)
	}
	if b.ResultTricks == nil || *b.ResultTricks != 11 {
		t.Fatalf("resultTricks wrong: %v", b.ResultTricks)
	}
	if b.IdxLabel == nil || *b.IdxLabel != 3 {
		t.Fatalf("idxLabel wrong: %v", b.IdxLabel)
	}
	var res map[string]any
	if err := json.Unmarshal(got.Results[0], &res); err != nil || res["dd_tricks"] != float64(11) {
		t.Fatalf("result passthrough wrong: %s", got.Results[0])
	}
}

func TestAnalyzeShareBulkAllowsNullResults(t *testing.T) {
	c := analyzeShareEnv(t)
	rr := httptest.NewRecorder()
	body := `{"kind":"bulk","boards":[` + oneBoard + `,` + oneBoard + `],"results":[` + oneResult + `,null]}`
	handleAnalyzeShareCreate(rr, analyzeReq("POST", "/api/analyze/share", body, c))
	if rr.Code != http.StatusOK {
		t.Fatalf("create bulk: want 200, got %d (%s)", rr.Code, rr.Body.String())
	}
	var created struct {
		ID string `json:"id"`
	}
	json.Unmarshal(rr.Body.Bytes(), &created)
	rr = httptest.NewRecorder()
	req := analyzeReq("GET", "/api/analyze/share/"+created.ID, "", nil)
	req.SetPathValue("id", created.ID)
	handleAnalyzeShareGet(rr, req)
	var got SharedAnalysis
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("get body: %v", err)
	}
	if got.Kind != "bulk" || len(got.Boards) != 2 {
		t.Fatalf("bulk snapshot wrong: %+v", got)
	}
	if string(got.Results[1]) != "null" {
		t.Fatalf("failed board should stay null, got %s", got.Results[1])
	}
}

func TestAnalyzeShareTruncatesDisplayText(t *testing.T) {
	c := analyzeShareEnv(t)
	long := strings.Repeat("x", 300)
	rr := httptest.NewRecorder()
	board := `{"title":"` + long + `","hands":["a","b","c","d"],"dealer":0,"vuln":0,"names":["` + long + `","","",""]}`
	body := `{"kind":"bulk","boards":[` + board + `],"results":[` + oneResult + `]}`
	handleAnalyzeShareCreate(rr, analyzeReq("POST", "/api/analyze/share", body, c))
	if rr.Code != http.StatusOK {
		t.Fatalf("create: want 200, got %d (%s)", rr.Code, rr.Body.String())
	}
	data, err := os.ReadFile(analysisPath(mustShareID(t, rr.Body.Bytes())))
	if err != nil {
		t.Fatal(err)
	}
	var got SharedAnalysis
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if len([]rune(got.Boards[0].Title)) != 120 {
		t.Errorf("title not capped: %d runes", len([]rune(got.Boards[0].Title)))
	}
	if len([]rune(got.Boards[0].Names[0])) != 60 {
		t.Errorf("name not capped: %d runes", len([]rune(got.Boards[0].Names[0])))
	}
}

func mustShareID(t *testing.T, body []byte) string {
	t.Helper()
	var v struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &v); err != nil || v.ID == "" {
		t.Fatalf("no id in %s", body)
	}
	return v.ID
}

func TestAnalyzeShareGetRejectsBadIDs(t *testing.T) {
	analyzeShareEnv(t)
	for _, id := range []string{"../users", "deadbeefdeadbeef", ""} {
		rr := httptest.NewRecorder()
		req := analyzeReq("GET", "/api/analyze/share/"+id, "", nil)
		req.SetPathValue("id", id)
		handleAnalyzeShareGet(rr, req)
		if rr.Code != http.StatusNotFound && rr.Code != http.StatusBadRequest {
			t.Errorf("id %q: want 400/404, got %d", id, rr.Code)
		}
	}
}
