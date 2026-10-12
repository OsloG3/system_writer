package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// drillHands is a fixed 52-card deal (the same one as the play-board tests):
// N holds AT7.KT943.A42.K8 and the 4H contract below is declared by North.
var drillHands = [4]string{
	"AT7.KT943.A42.K8", "J6543.75.K8.AQJT",
	"K982.AQ6.T73.952", "Q.J82.QJ965.7643",
}

// naivePlayLine scripts a complete legal 52-card play: every seat follows
// suit when able and plays its lowest-ranked holding otherwise.
func naivePlayLine(hands [4]string, opener int, trump byte) []DrillPlay {
	rem := map[int][]string{}
	for seat := 0; seat < 4; seat++ {
		rem[seat] = handCards(hands[seat])
	}
	var plays []DrillPlay
	next := opener
	for len(plays) < 52 {
		seat := next
		var card string
		if len(plays)%4 > 0 {
			lead := cardSuit(plays[(len(plays)/4)*4].Card)
			for _, c := range rem[seat] {
				if cardSuit(c) == lead {
					card = c
					break
				}
			}
		}
		if card == "" {
			card = rem[seat][len(rem[seat])-1]
		}
		for i, c := range rem[seat] {
			if c == card {
				rem[seat] = append(rem[seat][:i], rem[seat][i+1:]...)
				break
			}
		}
		plays = append(plays, DrillPlay{Seat: seat, Card: card})
		if len(plays)%4 == 0 {
			trick := make([]PlayCard, 4)
			for j := 0; j < 4; j++ {
				trick[j] = PlayCard{Seat: plays[len(plays)-4+j].Seat, Card: plays[len(plays)-4+j].Card}
			}
			next = trickWinner(trick, trump)
		} else {
			next = (seat + 1) % 4
		}
	}
	return plays
}

// drillProblem builds a valid defense problem: 4H by North, hero is East,
// one scripted trick and a claim of 10 tricks for +420 (N-S).
func drillProblem() DrillProblem {
	return DrillProblem{
		Title: "Lead problem", Role: "defense", HeroSeat: 1,
		Difficulty: 3, Dealer: 0, Vuln: 0,
		Hands:          drillHands,
		Calls:          []string{"1H", "P", "4H", "P", "P", "P"},
		BidComment:     "Standard opening.",
		ProblemComment: "Udca signals.",
		Plays: []DrillPlay{
			{Seat: 1, Card: "5S", Msg: "the obvious lead"},
			{Seat: 2, Card: "2S"},
			{Seat: 3, Card: "QS"},
			{Seat: 0, Card: "7S"},
		},
		Claim: &DrillClaim{Tricks: 10, ScoreNS: 420},
	}
}

func TestValidateHands(t *testing.T) {
	if err := validateHands(drillHands); err != nil {
		t.Fatalf("valid deck rejected: %v", err)
	}
	bad := drillHands
	bad[0] = "AT7.KT943.A42.K" // 12 cards
	if err := validateHands(bad); err == nil {
		t.Fatal("short hand accepted")
	}
	bad = drillHands
	bad[1] = "J6543.75.K8.AQJ9" // E swaps T♣ for 9♣: duplicates South's 9♣
	if err := validateHands(bad); err == nil {
		t.Fatal("duplicated card accepted")
	}
	bad = drillHands
	bad[0] = "AT7.KT943.A42.KJ" // J♣ is East's
	if err := validateHands(bad); err == nil {
		t.Fatal("duplicated JC accepted")
	}
	if err := validateHands([4]string{"AT7.KT943.A42.KX", drillHands[1], drillHands[2], drillHands[3]}); err == nil {
		t.Fatal("junk card accepted")
	}
}

func TestValidateAuction(t *testing.T) {
	ok := []struct {
		dealer int
		calls  []string
		level  int
		denom  string
		decl   int
		pen    int
	}{
		{0, []string{"1H", "P", "4H", "P", "P", "P"}, 4, "H", 0, 0},
		{2, []string{"1C", "X", "P", "P", "P"}, 1, "C", 2, 1},
		{2, []string{"1C", "X", "XX", "P", "P", "P"}, 1, "C", 2, 2},
		{0, []string{"1H", "P", "P", "2S", "P", "P", "P"}, 2, "S", 3, 0},
		{1, []string{"P", "1NT", "P", "P", "P"}, 1, "N", 2, 0},
	}
	for _, c := range ok {
		contract, err := validateAuction(c.dealer, c.calls)
		if err != nil {
			t.Fatalf("%v rejected: %v", c.calls, err)
		}
		if contract.Level != c.level || contract.Denom != c.denom ||
			contract.Declarer != c.decl || contract.Penalty != c.pen {
			t.Fatalf("%v -> %+v, want %d%s by %d pen %d", c.calls, contract, c.level, c.denom, c.decl, c.pen)
		}
	}
	bad := [][]string{
		{"P", "P", "P", "P"},                  // passed out
		{"1H", "P", "P"},                      // unfinished
		{"1H", "P", "P", "P", "P"},            // calls after the auction ended
		{"2C", "1H", "P", "P", "P"},           // bid does not outrank
		{"1H", "P", "X", "P", "P", "P"},       // double not directly over the bid
		{"1H", "X", "P", "XX", "P", "P", "P"}, // redouble not directly over the double
		{"1H", "1H", "P", "P", "P"},           // equal bid
		{"1H", "P", "4H", "P", "P", "P", "X"}, // double after the auction ended
		{"1H", "2S", "P", "P", "P", "P", "P"}, // trailing passes beyond three
	}
	for _, calls := range bad {
		if _, err := validateAuction(0, calls); err == nil {
			t.Fatalf("%v accepted, want rejection", calls)
		}
	}
	// A double directly over the bid is always by the next seat (an
	// opponent); a redouble by anyone else must be rejected.
	if _, err := validateAuction(0, []string{"1H", "P", "P", "XX"}); err == nil {
		t.Fatal("stray redouble accepted")
	}
}

func TestScoreContractNS(t *testing.T) {
	c := func(level int, denom string, declarer, penalty int) *PlayContract {
		return &PlayContract{Level: level, Denom: denom, Declarer: declarer, Penalty: penalty}
	}
	cases := []struct {
		contract *PlayContract
		tricks   int
		vuln     int
		want     int
	}{
		{c(3, "N", 0, 0), 9, 0, 400},   // 3NT NV making
		{c(3, "N", 2, 0), 9, 1, 600},   // 3NT V (N-S) by S making
		{c(4, "S", 2, 0), 10, 0, 420},  // 4S NV making
		{c(4, "S", 0, 0), 11, 1, 650},  // 4S V +1
		{c(4, "S", 0, 1), 10, 1, 790},  // 4SX V making
		{c(4, "S", 0, 1), 11, 1, 990},  // 4SX V +1
		{c(3, "N", 2, 1), 9, 0, 550},   // 3NTX NV making
		{c(3, "N", 2, 0), 7, 0, -100},  // 3NT NV -2
		{c(3, "N", 2, 1), 7, 0, -300},  // 3NTX NV -2
		{c(3, "N", 2, 1), 7, 1, -400},  // 3NTX V -2 (N-S vul, S declares)
		{c(3, "N", 2, 2), 7, 0, -600},  // 3NTXX NV -2
		{c(2, "H", 1, 0), 8, 0, -110},  // 2H by E making: N-S pays 110
		{c(6, "S", 0, 0), 12, 0, 980},  // small slam NV
		{c(7, "N", 2, 0), 13, 1, 2220}, // grand slam V
		{c(4, "S", 3, 1), 9, 0, 100},   // 4SX NV by W -1: N-S +100
		{c(5, "D", 1, 1), 11, 3, -750}, // 5DX V by E making: N-S -750
		{c(1, "C", 2, 0), 7, 0, 70},    // partscore NV
		{c(3, "N", 2, 2), 7, 3, -800},  // 3NTXX both vul by S -2
	}
	for _, tc := range cases {
		got := scoreContractNS(tc.contract, tc.tricks, tc.vuln)
		if got != tc.want {
			t.Errorf("score(%s, %d tricks, vuln %d) = %d, want %d",
				tc.contract, tc.tricks, tc.vuln, got, tc.want)
		}
	}
}

func TestValidateDrill(t *testing.T) {
	p := drillProblem()
	if _, err := validateDrill(&p); err != nil {
		t.Fatalf("valid problem rejected: %v", err)
	}
	if p.Plays[0].Card != "5S" {
		t.Fatal("scripted card not normalized in place")
	}

	mutate := func(name string, f func(*DrillProblem)) {
		t.Run(name, func(t *testing.T) {
			q := drillProblem()
			f(&q)
			if _, err := validateDrill(&q); err == nil {
				t.Fatalf("%s: invalid problem accepted", name)
			}
		})
	}
	mutate("bad role", func(q *DrillProblem) { q.Role = "both" })
	mutate("stars", func(q *DrillProblem) { q.Difficulty = 7 })
	mutate("broken deck", func(q *DrillProblem) { q.Hands[0] = "AT7.KT943.A42.K" })
	mutate("pass-out", func(q *DrillProblem) { q.Calls = []string{"P", "P", "P", "P"} })
	mutate("declarer role wrong hero", func(q *DrillProblem) {
		q.Role, q.HeroSeat = "declarer", 2 // S is a defender here
	})
	mutate("defense hero declares", func(q *DrillProblem) { q.HeroSeat = 0 })
	mutate("no plays", func(q *DrillProblem) { q.Plays = nil })
	mutate("out of turn", func(q *DrillProblem) {
		q.Plays[0], q.Plays[1] = q.Plays[1], q.Plays[0]
	})
	mutate("card not held", func(q *DrillProblem) { q.Plays[1].Card = "AS" })
	mutate("revoke", func(q *DrillProblem) { q.Plays[1].Card = "AH" }) // S holds 2S and spades were led
	mutate("msg on non-hero seat", func(q *DrillProblem) { q.Plays[2].Msg = "why" })
	mutate("claim off trick boundary", func(q *DrillProblem) {
		q.Plays = q.Plays[:3]
	})
	mutate("claim tricks impossible", func(q *DrillProblem) { q.Claim.Tricks = 14 })
	mutate("claim tricks below won", func(q *DrillProblem) {
		q.Plays = naivePlayLine(drillHands, 1, 'H')[:8]
		q.Claim.Tricks = 0
	})
	mutate("no claim short line", func(q *DrillProblem) {
		q.Claim = nil
		q.Plays = naivePlayLine(drillHands, 1, 'H')[:8]
	})

	// A declarer problem played out to the end needs no claim.
	full := drillProblem()
	full.Role, full.HeroSeat, full.Claim = "declarer", 0, nil
	full.Plays = naivePlayLine(drillHands, 1, 'H')
	if _, err := validateDrill(&full); err != nil {
		t.Fatalf("played-out declarer problem rejected: %v", err)
	}
	if len(full.Plays) != 52 {
		t.Fatalf("naive line has %d plays", len(full.Plays))
	}
}

func TestDrillSessionPlayedOut(t *testing.T) {
	p := drillProblem()
	p.Role, p.HeroSeat, p.Claim = "declarer", 0, nil
	p.Plays = naivePlayLine(drillHands, 1, 'H')
	s, err := newDrillSession(&p, "alice")
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if s.Done || len(s.Plays) != 1 {
		t.Fatalf("session should stop at the hero's first turn (dummy, trick 1) with the opening lead auto-played, got %d (done=%v)", len(s.Plays), s.Done)
	}
	st := s.stateLocked()
	if st["phase"] != "play" || st["yourTurn"] != true || st["actSeat"] != 2 {
		t.Fatalf("bad initial state: %+v", st)
	}
	if st["dummyHand"] != drillHands[2] {
		t.Fatal("dummy must be tabled after the opening lead")
	}
	for !s.Done {
		if s.heroControls(s.nextSeatLocked()) {
			s.applyLocked(s.Problem.Plays[len(s.Plays)])
		}
		s.advanceLocked()
	}
	if s.Result == nil || s.Result.Claimed {
		t.Fatalf("bad result: %+v", s.Result)
	}
	if s.Result.Tricks != s.DeclTricks {
		t.Fatalf("result tricks %d != counted %d", s.Result.Tricks, s.DeclTricks)
	}
	want := scoreContractNS(s.contract, s.DeclTricks, p.Vuln)
	if s.Result.ScoreNS != want {
		t.Fatalf("result score %d != scored %d", s.Result.ScoreNS, want)
	}
	st = s.stateLocked()
	if st["phase"] != "done" || st["hands"] == nil || st["line"] == nil {
		t.Fatalf("done state must reveal hands and the line: %+v", st)
	}
}

// ---- Handler-level tests ----

func drillLogin(t *testing.T, username string) string {
	t.Helper()
	token := "drilltoken-" + username
	sessionsMu.Lock()
	sessions[token] = Session{Username: username, Expires: time.Now().Add(time.Hour)}
	sessionsMu.Unlock()
	t.Cleanup(func() {
		sessionsMu.Lock()
		delete(sessions, token)
		sessionsMu.Unlock()
	})
	return token
}

func drillReq(t *testing.T, method, path, body, token string) *http.Request {
	t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, rd)
	if token != "" {
		req.AddCookie(&http.Cookie{Name: sessionCookie, Value: token})
	}
	return req
}

func TestDrillHandlersCRUDAndSession(t *testing.T) {
	old := dataDir
	dataDir = t.TempDir()
	defer func() { dataDir = old }()
	if err := os.MkdirAll(drillsDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		drillsMu.Lock()
		drills = map[string]*DrillProblem{}
		drillsMu.Unlock()
		drillSessMu.Lock()
		drillSess = map[string]*DrillSession{}
		drillSessMu.Unlock()
	})

	alice, bob := drillLogin(t, "alice"), drillLogin(t, "bob")
	problem := drillProblem()
	body, _ := json.Marshal(problem)

	// logged out
	rr := httptest.NewRecorder()
	handleDrillsList(rr, drillReq(t, "GET", "/api/drills", "", ""))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("list logged out: %d", rr.Code)
	}

	// create
	rr = httptest.NewRecorder()
	handleDrillCreate(rr, drillReq(t, "POST", "/api/drills", string(body), alice))
	if rr.Code != http.StatusOK {
		t.Fatalf("create: %d %s", rr.Code, rr.Body.String())
	}
	var created struct {
		ID string `json:"id"`
	}
	json.Unmarshal(rr.Body.Bytes(), &created)
	id := created.ID
	if id == "" {
		t.Fatal("create returned no id")
	}
	if _, err := os.Stat(drillPath(id)); err != nil {
		t.Fatalf("problem not persisted: %v", err)
	}

	// invalid problem rejected
	bad := drillProblem()
	bad.Difficulty = 0
	badBody, _ := json.Marshal(bad)
	rr = httptest.NewRecorder()
	handleDrillCreate(rr, drillReq(t, "POST", "/api/drills", string(badBody), alice))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("invalid create: %d", rr.Code)
	}

	// library metadata carries no spoilers
	rr = httptest.NewRecorder()
	handleDrillsList(rr, drillReq(t, "GET", "/api/drills", "", bob))
	if rr.Code != http.StatusOK {
		t.Fatalf("list: %d", rr.Code)
	}
	listBody := rr.Body.String()
	for _, leak := range []string{`"hands"`, `"plays"`, "Udca", "5S", "KT943"} {
		if strings.Contains(listBody, leak) {
			t.Fatalf("library list leaks %s", leak)
		}
	}
	var listed struct {
		Problems []map[string]any `json:"problems"`
	}
	json.Unmarshal(rr.Body.Bytes(), &listed)
	if len(listed.Problems) != 1 || listed.Problems[0]["contract"] != "4H by N" {
		t.Fatalf("bad library listing: %s", listBody)
	}

	// full read: owner only
	rr = httptest.NewRecorder()
	req := drillReq(t, "GET", "/api/drills/"+id, "", bob)
	req.SetPathValue("id", id)
	handleDrillGet(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("foreign get: %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	req = drillReq(t, "GET", "/api/drills/"+id, "", alice)
	req.SetPathValue("id", id)
	handleDrillGet(rr, req)
	if rr.Code != http.StatusOK || !strings.Contains(rr.Body.String(), "Udca") {
		t.Fatalf("owner get: %d %s", rr.Code, rr.Body.String())
	}

	// edit: owner only, ownership preserved
	edited := drillProblem()
	edited.Title = "Renamed"
	edited.Difficulty = 5
	editedBody, _ := json.Marshal(edited)
	rr = httptest.NewRecorder()
	req = drillReq(t, "PUT", "/api/drills/"+id, string(editedBody), bob)
	req.SetPathValue("id", id)
	handleDrillPut(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("foreign put: %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	req = drillReq(t, "PUT", "/api/drills/"+id, string(editedBody), alice)
	req.SetPathValue("id", id)
	handleDrillPut(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("owner put: %d %s", rr.Code, rr.Body.String())
	}
	p := lookupDrill(id)
	if p.Title != "Renamed" || p.Difficulty != 5 || p.Owner != "alice" || p.Created == "" {
		t.Fatalf("edit not applied: %+v", p)
	}

	// practice session for another user
	rr = httptest.NewRecorder()
	req = drillReq(t, "POST", "/api/drills/"+id+"/session", "", bob)
	req.SetPathValue("id", id)
	handleDrillSessionNew(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("session new: %d %s", rr.Code, rr.Body.String())
	}
	var st map[string]any
	json.Unmarshal(rr.Body.Bytes(), &st)
	sid := st["id"].(string)
	t.Cleanup(func() {
		drillSessMu.Lock()
		delete(drillSess, sid)
		drillSessMu.Unlock()
	})
	if st["yourTurn"] != true || st["hand"] != drillHands[1] || st["dummyHand"] != "" {
		t.Fatalf("bad initial session state: %+v", st)
	}
	// the script's messages must not travel with the live state
	if strings.Contains(rr.Body.String(), "obvious lead") {
		t.Fatal("session state leaks the hero message")
	}

	card := func(payload string) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		req := drillReq(t, "POST", "/api/drill/session/"+sid+"/card", payload, bob)
		req.SetPathValue("sid", sid)
		handleDrillSessionCard(rr, req)
		return rr
	}
	// a card East does not hold
	if rr := card(`{"card":"2S"}`); rr.Code != http.StatusBadRequest {
		t.Fatalf("card not held: %d", rr.Code)
	}
	// legal but not the authored card: refused, hint returned, nothing played
	rr = card(`{"card":"7H"}`)
	if rr.Code != http.StatusConflict {
		t.Fatalf("wrong card: %d %s", rr.Code, rr.Body.String())
	}
	var wrong struct {
		Hint  string `json:"hint"`
		Wrong int    `json:"wrong"`
	}
	json.Unmarshal(rr.Body.Bytes(), &wrong)
	if wrong.Hint != "the obvious lead" || wrong.Wrong != 1 {
		t.Fatalf("bad wrong-pick response: %+v", wrong)
	}
	if s := lookupDrillSession(sid, "bob"); s == nil || len(s.Plays) != 0 || s.Wrong != 1 {
		t.Fatal("a wrong pick must not play anything")
	}
	// the authored card: played, the rest of the trick auto-plays, claim ends
	// the hand with the author's score
	rr = card(`{"card":"5S"}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("right card: %d %s", rr.Code, rr.Body.String())
	}
	json.Unmarshal(rr.Body.Bytes(), &st)
	if st["phase"] != "done" || st["wrong"] != float64(1) {
		t.Fatalf("session not done after the claim: %+v", st)
	}
	res := st["result"].(map[string]any)
	if res["tricks"] != float64(10) || res["scoreNs"] != float64(420) || res["claimed"] != true {
		t.Fatalf("bad result: %+v", res)
	}
	if len(st["plays"].([]any)) != 4 || st["hands"] == nil || st["line"] == nil {
		t.Fatalf("done state incomplete: %+v", st)
	}
	for _, pl := range st["line"].([]any) {
		m := pl.(map[string]any)
		if m["seat"] == float64(1) && m["msg"] != "the obvious lead" {
			t.Fatalf("review line lost the message: %+v", m)
		}
	}

	// sessions are user-isolated
	rr = httptest.NewRecorder()
	req = drillReq(t, "GET", "/api/drill/session/"+sid, "", alice)
	req.SetPathValue("sid", sid)
	handleDrillSessionGet(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Fatalf("foreign session get: %d", rr.Code)
	}

	// default message fallback for unannotated hero cards
	def := drillProblem()
	def.Plays[0].Msg = ""
	s, err := newDrillSession(&def, "alice")
	if err != nil {
		t.Fatalf("default-msg session: %v", err)
	}
	s.Wrong = 0
	scripted := s.Problem.Plays[0]
	if msg := defaultHeroMsg(scripted.Card); msg != "I just decided to play the 5♠ instead." {
		t.Fatalf("bad default message: %s", msg)
	}

	// delete: owner only
	rr = httptest.NewRecorder()
	req = drillReq(t, "DELETE", "/api/drills/"+id, "", bob)
	req.SetPathValue("id", id)
	handleDrillDelete(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("foreign delete: %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	req = drillReq(t, "DELETE", "/api/drills/"+id, "", alice)
	req.SetPathValue("id", id)
	handleDrillDelete(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("owner delete: %d", rr.Code)
	}
	if _, err := os.Stat(drillPath(id)); !os.IsNotExist(err) {
		t.Fatalf("problem file still there: %v", err)
	}
}

func TestDrillScoreHandler(t *testing.T) {
	token := drillLogin(t, "alice")
	rr := httptest.NewRecorder()
	handleDrillScore(rr, drillReq(t, "POST", "/api/drills/score",
		`{"level":4,"denom":"S","penalty":0,"declarer":0,"vuln":1,"tricks":11}`, token))
	if rr.Code != http.StatusOK {
		t.Fatalf("score: %d %s", rr.Code, rr.Body.String())
	}
	var out struct {
		ScoreNS int `json:"scoreNs"`
	}
	json.Unmarshal(rr.Body.Bytes(), &out)
	if out.ScoreNS != 650 {
		t.Fatalf("scoreNs = %d, want 650", out.ScoreNS)
	}
	rr = httptest.NewRecorder()
	handleDrillScore(rr, drillReq(t, "POST", "/api/drills/score",
		`{"level":8,"denom":"S","penalty":0,"declarer":0,"vuln":0,"tricks":11}`, token))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("bad level accepted: %d", rr.Code)
	}
}

func TestDrillLoadFromDisk(t *testing.T) {
	old := dataDir
	dataDir = t.TempDir()
	defer func() { dataDir = old }()
	t.Cleanup(func() {
		drillsMu.Lock()
		drills = map[string]*DrillProblem{}
		drillsMu.Unlock()
	})
	if err := os.MkdirAll(drillsDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	p := drillProblem()
	p.ID, p.Owner = "abc123", "alice"
	drillsMu.Lock()
	if err := saveDrillLocked(&p); err != nil {
		drillsMu.Unlock()
		t.Fatal(err)
	}
	drillsMu.Unlock()
	if _, err := os.Stat(filepath.Join(drillsDir(), "abc123.json")); err != nil {
		t.Fatal(err)
	}
	drillsMu.Lock()
	drills = map[string]*DrillProblem{}
	drillsMu.Unlock()
	initDrills()
	loaded := lookupDrill("abc123")
	if loaded == nil || loaded.Owner != "alice" || loaded.Role != "defense" {
		t.Fatalf("reload failed: %+v", loaded)
	}
}
