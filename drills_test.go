package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
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
// one scripted trick and a claim of 10 tricks (result shown as 4H=).
func drillProblem() DrillProblem {
	return DrillProblem{
		Title: "Lead problem", Role: "defense", HeroSeat: 1,
		Difficulty: 3, Dealer: 0, Vuln: 0,
		Hands:          drillHands,
		Calls:          []string{"1H", "P", "4H", "P", "P", "P"},
		BidComment:     "Standard opening.",
		ProblemComment: "Udca signals.",
		Lines: []DrillLine{{
			Plays: []DrillPlay{
				{Seat: 1, Card: "5S", Msg: "the obvious lead"},
				{Seat: 2, Card: "2S"},
				{Seat: 3, Card: "QS"},
				{Seat: 0, Card: "7S"},
			},
			Claim: &DrillClaim{Tricks: 10},
		}},
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

func TestValidateDrill(t *testing.T) {
	p := drillProblem()
	if _, err := validateDrill(&p); err != nil {
		t.Fatalf("valid problem rejected: %v", err)
	}
	if p.Lines[0].Plays[0].Card != "5S" {
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
	mutate("no plays", func(q *DrillProblem) { q.Lines[0].Plays = nil })
	mutate("out of turn", func(q *DrillProblem) {
		l := q.Lines[0].Plays
		l[0], l[1] = l[1], l[0]
	})
	mutate("card not held", func(q *DrillProblem) { q.Lines[0].Plays[1].Card = "AS" })
	mutate("revoke", func(q *DrillProblem) { q.Lines[0].Plays[1].Card = "AH" }) // S holds 2S and spades were led
	mutate("msg on non-hero seat", func(q *DrillProblem) { q.Lines[0].Plays[2].Msg = "why" })
	mutate("alts on non-hero seat", func(q *DrillProblem) { q.Lines[0].Plays[2].Alts = []string{"QC"} })
	mutate("alt not held", func(q *DrillProblem) { q.Lines[0].Plays[0].Alts = []string{"2S"} })
	mutate("alt is the card itself", func(q *DrillProblem) { q.Lines[0].Plays[0].Alts = []string{"5S"} })
	mutate("claim off trick boundary", func(q *DrillProblem) {
		q.Lines[0].Plays = q.Lines[0].Plays[:3]
	})
	mutate("claim tricks impossible", func(q *DrillProblem) { q.Lines[0].Claim.Tricks = 14 })
	mutate("claim tricks below won", func(q *DrillProblem) {
		q.Lines[0].Plays = naivePlayLine(drillHands, 1, 'H')[:8]
		q.Lines[0].Claim.Tricks = 0
	})
	mutate("no claim short line", func(q *DrillProblem) {
		q.Lines[0].Claim = nil
		q.Lines[0].Plays = naivePlayLine(drillHands, 1, 'H')[:8]
	})
	mutate("claim and bots together", func(q *DrillProblem) {
		q.Lines[0].Bot = &DrillBotEnd{}
	})
	mutate("bot line already 52", func(q *DrillProblem) {
		q.Lines[0].Claim = nil
		q.Lines[0].Plays = naivePlayLine(drillHands, 1, 'H')
		q.Lines[0].Bot = &DrillBotEnd{}
	})
	mutate("target out of range", func(q *DrillProblem) { q.Target = 14 })
	mutate("duplicate lines", func(q *DrillProblem) {
		q.Lines = append(q.Lines, q.Lines[0])
	})
	mutate("lines diverge on an auto seat", func(q *DrillProblem) {
		other := q.Lines[0]
		other.Plays = append([]DrillPlay(nil), q.Lines[0].Plays...)
		other.Plays[1] = DrillPlay{Seat: 2, Card: "8S"} // South is not the hero
		q.Lines = append(q.Lines, other)
	})

	// A bot-ended line: the script stops after one trick and the sidecar
	// plays the rest, graded against the problem's target.
	bot := drillProblem()
	bot.Lines[0].Claim = nil
	bot.Lines[0].Bot = &DrillBotEnd{}
	bot.Target = 3
	if _, err := validateDrill(&bot); err != nil {
		t.Fatalf("bot-ended problem rejected: %v", err)
	}

	// A declarer problem played out to the end needs no ending, and a
	// problem without a target inherits the count of its scripted line.
	full := drillProblem()
	full.Role, full.HeroSeat = "declarer", 0
	full.Lines = []DrillLine{{Plays: naivePlayLine(drillHands, 1, 'H')}}
	full.Target = 0
	c, err := validateDrill(&full)
	if err != nil {
		t.Fatalf("played-out declarer problem rejected: %v", err)
	}
	if len(full.Lines[0].Plays) != 52 {
		t.Fatalf("naive line has %d plays", len(full.Lines[0].Plays))
	}
	if _, decl, err := simulateLine(&full, c, full.Lines[0].Plays); err != nil {
		t.Fatalf("simulate: %v", err)
	} else if full.Target != decl {
		t.Fatalf("target not counted from the script: %d, want %d", full.Target, decl)
	}

	// The legacy single Plays+Claim shape still validates and migrates.
	legacy := drillProblem()
	legacy.Plays, legacy.Claim = legacy.Lines[0].Plays, legacy.Lines[0].Claim
	legacy.Lines = nil
	if _, err := validateDrill(&legacy); err != nil {
		t.Fatalf("legacy problem rejected: %v", err)
	}
	if len(legacy.Lines) != 1 || legacy.Lines[0].Claim == nil || legacy.Plays != nil {
		t.Fatalf("legacy shape not migrated: %+v", legacy)
	}
	// the target folds in from the legacy claim: 10 declarer tricks leave
	// the defending hero's side 3
	if legacy.Target != 3 {
		t.Fatalf("legacy target not migrated: %d", legacy.Target)
	}
}

// twoTrickEquivProblem: trick 1 E leads a spade (any of J6543 -- West's queen
// wins regardless), North takes it... no: West's QS wins trick 1; trick 2 has
// West on lead. Used by the equivalence tests.
func twoTrickEquivProblem() DrillProblem {
	p := drillProblem()
	p.Lines = []DrillLine{{
		Plays: []DrillPlay{
			{Seat: 1, Card: "3S", Msg: "fourth best", Alts: []string{"JS", "6S", "5S", "4S"}},
			{Seat: 2, Card: "2S"},
			{Seat: 3, Card: "QS"},
			{Seat: 0, Card: "7S"},
			// West won; a diamond trick where East's 8D loses to South's TD
			{Seat: 3, Card: "5D"},
			{Seat: 0, Card: "2D"},
			{Seat: 1, Card: "8D", Msg: "keep the king"},
			{Seat: 2, Card: "TD"},
		},
		Claim: &DrillClaim{Tricks: 10},
	}}
	return p
}

func TestDrillEquivalence(t *testing.T) {
	p := twoTrickEquivProblem()
	if _, err := validateDrill(&p); err != nil {
		t.Fatalf("equivalent leads rejected: %v", err)
	}
	// The 8D/4S class stays sound: swapping the diamond order keeps South's
	// ten winning trick 2. But declaring the KING of diamonds equivalent to
	// the eight moves the trick-2 winner from South to East -- rejected.
	bad := twoTrickEquivProblem()
	bad.Lines[0].Plays[6].Alts = []string{"KD"}
	if _, err := validateDrill(&bad); err == nil {
		t.Fatal("a trick-winning swap accepted as equivalent")
	} else if !strings.Contains(err.Error(), "interchangeable") {
		t.Fatalf("unexpected error: %v", err)
	}
	// A cross-suit "alternative" breaks follow-suit: rejected.
	bad2 := twoTrickEquivProblem()
	bad2.Lines[0].Plays[6].Alts = []string{"5H"}
	if _, err := validateDrill(&bad2); err == nil {
		t.Fatal("a revoke-by-swap accepted as equivalent")
	}

	// Session: the hero swaps within the class on the lead, and the class
	// keeps covering the LATER spade play with the members still held.
	s, err := newDrillSession(&p, "alice")
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if !s.lineAccepts(s.Live[0], 1, "6S") || !s.lineAccepts(s.Live[0], 1, "JS") ||
		s.lineAccepts(s.Live[0], 1, "7H") || s.lineAccepts(s.Live[0], 1, "2S") {
		t.Fatal("class acceptance wrong at the lead")
	}
	// hero plays the 6S instead of the scripted 3S
	s.applyLocked(DrillPlay{Seat: 1, Card: "6S"})
	s.advanceLocked() // S 2S, W QS, N 7S -- West wins and leads the diamond
	if len(s.Plays) != 6 || s.Done {
		t.Fatalf("expected 6 plays up to East's diamond, got %d (done=%v)", len(s.Plays), s.Done)
	}
	// East's scripted 8D has no alts of its own, but it never entered a class:
	// only the spades are interchangeable.
	if s.lineAccepts(s.Live[0], 1, "KD") {
		t.Fatal("an undeclared equivalent was accepted")
	}
	if !s.lineAccepts(s.Live[0], 1, "8D") {
		t.Fatal("the scripted card must be accepted")
	}
	s.applyLocked(DrillPlay{Seat: 1, Card: "8D"})
	s.advanceLocked() // S TD wins; the script is over -> the claim ends it
	if !s.Done || s.Result == nil || !s.Result.Claimed || s.Result.Tricks != 10 {
		t.Fatalf("claim did not end the hand: done=%v result=%+v", s.Done, s.Result)
	}

	// The user's exact scenario: after swapping the lead, the LATER spade
	// play accepts every class member still held. Trick 1: E 3S, S 2S, W QS,
	// N AS (North wins). Trick 2: N TS, E 4S, S 8S, W 3D. The class
	// {3S,6S,5S,4S} survives both positions; the jack may NOT join it --
	// played at trick 2 it would beat the ten and move the winner.
	p2 := drillProblem()
	p2.Lines = []DrillLine{{
		Plays: []DrillPlay{
			{Seat: 1, Card: "3S", Msg: "lead", Alts: []string{"6S", "5S", "4S"}},
			{Seat: 2, Card: "2S"},
			{Seat: 3, Card: "QS"},
			{Seat: 0, Card: "AS"},
			{Seat: 0, Card: "TS"},
			{Seat: 1, Card: "4S", Msg: "second spade"},
			{Seat: 2, Card: "8S"},
			{Seat: 3, Card: "6D"},
		},
		Claim: &DrillClaim{Tricks: 10},
	}}
	if _, err := validateDrill(&p2); err != nil {
		t.Fatalf("second-spade line rejected: %v", err)
	}
	// Adding the jack to the class must be rejected: (5S at trick 1, JS at
	// trick 2) is a legal combination that moves the trick-2 winner to East.
	badP2 := drillProblem()
	badP2.Lines = []DrillLine{{Plays: append([]DrillPlay(nil), p2.Lines[0].Plays...), Claim: &DrillClaim{Tricks: 10}}}
	badP2.Lines[0].Plays[0].Alts = []string{"JS", "6S", "5S", "4S"}
	if _, err := validateDrill(&badP2); err == nil {
		t.Fatal("a class whose later swap moves a trick winner was accepted")
	}

	s2, err := newDrillSession(&p2, "alice")
	if err != nil {
		t.Fatalf("session 2: %v", err)
	}
	s2.applyLocked(DrillPlay{Seat: 1, Card: "5S"}) // swap the lead
	s2.advanceLocked()                             // S 2S, W QS, N AS; N leads TS
	if len(s2.Plays) != 5 || s2.nextSeatLocked() != 1 {
		t.Fatalf("bad position: %d plays, next %d", len(s2.Plays), s2.nextSeatLocked())
	}
	// East holds J,6,4,3 of spades now: every remaining class member is
	// accepted for the scripted 4S; the out-of-class jack is not.
	for _, c := range []string{"6S", "4S", "3S"} {
		if !s2.lineAccepts(s2.Live[0], 1, c) {
			t.Fatalf("%s should still be an acceptable spade", c)
		}
	}
	if s2.lineAccepts(s2.Live[0], 1, "JS") {
		t.Fatal("the jack was never declared equivalent")
	}
	s2.applyLocked(DrillPlay{Seat: 1, Card: "6S"})
	s2.advanceLocked()
	if !s2.Done || s2.Result.Tricks != 10 || !s2.Result.Claimed {
		t.Fatalf("hand should end on the claim: %+v", s2.Result)
	}
}

func TestDrillMultipleLines(t *testing.T) {
	// Line 1: spade lead (West wins the trick). Line 2: club lead, East's ace
	// wins -- a genuinely different continuation, so two lines are required.
	p := drillProblem()
	p.Lines = []DrillLine{
		{Plays: []DrillPlay{
			{Seat: 1, Card: "3S", Msg: "attack spades"},
			{Seat: 2, Card: "2S"}, {Seat: 3, Card: "QS"}, {Seat: 0, Card: "7S"},
		}, Claim: &DrillClaim{Tricks: 9}},
		{Plays: []DrillPlay{
			{Seat: 1, Card: "AC", Msg: "cash the ace"},
			{Seat: 2, Card: "2C"}, {Seat: 3, Card: "3C"}, {Seat: 0, Card: "8C"},
		}, Claim: &DrillClaim{Tricks: 11}},
	}
	if _, err := validateDrill(&p); err != nil {
		t.Fatalf("two-line problem rejected: %v", err)
	}
	s, err := newDrillSession(&p, "alice")
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if len(s.Live) != 2 || s.Problem.Lines[0].Plays[0].Card != "3S" {
		t.Fatalf("both lines should start alive: %+v", s.Live)
	}
	// Neither line's card: refused, hint from the first live line.
	if s.lineAccepts(s.Live[0], 1, "7H") || s.lineAccepts(s.Live[1], 1, "7H") {
		t.Fatal("7H must not be accepted by either line")
	}
	// The club lead matches line 2 only.
	if s.lineAccepts(0, 1, "AC") {
		t.Fatal("line 1 must not accept the club lead")
	}
	if !s.lineAccepts(1, 1, "AC") {
		t.Fatal("line 2 must accept the club lead")
	}
	s.Live = []int{1}
	s.applyLocked(DrillPlay{Seat: 1, Card: "AC"})
	s.advanceLocked()
	if !s.Done || s.Result == nil || s.Result.Tricks != 11 || !s.Result.Claimed {
		t.Fatalf("line 2's claim should decide the result: %+v", s.Result)
	}
	// The review follows the applied cards with the FOLLOWED line's message.
	st := s.stateLocked()
	line := st["line"].([]map[string]any)
	if line[0]["card"] != "AC" || line[0]["msg"] != "cash the ace" {
		t.Fatalf("bad review line: %+v", line[0])
	}
}

func TestDrillSessionPlayedOut(t *testing.T) {
	p := drillProblem()
	p.Role, p.HeroSeat = "declarer", 0
	p.Lines = []DrillLine{{Plays: naivePlayLine(drillHands, 1, 'H')}}
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
			s.applyLocked(s.Problem.Lines[0].Plays[len(s.Plays)])
		}
		if err := s.advanceLocked(); err != nil {
			t.Fatalf("advance: %v", err)
		}
	}
	if s.Result == nil || s.Result.Claimed || s.Result.Botted {
		t.Fatalf("bad result: %+v", s.Result)
	}
	if s.Result.Tricks != s.DeclTricks {
		t.Fatalf("result tricks %d != counted %d", s.Result.Tricks, s.DeclTricks)
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
	if res["tricks"] != float64(10) || res["claimed"] != true {
		t.Fatalf("bad result: %+v", res)
	}
	if _, has := res["scoreNs"]; has {
		t.Fatalf("result must not carry a score: %+v", res)
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
	def.Lines[0].Plays[0].Msg = ""
	s, err := newDrillSession(&def, "alice")
	if err != nil {
		t.Fatalf("default-msg session: %v", err)
	}
	s.Wrong = 0
	scripted := s.Problem.Lines[0].Plays[0]
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

// ---- Bot-finished lines ----

type fakePlayBot struct {
	calls int32
	drops int32
}

// newFakePlayServer mimics the bidding-dt sidecar's /play/choose with a
// deterministic legal-card picker (highest card of the led suit, else the
// first holding) and counts /play/drop calls.
func newFakePlayServer(t *testing.T) (*httptest.Server, *fakePlayBot) {
	t.Helper()
	fb := &fakePlayBot{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/play/choose":
			atomic.AddInt32(&fb.calls, 1)
			var body struct {
				Known map[string]string `json:"known"`
				Plays [][]any           `json:"plays"`
				ToAct int               `json:"to_act"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), 400)
				return
			}
			played := map[string]bool{}
			for _, pl := range body.Plays {
				if len(pl) == 2 {
					if seat, ok := pl[0].(float64); ok && int(seat) == body.ToAct {
						if c, ok := pl[1].(string); ok {
							played[c] = true
						}
					}
				}
			}
			var rem []string
			for _, c := range handCards(body.Known[strconv.Itoa(body.ToAct)]) {
				if !played[c] {
					rem = append(rem, c)
				}
			}
			if len(rem) == 0 {
				http.Error(w, "no cards left", 500)
				return
			}
			card := rem[0]
			if len(body.Plays)%4 != 0 && len(body.Plays) > 0 {
				if lead, ok := body.Plays[(len(body.Plays)/4)*4][1].(string); ok && len(lead) == 2 {
					for _, c := range rem {
						if cardSuit(c) == cardSuit(lead) {
							card = c
							break
						}
					}
				}
			}
			writeJSON(w, map[string]any{"seat": body.ToAct, "card": card})
		case "/play/drop":
			atomic.AddInt32(&fb.drops, 1)
			writeJSON(w, map[string]any{})
		default:
			http.Error(w, "unexpected path "+r.URL.Path, 404)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, fb
}

// botProblem scripts one trick (West's queen wins) and hands the rest to the
// bots, graded against the problem's target (defenders' tricks).
func botProblem(target int) DrillProblem {
	p := drillProblem()
	p.Target = target
	p.Lines = []DrillLine{{
		Plays: []DrillPlay{
			{Seat: 1, Card: "5S", Msg: "the obvious lead"},
			{Seat: 2, Card: "2S"},
			{Seat: 3, Card: "QS"},
			{Seat: 0, Card: "7S"},
		},
		Bot: &DrillBotEnd{},
	}}
	return p
}

func TestDrillBotFinish(t *testing.T) {
	srv, fb := newFakePlayServer(t)
	oldURL := botURL
	botURL = srv.URL
	defer func() { botURL = oldURL }()

	p := botProblem(10)
	s, err := newDrillSession(&p, "alice")
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if s.Bot || len(s.Plays) != 0 || !s.heroControls(s.nextSeatLocked()) {
		t.Fatalf("session should start at the hero's lead: bot=%v plays=%d", s.Bot, len(s.Plays))
	}
	// the target is revealed before the first card, not when the hand ends
	if st0 := s.stateLocked(); st0["target"] != 10 {
		t.Fatalf("target not revealed up front: %v", st0["target"])
	}
	// The hero plays the scripted lead; the remaining scripted cards play
	// themselves and the bots take over -- one bot card per advance call.
	s.applyLocked(DrillPlay{Seat: 1, Card: "5S"})
	if err := s.advanceLocked(); err != nil {
		t.Fatalf("advance into the bot phase: %v", err)
	}
	if !s.Bot || s.Done {
		t.Fatalf("bots should have taken over: bot=%v done=%v plays=%d", s.Bot, s.Done, len(s.Plays))
	}
	if atomic.LoadInt32(&fb.calls) == 0 {
		t.Fatal("the sidecar was never consulted")
	}
	if len(s.Plays) != 5 {
		t.Fatalf("expected the scripted trick plus exactly ONE bot card, got %d plays", len(s.Plays))
	}
	before := len(s.Plays)
	if err := s.advanceLocked(); err != nil {
		t.Fatalf("second bot advance: %v", err)
	}
	if len(s.Plays) != before+1 {
		t.Fatalf("bot phase must advance one card per call: %d -> %d", before, len(s.Plays))
	}
	// the hero may play ANY legal card from here; drive the hand out
	for !s.Done {
		seat := s.nextSeatLocked()
		if s.heroControls(seat) {
			legal := s.legalCardsLocked(seat)
			if len(legal) == 0 {
				t.Fatal("hero has no legal card")
			}
			s.applyLocked(DrillPlay{Seat: seat, Card: legal[len(legal)-1]})
			if len(s.Plays) == 52 {
				s.finishBotLocked()
				break
			}
			continue
		}
		if err := s.advanceLocked(); err != nil {
			t.Fatalf("bot advance: %v", err)
		}
	}
	if len(s.Plays) != 52 {
		t.Fatalf("hand played %d cards", len(s.Plays))
	}
	r := s.Result
	if r == nil || !r.Botted || r.Claimed {
		t.Fatalf("bad bot result: %+v", r)
	}
	if r.Tricks != s.DeclTricks || r.Target != 10 ||
		r.Success != (s.Problem.heroSide(s.DeclTricks) >= 10) {
		t.Fatalf("bad grading: %+v (decl %d)", r, s.DeclTricks)
	}
	if atomic.LoadInt32(&fb.drops) == 0 {
		t.Fatal("the sidecar sessions were never dropped")
	}

	// the same deterministic bots replayed with the target set to what the
	// hero's side actually took grade the hand a success
	p2 := botProblem(s.Problem.heroSide(s.DeclTricks))
	s2, err := newDrillSession(&p2, "alice")
	if err != nil {
		t.Fatalf("session 2: %v", err)
	}
	s2.applyLocked(DrillPlay{Seat: 1, Card: "5S"})
	for !s2.Done {
		seat := s2.nextSeatLocked()
		if s2.heroControls(seat) {
			legal := s2.legalCardsLocked(seat)
			s2.applyLocked(DrillPlay{Seat: seat, Card: legal[len(legal)-1]})
			if len(s2.Plays) == 52 {
				s2.finishBotLocked()
				break
			}
			continue
		}
		if err := s2.advanceLocked(); err != nil {
			t.Fatalf("bot advance 2: %v", err)
		}
	}
	if s2.DeclTricks != s.DeclTricks || s2.Result == nil || !s2.Result.Success {
		t.Fatalf("replay should succeed deterministically: decl %d vs %d, result %+v", s2.DeclTricks, s.DeclTricks, s2.Result)
	}
	// the target is public information in the session state, from the start
	st := s2.stateLocked()
	if st["bot"] != true || st["target"] != s2.Problem.Target {
		t.Fatalf("bot phase state wrong: bot=%v target=%v", st["bot"], st["target"])
	}
}

func TestDrillAllowWrong(t *testing.T) {
	srv, fb := newFakePlayServer(t)
	oldURL := botURL
	botURL = srv.URL
	defer func() { botURL = oldURL }()

	p := drillProblem()
	p.AllowWrong = true
	s, err := newDrillSession(&p, "alice")
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	// West (hero) on lead strays from the scripted 5S: the card stands and
	// the bots take over instead of the pick being refused
	if err := s.strayLocked(1, "3S"); err != nil {
		t.Fatalf("stray: %v", err)
	}
	if !s.Bot || s.Wrong != 1 || len(s.Plays) < 2 {
		t.Fatalf("stray should hand the hand to the bots: bot=%v wrong=%d plays=%d", s.Bot, s.Wrong, len(s.Plays))
	}
	if atomic.LoadInt32(&fb.calls) == 0 {
		t.Fatal("the sidecar was never consulted after the stray")
	}
	// drive the hand out; the grade is against the target, not a script
	for !s.Done {
		seat := s.nextSeatLocked()
		if s.heroControls(seat) {
			legal := s.legalCardsLocked(seat)
			s.applyLocked(DrillPlay{Seat: seat, Card: legal[len(legal)-1]})
			if len(s.Plays) == 52 {
				s.finishBotLocked()
				break
			}
			continue
		}
		if err := s.advanceLocked(); err != nil {
			t.Fatalf("bot advance: %v", err)
		}
	}
	r := s.Result
	if r == nil || !r.Botted || r.Target != s.Problem.Target || r.Target == 0 ||
		r.Success != (s.Problem.heroSide(r.Tricks) >= r.Target) {
		t.Fatalf("bad stray grading: %+v", r)
	}
	// the strayed card carries no script message in the review
	st := s.stateLocked()
	line, _ := st["line"].([]map[string]any)
	if len(line) > 0 {
		if _, ok := line[0]["msg"]; ok {
			t.Fatalf("strayed card carries a script message: %+v", line[0])
		}
	}
}

func TestDrillRating(t *testing.T) {
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
	body, _ := json.Marshal(drillProblem())
	rr := httptest.NewRecorder()
	handleDrillCreate(rr, drillReq(t, "POST", "/api/drills", string(body), alice))
	var created struct {
		ID string `json:"id"`
	}
	json.Unmarshal(rr.Body.Bytes(), &created)
	id := created.ID

	rate := func(tok, payload string) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		req := drillReq(t, "POST", "/api/drills/"+id+"/rate", payload, tok)
		req.SetPathValue("id", id)
		handleDrillRate(rr, req)
		return rr
	}

	// a session that has NOT finished cannot rate yet
	rr = httptest.NewRecorder()
	req := drillReq(t, "POST", "/api/drills/"+id+"/session", "", bob)
	req.SetPathValue("id", id)
	handleDrillSessionNew(rr, req)
	var st map[string]any
	json.Unmarshal(rr.Body.Bytes(), &st)
	sid := st["id"].(string)
	t.Cleanup(func() {
		drillSessMu.Lock()
		delete(drillSess, sid)
		drillSessMu.Unlock()
	})
	if rr := rate(bob, `{"stars":5,"sessionId":"`+sid+`"}`); rr.Code != http.StatusForbidden {
		t.Fatalf("rating an unfinished run: %d", rr.Code)
	}
	// unknown session
	if rr := rate(bob, `{"stars":5,"sessionId":"nope"}`); rr.Code != http.StatusForbidden {
		t.Fatalf("rating with a foreign session: %d", rr.Code)
	}
	// bad stars
	card := func(c string) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		req := drillReq(t, "POST", "/api/drill/session/"+sid+"/card", `{"card":"`+c+`"}`, bob)
		req.SetPathValue("sid", sid)
		handleDrillSessionCard(rr, req)
		return rr
	}
	// finish the hand
	if rr := card("5S"); rr.Code != http.StatusOK {
		t.Fatalf("finishing play: %d %s", rr.Code, rr.Body.String())
	}
	if rr := rate(bob, `{"stars":9,"sessionId":"`+sid+`"}`); rr.Code != http.StatusBadRequest {
		t.Fatalf("stars out of range: %d", rr.Code)
	}
	// the author cannot rate their own problem
	if rr := rate(alice, `{"stars":5,"sessionId":"`+sid+`"}`); rr.Code != http.StatusForbidden {
		t.Fatalf("author self-rating: %d", rr.Code)
	}
	// bob rates 4 stars
	rr = rate(bob, `{"stars":4,"sessionId":"`+sid+`"}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("rate: %d %s", rr.Code, rr.Body.String())
	}
	var out struct {
		Rating     float64 `json:"rating"`
		Ratings    int     `json:"ratings"`
		YourRating int     `json:"yourRating"`
	}
	json.Unmarshal(rr.Body.Bytes(), &out)
	if out.Rating != 4 || out.Ratings != 1 || out.YourRating != 4 {
		t.Fatalf("bad rating response: %+v", out)
	}
	// the done state now carries the live summary
	rr = httptest.NewRecorder()
	req = drillReq(t, "GET", "/api/drill/session/"+sid, "", bob)
	req.SetPathValue("sid", sid)
	handleDrillSessionGet(rr, req)
	json.Unmarshal(rr.Body.Bytes(), &st)
	if st["rating"] != float64(4) || st["ratings"] != float64(1) || st["yourRating"] != float64(4) {
		t.Fatalf("done state missing rating: %+v", st)
	}
	// re-rating replaces, it does not stack
	rr = rate(bob, `{"stars":2,"sessionId":"`+sid+`"}`)
	json.Unmarshal(rr.Body.Bytes(), &out)
	if out.Rating != 2 || out.Ratings != 1 {
		t.Fatalf("re-rating should replace: %+v", out)
	}
	// the library sorts by rating and carries yourRating for the caller
	p2 := drillProblem()
	p2.Title = "Second"
	b2, _ := json.Marshal(p2)
	rr = httptest.NewRecorder()
	handleDrillCreate(rr, drillReq(t, "POST", "/api/drills", string(b2), alice))
	var c2 struct {
		ID string `json:"id"`
	}
	json.Unmarshal(rr.Body.Bytes(), &c2)
	rr = httptest.NewRecorder()
	handleDrillsList(rr, drillReq(t, "GET", "/api/drills", "", bob))
	var listed struct {
		Problems []map[string]any `json:"problems"`
	}
	json.Unmarshal(rr.Body.Bytes(), &listed)
	if len(listed.Problems) != 2 {
		t.Fatalf("want 2 problems, got %d", len(listed.Problems))
	}
	first := listed.Problems[0]
	if first["id"] != id || first["rating"] != float64(2) || first["yourRating"] != float64(2) {
		t.Fatalf("rated problem should sort first: %+v", first)
	}
	if listed.Problems[1]["rating"] != float64(0) || listed.Problems[1]["ratings"] != float64(0) {
		t.Fatalf("unrated problem second: %+v", listed.Problems[1])
	}
	// no ending spoilers in the metadata
	for _, leak := range []string{"claimed", `"bot"`} {
		if strings.Contains(rr.Body.String(), leak) {
			t.Fatalf("library metadata leaks %s", leak)
		}
	}
	// editing the problem preserves its ratings
	edited := drillProblem()
	edited.Title = "Renamed"
	eb, _ := json.Marshal(edited)
	rr = httptest.NewRecorder()
	req = drillReq(t, "PUT", "/api/drills/"+id, string(eb), alice)
	req.SetPathValue("id", id)
	handleDrillPut(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("put: %d", rr.Code)
	}
	if p := lookupDrill(id); p.Ratings["bob"] != 2 {
		t.Fatalf("ratings lost on edit: %+v", p.Ratings)
	}
}
