package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---- Play vs bots ----
//
// The human sits South at a table with three bots served by the bidding-dt
// sidecar (see bidding-dt/src/bidding_dt/play_server.py). Go owns the board
// state and the per-user history; the sidecar owns the model, deal generation,
// legality and double-dummy par scoring. Every finished board scores IMPs
// versus par from the N-S (human team) perspective.

const (
	defaultBotURL   = "http://127.0.0.1:8081"
	humanSeat       = 2  // South
	maxAuctionCalls = 24 // model sequence cap (MAX_CALLS in bidding-dt)
	playBoardTTL    = 6 * time.Hour
	playRecentMax   = 50

	botTimeout      = 30 * time.Second // deal / bid / legal
	botScoreTimeout = 90 * time.Second // score (cold double-dummy solve)
)

var (
	botURL    = defaultBotURL
	botModels = map[int]string{} // bot seat -> sidecar model name ("" = default)

	playMu     sync.Mutex // guards playBoards and playStats
	playBoards = map[string]*PlayBoard{}
	playStats  = map[string]PlayStats{}
)

type PlayContract struct {
	Level    int    `json:"level"`
	Denom    string `json:"denom"` // C D H S N
	Declarer int    `json:"declarer"`
	Penalty  int    `json:"penalty"` // 0 none, 1 doubled, 2 redoubled
}

func (c *PlayContract) String() string {
	if c == nil {
		return "passed out"
	}
	decl := "?"
	if c.Declarer >= 0 && c.Declarer < 4 {
		decl = string("NESW"[c.Declarer])
	}
	pen := ""
	if c.Penalty == 1 {
		pen = "X"
	} else if c.Penalty >= 2 {
		pen = "XX"
	}
	return fmt.Sprintf("%d%s%s by %s", c.Level, c.Denom, pen, decl)
}

type PlayResult struct {
	Contract *PlayContract `json:"contract"` // nil = passed out
	Tricks   *int          `json:"tricks"`   // double-dummy tricks for the contract
	ParNS    int           `json:"parNs"`
	ScoreNS  int           `json:"scoreNs"`
	Imps     float64       `json:"imps"` // human team (N-S) versus par
}

type PlayBoard struct {
	mu       sync.Mutex // guards the fields below
	ID       string
	User     string
	Hands    [4]string // N,E,S,W 'S.H.D.C'
	Dealer   int
	Vuln     int
	Calls    []string
	Legal    []string // legal calls for the human at LegalLen calls
	LegalLen int
	Result   *PlayResult
	Created  time.Time
}

type PlayRecord struct {
	Ts       string    `json:"ts"`
	Dealer   int       `json:"dealer"`
	Vuln     int       `json:"vuln"`
	Calls    []string  `json:"calls"`
	Contract string    `json:"contract"`
	Tricks   *int      `json:"tricks"`
	ParNS    int       `json:"parNs"`
	ScoreNS  int       `json:"scoreNs"`
	Imps     float64   `json:"imps"`
	Hands    [4]string `json:"hands,omitempty"` // all four hands, N,E,S,W
}

// PlayStats is the persisted per-user history (data/play_stats.json)
type PlayStats struct {
	Boards    int          `json:"boards"`
	ImpsTotal float64      `json:"impsTotal"`
	Recent    []PlayRecord `json:"recent"`
}

func initPlay() {
	if u := strings.TrimSpace(os.Getenv("BIDDING_DT_URL")); u != "" {
		botURL = strings.TrimSuffix(u, "/")
	}
	seatOf := map[string]int{"N": 0, "E": 1, "S": 2, "W": 3}
	for _, kv := range strings.Split(os.Getenv("PLAY_BOTS"), ",") {
		seat, model, ok := strings.Cut(strings.TrimSpace(kv), "=")
		if !ok {
			continue
		}
		if s, found := seatOf[strings.ToUpper(seat)]; found && s != humanSeat && model != "" {
			botModels[s] = model
		}
	}
	loadPlayStats()
	log.Printf("Play bots: sidecar %s, seat models %v", botURL, botModels)
}

func playStatsPath() string { return filepath.Join(dataDir, "play_stats.json") }

func loadPlayStats() {
	data, err := os.ReadFile(playStatsPath())
	if err != nil {
		return
	}
	var m map[string]PlayStats
	if err := json.Unmarshal(data, &m); err != nil {
		log.Printf("Could not parse play_stats.json: %v", err)
		return
	}
	playMu.Lock()
	playStats = m
	playMu.Unlock()
	log.Printf("Loaded play stats for %d user(s)", len(m))
}

// savePlayStatsLocked writes the history to disk. Callers must hold playMu.
func savePlayStatsLocked() error {
	return writeJSONFile(playStatsPath(), playStats)
}

func getPlayStats(username string) PlayStats {
	playMu.Lock()
	defer playMu.Unlock()
	return playStats[username]
}

// recordPlayResult folds one finished board into the user's history.
// Called with b.mu held (lock order: board -> playMu).
func recordPlayResult(b *PlayBoard) {
	playMu.Lock()
	defer playMu.Unlock()
	s := playStats[b.User]
	s.Boards++
	s.ImpsTotal += b.Result.Imps
	rec := PlayRecord{
		Ts:       time.Now().UTC().Format(time.RFC3339),
		Dealer:   b.Dealer,
		Vuln:     b.Vuln,
		Calls:    append([]string(nil), b.Calls...),
		Contract: b.Result.Contract.String(),
		Tricks:   b.Result.Tricks,
		ParNS:    b.Result.ParNS,
		ScoreNS:  b.Result.ScoreNS,
		Imps:     b.Result.Imps,
		Hands:    b.Hands,
	}
	s.Recent = append([]PlayRecord{rec}, s.Recent...)
	if len(s.Recent) > playRecentMax {
		s.Recent = s.Recent[:playRecentMax]
	}
	playStats[b.User] = s
	if err := savePlayStatsLocked(); err != nil {
		log.Printf("Could not persist play stats: %v", err)
	}
}

// ---- Sidecar client ----

func botPost(path string, body any, out any, timeout time.Duration) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, botURL+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("unreachable: %v", err)
	}
	defer resp.Body.Close()
	respData, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		msg := ""
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(respData, &e) == nil {
			msg = e.Error
		}
		if msg == "" {
			msg = fmt.Sprintf("sidecar returned %d", resp.StatusCode)
		}
		return errors.New(msg)
	}
	return json.Unmarshal(respData, out)
}

func botUnavailableMsg(err error) string {
	return "bot service error: " + err.Error()
}

// ---- Board logic (callers hold b.mu) ----

func callsOrEmpty(c []string) []string {
	if c == nil {
		return []string{}
	}
	return c
}

func (b *PlayBoard) nextSeat() int { return (b.Dealer + len(b.Calls)) % 4 }

func (b *PlayBoard) over() bool {
	n := len(b.Calls)
	return n >= 4 && b.Calls[n-1] == "P" && b.Calls[n-2] == "P" && b.Calls[n-3] == "P"
}

// advance runs the bots until it is the human's turn or the auction ends,
// then scores the finished board exactly once.
func (b *PlayBoard) advance() error {
	for !b.over() && len(b.Calls) < maxAuctionCalls && b.nextSeat() != humanSeat {
		body := map[string]any{
			"hands":  b.Hands,
			"dealer": b.Dealer,
			"vuln":   b.Vuln,
			"calls":  callsOrEmpty(b.Calls),
		}
		if m := botModels[b.nextSeat()]; m != "" {
			body["model"] = m
		}
		var out struct {
			Seat int    `json:"seat"`
			Call string `json:"call"`
		}
		if err := botPost("/bid", body, &out, botTimeout); err != nil {
			return err
		}
		if out.Seat != b.nextSeat() || normalizeCall(out.Call) == "" {
			return errors.New("sidecar returned an inconsistent bid")
		}
		b.Calls = append(b.Calls, out.Call)
	}
	if b.over() || len(b.Calls) >= maxAuctionCalls {
		return b.score()
	}
	return b.ensureLegal()
}

func (b *PlayBoard) score() error {
	if b.Result != nil {
		return nil
	}
	var out struct {
		Imps     float64       `json:"imps"`
		ParNS    int           `json:"par_ns"`
		ScoreNS  int           `json:"score_ns"`
		Tricks   *int          `json:"tricks"`
		Contract *PlayContract `json:"contract"`
	}
	body := map[string]any{
		"hands":  b.Hands,
		"dealer": b.Dealer,
		"vuln":   b.Vuln,
		"calls":  callsOrEmpty(b.Calls),
	}
	if err := botPost("/score", body, &out, botScoreTimeout); err != nil {
		return err
	}
	b.Result = &PlayResult{
		Contract: out.Contract, Tricks: out.Tricks,
		ParNS: out.ParNS, ScoreNS: out.ScoreNS, Imps: out.Imps,
	}
	recordPlayResult(b)
	return nil
}

func (b *PlayBoard) ensureLegal() error {
	if b.Result != nil {
		b.Legal = nil
		b.LegalLen = -1
		return nil
	}
	if b.Legal != nil && b.LegalLen == len(b.Calls) {
		return nil
	}
	var out struct {
		Seat  int      `json:"seat"`
		Over  bool     `json:"over"`
		Legal []string `json:"legal"`
	}
	body := map[string]any{"dealer": b.Dealer, "vuln": b.Vuln, "calls": callsOrEmpty(b.Calls)}
	if err := botPost("/legal", body, &out, botTimeout); err != nil {
		return err
	}
	b.Legal = out.Legal
	b.LegalLen = len(b.Calls)
	return nil
}

// stateLocked renders the client-visible board. Only the human's hand is
// exposed while the auction runs; all four are revealed once it is scored.
func (b *PlayBoard) stateLocked(stats PlayStats) map[string]any {
	st := map[string]any{
		"id":        b.ID,
		"dealer":    b.Dealer,
		"vuln":      b.Vuln,
		"humanSeat": humanSeat,
		"hand":      b.Hands[humanSeat],
		"calls":     callsOrEmpty(b.Calls),
		"nextSeat":  b.nextSeat(),
		"yourTurn":  b.Result == nil && b.nextSeat() == humanSeat,
		"done":      b.Result != nil,
		"bots":      botLabels(),
		"stats":     statsSummary(stats),
	}
	if b.Result != nil {
		st["result"] = b.Result
		st["hands"] = b.Hands
	} else if b.nextSeat() == humanSeat && b.Legal != nil && b.LegalLen == len(b.Calls) {
		st["legal"] = b.Legal
	}
	// Alerts from the model's self-play tree: what each bot bid promised and
	// what each of the human's legal calls would show. Both are omitted
	// until the tree finishes loading in the background.
	if t := alertTreeReady(); t != nil {
		if a := t.callAlerts(b.Dealer, b.Calls, 1<<humanSeat); a != nil {
			st["alerts"] = a
		}
		if st["legal"] != nil {
			if a := t.optionAlerts(b.Calls, b.Legal); a != nil {
				st["optionAlerts"] = a
			}
		}
	}
	return st
}

func botLabels() map[string]string {
	m := make(map[string]string, len(botModels))
	for seat, name := range botModels {
		m[strconv.Itoa(seat)] = name
	}
	return m
}

func statsSummary(s PlayStats) map[string]any {
	avg := 0.0
	if s.Boards > 0 {
		avg = s.ImpsTotal / float64(s.Boards)
	}
	recent := s.Recent
	if recent == nil {
		recent = []PlayRecord{}
	}
	return map[string]any{
		"boards":    s.Boards,
		"impsTotal": s.ImpsTotal,
		"avgImps":   avg,
		"recent":    recent,
	}
}

// normalizeCall maps user/bot input onto the canonical vocab tokens
// (P, X, XX, 1C..7N); returns "" when the input is not a call.
func normalizeCall(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	switch s {
	case "PASS":
		s = "P"
	case "DBL", "DOUBLE":
		s = "X"
	case "RDBL", "REDOUBLE":
		s = "XX"
	}
	if s == "P" || s == "X" || s == "XX" {
		return s
	}
	if len(s) == 3 && s[1:] == "NT" {
		s = s[:1] + "N"
	}
	if len(s) == 2 && s[0] >= '1' && s[0] <= '7' && strings.IndexByte("CDHSN", s[1]) >= 0 {
		return s
	}
	return ""
}

func lookupPlayBoard(id, username string) *PlayBoard {
	playMu.Lock()
	defer playMu.Unlock()
	b := playBoards[id]
	if b == nil || b.User != username {
		return nil
	}
	return b
}

func gcPlayBoardsLocked() {
	now := time.Now()
	for id, b := range playBoards {
		if now.Sub(b.Created) > playBoardTTL {
			delete(playBoards, id)
		}
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// ---- Handlers ----

// POST /api/play/new - deals a fresh board and runs the bots up to the human
func handlePlayNew(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	var deal struct {
		Hands  [4]string `json:"hands"`
		Dealer int       `json:"dealer"`
		Vuln   int       `json:"vuln"`
	}
	if err := botPost("/deal", struct{}{}, &deal, botTimeout); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	b := &PlayBoard{
		ID: generateID(), User: username, Hands: deal.Hands,
		Dealer: deal.Dealer, Vuln: deal.Vuln,
		LegalLen: -1, Created: time.Now(),
	}
	if err := b.advance(); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	playMu.Lock()
	gcPlayBoardsLocked()
	playBoards[b.ID] = b
	stats := playStats[username]
	playMu.Unlock()

	b.mu.Lock()
	st := b.stateLocked(stats)
	b.mu.Unlock()
	writeJSON(w, st)
}

// GET /api/play/{id} - current board state (re-advances stalled bot turns)
func handlePlayGet(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid board ID")
		return
	}
	b := lookupPlayBoard(id, username)
	if b == nil {
		jsonError(w, http.StatusNotFound, "no such board")
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Result == nil {
		if err := b.advance(); err != nil {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
			return
		}
	}
	writeJSON(w, b.stateLocked(getPlayStats(username)))
}

// POST /api/play/{id}/call - the human's call, then bots until the human is
// up again or the auction ends (scored and recorded once)
func handlePlayCall(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid board ID")
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Call string `json:"call"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	call := normalizeCall(body.Call)
	if call == "" {
		jsonError(w, http.StatusBadRequest, "invalid call")
		return
	}
	b := lookupPlayBoard(id, username)
	if b == nil {
		jsonError(w, http.StatusNotFound, "no such board")
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Result != nil {
		jsonError(w, http.StatusConflict, "board is finished")
		return
	}
	if b.nextSeat() != humanSeat {
		// recover a turn that stalled on a sidecar failure
		if err := b.advance(); err != nil {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
			return
		}
		if b.Result != nil {
			writeJSON(w, b.stateLocked(getPlayStats(username)))
			return
		}
	}
	if err := b.ensureLegal(); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	if !slices.Contains(b.Legal, call) {
		jsonError(w, http.StatusBadRequest, "illegal call: "+call)
		return
	}
	b.Calls = append(b.Calls, call)
	b.Legal = nil
	b.LegalLen = -1
	if err := b.advance(); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	writeJSON(w, b.stateLocked(getPlayStats(username)))
}

// GET /api/play/stats - the user's history vs the bots
func handlePlayStats(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	writeJSON(w, statsSummary(getPlayStats(username)))
}
